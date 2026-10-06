"""
FiqhQA backend — one FastAPI server for all four website tabs
=============================================================

The website never talks to the models directly. The browser calls api/ask.php,
which forwards the request here (with the secret token), and this server runs the
same code as the four experiment notebooks:

  tab                    system       model    notebook reproduced here
  ---------------------  -----------  -------  -----------------------------------------
  Gemini Baseline        baseline     gemini   No RAG - Gemini_original 10 JAN.ipynb
  SILMA Baseline         baseline     silma    No RAG - Silma 10 JAN.ipynb
  FiqhQA-RAG + SILMA     fiqhqa-rag   silma    Good version_Silma_10 JAN.ipynb
  FiqhQA-RAG + Gemini    fiqhqa-rag   gemini   Good version_Gemini.ipynb

POST /ask    { "question": str, "model": "gemini"|"silma", "system": "baseline"|"fiqhqa-rag" }
GET  /health { ok, ready, models: {...} }

Run:   pip install -r requirements.txt
       export GEMINI_API_KEY=...  BACKEND_TOKEN=...  DATA_PATH=/path/LastRAG_dataset.csv
       uvicorn main:app --host 0.0.0.0 --port 8000
SILMA needs a GPU (a free Colab T4 is enough). See run_backend_colab.ipynb.
"""
from __future__ import annotations

import gc
import math
import os
import re
import threading
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration (environment variables). Defaults reproduce the notebooks.
# ---------------------------------------------------------------------------
env = os.getenv
DATA_PATH = env("DATA_PATH", "LastRAG_dataset.csv")
BACKEND_TOKEN = env("BACKEND_TOKEN", "")                 # shared secret with api/config.php
GEMINI_API_KEY = env("GEMINI_API_KEY", "") or env("GOOGLE_API_KEY", "")

# Models — exactly as in the notebooks
GEMINI_BASELINE_MODEL = env("GEMINI_BASELINE_MODEL", "gemini-2.5-flash")       # No RAG - Gemini
GEMINI_RAG_MODEL = env("GEMINI_RAG_MODEL", "gemini-2.5-flash-lite")            # Good version_Gemini
SILMA_MODEL = env("SILMA_MODEL", "silma-ai/SILMA-Kashif-2B-Instruct-v1.0")     # both SILMA notebooks
EMBED_MODEL_NAME = env("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
RERANKER_NAME = env("RERANKER_NAME", "oddadmix/arabic-reranker")

# "notebook" = Good version_Gemini.ipynb as written (BM25 top-5 + dense top-5, no RRF/reranker)
# "fiqhqa"   = the full FiqhQA pipeline (shared with the SILMA tab) + Gemini
GEMINI_RAG_PIPELINE = env("GEMINI_RAG_PIPELINE", "notebook")

# "improved" (default, for the live site) or "notebook" (exact notebook behaviour, to reproduce the comparison experiments)
RAG_MODE = env("RAG_MODE", "improved")
INDEX_SCOPE = env("INDEX_SCOPE", "all")                   # improved mode: "all" rows (live demo) or "train" (evaluation)
GEN_TOP_K = int(env("GEN_TOP_K", "5"))                    # passages given to the generator (after reranking)
GEMINI_V2_MODEL = env("GEMINI_V2_MODEL", "gemini-2.5-flash")
SILMA_V2_MAX_INPUT_TOKENS = int(env("SILMA_V2_MAX_INPUT_TOKENS", "2048"))
SILMA_V2_MAX_NEW_TOKENS = int(env("SILMA_V2_MAX_NEW_TOKENS", "400"))
NO_ANSWER = "لا يوجد في النصوص المسترجعة ما يجيب عن هذا السؤال"
# Speed (improved mode): rerank only the best RRF candidates, with shorter inputs; cache repeated questions
RERANK_CANDIDATES = int(env("RERANK_CANDIDATES", "20"))
RERANK_MAX_LEN = int(env("RERANK_MAX_LEN", "384"))
CACHE_SIZE = int(env("CACHE_SIZE", "500"))

TEST_SIZE, RANDOM_STATE = 0.2, 42
TOP_K_DENSE = TOP_K_BM25 = 30
TOP_K_FINAL = 10
RRF_K = 60
K_SIMPLE = 5                                             # Gemini notebook: K_BM25 = K_DENSE = 5
SILMA_MAX_NEW_TOKENS = int(env("SILMA_MAX_NEW_TOKENS", "512"))
SILMA_MAX_INPUT_TOKENS = int(env("SILMA_MAX_INPUT_TOKENS", "512"))  # notebook: max_length(1024) - max_new_tokens(512)
SILMA_NUM_BEAMS = int(env("SILMA_NUM_BEAMS", "4"))       # notebook: 4. Use 1 on a CPU-only host (much faster)
# Abstain + refer to a specialist when the best passage is weakly related (challenge track 01 success criterion).
# The default is a starting point; calibrate it with eval/evaluate.py --calibrate on the real reranker.
MIN_GROUNDING = float(env("MIN_GROUNDING", "0"))   # legacy gate on sigmoid(score); saturates with this reranker
# Preferred gate: the RAW reranker score of the best passage (set from eval/evaluate.py --calibrate).
# The reranker gives very large scores, so sigmoid(score) is ~1.0 even for unrelated passages.
_mrs = env("MIN_RERANK_SCORE", "")
MIN_RERANK_SCORE = float(_mrs) if _mrs.strip() else None
RERANK_SCORE_MAX = float(env("RERANK_SCORE_MAX", "12.34"))  # saturation score of the reranker (see calibration)
RATE_LIMIT_PER_MIN = int(env("RATE_LIMIT_PER_MIN", "20"))  # per client IP; protects the Gemini quota during judging
LOAD_SILMA = env("LOAD_SILMA", "1") == "1"               # set 0 to run Gemini tabs only (no GPU)
FAKE_MODELS = env("FIQHQA_FAKE_MODELS", "0") == "1"      # offline smoke test: tiny stand-in models

MODEL_NAMES = {
    ("baseline", "gemini"): GEMINI_BASELINE_MODEL,
    ("baseline", "silma"): SILMA_MODEL.split("/")[-1],
    ("fiqhqa-rag", "silma"): SILMA_MODEL.split("/")[-1],
    ("fiqhqa-rag", "gemini"): GEMINI_V2_MODEL if RAG_MODE == "improved" else
                              (GEMINI_RAG_MODEL if GEMINI_RAG_PIPELINE == "notebook" else GEMINI_BASELINE_MODEL),
}


def family(model: str) -> str:
    """Accept the site's old ids too ('gemini-2.5-flash', 'silma-9b-instruct')."""
    m = model.lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("silma"):
        return "silma"
    raise HTTPException(400, f"unknown model: {model}")


# ---------------------------------------------------------------------------
# Arabic normalisation — copied from the notebooks
# ---------------------------------------------------------------------------
def normalize_arabic(text) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[ؗ-ًؚ-ْ]", "", text)
    text = re.sub("[إأآٱا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = text.replace("ـ", "")
    return re.sub(r"\s+", " ", text).strip()


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()


# ---------------------------------------------------------------------------
# Stand-in models for FIQHQA_FAKE_MODELS=1 (no downloads, CPU, for testing only)
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    dim = 512

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, **_):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            t = " " + normalize_arabic(t) + " "
            for j in range(len(t) - 2):
                out[i, hash(t[j:j + 3]) % self.dim] += 1.0
        return out


def _fake_rerank_scores(query, texts):
    q = set(normalize_arabic(query).split())
    return [4.0 * len(q & set(normalize_arabic(t).split())) / (len(q) or 1) - 2.0 for t in texts]


# ---------------------------------------------------------------------------
# Data + indices
# ---------------------------------------------------------------------------
class Doc(dict):
    """{text, book, source, ref, id} plus ranking info added during retrieval."""


class State:
    ready = False
    error = ""
    device = "cpu"
    embedder = None
    rr_tok = rr_model = None
    gen_tok = gen_model = None
    silma_lock = threading.Lock()
    # FiqhQA hybrid index (SILMA notebook): train split only, raw text
    train_docs: list = []
    faiss_l2 = None
    bm25_train = None
    # Gemini notebook index: all rows (dropna), normalised text
    all_docs: list = []
    faiss_ip = None
    bm25_all = None
    # Improved index: question + answer, E5 prefixes, cosine similarity, normalised BM25 tokens
    v2_docs: list = []
    faiss_v2 = None
    bm25_v2 = None


S = State()


def _row_doc(row, text) -> Doc:
    return Doc(text=text, book=_s(row.get("book")), source=_s(row.get("Source Updated (auto)")),
               ref=_s(row.get("Q# & Doc#")), id=_s(row.get("id")))


def _unit(x):
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


_PUNCT = re.compile(r"[^\u0621-\u064A0-9a-zA-Z\s]")
_STOP = set("في من على عن الى إلى او أو و ما ماذا هل كم مع ذلك هذا هذه التي الذي ان أن".split())


def bm25_tokens(text):
    """Normalised, punctuation-free tokens (the notebook used raw str.split())."""
    t = _PUNCT.sub(" ", normalize_arabic(text))
    return [w for w in t.split() if len(w) > 1 and w not in _STOP]


def load_everything():
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    if FAKE_MODELS:
        S.device = "cpu"
        S.embedder = _FakeEmbedder()
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        S.device = "cuda" if torch.cuda.is_available() else "cpu"
        if S.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 2)
        S.embedder = SentenceTransformer(EMBED_MODEL_NAME, device=S.device)
        S.rr_tok = AutoTokenizer.from_pretrained(RERANKER_NAME)
        S.rr_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME).to(S.device).eval()
        if LOAD_SILMA:
            from transformers import AutoModelForCausalLM

            S.gen_tok = AutoTokenizer.from_pretrained(SILMA_MODEL)
            S.gen_model = AutoModelForCausalLM.from_pretrained(
                SILMA_MODEL, torch_dtype=torch.float16 if S.device == "cuda" else torch.float32
            ).to(S.device).eval()

    # --- Dataset: local file, or a PRIVATE Hugging Face dataset repo (keeps the data out of the public GitHub repo)
    path = DATA_PATH
    if not os.path.exists(path) and env("DATA_HF_REPO"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(env("DATA_HF_REPO"), env("DATA_HF_FILE", "LastRAG_dataset.csv"),
                               repo_type="dataset", token=env("HF_TOKEN") or None)

    # Support reading from both CSV and Excel spreadsheet files
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # --- Good version_Silma: split, embed raw train answers, FAISS L2, BM25 on raw text
    df.columns = df.columns.str.strip()
    if RAG_MODE == "notebook":
        _build_notebook_indices(df)
    _build_improved_index(df)
    S.ready = True
    print(f"[FiqhQA] ready in {time.time() - t0:.0f}s on {S.device}: "
          f"{len(S.train_docs)} train docs, {len(S.all_docs)} docs (all), {len(S.v2_docs)} improved-index docs, "
          f"RAG_MODE={RAG_MODE}, SILMA={'on' if S.gen_model or FAKE_MODELS else 'off'}")


def _build_notebook_indices(df):
    """Indices of the original notebooks (only needed for RAG_MODE=notebook)."""
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.model_selection import train_test_split
    train_df, _ = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True)
    S.train_docs = [_row_doc(r, str(r["answer"])) for _, r in train_df.iterrows()]
    texts = [d["text"] for d in S.train_docs]
    emb = np.asarray(S.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype="float32")
    S.faiss_l2 = faiss.IndexFlatL2(emb.shape[1])
    S.faiss_l2.add(emb)
    S.bm25_train = BM25Okapi([t.split() for t in texts])

    # --- Good version_Gemini: dropna, normalise, index ALL answers, FAISS inner product
    data = df.dropna().reset_index(drop=True)
    S.all_docs = [_row_doc(r, normalize_arabic(r["answer"])) for _, r in data.iterrows()]
    texts = [d["text"] for d in S.all_docs]
    emb = np.asarray(S.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype="float32")
    S.faiss_ip = faiss.IndexFlatIP(emb.shape[1])
    S.faiss_ip.add(emb)
    S.bm25_all = BM25Okapi([t.split() for t in texts])



def _build_improved_index(df):
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.model_selection import train_test_split
    rows = df.dropna(subset=["question", "answer"])
    if INDEX_SCOPE == "train":
        rows, _ = train_test_split(rows, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    S.v2_docs = []
    for _, r in rows.reset_index(drop=True).iterrows():
        d = _row_doc(r, str(r["answer"]).strip())
        d["question"] = str(r["question"]).strip()
        d["rr_text"] = d["question"] + "\n" + d["text"]           # what the reranker sees
        S.v2_docs.append(d)
    passages = ["passage: " + normalize_arabic(d["rr_text"]) for d in S.v2_docs]
    emb = _unit(S.embedder.encode(passages, convert_to_numpy=True, show_progress_bar=False))
    S.faiss_v2 = faiss.IndexFlatIP(emb.shape[1])
    S.faiss_v2.add(emb)
    S.bm25_v2 = BM25Okapi([bm25_tokens(d["rr_text"]) for d in S.v2_docs])


# ---------------------------------------------------------------------------
# Retrieval — FiqhQA hybrid (Good version_Silma cells 9-14)
# ---------------------------------------------------------------------------
def dense_search(query, k=TOP_K_DENSE):
    q = np.asarray(S.embedder.encode([query], convert_to_numpy=True), dtype="float32")
    _, idx = S.faiss_l2.search(q, k)
    return [Doc(S.train_docs[int(i)], rank_dense=r) for r, i in enumerate(idx[0], 1) if i >= 0]


def bm25_search(query, k=TOP_K_BM25):
    scores = S.bm25_train.get_scores(query.split())
    top = np.argsort(scores)[::-1][:k]
    return [Doc(S.train_docs[int(i)], rank_bm25=r, bm25_score=float(scores[int(i)])) for r, i in enumerate(top, 1)]


def reciprocal_rank_fusion(dense_docs, sparse_docs, k=RRF_K):
    fused = {}
    for lst, key in ((dense_docs, "rank_dense"), (sparse_docs, "rank_bm25")):
        for rank, d in enumerate(lst, 1):
            _id = d["text"][:64]                       # same document id as the notebook
            e = fused.setdefault(_id, Doc(d, rrf=0.0))
            e["rrf"] += 1.0 / (k + rank)
            e[key] = d[key]
    return sorted(fused.values(), key=lambda d: d["rrf"], reverse=True)


def rerank(query, candidates, max_length=512):
    if not candidates:
        return []
    texts = [c.get("rr_text", c["text"]) for c in candidates]
    if FAKE_MODELS:
        scores = _fake_rerank_scores(query, texts)
    else:
        import torch

        enc = S.rr_tok([query] * len(texts), texts, truncation=True, padding=True,
                       return_tensors="pt", max_length=max_length)
        enc = {k: v.to(S.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = S.rr_model(**enc).logits
        scores = (logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 1]).float().cpu().tolist()
    for c, s in zip(candidates, scores):
        c["rerank"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank"], reverse=True)


def hybrid_retrieve(query):
    return rerank(query, reciprocal_rank_fusion(dense_search(query), bm25_search(query)))[:TOP_K_FINAL]


# Retrieval — Good version_Gemini (BM25 top-5 + dense top-5, de-duplicated)
def simple_retrieve(query):
    q = normalize_arabic(query)
    scores = S.bm25_all.get_scores(q.split())
    bm = [int(i) for i in np.argsort(scores)[::-1][:K_SIMPLE]]
    emb = np.asarray(S.embedder.encode([q], convert_to_numpy=True), dtype="float32")
    _, idx = S.faiss_ip.search(emb, K_SIMPLE)
    dn = [int(i) for i in idx[0] if i >= 0]
    out, seen = [], set()
    for i in bm + dn:
        if S.all_docs[i]["text"] in seen:
            continue
        seen.add(S.all_docs[i]["text"])
        out.append(Doc(S.all_docs[i], rank_bm25=(bm.index(i) + 1) if i in bm else None,
                       rank_dense=(dn.index(i) + 1) if i in dn else None))
    return out


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
# Fallbacks, tried in order when the main model is busy, out of quota or retired (404).
# gemini-2.0-flash was shut down on 1 June 2026, so it is no longer in the list.
GEMINI_FALLBACK = [m.strip() for m in env("GEMINI_FALLBACK_MODELS",
                   "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-2.5-flash-lite").split(",") if m.strip()]
_RETRY_STATUS = {429, 500, 502, 503, 504}               # rate limit + "model overloaded / high demand"
_gem_local = threading.local()


def _gen_config_for(model, generation_config):
    cfg = dict(generation_config or {})
    if not model.startswith("gemini-2.5"):
        cfg.pop("thinkingConfig", None)                  # thinkingBudget is a 2.5 setting
    if model.startswith("gemini-3"):
        cfg.pop("temperature", None)                     # Google recommends the default temperature for Gemini 3
        if "maxOutputTokens" in cfg:                     # Gemini 3 thinks by default; thinking counts toward this limit
            cfg["maxOutputTokens"] = max(cfg["maxOutputTokens"], 4096)
    return cfg


def grounding_meter(sc):
    """Meter for the site. The reranker saturates: its highest score is ~RERANK_SCORE_MAX (12.34 for
    oddadmix/arabic-reranker, measured in calibration.csv), and a verbatim match reaches it.
    With a threshold: threshold -> 50%, saturation -> 100%, below the threshold falls towards 0%."""
    thr = MIN_RERANK_SCORE if MIN_RERANK_SCORE is not None else RERANK_SCORE_MAX - 4
    span = max(0.5, RERANK_SCORE_MAX - thr)
    if sc >= thr:
        g = 0.5 + 0.5 * min(1.0, (sc - thr) / span)
    else:
        g = 0.5 * math.exp(-(thr - sc) / span)
    return round(g, 3)


def gemini_generate(model_name, prompt, generation_config=None, max_retries=3):
    """Call Gemini; retry busy errors, then fall back to the next model in GEMINI_FALLBACK_MODELS.
    A model that is retired (404) or out of quota is skipped at once. Only a bad/missing key stops everything.
    The model that actually answered is stored in _gem_local.used (shown on the site)."""
    if FAKE_MODELS:
        _gem_local.used = model_name
        return f"[FAKE {model_name}] " + prompt[-300:]
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not set on the backend")
    errors = []
    for m in [model_name] + [f for f in GEMINI_FALLBACK if f != model_name]:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        cfg = _gen_config_for(m, generation_config)
        if cfg:
            body["generationConfig"] = cfg
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        err = "no response"
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json=body, timeout=120,
                                  headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"})
            except requests.RequestException as e:
                err = f"connection error: {e}"
                time.sleep(2 ** attempt)
                continue
            d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 200:
                parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                _gem_local.used = m
                if m != model_name:
                    print(f"[FiqhQA] {model_name} unavailable -> answered with fallback {m}", flush=True)
                return text or "[No output – filtered by Gemini safety system]"
            msg = d.get("error", {}).get("message", "") or f"HTTP {r.status_code}"
            err = f"HTTP {r.status_code}: {msg[:200]}"
            low = msg.lower()
            if r.status_code in (401, 403) or "api key" in low or "api_key" in low:
                print(f"[FiqhQA] Gemini key rejected: {err}", flush=True)
                raise HTTPException(502, "Gemini API key rejected: " + msg[:200])
            if r.status_code not in _RETRY_STATUS or "quota" in low or "per day" in low:
                break                                    # retired model / daily quota: try the next model now
            time.sleep(min(20, 2 * 2 ** attempt))        # 2, 4, 8 s
        print(f"[FiqhQA] Gemini {m} failed: {err}", flush=True)
        errors.append(f"{m} → {err}")
    raise HTTPException(503, "Gemini unavailable: " + " | ".join(errors))


def _silma_generate_ids(prompt_ids, **gen_kwargs):
    import torch

    with S.silma_lock, torch.no_grad():                  # one GPU, one generation at a time
        ids = torch.tensor([prompt_ids], device=S.device)
        out = S.gen_model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   pad_token_id=S.gen_tok.eos_token_id,
                                   eos_token_id=S.gen_tok.eos_token_id, **gen_kwargs)
        new = out[0][len(prompt_ids):]
        text = S.gen_tok.decode(new, skip_special_tokens=True)
    if S.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return text


def _need_silma():
    if FAKE_MODELS:
        return
    if S.gen_model is None:
        raise HTTPException(503, "SILMA is not loaded on this backend (LOAD_SILMA=0 or no GPU)")


def silma_baseline(question):
    """No RAG - Silma 10 JAN.ipynb, cell 8."""
    _need_silma()
    prompt = f"سؤال: {question}\nالإجابة:"
    if FAKE_MODELS:
        return "[FAKE SILMA baseline] " + question
    ids = S.gen_tok(prompt)["input_ids"]
    text = _silma_generate_ids(ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    return text.split("الإجابة:")[-1].strip()


SILMA_INSTRUCTION = (
    "أنت فقيه وعالم متخصص في الشريعة الإسلامية. "
    "أجب باللغة العربية الفصحى فقط من غير أي كلمات أجنبية أو ترجمة، "
    "واستخدم أسلوبًا علميًا مختصرًا وواضحًا يعتمد فقط على النصوص الفقهية التالية. "
    "اذكر المرجع (الكتاب والمصدر) في نهاية الإجابة داخل قوسين.\n\n"
    "النصوص الفقهية:\n"
)
OPENING = "بسم الله الرحمن الرحيم، الحمد لله، أما بعد، "


def build_fiqhqa_prompt(query, docs, budget_tokens=None, tokenizer=None):
    """Prompt of Good version_Silma cell 13 (original FiqhQA prompt).

    Fix vs. the notebook: the notebook truncated the *end* of the prompt to 512 tokens, which
    could cut off the question itself. Here the question and answer trigger are always kept and
    the retrieved passages are trimmed (lowest-ranked first) to fit the same token budget."""
    tail = f"\n\nالسؤال: {query}\n==============================\nالجواب التفصيلي:\n{OPENING}"
    blocks = [f"📖 من {d['book']} — {d['source']}:\n{d['text']}" for d in docs]
    if not tokenizer or not budget_tokens:
        return SILMA_INSTRUCTION + "\n\n".join(blocks) + tail, len(blocks)
    n = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])
    room = budget_tokens - n(SILMA_INSTRUCTION) - n(tail) - 2
    kept = []
    for b in blocks:
        cost = n(b + "\n\n")
        if cost <= room:
            kept.append(b)
            room -= cost
        else:
            if room > 40:                                 # keep a partial passage if it is meaningful
                ids = tokenizer(b, add_special_tokens=False)["input_ids"][: room - 4]
                kept.append(tokenizer.decode(ids))
            break
    return SILMA_INSTRUCTION + "\n\n".join(kept) + tail, len(kept)


def citations_text(docs):
    seen, cites = set(), []
    for d in docs:
        c = f"({d['book']} – {d['source']})"
        if (d["book"] or d["source"]) and c not in seen:
            seen.add(c)
            cites.append(c)
    return ("المراجع الفقهية المعتمدة:\n" + "، ".join(cites)) if cites else ""


def silma_rag(question, docs):
    """Good version_Silma cell 13 (generation + post-processing)."""
    _need_silma()
    if FAKE_MODELS:
        text = "[FAKE SILMA RAG] " + docs[0]["text"][:200]
    else:
        prompt, _ = build_fiqhqa_prompt(question, docs, SILMA_MAX_INPUT_TOKENS, S.gen_tok)
        ids = S.gen_tok(prompt)["input_ids"]
        text = OPENING + _silma_generate_ids(
            ids, max_new_tokens=SILMA_MAX_NEW_TOKENS, temperature=0.6, top_p=0.9, do_sample=True,
            num_beams=SILMA_NUM_BEAMS, repetition_penalty=1.8, no_repeat_ngram_size=4)
    text = re.sub(r"[A-Za-z]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    cites = citations_text(docs)
    return text + ("\n\n" + cites if cites else "")


def gemini_rag(question, docs):
    if GEMINI_RAG_PIPELINE == "notebook":                 # Good version_Gemini: generate_with_retry
        context = "\n".join(d["text"] for d in docs)
        prompt = f"السؤال: {normalize_arabic(question)}\n\nالسياق:\n{context}\n\nالإجابة:"
        return gemini_generate(GEMINI_RAG_MODEL, prompt)
    prompt, _ = build_fiqhqa_prompt(question, docs)        # original FiqhQA prompt and generation settings
    answer = gemini_generate(GEMINI_BASELINE_MODEL, prompt, {
        "temperature": 0.6, "topP": 0.9, "maxOutputTokens": 512, "thinkingConfig": {"thinkingBudget": 0}})
    cites = citations_text(docs)
    return answer + ("\n\n" + cites if cites else "")


# ---------------------------------------------------------------------------
# Improved RAG (RAG_MODE=improved)
# ---------------------------------------------------------------------------
def v2_retrieve(query):
    qn = normalize_arabic(query)
    qe = _unit(S.embedder.encode(["query: " + qn], convert_to_numpy=True))
    _, idx = S.faiss_v2.search(qe, TOP_K_DENSE)
    dense = [Doc(S.v2_docs[int(i)], rank_dense=r) for r, i in enumerate(idx[0], 1) if i >= 0]
    scores = S.bm25_v2.get_scores(bm25_tokens(query))
    top = np.argsort(scores)[::-1][:TOP_K_BM25]
    sparse = [Doc(S.v2_docs[int(i)], rank_bm25=r) for r, i in enumerate(top, 1) if scores[int(i)] > 0]
    fused = {}
    for lst, key in ((dense, "rank_dense"), (sparse, "rank_bm25")):
        for rank, d in enumerate(lst, 1):
            e = fused.setdefault(d["id"] or d["text"][:64], Doc(d, rrf=0.0))
            e["rrf"] += 1.0 / (RRF_K + rank)
            e[key] = d[key]
    ranked = sorted(fused.values(), key=lambda d: d["rrf"], reverse=True)
    return rerank(query, ranked[:RERANK_CANDIDATES], max_length=RERANK_MAX_LEN)[:TOP_K_FINAL]


V2_INSTRUCTION = (
    "أنت باحث متخصص في الفقه الإسلامي. أجب عن السؤال اعتمادًا على النصوص الفقهية المرقّمة أدناه فقط، "
    "دون أي معلومة من خارجها.\n"
    "- اكتب بالعربية الفصحى بإيجاز ووضوح: اذكر الحكم أولًا ثم دليله كما ورد في النصوص، "
    "وانقل الآيات والأحاديث بنصها دون تغيير.\n"
    "- بعد كل حكم ضع رقم النص الذي استندت إليه بين معقوفين، مثل [1].\n"
    "- إن كانت في النصوص أقوال متعددة فاذكرها كما وردت.\n"
    f"- إن لم تتضمن النصوص جوابًا عن السؤال فاكتب هذه الجملة فقط: {NO_ANSWER}.\n\n"
)


def v2_prompt(question, docs, budget_tokens=None, tokenizer=None):
    blocks = [f"[{i}] ({d['book']} — {d['source'] or d['book']})\nس: {d.get('question', '')}\nج: {d['text']}"
              for i, d in enumerate(docs, 1)]
    tail = f"\n\nالسؤال: {question}\nالجواب:"
    if not tokenizer or not budget_tokens:
        return V2_INSTRUCTION + "النصوص:\n" + "\n\n".join(blocks) + tail
    n = lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"])
    room = budget_tokens - n(V2_INSTRUCTION + "النصوص:\n" + tail) - 32
    kept = []
    for b in blocks:                                       # keep whole passages, best-ranked first
        c = n(b + "\n\n")
        if c > room:
            break
        kept.append(b); room -= c
    return V2_INSTRUCTION + "النصوص:\n" + "\n\n".join(kept or blocks[:1]) + tail


def v2_references(answer, docs):
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer) if 1 <= int(m) <= len(docs)}) or [1]
    lines = [f"[{i}] {docs[i-1]['book']} – {docs[i-1]['source'] or docs[i-1]['book']}"
             + (f" ({docs[i-1]['ref']})" if docs[i-1].get("ref") else "") for i in cited]
    return "المراجع:\n" + "\n".join(lines), cited


def silma_rag_v2(question, docs):
    _need_silma()
    if FAKE_MODELS:
        return "[FAKE SILMA v2] " + docs[0]["text"][:200] + " [1]"
    prompt = v2_prompt(question, docs, SILMA_V2_MAX_INPUT_TOKENS, S.gen_tok)
    chat = S.gen_tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    ids = S.gen_tok(chat, add_special_tokens=False)["input_ids"]
    # deterministic decoding: no sampling, mild repetition penalty (1.8 distorted quoted hadith text)
    return _silma_generate_ids(ids, max_new_tokens=SILMA_V2_MAX_NEW_TOKENS, do_sample=False,
                               num_beams=1, repetition_penalty=1.1).strip()


def gemini_rag_v2(question, docs):
    return gemini_generate(GEMINI_V2_MODEL, v2_prompt(question, docs), {
        "temperature": 0.2, "topP": 0.9, "maxOutputTokens": 1024, "thinkingConfig": {"thinkingBudget": 0}})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="FiqhQA API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=env("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


PREWARM_QUESTIONS = [            # the example buttons on the website (script.js EXAMPLES + out-of-scope)
    "كم مدة المسح على الخفين والعمامة والخمار؟ مع الدليل",
    "ما الدليل على تحريم أواني الذهب والفضة؟",
    "ما حكم استعمال آنية الكفار وثيابهم؟",
    "ما هي الأحكام الشرعية الخمسة؟",
    "ما هو الفقه لغة وشرعا؟",
    "ما حكم تداول العملات الرقمية؟",
]


def _prewarm():
    """Answer the website's example questions once (Gemini tabs; SILMA too if on GPU) so clicks are instant."""
    if env("PREWARM", "1") != "1":
        return
    combos = [("fiqhqa-rag", "gemini"), ("baseline", "gemini")]
    if S.device == "cuda":
        combos += [("fiqhqa-rag", "silma"), ("baseline", "silma")]
    n = 0
    for system, model in combos:
        for q in PREWARM_QUESTIONS:
            try:
                ask(AskIn(question=q, model=model, system=system), None, BACKEND_TOKEN or None, "prewarm")
                n += 1
            except Exception as e:
                print(f"[FiqhQA] prewarm skipped ({system}/{model}): {e}")
    print(f"[FiqhQA] prewarm done: {n} example answers cached")


@app.on_event("startup")
def _startup():
    def run():
        try:
            load_everything()
            _prewarm()
        except Exception as e:                            # surfaced by /health
            S.error = f"{type(e).__name__}: {e}"
            print("[FiqhQA] startup failed:", S.error)
    threading.Thread(target=run, daemon=True).start()


class AskIn(BaseModel):
    question: str = Field(min_length=4, max_length=500)
    model: str = "gemini"
    system: Literal["baseline", "fiqhqa-rag"] = "fiqhqa-rag"
    retrieve_only: bool = False          # skip generation (used by eval/evaluate.py --calibrate)


def _check_token(token):
    if BACKEND_TOKEN and token != BACKEND_TOKEN:
        raise HTTPException(401, "bad or missing X-API-Key")


_cache: dict = {}                                       # repeated questions answer instantly
_cache_lock = threading.Lock()
_hits: dict = {}
_hits_lock = threading.Lock()


def _rate_limit(ip):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    with _hits_lock:
        q = [t for t in _hits.get(ip, []) if now - t < 60]
        if len(q) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(429, "عدد كبير من الطلبات؛ انتظر دقيقة ثم أعد المحاولة")
        q.append(now)
        _hits[ip] = q


REFERRAL = ("لم يجد النظام في المصدر نصًا فقهيًا كافيًا للإجابة عن هذا السؤال، لذلك امتنع عن الإجابة. "
            "يُرجى الرجوع إلى أهل العلم المختصين أو الجهات الشرعية الرسمية.")


@app.get("/")
def root():
    return {"service": "FiqhQA API", "ready": S.ready, "endpoints": ["GET /health", "POST /ask"]}


@app.get("/health")
def health():
    return {"ok": not S.error, "ready": S.ready, "error": S.error or None, "device": S.device,
            "fake_models": FAKE_MODELS, "gemini_key": bool(GEMINI_API_KEY),
            "silma_loaded": bool(S.gen_model) or FAKE_MODELS,
            "gemini_rag_pipeline": GEMINI_RAG_PIPELINE, "min_grounding": MIN_GROUNDING,
            "min_rerank_score": MIN_RERANK_SCORE,
            "rag_mode": RAG_MODE, "index_scope": INDEX_SCOPE, "gen_top_k": GEN_TOP_K,
            "models": {f"{s}/{m}": n for (s, m), n in MODEL_NAMES.items()}}


def _source(d):
    return {"text": d["text"], "book_title": d["book"], "fiqh_source": d["source"] or d["book"],
            "ref": " | ".join(x for x in (d.get("id"), d.get("ref")) if x),
            "rank_bm25": d.get("rank_bm25"), "rank_dense": d.get("rank_dense"),
            "rrf": d.get("rrf"), "rerank": d.get("rerank")}


@app.post("/ask")
def ask(body: AskIn, request: Request, x_api_key: Optional[str] = Header(default=None),
        x_forwarded_for: Optional[str] = Header(default=None)):
    _check_token(x_api_key)
    _rate_limit((x_forwarded_for or "").split(",")[0].strip() or (request.client.host if request.client else "?"))
    if S.error:
        raise HTTPException(500, "backend failed to start: " + S.error)
    if not S.ready:
        raise HTTPException(503, "models are still loading — try again in a minute")
    t0 = time.perf_counter()
    _gem_local.used = None
    fam = family(body.model)
    q = body.question.strip()
    ckey = (body.system, fam, normalize_arabic(q))
    if CACHE_SIZE and not body.retrieve_only and ckey in _cache:
        hit = dict(_cache[ckey]); hit["cached"] = True
        hit["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return hit
    out = {"system": body.system, "model": fam, "model_name": MODEL_NAMES[(body.system, fam)],
           "citation": None, "sources": [], "grounding": None}

    if body.system == "baseline":
        out["answer"] = gemini_generate(GEMINI_BASELINE_MODEL, f"سؤال: {q}\nالإجابة:") if fam == "gemini" \
            else silma_baseline(q)
        out["status"] = "ok"
    else:
        if RAG_MODE == "improved":
            docs = v2_retrieve(q)
            pipeline = (f"improved: question+answer index ({INDEX_SCOPE}), BM25 + E5 (query/passage) → RRF → "
                        f"arabic-reranker → top-{GEN_TOP_K} to the generator")
        elif fam == "gemini" and GEMINI_RAG_PIPELINE == "notebook":
            docs, pipeline = simple_retrieve(q), "BM25 top-5 + E5 top-5 (Good version_Gemini)"
        else:
            docs, pipeline = hybrid_retrieve(q), "BM25 + E5 → RRF(k=60) → arabic-reranker top-10"
        out["pipeline"] = pipeline
        if docs and docs[0].get("rerank") is None:
            # Gemini notebook pipeline has no reranker: score its passages (order and prompt unchanged)
            # only so the abstain-and-refer gate and the grounding meter work for this tab too.
            scored = rerank(q, [Doc(d) for d in docs])
            best = scored[0]
            for d in docs:
                d["rerank"] = next(s["rerank"] for s in scored if s["text"] == d["text"])
            top = best
        else:
            top = docs[0] if docs else None
        if top is not None and top.get("rerank") is not None:
            sc = top["rerank"]
            out["top_score"] = round(sc, 3)                                   # raw score, for calibration
            out["grounding"] = grounding_meter(sc)
        if MIN_RERANK_SCORE is not None:
            weak = top is not None and top.get("rerank") is not None and top["rerank"] < MIN_RERANK_SCORE
        else:
            weak = out["grounding"] is not None and MIN_GROUNDING and out["grounding"] < MIN_GROUNDING
        if body.retrieve_only:
            out.update(status="retrieved", answer="", sources=[_source(d) for d in docs])
        elif not docs or weak:
            # FiqhQA abstention + track 01: abstain and refer to a specialist
            out.update(status="retrieval_failure", answer="", referral=REFERRAL,
                       nearest=[_source(d) for d in docs[:3]])
        elif RAG_MODE == "improved":
            gen_docs = docs[:GEN_TOP_K]
            ans = silma_rag_v2(q, gen_docs) if fam == "silma" else gemini_rag_v2(q, gen_docs)
            if NO_ANSWER in ans or "لا يوجد في النصوص" in ans[:80]:
                # second safety layer: the generator itself found no answer in the passages
                out.update(status="retrieval_failure", answer="", referral=REFERRAL, llm_abstained=True,
                           nearest=[_source(d) for d in docs[:3]])
            else:
                refs, cited = v2_references(ans, gen_docs)
                main_doc = gen_docs[cited[0] - 1]
                out.update(status="ok", answer=ans + "\n\n" + refs,
                           citation={"book_title": main_doc["book"], "fiqh_source": main_doc["source"] or main_doc["book"]},
                           sources=[_source(d) for d in gen_docs])
        else:
            out["answer"] = silma_rag(q, docs) if fam == "silma" else gemini_rag(q, docs)
            out["status"] = "ok"
            out["citation"] = {"book_title": top["book"], "fiqh_source": top["source"] or top["book"]}
            out["sources"] = [_source(d) for d in docs]
    used = getattr(_gem_local, "used", None)
    if fam == "gemini" and used and used != out["model_name"]:
        out["model_name"] = f"{used} (fallback: {out['model_name']} busy)"
    _gem_local.used = None
    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    if CACHE_SIZE and not body.retrieve_only and out.get("status") in ("ok", "retrieval_failure"):
        with _cache_lock:
            _cache[ckey] = out
            while len(_cache) > CACHE_SIZE:
                _cache.pop(next(iter(_cache)))
    return out
