"""
FiqhQA — fair re-evaluation through the live API (the same code the website uses)
=================================================================================
Fixes the problems found in the original result CSVs:
  * API errors (e.g. Gemini 429) are retried and NEVER scored as answers; they are reported separately.
  * Every system answers the SAME held-out test questions (80/20 split, random_state=42).
  * The FiqhQA-RAG index contains only the training answers (see backend: hybrid index = train split).
  * Only the model's answer is scored (the appended citation list is removed first).
  * Abstention is measured: in-scope questions (should answer) and out-of-scope questions (should refuse).

Usage
  python evaluate.py --url http://localhost:8000 --data LastRAG_dataset.csv --n 100            # answers + abstention
  python evaluate.py --url ... --data ... --calibrate                                          # suggest MIN_RERANK_SCORE
  python evaluate.py --url ... --data ... --n 100 --metrics                                    # + BERTScore/ROUGE/BLEU/...
Run the server with RATE_LIMIT_PER_MIN=0 while evaluating. Outputs go to ./eval_out/ (one CSV per run + summary.json).
Fair comparison: start the server with INDEX_SCOPE=train so the test answers are NOT in the retrieval index.
Compare RAG_MODE=notebook (original experiment pipeline) vs RAG_MODE=improved by running this script once against each.
"""
import argparse, json, os, re, time
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

SYSTEMS = [("baseline", "gemini"), ("baseline", "silma"), ("fiqhqa-rag", "silma"), ("fiqhqa-rag", "gemini")]

# Questions outside the scope of al-As'ilah wa al-Ajwibah al-Fiqhiyyah (modern issues / non-fiqh).
# The RAG systems should abstain and refer the user to a specialist.
OUT_OF_SCOPE = [
    "ما حكم تداول العملات الرقمية مثل البيتكوين؟",
    "ما حكم التأمين التجاري على السيارات؟",
    "ما حكم أطفال الأنابيب؟",
    "ما حكم التبرع بالأعضاء بعد الوفاة الدماغية؟",
    "ما حكم بيع الأسهم في الشركات المختلطة؟",
    "ما حكم استخدام الذكاء الاصطناعي في إصدار الفتاوى؟",
    "ما حكم التصوير الفوتوغرافي بالهاتف المحمول؟",
    "ما حكم بطاقات الائتمان التي تفرض فوائد على التأخير؟",
    "ما حكم الصلاة في الطائرة إذا لم يمكن استقبال القبلة؟",
    "ما حكم الاستنساخ البشري؟",
    "كيف أطبخ الكبسة بالدجاج؟",
    "من فاز بكأس العالم لكرة القدم عام 2022؟",
    "ما عاصمة اليابان؟",
    "كيف أتعلم لغة بايثون؟",
    "ما أفضل هاتف ذكي في السوق حاليًا؟",
]


def strip_citations(t):
    return re.split(r"المراجع الفقهية المعتمدة", str(t))[0].strip()


def ask(url, token, q, system, model, retrieve_only=False, tries=6):
    headers = {"X-API-Key": token} if token else {}
    for a in range(tries):
        try:
            r = requests.post(url.rstrip("/") + "/ask", headers=headers, timeout=300,
                              json={"question": q, "system": system, "model": model, "retrieve_only": retrieve_only})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(90, 10 * 2 ** a)); continue
            d = r.json()
            if r.status_code != 200:
                return {"status": "api_error", "error": str(d.get("detail") or d)}
            if str(d.get("answer", "")).startswith(("[Error", "[No output")):
                time.sleep(min(90, 10 * 2 ** a)); continue
            return d
        except Exception as e:
            err = str(e); time.sleep(min(90, 10 * 2 ** a))
    return {"status": "api_error", "error": "failed after retries"}


def test_questions(data_path, n):
    df = pd.read_csv(data_path); df.columns = df.columns.str.strip()
    _, test = train_test_split(df, test_size=0.2, random_state=42)
    test = test.dropna(subset=["question", "answer"])
    return test.sample(n=min(n, len(test)), random_state=42).reset_index(drop=True)


def calibrate(args):
    """Suggest MIN_RERANK_SCORE: the raw reranker score that best separates in-scope test questions
    (should be answered) from out-of-scope questions (should be declined). Retrieval only, no generation."""
    test = test_questions(args.data, args.n)
    rows = []
    for system, model in [s for s in SYSTEMS if s[0] == "fiqhqa-rag"][:1]:   # same retriever for both RAG tabs
        for q, label in [(q, 1) for q in test.question] + [(q, 0) for q in OUT_OF_SCOPE]:
            d = ask(args.url, args.token, q, system, model, retrieve_only=True)
            rows.append({"in_scope": label, "top_score": d.get("top_score"), "question": q})
    df = pd.DataFrame(rows).dropna(subset=["top_score"])
    os.makedirs(args.out, exist_ok=True); df.to_csv(f"{args.out}/calibration.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("in_scope").top_score.describe().round(2).rename(index={0: "out-of-scope", 1: "in-scope"}))
    cands = sorted(set(df.top_score.round(2)))
    def bal_acc(t):
        pred = df.top_score >= t
        return ((pred & (df.in_scope == 1)).sum() / max(1, (df.in_scope == 1).sum())
                + (~pred & (df.in_scope == 0)).sum() / max(1, (df.in_scope == 0).sum())) / 2
    best = max(cands, key=bal_acc)
    ins = df[df.in_scope == 1].top_score; oos = df[df.in_scope == 0].top_score
    print(f"\nSuggested MIN_RERANK_SCORE = {best:.2f}  (balanced accuracy {bal_acc(best):.3f})")
    print(f"  in-scope answered: {(ins >= best).mean():.0%}   out-of-scope declined: {(oos < best).mean():.0%}")
    print("  Note: the generator also declines by itself when the passages do not answer the question.")


def metrics_fn():
    """Same metric definitions as the experiment notebooks (compute_comprehensive_metrics)."""
    import torch, nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_calc
    from sentence_transformers import SentenceTransformer, util
    nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sem = SentenceTransformer("intfloat/multilingual-e5-base", device=dev)
    rouge, smooth = rouge_scorer.RougeScorer(["rougeL"]), SmoothingFunction().method1
    norm = lambda t: re.sub(r"[ىة]", lambda m: {"ى": "ي", "ة": "ه"}[m.group()],
                            re.sub(r"[إأآ]", "ا", re.sub(r"[ً-ْـ]", "", str(t)))).strip()

    def f(gen, ref):
        g, r = norm(gen), norm(ref)
        out = {"bleu": sentence_bleu([r.split()], g.split(), smoothing_function=smooth),
               "rougeL": rouge.score(r, g)["rougeL"].fmeasure,
               "meteor": meteor_score([r.split()], g.split()),
               "bertscore_f1": bert_score_calc([gen], [ref], lang="ar", model_type="aubmindlab/bert-base-arabertv02",
                                               num_layers=12, device=dev)[2].item()}
        split = lambda t: [s.strip() for s in re.split(r"[.\n،؟]", t) if len(s.strip()) > 5] or [t]
        sim = util.cos_sim(sem.encode(split(ref), convert_to_tensor=True), sem.encode(split(gen), convert_to_tensor=True))
        comp, grd = sim.max(dim=1).values.mean().item(), sim.max(dim=0).values.mean().item()
        out.update(completeness=comp, hallucination=max(0, 1 - grd), irrelevance=max(0, 1 - comp - grd))
        return out
    return f


def run(args):
    test = test_questions(args.data, args.n)
    os.makedirs(args.out, exist_ok=True)
    score = metrics_fn() if args.metrics else None
    summary = {}
    systems = [x for x in SYSTEMS if args.systems == 'all' or x[0] == args.systems]
    for system, model in systems:
        name = f"{system}/{model}"
        rows = []
        for i, r in test.iterrows():
            d = ask(args.url, args.token, r.question, system, model)
            row = {"question": r.question, "reference": r.answer, "status": d.get("status"),
                   "answer": strip_citations(d.get("answer", "")), "grounding": d.get("grounding"),
                   "latency_ms": d.get("latency_ms"), "error": d.get("error")}
            if score and row["status"] == "ok" and row["answer"]:
                row.update(score(row["answer"], r.answer))
            rows.append(row); print(name, i + 1, row["status"])
        oos = [ask(args.url, args.token, q, system, model).get("status") for q in OUT_OF_SCOPE] if system == "fiqhqa-rag" else []
        df = pd.DataFrame(rows); df.to_csv(f"{args.out}/{name.replace('/', '_')}.csv", index=False, encoding="utf-8-sig")
        ok = df[df.status == "ok"]
        s = {"n": len(df), "answered": len(ok), "abstained_in_scope": int((df.status == "retrieval_failure").sum()),
             "api_errors": int((df.status == "api_error").sum()),
             "abstained_out_of_scope": f"{sum(x == 'retrieval_failure' for x in oos)}/{len(oos)}" if oos else "n/a"}
        for m in ["bertscore_f1", "completeness", "hallucination", "irrelevance", "meteor", "bleu", "rougeL"]:
            if m in ok: s[m] = round(float(ok[m].mean()), 4)
        summary[name] = s; print(name, s)
    json.dump(summary, open(f"{args.out}/summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--token", default=os.getenv("BACKEND_TOKEN", ""))
    p.add_argument("--data", required=True); p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default="eval_out"); p.add_argument("--metrics", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--systems", default="all", choices=["all", "baseline", "fiqhqa-rag"])
    a = p.parse_args()
    calibrate(a) if a.calibrate else run(a)
