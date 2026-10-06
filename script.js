/* =========================================================
   FiqhQA — script.js
   ---------------------------------------------------------
   كل التبويبات الأربعة متصلة بالنماذج الفعلية:
     المتصفح → api/ask.php → خادم FiqhQA (backend/main.py) → Gemini / SILMA + الاسترجاع
   الأوضاع (غيّري MODE، أو أضيفي ?mode=replay للرابط):
     'live'   : اتصال مباشر بالنماذج (الافتراضي). إن تعطّل الخادم تُعرض الإجابة المسجّلة إن وُجدت.
     'replay' : الإجابات المسجّلة فقط من ملفات نتائج التجارب (للعرض دون خادم).
   ========================================================= */
const CONFIG = {
  // استضافة PHP (Hostinger): 'api/ask.php'
  // استضافة ثابتة بلا PHP (Cloudflare Pages / Netlify): رابط الخادم مباشرة، مثل
  //   'https://USERNAME-fiqhqa-api.hf.space/ask'   (لا مفاتيح في المتصفح؛ مفتاح Gemini في الخادم فقط)
  API_URL: 'https://fatimashanaq-fiqhqa-api.hf.space/ask',
  MODE: 'live',                 // 'live' | 'replay'
  TIMEOUT_MS: 180000,           // SILMA + الاسترجاع قد يستغرق ~30 ث على T4
  FALLBACK_TO_RECORDED: true,   // إن تعطّل الخادم: اعرض الإجابة المسجّلة من التجارب (مع تنبيه)
  RECORDED_URL: 'assets/results.js?v=2',
  RRF_K: 60, CHUNK_SIZE: 256,
};

(() => {
  const p = new URLSearchParams(location.search).get('mode');
  if (['live', 'replay'].includes(p)) CONFIG.MODE = p;
})();

/* ---------------------------------------------------------
   عقد الـ API (api/ask.php)
   POST { question, model: 'gemini' | 'silma', system: 'baseline' | 'fiqhqa-rag' }
   → { status: 'ok' | 'retrieval_failure', answer, citation: {book_title, fiqh_source} | null,
       sources: [{ text, book_title, fiqh_source, ref, rank_bm25, rank_dense, rrf, rerank }],
       grounding: 0..1 | null, model, model_name, pipeline, latency_ms }
   GET ?health=1 → { ok, ready, ... }
   --------------------------------------------------------- */

const EXAMPLES = [
  'كم مدة المسح على الخفين والعمامة والخمار؟ مع الدليل',
  'ما الدليل على تحريم أواني الذهب والفضة؟',
  'ما حكم استعمال آنية الكفار وثيابهم؟',
  'ما هي الأحكام الشرعية الخمسة؟',
  'ما هو الفقه لغة وشرعا؟',
];
const OUT_OF_SCOPE = 'ما حكم تداول العملات الرقمية؟';

/* التبويبات الأربعة ← الدفاتر (notebooks) التي يشغّلها الخادم */
const TABS = [
  { id: 'gemini-base', label: 'Gemini Baseline',     system: 'baseline',   model: 'gemini' }, // No RAG - Gemini_original 10 JAN
  { id: 'silma-base',  label: 'SILMA Baseline',      system: 'baseline',   model: 'silma'  }, // No RAG - Silma 10 JAN
  { id: 'rag-silma',   label: 'FiqhQA-RAG + SILMA',  system: 'fiqhqa-rag', model: 'silma'  }, // Good version_Silma_10 JAN
  { id: 'rag-gemini',  label: 'FiqhQA-RAG + Gemini', system: 'fiqhqa-rag', model: 'gemini' }, // Good version_Gemini
];
const DEFAULT_TAB = 3;
const HALLUCINATION = { 'gemini-base': '61.3%', 'silma-base': '45.3%' }; // من جدول نتائج التجارب المقارنة
const MODEL_NAMES = { gemini: 'Gemini', silma: 'SILMA' };

/* =========================================================
   1) الإجابات المسجّلة من التجارب (assets/results.js، تُحمَّل عند الحاجة)
   ========================================================= */
const qKey = (s) => String(s || '')
  .replace(/[ؗ-ًؚ-ْـ]/g, '')
  .replace(/[إأآٱ]/g, 'ا').replace(/ى/g, 'ي').replace(/ة/g, 'ه')
  .replace(/[^ء-ي0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();

let recordedPromise = null;
function loadRecorded() {
  if (window.FIQHQA_RECORDED) return Promise.resolve(window.FIQHQA_RECORDED);
  if (!recordedPromise) recordedPromise = new Promise((resolve) => {
    const s = document.createElement('script');
    s.src = CONFIG.RECORDED_URL;
    s.onload = () => resolve(window.FIQHQA_RECORDED || null);
    s.onerror = () => resolve(null);
    document.head.appendChild(s);
  });
  return recordedPromise;
}

// يطابق السؤال مع الأسئلة المسجّلة حتى لو اختلفت الصياغة قليلًا (تطابق الكلمات الجوهرية)
const STOPW = new Set(['ما','ماذا','هل','كم','في','من','علي','عن','الي','او','و','هو','هي','مع','بين','اذكر','وضح','ذلك','تقول','الدليل','دليل','حكم','واذكر','مع']);
const toks = (s) => new Set(qKey(s).split(' ').map(t => t.replace(/^(وال|بال|فال|لل|ال|و)/, '')).filter(t => t.length > 1 && !STOPW.has(t)));
function findRecorded(db, question) {
  if (!db) return null;
  if (db.items[qKey(question)]) return db.items[qKey(question)];
  const q = toks(question);
  if (q.size < 2) return null;
  let best = null, bestScore = 0;
  for (const it of Object.values(db.items)) {
    const t = toks(it.q);
    const inter = [...q].filter(x => t.has(x)).length;
    const score = inter / q.size;                      // نسبة كلمات السؤال الموجودة في السؤال المسجّل
    if (score > bestScore) { best = it; bestScore = score; }
  }
  return bestScore >= 0.8 ? best : null;
}

async function recordedAnswer(question, tab) {
  const db = await loadRecorded();
  const it = findRecorded(db, question);
  const rec = it && it.a[tab.id];
  if (!rec) return null;
  return {
    status: 'ok', system: tab.system, model: tab.model, answer: rec.answer,
    citation: null, sources: [], grounding: null, latency_ms: null,
    recorded: true, recorded_metrics: rec.m, recorded_q: it.q,
  };
}

/* =========================================================
   2) طبقة الاتصال
   ========================================================= */
async function callApi(question, tab) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CONFIG.TIMEOUT_MS);
  try {
    const res = await fetch(CONFIG.API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model: tab.model, system: tab.system }),
      signal: ctrl.signal,
    });
    let d = null;
    try { d = await res.json(); } catch { /* رد غير JSON */ }
    if (!res.ok || !d) throw new Error((d && (d.error || d.detail)) || ('HTTP ' + res.status));
    return { ...adapt(d, tab), system: tab.system };
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('انتهت مهلة الانتظار دون رد من النموذج');
    if (location.protocol === 'file:') throw new Error('خادم النماذج غير مشغَّل، فلا يمكن توليد إجابة جديدة. الأسئلة المسجّلة من التجارب المقارنة فقط متاحة الآن');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function askFiqhQA(question, tab) {
  if (CONFIG.MODE === 'replay') {
    const r = await recordedAnswer(question, tab);
    if (r) return r;
    throw new Error('لا توجد إجابة مسجّلة لهذا السؤال في هذا التبويب.' + (recordedHint(await loadRecorded(), question) || ' اختاري سؤالًا من «أسئلة من مجموعة الاختبار».'));
  }
  try {
    return await callApi(question, tab);
  } catch (err) {
    if (!CONFIG.FALLBACK_TO_RECORDED) throw err;
    const r = await recordedAnswer(question, tab);
    if (!r) throw new Error(err.message + recordedHint(await loadRecorded(), question));
    r.fallback = err.message;
    return r;
  }
}

// يذكر التبويبات التي لها إجابة مسجّلة لهذا السؤال
function recordedHint(db, question) {
  const it = findRecorded(db, question);
  if (!it) return '';
  return ' — توجد إجابة مسجّلة لهذا السؤال في: ' + Object.keys(it.a).map(id => TABS.find(t => t.id === id).label).join('، ');
}

// يوحّد شكل الاستجابة إن اختلفت أسماء الحقول في الخادم
function adapt(d, tab) {
  return {
    status: d.status || (d.answer ? 'ok' : 'retrieval_failure'),
    answer: d.answer || '',
    citation: d.citation || null,
    sources: (d.sources || d.contexts || []).map(s => ({
      text: s.text || s.chunk || '', book_title: s.book_title || '', fiqh_source: s.fiqh_source || '',
      ref: s.ref || '', rank_bm25: s.rank_bm25 ?? null, rank_dense: s.rank_dense ?? null,
      rrf: s.rrf ?? null, rerank: s.rerank ?? s.score ?? null,
    })),
    grounding: typeof d.grounding === 'number' ? d.grounding : null,
    referral: d.referral || null,
    llm_abstained: !!d.llm_abstained,
    nearest: d.nearest || [],
    model: d.model || tab.model,
    model_name: d.model_name || null,
    pipeline: d.pipeline || null,
    latency_ms: d.latency_ms ?? null,
  };
}

/* =========================================================
   3) الواجهة
   ========================================================= */
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const MODE_NAMES = {
  live: 'متصل بالنماذج مباشرة',
  checking: 'جارٍ فحص اتصال الخادم…',
  offline: 'الخادم غير متاح: تُعرض الإجابات المسجّلة',
  loading: 'الخادم يحمّل النماذج…',
  replay: 'وضع الإجابات المسجّلة من التجارب المقارنة',
};

// تنسيق خفيف لإجابات النماذج (Markdown بسيط) بعد التهريب
function fmtAnswer(s) {
  return esc(s).split(/\n/).map(line => {
    let l = line.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    if (/^\s*([*\-•]|\d+[.)])\s+/.test(l)) l = '• ' + l.replace(/^\s*([*\-•]|\d+[.)])\s+/, '');
    l = l.replace(/^\s*#{1,6}\s*/, '');
    return l;
  }).join('<br>').replace(/(<br>){3,}/g, '<br><br>');
}

const form = $('#askForm'), qEl = $('#question'), btn = $('#askBtn'), out = $('#out');

function setBadge(state) {
  const badge = $('#modeBadge');
  badge.textContent = MODE_NAMES[state];
  badge.classList.toggle('mode--live', state === 'live');
}

async function checkHealth() {
  if (CONFIG.MODE !== 'live') return;
  setBadge('checking');
  try {
    const healthUrl = /\/ask\/?$/.test(CONFIG.API_URL) ? CONFIG.API_URL.replace(/\/ask\/?$/, '/health') : CONFIG.API_URL + '?health=1';
    const res = await fetch(healthUrl, { cache: 'no-store' });
    const h = await res.json();
    if (h.ready) setBadge('live');
    else if (h.ok && !h.error) { setBadge('loading'); setTimeout(checkHealth, 15000); }
    else { setBadge('offline'); addRecordedChips(); }
  } catch {
    setBadge('offline'); addRecordedChips();
  }
}

function addChip(container, q, cls = '', title = '', recordedTabs = null) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'chip' + (cls ? ' ' + cls : '');
  b.textContent = q.length > 70 ? q.slice(0, 68) + '…' : q;
  b.title = title || q;
  b.dataset.q = q;
  b.addEventListener('click', () => {
    // سؤال مسجّل: انتقل إلى تبويب له إجابة مسجّلة إن لم يكن التبويب الحالي منها
    const cur = TABS[+$('input[name="tab"]:checked').value];
    const rt = recordedTabs || (b.dataset.tabs ? b.dataset.tabs.split(',') : null);
    if (rt && !rt.includes(cur.id)) {
      const i = TABS.findIndex(t => t.id === rt[0]);
      $(`input[name="tab"][value="${i}"]`).checked = true;
    }
    qEl.value = q; run();
  });
  container.appendChild(b);
}

let recordedChipsShown = false;
async function addRecordedChips() {
  if (recordedChipsShown) return;
  recordedChipsShown = true;
  const db = await loadRecorded();
  if (!db) return;
  const chips = $('#chips');
  // دون خادم: أخفِ الأمثلة التي ليس لها إجابة مسجّلة (لا يمكن توليدها الآن)
  [...chips.querySelectorAll('.chip')].forEach(c => { const it = findRecorded(db, c.dataset.q); if (!it) c.remove(); else c.dataset.tabs = Object.keys(it.a).join(','); });
  const label = document.createElement('span');
  label.textContent = 'أسئلة من مجموعة الاختبار (لها إجابات مسجّلة):';
  label.style.cssText = 'flex-basis:100%;font-size:.8rem;color:rgba(255,255,255,.6);margin-top:.4rem';
  chips.appendChild(label);
  // أولًا أسئلة لها إجابة مسجّلة في FiqhQA-RAG + Gemini، ثم الأسئلة المسجّلة لتبويبين
  const all = Object.values(db.items);
  const pick = [...all.filter(it => it.a['rag-gemini']).slice(0, 3), ...all.filter(it => !it.a['rag-gemini'] && Object.keys(it.a).length > 1).slice(0, 3)];
  pick.forEach(it => {
    const labels = Object.keys(it.a).map(id => TABS.find(t => t.id === id).label);
    addChip(chips, it.q, '', 'إجابات مسجّلة لـ: ' + labels.join('، '), Object.keys(it.a));
  });
}

function initUI() {
  setBadge(CONFIG.MODE === 'replay' ? 'replay' : 'checking');
  const chips = $('#chips');
  if (CONFIG.MODE === 'replay') addRecordedChips();
  else {
    EXAMPLES.forEach(q => addChip(chips, q));
    addChip(chips, OUT_OF_SCOPE, 'chip--out', 'سؤال خارج نطاق المصدر لاختبار سلوك النظام');
  }
  form.addEventListener('submit', e => { e.preventDefault(); run(); });
  qEl.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run(); });
  checkHealth();
}

let busy = false;
async function run() {
  const question = qEl.value.trim();
  if (busy) return;
  if (question.length < 4) { qEl.focus(); qEl.placeholder = 'اكتب سؤالًا فقهيًا أولًا، أو اختر مثالًا من الأسفل.'; return; }

  const tab = TABS[+$('input[name="tab"]:checked').value];
  busy = true; btn.disabled = true;
  btn.textContent = tab.system === 'baseline' ? 'جارٍ التوليد…' : 'جارٍ الاسترجاع والتوليد…';
  out.innerHTML = loadingTpl(tab);

  let r;
  try { r = await askFiqhQA(question, tab); }
  catch (err) { r = { status: 'error', error: err.message }; }

  render(r, question, tab);
  if (innerWidth < 920) out.scrollIntoView({ behavior: 'smooth', block: 'start' });
  busy = false; btn.disabled = false; btn.textContent = 'اسأل FiqhQA';
}

function loadingTpl(tab) {
  const slow = tab.model === 'silma' ? '<p class="meter__hint">SILMA يعمل على GPU خاص بالمشروع؛ قد تستغرق الإجابة نصف دقيقة.</p>' : '';
  return `<div class="result loading"><div class="answer"><div class="answer__head"><h3>الإجابة: ${esc(tab.label)}</h3></div><div class="answer__body">.</div></div>
    <div class="side"><div class="meter"><p class="meter__label">مؤشر الاستناد إلى المصدر</p><div class="meter__track"><div class="meter__fill"></div></div>${slow}</div></div></div>`;
}

function level(g) {
  if (g == null) return { t: 'غير متاح', c: 'rgba(255,255,255,.7)' };
  if (g >= 0.75) return { t: 'استناد مرتفع', c: '#8FD9C7' };
  if (g >= 0.55) return { t: 'استناد متوسط', c: '#F0D9A6' };
  return { t: 'استناد منخفض', c: '#F3B8A8' };
}

function banner(r) {
  if (r.fallback) return `<p class="banner">تعذّر الحصول على إجابة لحظية من النموذج الآن، فهذه إجابة مسجّلة من التجارب المقارنة (وليست توليدًا لحظيًا) لسؤال الاختبار: «${esc(r.recorded_q)}».<br><small dir="ltr" style="opacity:.75">${esc(r.fallback)}</small></p>`;
  if (r.recorded) return `<p class="banner">إجابة مسجّلة من التجارب المقارنة لسؤال الاختبار: «${esc(r.recorded_q)}»، وليست توليدًا لحظيًا.</p>`;
  return '';
}

function metaSpans(r, extra = []) {
  const name = r.model_name || MODEL_NAMES[r.model] || r.model;
  const spans = [`<span dir="ltr">${esc(name)}</span>`];
  if (r.latency_ms != null) spans.push(`<span>${r.latency_ms} ms</span>`);
  if (r.recorded_metrics) {
    const [b, c, h] = r.recorded_metrics;
    spans.push(`<span dir="ltr">BERTScore ${b.toFixed(3)}</span>`, `<span>الاكتمال ${c.toFixed(3)}</span>`, `<span>الهلوسة ${h.toFixed(3)}</span>`);
  }
  return spans.concat(extra).join('');
}

function bindCopy(text, label) {
  $('#copyBtn').addEventListener('click', async (e) => {
    try { await navigator.clipboard.writeText(text); e.target.textContent = 'نُسخت'; }
    catch { e.target.textContent = 'تعذّر النسخ'; }
    setTimeout(() => (e.target.textContent = label), 1800);
  });
}

function renderBaseline(r, question, tab, target) {
  target.innerHTML = banner(r) + `
    <div class="result">
      <article class="answer">
        <div class="answer__head"><h3>الإجابة: ${esc(tab.label)}</h3><button class="copy" type="button" id="copyBtn">نسخ</button></div>
        <div class="answer__body">${fmtAnswer(r.answer)}</div>
        <div class="answer__meta">${metaSpans(r, ['<span>بدون RAG</span>', '<span>0 نصوص مسترجَعة</span>'])}</div>
      </article>
      <aside class="side">
        <div class="meter">
          <div class="meter__top"><span class="meter__label">مؤشر الاستناد إلى المصدر</span><span class="meter__val">—</span></div>
          <div class="meter__track"><div class="meter__fill"></div></div>
          <p class="meter__level" style="color:#F3B8A8">بدون مصدر</p>
          <p class="meter__hint">إجابة من معرفة النموذج المخزَّنة، بلا استرجاع ولا كتاب ولا مرجع. معدل الهلوسة لهذا الإعداد في تجاربنا المقارنة: <span dir="ltr">${HALLUCINATION[tab.id]}</span>.</p>
        </div>
      </aside>
    </div>`;
  bindCopy(`س: ${question}\nج (${tab.label}، بدون مصدر): ${r.answer}\n\nملاحظة: إجابة نموذج دون استرجاع، وليست فتوى.`, 'نسخ');
}

function render(r, question, tab = TABS[DEFAULT_TAB]) {
  const target = out;

  if (r.status === 'error') {
    target.innerHTML = `<div class="fail"><h3>تعذّر الوصول إلى ${esc(tab.label)}</h3><p>${esc(r.error)}</p>
      <p>للإجابات الحية: شغّلي خادم النماذج (backend) وضعي رابطه في API_URL داخل script.js. أو اختاري سؤالًا من «أسئلة من مجموعة الاختبار».</p></div>`;
    return;
  }

  if (r.system === 'baseline') { renderBaseline(r, question, tab, target); return; }

  if (r.status !== 'ok') {
    const g = (r.grounding == null || r.llm_abstained) ? '' : ` (مؤشر الاستناد ${Math.round(r.grounding * 100)}%)`;
    const why = r.llm_abstained
      ? 'امتنع النظام عن الإجابة: النصوص المسترجَعة لا تتضمن جوابًا عن هذا السؤال'
      : `امتنع النظام عن الإجابة: لا يوجد سند كافٍ في المصدر${g}`;
    const near = (r.nearest || []).length ? `<p style="margin-top:.8rem"><b>أقرب ما في المصدر</b> (للاطلاع فقط، وليس جوابًا):</p>
      <ul style="margin:.3rem 0 0;padding-inline-start:1.2rem">${r.nearest.map(s =>
        `<li>${esc(s.fiqh_source || s.book_title)}${s.ref ? ` <span dir="ltr">— ${esc(s.ref)}</span>` : ''}</li>`).join('')}</ul>` : '';
    target.innerHTML = `<div class="fail">
      <h3>${why}</h3>
      <p>${esc(r.referral || 'لم يعثر النظام على نص فقهي ذي صلة كافية، لذلك لم يولِّد إجابة. الامتناع أَولى من إجابة بلا سند. يُرجى الرجوع إلى أهل العلم المختصين.')}</p>
      ${near}
    </div>`;
    return;
  }

  const lv = level(r.grounding);
  const pct = r.grounding == null ? '—' : Math.round(r.grounding * 100) + '%';
  const cite = r.citation ? `(${esc(r.citation.fiqh_source || r.citation.book_title)})` : '';
  const fmt = (v, d = 3) => v == null ? '—' : Number(v).toFixed(d);
  const extra = r.recorded ? [] : [`<span>k = ${CONFIG.RRF_K}</span>`, `<span>${r.sources.length} نصوص مسترجَعة</span>`];

  const sourcesHtml = r.sources.length ? `
    <section class="sources">
      <h3>النصوص المسترجَعة التي استندت إليها الإجابة</h3>
      ${r.pipeline ? `<p class="src__ref" style="padding:0;margin:-.4rem 0 .8rem" dir="ltr">${esc(r.pipeline)}</p>` : ''}
      <ol class="src-list">
        ${r.sources.map((s, i) => `
          <li class="src">
            <div class="src__top"><span class="src__rank">${i + 1}</span><span class="src__book">${esc(s.fiqh_source || s.book_title)}</span></div>
            ${s.ref ? `<p class="src__ref">${esc(s.ref)}</p>` : ''}
            <p class="src__text">${esc(s.text.length > 600 ? s.text.slice(0, 600) + '…' : s.text)}</p>
            <div class="src__scores">
              ${s.rank_bm25 != null ? `<span>BM25 #${s.rank_bm25}</span>` : ''}
              ${s.rank_dense != null ? `<span>E5 #${s.rank_dense}</span>` : ''}
              ${s.rrf != null ? `<span>RRF ${fmt(s.rrf, 4)}</span>` : ''}
              ${s.rerank != null ? `<span>rerank ${fmt(s.rerank, 2)}</span>` : ''}
            </div>
          </li>`).join('')}
      </ol>
    </section>` : '';

  target.innerHTML = banner(r) + `
    <div class="result">
      <article class="answer">
        <div class="answer__head"><h3>الإجابة: ${esc(tab.label)}</h3><button class="copy" type="button" id="copyBtn">نسخ مع المرجع</button></div>
        <div class="answer__body" id="ansText">${fmtAnswer(r.answer)}</div>
        ${cite ? `<p class="answer__cite">${cite}</p>` : ''}
        <div class="answer__meta">${metaSpans(r, extra)}</div>
      </article>
      <aside class="side">
        <div class="meter">
          <div class="meter__top"><span class="meter__label">مؤشر الاستناد إلى المصدر</span><span class="meter__val">${pct}</span></div>
          <div class="meter__track"><div class="meter__fill" id="meterFill"></div></div>
          <p class="meter__level" style="color:${lv.c}">${lv.t}</p>
          <p class="meter__hint">${r.recorded
            ? 'الإجابة المسجّلة لا تتضمن النصوص المسترجَعة؛ المقاييس أعلاه من ملف نتائج التجربة لهذا السؤال.'
            : r.grounding == null
              ? 'هذا الإعداد لا يستخدم مُعيد الترتيب، فلا تتوفر درجة استناد.'
              : 'مستمد من درجة مُعيد الترتيب (oddadmix/arabic-reranker) لأعلى نص مسترجَع.'}</p>
        </div>
      </aside>
    </div>` + sourcesHtml;

  requestAnimationFrame(() => { const f = $('#meterFill'); if (f) f.style.width = (r.grounding ?? 0) * 100 + '%'; });
  bindCopy(`س: ${question}\nج: ${r.answer}\n${cite}\n\nملاحظة: FiqhQA مساعد معرفي وليس فتوى.`, 'نسخ مع المرجع');
}

/* =========================================================
   4) تفاصيل الصفحة: الشريط العلوي، القائمة، الرسم البياني
   ========================================================= */
function initPage() {
  const bar = $('#topbar');
  const onScroll = () => bar.classList.toggle('is-solid', scrollY > 40);
  onScroll(); addEventListener('scroll', onScroll, { passive: true });

  const menu = $('#menuBtn'), nav = $('#nav');
  menu.addEventListener('click', () => {
    const open = nav.classList.toggle('is-open');
    menu.setAttribute('aria-expanded', open);
  });
  nav.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
    nav.classList.remove('is-open'); menu.setAttribute('aria-expanded', 'false');
  }));

  const chart = $('#hallChart');
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((es, o) => es.forEach(e => {
      if (e.isIntersecting) { chart.classList.add('is-in'); o.disconnect(); }
    }), { threshold: .35 }).observe(chart);
  } else chart.classList.add('is-in');
}

/* ---------- سند: تشغيل الوكيل مباشرة عند الضغط ---------- */
function initSanad() {
  const trigger = document.getElementById('sanadStart');
  if (!trigger) return;

  // يبحث عن زر بدء المحادثة داخل الويدجت (قد يكون داخل Shadow DOM متداخل)
  const findStartButton = (root, depth = 0) => {
    if (!root || depth > 6) return null;
    const btns = Array.from(root.querySelectorAll('button'));
    const wanted = btns.find(b => /start|call|talk|chat|ابدأ|تحدث|اتصال/i.test(
      (b.getAttribute('aria-label') || '') + ' ' + (b.textContent || '')));
    if (wanted) return wanted;
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) {
        const found = findStartButton(el.shadowRoot, depth + 1);
        if (found) return found;
      }
    }
    return btns[0] || null;
  };

  const startSanad = () => {
    const widget = document.querySelector('elevenlabs-convai');
    if (!widget) return;
    widget.scrollIntoView({ behavior: 'smooth', block: 'center' });
    let tries = 0;
    const attempt = () => {
      const btn = widget.shadowRoot ? findStartButton(widget.shadowRoot) : null;
      if (btn) { btn.click(); return; }
      if (++tries < 20) setTimeout(attempt, 250); // انتظار تحميل الويدجت
    };
    attempt();
  };

  trigger.addEventListener('click', startSanad);
}

initUI();
initPage();
initSanad();
