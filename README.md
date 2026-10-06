# FiqhQA — إجابات فقهية موثَّقة المصدر

**المسار:** 01 الحوار المعرفي والإجابات الموثوقة — تحدي الذكاء الاصطناعي في خدمة المحتوى الإسلامي 2026.
**مؤشر النجاح:** يقدّم إجابة صحيحة وواضحة يمكن تتبّعها إلى مصدر معتمد، ويمتنع ويحيل إلى المختص عندما لا يجد مرجعية كافية.

## فتح الموقع

| الطريقة | الخطوات | ما يعمل |
|---|---|---|
| **سريعة** | انقري مرتين على `index.html` | الموقع كاملًا + الإجابات المسجّلة لأسئلة الاختبار |
| **محلية** | Windows: `start-site-windows.bat` — Mac: `start-site-mac.command` (يحتاج Python) | نفس السابق عبر `http://localhost:8080`؛ ومع ضبط `API_URL` في `script.js` على رابط الخادم تعمل النماذج مباشرة |
| **أونلاين** | انظري «النشر» أدناه | رابط Live Demo للمحكّمين |

## المعمارية: النموذج كاملًا عبر خادم API

```
المتصفح (script.js) ──► api/ask.php (اختياري، لاستضافة PHP) ──► backend/main.py (FastAPI)
                    └──────────── أو مباشرة (استضافة ثابتة) ──────┘     BM25 + E5 → RRF → reranker
                                                                       → Gemini / SILMA → إجابة + مراجع
                                                                       أو امتناع + إحالة إلى المختص
```

| التبويب | الدفتر الأصلي | النموذج | الاسترجاع |
|---|---|---|---|
| Gemini Baseline | No RAG - Gemini_original 10 JAN | gemini-2.5-flash | لا يوجد |
| SILMA Baseline | No RAG - Silma 10 JAN | SILMA-Kashif-2B-Instruct-v1.0 | لا يوجد |
| FiqhQA-RAG + SILMA | Good version_Silma_10 JAN | SILMA-Kashif-2B-Instruct-v1.0 | BM25 + E5 → RRF(k=60) → arabic-reranker أعلى 10 (فهرس التدريب) |
| FiqhQA-RAG + Gemini | Good version_Gemini | gemini-2.5-flash (+ بدائل احتياطية 3.8-flash / 3.5-flash-lite) | نفس استرجاع FiqhQA الهجين وإعادة الترتيب (`RAG_MODE=notebook` يعيد إعداد الدفتر الأصلي) |

- **الامتناع والإحالة:** طبقتان: (1) إن كانت درجة مُعيد الترتيب لأفضل نص أقل من `MIN_RERANK_SCORE` (تُحدَّد بالمعايرة) لا يولِّد النظام إجابة؛ (2) إن لم يجد النموذج الجواب في النصوص المسترجَعة يمتنع بنفسه. في الحالتين تُعرض رسالة إحالة إلى المختص وأقرب المواضع في المصدر.
- **احتياط:** إن تعطّل الخادم تُعرض الإجابة المسجّلة من التجارب المقارنة لأسئلة الاختبار (`assets/results.js`)، مع تنبيه. `?mode=replay` للعرض دون خادم.

## النشر (رابط Live Demo يعمل طوال التحكيم: 7–22 أكتوبر)

1. **الخادم — Hugging Face Space (Docker):** أنشئي Space بنوع Docker، وارفعي محتوى `backend/` وأعيدي تسمية `SPACE_README.md` إلى `README.md`.
   - Secrets: `GEMINI_API_KEY` (مفتاح **جديد**)، و`HF_TOKEN` + `DATA_HF_REPO` إن كانت البيانات في مستودع بيانات خاص (أو ضعي الملف في `data/LastRAG_dataset.csv`).
   - العتاد المجاني CPU (16GB) يكفي لتبويبي Gemini والاسترجاع؛ SILMA يعمل ببطء (`SILMA_NUM_BEAMS=1`). للسرعة: ترقية Space إلى T4 خلال نافذتي التحكيم.
   - بديل مؤقت: `backend/run_backend_colab.ipynb` (رابط يتغير ويتوقف؛ لا يصلح لفترة التحكيم).
2. **الواجهة:**
   - Cloudflare Pages / Netlify (ثابتة): في `script.js` ضعي `API_URL: 'https://USERNAME-SPACE.hf.space/ask'`.
   - أو استضافة PHP: أبقي `API_URL: 'api/ask.php'` وضعي `BACKEND_URL` في متغيرات البيئة أو `api/config.local.php`.
3. **فحص:** `/health` يُرجع `"ready": true`، وشارة العرض الحي «متصل بالنماذج مباشرة». جرّبي التبويبات الأربعة وسؤالًا خارج النطاق.

## إعدادات الدقة (RAG_MODE=improved، الافتراضي)

| التحسين | قبل (الدفاتر) | بعد |
|---|---|---|
| ما يُفهرَس | نص الإجابة فقط | السؤال + الإجابة (أسئلة المستخدمين تشبه أسئلة الكتاب) |
| E5 | بلا بادئات، مسافة L2 | `query:`/`passage:` كما يتطلب النموذج، تشابه جيب التمام |
| BM25 | نص خام، `split()` | تطبيع + حذف علامات الترقيم والكلمات الوظيفية |
| ما يُمرَّر للنموذج | 10 مقاطع مقطوعة عند 512 رمزًا | أفضل 5 أزواج س/ج كاملة، بميزانية 2048 رمزًا |
| SILMA | موجّه خام، عيّنات + beam + repetition_penalty=1.8 | قالب المحادثة، توليد حتمي، 1.1 |
| Gemini RAG | 2.5-flash-lite، موجّه بسيط | 2.5-flash، حرارة 0.2، موجّه منظم |
| الاستشهاد | قائمة كل الكتب | أرقام [n] بعد كل حكم + المراجع المستشهد بها فقط |
| الامتناع | عتبة الاسترجاع | العتبة + امتناع النموذج نفسه إذا لم يجد الجواب في النصوص |

`RAG_MODE=notebook` يعيد سلوك الدفاتر حرفيًا (لاستنساخ نتائج التجارب المقارنة).

## إعادة التقييم العادلة (للعرض والمعيار «تحقيق النفع» 20%)

```bash
RATE_LIMIT_PER_MIN=0 uvicorn main:app --port 8000          # الخادم
python eval/evaluate.py --url http://localhost:8000 --data LastRAG_dataset.csv --calibrate   # يقترح MIN_RERANK_SCORE
python eval/evaluate.py --url http://localhost:8000 --data LastRAG_dataset.csv --n 100 --metrics
```
يقيس كل الأنظمة على نفس أسئلة الاختبار، ولا يحتسب أخطاء الـAPI كإجابات، ويقيس الامتناع على أسئلة خارج النطاق.

## قائمة التسليم (حتى 6 أكتوبر 11:59 م بتوقيت الرياض)
- [ ] رابط Live Demo يعمل بالتبويبات الأربعة ومختبَر.
- [ ] مستودع GitHub **عام**، بلا مفاتيح أو كلمات مرور (حُذف المفتاح من دفاتر Gemini وأُلغي من Google AI Studio).
- [ ] `SOURCES_AND_LICENSES.md` مكتمل، ونسخة البداية موثّقة.
- [ ] عرض PDF/PowerPoint: المشكلة، الحل، آلية العمل، القيمة المضافة، التقنيات، النتائج، خطة الاستمرار.
- [ ] فيديو توضيحي ≤ دقيقتين.

## الملفات
```
index.html, style.css, script.js     الواجهة
assets/results.js                    الإجابات المسجّلة (احتياط)
api/ask.php, config.php, .htaccess   وكيل PHP اختياري
backend/main.py                      الخادم (الدفاتر الأربعة خلف /ask)
backend/Dockerfile, SPACE_README.md  نشر Hugging Face Space
backend/run_backend_colab.ipynb      تشغيل مؤقت على Colab
eval/evaluate.py                     إعادة التقييم والمعايرة
SOURCES_AND_LICENSES.md              سجل المصادر والتراخيص ونسخة البداية
```
