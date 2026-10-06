# سجل المصادر والأدوات والتراخيص — FiqhQA

> مطلوب في دليل المشارك (رابط الحل والتوثيق البرمجي، البند 06).

## المصدر الشرعي
| العنصر | التفاصيل | الحالة |
|---|---|---|
| الكتاب | «الأسئلة والأجوبة الفقهية» — عبدالعزيز بن محمد السلمان | النص مملوك لأصحابه؛ استُخدم لأغراض البحث والعرض، ولا يُنشر النص الكامل في المستودع العام |
| مصدر النسخة الرقمية | المكتبة الشاملة (HTML ← DOCX ← CSV) | |
| مجموعة البيانات | FiqhQA: 1,140 زوج سؤال/جواب مع الكتاب والباب والمرجع | مستثناة من GitHub عبر `.gitignore`؛ يحمّلها الخادم من مستودع بيانات خاص (`DATA_HF_REPO`) |

## النماذج
| النموذج | الدور | الترخيص |
|---|---|---|
| Gemini 2.5 Flash (أساسي) + بدائل احتياطية تلقائية عند الانشغال أو نفاد الحصة: Gemini 3.8 Flash، 3.5 Flash-Lite، 2.5 Flash-Lite (Google) | التوليد (API)؛ يُعرض اسم النموذج المستخدم فعليًا مع كل إجابة | شروط Google Gemini API |
| silma-ai/SILMA-Kashif-2B-Instruct-v1.0 | التوليد (محلي) | Gemma Terms of Use (مبني على Google Gemma) |
| intfloat/multilingual-e5-base | التضمين الدلالي | MIT |
| oddadmix/arabic-reranker | إعادة الترتيب | لا يذكر بطاقة النموذج ترخيصًا صريحًا؛ مبني على aubmindlab/bert-base-arabertv02. استُخدم لأغراض بحثية غير تجارية |
| aubmindlab/bert-base-arabertv02 | BERTScore في التقييم فقط | لا يذكر بطاقة النموذج ترخيصًا صريحًا؛ استُخدم للتقييم البحثي فقط |

## المكتبات
FastAPI (MIT)، Uvicorn (BSD-3)، Pydantic (MIT)، rank-bm25 (Apache-2.0)، faiss-cpu (MIT)، sentence-transformers (Apache-2.0)، transformers (Apache-2.0)، PyTorch (BSD-3)، pandas (BSD-3)، scikit-learn (BSD-3)، requests (Apache-2.0).

## الواجهة
خطوط Google Fonts: Amiri و IBM Plex Sans Arabic (SIL Open Font License).
المساعد الصوتي «سند»: أداة ElevenLabs Conversational AI (`@elevenlabs/convai-widget-embed`) وفق شروط ElevenLabs؛ مساعد تعريفي بالمشروع فقط ولا يقدّم فتاوى.

## الاستضافة
الخادم: Hugging Face Spaces (Docker). الواجهة: Netlify. التقييم: Google Colab.

## نسخة البداية (Starting version)
وفق الأسئلة الشائعة (يُقيَّم ما أُنجز من 4 إلى 6 أكتوبر فقط):
- **ما كان موجودًا قبل 4 أكتوبر 2026:** دفاتر التجارب الأربعة، مجموعة بيانات FiqhQA، وواجهة الموقع الثابتة (وضع عرض محاكى).
- **ما أُنجز خلال أيام التحدي:** خادم API موحّد يشغّل النماذج الأربعة، ربط الموقع بالنماذج فعليًا، الامتناع والإحالة إلى المختص، الإجابات المسجّلة احتياطًا، النشر، وإعادة التقييم العادلة (`eval/evaluate.py`).
- إضافات 6 أكتوبر: نماذج Gemini احتياطية، ومعايرة عتبة الامتناع (`MIN_RERANK_SCORE=10`) على 50 سؤال اختبار و15 سؤالًا خارج النطاق، ومؤشر استناد مُعاير.
- أول commit في المستودع (الوسم `start-version`) يمثّل نسخة البداية، وما بعده هو عمل أيام التحدي.
