<?php
/* =========================================================
   FiqhQA — الإعدادات
   لا تكتبي القيم السرية هنا (هذا الملف يُرفع إلى GitHub العام).
   ضعيها كمتغيرات بيئة في الاستضافة، أو في api/config.local.php (مستثنى في .gitignore):
     <?php return ['BACKEND_URL' => 'https://...', 'BACKEND_TOKEN' => '...'];
   مفتاح Gemini ليس هنا: هو في الخادم (backend) فقط.
   ========================================================= */
$cfg = [
  // رابط خادم FiqhQA (backend/main.py) — التبويبات الأربعة كلها تمر عبره
  // مثال: https://USERNAME-fiqhqa-api.hf.space
  'BACKEND_URL'   => getenv('BACKEND_URL') ?: '',
  // السر المشترك مع الخادم (نفس قيمة BACKEND_TOKEN هناك)، اختياري
  'BACKEND_TOKEN' => getenv('BACKEND_TOKEN') ?: '',
  // SILMA مع الاسترجاع يستغرق ~30 ثانية على T4، فالمهلة أطول
  'TIMEOUT'       => 170,
];
$local = __DIR__ . '/config.local.php';
return is_file($local) ? array_merge($cfg, (array) require $local) : $cfg;
