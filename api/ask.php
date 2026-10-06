<?php
/* =========================================================
   FiqhQA — نقطة الاتصال الوحيدة للواجهة (وكيل/Proxy)
   الواجهة → api/ask.php → خادم FiqhQA (backend/main.py) → النماذج
   - يُخفي رابط الخادم والسر عن المتصفح
   - التبويبات الأربعة: baseline/fiqhqa-rag × gemini/silma
   GET  ?health=1  → حالة الخادم
   POST { question, model, system } → الإجابة
   ========================================================= */
header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');

$cfg = require __DIR__ . '/config.php';
$base = rtrim((string)$cfg['BACKEND_URL'], '/');
$headers = ['Content-Type: application/json'];
if ($cfg['BACKEND_TOKEN']) $headers[] = 'X-API-Key: ' . $cfg['BACKEND_TOKEN'];

/* ---------- فحص الحالة ---------- */
if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['health'])) {
  if (!$base) out(['ok' => false, 'ready' => false, 'error' => 'BACKEND_URL غير مضبوط في api/config.php']);
  [$code, $body] = http('GET', $base . '/health', $headers, null, 8, false);
  $d = json_decode((string)$body, true);
  out(is_array($d) ? $d : ['ok' => false, 'ready' => false, 'error' => 'الخادم لا يستجيب (HTTP ' . $code . ')']);
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') fail(405, 'POST only');

$in = json_decode(file_get_contents('php://input'), true) ?: [];
$question = trim((string)($in['question'] ?? ''));
$model    = (string)($in['model'] ?? 'gemini');
$system   = (string)($in['system'] ?? 'fiqhqa-rag');

if (mb_strlen($question) < 4 || mb_strlen($question) > 500) fail(400, 'طول السؤال يجب أن يكون بين 4 و500 حرف');
if (!in_array($model, ['gemini', 'silma', 'gemini-2.5-flash', 'silma-9b-instruct'], true)) fail(400, 'نموذج غير معروف');
if (!in_array($system, ['baseline', 'fiqhqa-rag'], true)) fail(400, 'إعداد غير معروف');
if (!$base) fail(503, 'خادم FiqhQA غير مربوط بعد. ضعي رابطه في BACKEND_URL داخل api/config.php');

[$code, $body] = http('POST', $base . '/ask', $headers,
  ['question' => $question, 'model' => $model, 'system' => $system], (int)$cfg['TIMEOUT']);
$d = json_decode((string)$body, true);
if ($code !== 200 || !is_array($d)) {
  $msg = is_array($d) ? ($d['detail'] ?? $d['error'] ?? null) : null;
  if (is_array($msg)) $msg = json_encode($msg, JSON_UNESCAPED_UNICODE);
  fail($code >= 400 && $code < 600 ? $code : 502, 'خادم FiqhQA: ' . ($msg ?: 'لم يُرجع ردًا صالحًا (HTTP ' . $code . ')'));
}
out($d);

/* ---------- أدوات ---------- */
function http($method, $url, $headers, $payload, $timeout, $failOnError = true) {
  $ch = curl_init($url);
  $opts = [CURLOPT_RETURNTRANSFER => true, CURLOPT_HTTPHEADER => $headers,
           CURLOPT_TIMEOUT => $timeout, CURLOPT_CONNECTTIMEOUT => 10];
  if ($method === 'POST') {
    $opts[CURLOPT_POST] = true;
    $opts[CURLOPT_POSTFIELDS] = json_encode($payload, JSON_UNESCAPED_UNICODE);
  }
  curl_setopt_array($ch, $opts);
  $body = curl_exec($ch);
  $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
  if ($body === false) {
    $e = curl_error($ch);
    curl_close($ch);
    if ($failOnError) fail(502, 'تعذّر الاتصال بخادم FiqhQA: ' . $e);
    return [0, null];
  }
  curl_close($ch);
  return [$code, $body];
}
function out($d) { echo json_encode($d, JSON_UNESCAPED_UNICODE); exit; }
function fail($code, $msg) { http_response_code($code); out(['error' => $msg]); }
