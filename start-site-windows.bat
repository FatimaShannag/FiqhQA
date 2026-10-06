@echo off
REM FiqhQA - local preview. Needs Python (python.org). Keep this window open while browsing.
cd /d "%~dp0"
start "" http://localhost:8080
python -m http.server 8080
