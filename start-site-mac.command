#!/bin/bash
# FiqhQA - local preview. Double-click; keep the Terminal window open while browsing.
cd "$(dirname "$0")"
(sleep 1; open http://localhost:8080) &
python3 -m http.server 8080
