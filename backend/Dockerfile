# Hugging Face Space (Docker SDK) — permanent public URL for the FiqhQA backend.
# Free "CPU basic" (16 GB RAM) runs retrieval + Gemini tabs, and SILMA slowly (set SILMA_NUM_BEAMS=1).
# Upgrade the Space hardware to "T4 small" during the judging windows for fast SILMA answers.
# Space secrets: GEMINI_API_KEY (required), BACKEND_TOKEN (only if you keep the PHP proxy).
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
# default torch wheel includes CUDA, so the same image works on CPU and after upgrading to a T4
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .

USER user
ENV HF_HOME=/home/user/.cache/huggingface \
    DATA_PATH=/app/data/LastRAG_dataset.csv \
    SILMA_NUM_BEAMS=1 \
    CORS_ORIGINS=*
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
