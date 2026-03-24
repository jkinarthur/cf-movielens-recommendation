# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Security: run as non-root
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY model.py          model.py
COPY app.py            app.py
COPY model_artifacts.pt model_artifacts.pt

RUN chown -R appuser:appuser /app
USER appuser

# ── Expose and run ────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
