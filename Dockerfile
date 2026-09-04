# =============================================================================
# Code_Down_ML — container image
# Transaction Classification Automation
#
# Lightweight by default: TF-IDF similarity + Logistic Regression (no torch).
# Semantic / SetFit engines are optional and degrade gracefully when absent.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Demo behaviour + Hugging Face persistent storage (mounts at /data).
    FILLDOWN_DEMO=1 \
    FILLDOWN_DATA_DIR=/data \
    FILLDOWN_MAX_ROWS=20000 \
    HF_HOME=/data/.huggingface

# HF Spaces run as UID 1000 — create a matching non-root user.
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python deps first so this layer caches across code changes.
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY --chown=user:user . .

# Writable runtime dir (works whether or not HF persistent storage is attached).
RUN mkdir -p /data && chown -R user:user /data /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 8501

CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
