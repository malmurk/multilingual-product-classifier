FROM python:3.11-slim

# System deps — onnxruntime needs libgomp for OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY classifier/requirements.txt /app/classifier/requirements.txt
RUN pip install --no-cache-dir -r /app/classifier/requirements.txt

# Pre-download the XLM-R tokenizer so the container can start offline
# (must match the backbone used at training time)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('xlm-roberta-base')"

# App code
COPY classifier /app/classifier

# Bake ONNX models (~3.3 GB) into the image so the deploy bundle is
# self-contained — IT just `docker load` and `docker compose up`.
# Retrains require rebuild + reupload of the image; acceptable given
# infrequent cadence.
COPY models/onnx /app/models

ENV MODEL_DIR=/app/models \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Default = HTTP service. Override CMD in compose for the worker.
CMD ["uvicorn", "classifier.service:app", "--host", "0.0.0.0", "--port", "8000"]
