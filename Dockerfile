FROM python:3.12-slim

WORKDIR /app

ENV HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip wheel \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY models /app/models

ENTRYPOINT ["python", "src/predict.py"]

