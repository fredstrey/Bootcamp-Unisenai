FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 libgl1 libxcb1 libx11-6 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY app ./app
COPY static ./static
COPY templates ./templates
COPY best.pt ./best.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
