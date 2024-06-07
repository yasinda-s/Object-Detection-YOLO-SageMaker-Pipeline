FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip cache purge && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=120 \
    numpy \
    pandas \
    boto3 \
    ultralytics

ENV PYTHONUNBUFFERED=1