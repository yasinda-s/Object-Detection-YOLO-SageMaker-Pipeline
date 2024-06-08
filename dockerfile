FROM python:3.8-slim

WORKDIR /app

# Install OS dependencies for PyTorch and other imaging libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6

# Upgrade pip and install Python dependencies, including PyTorch and ultralytics
RUN pip install pip install --no-cache-dir --default-timeout=120 --upgrade pip && \
    pip install torch==2.3.1 numpy pandas boto3 ultralytics

ENV PYTHONUNBUFFERED=1

COPY . /app