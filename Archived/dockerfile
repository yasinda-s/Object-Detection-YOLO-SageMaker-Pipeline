FROM python:3.8-alpine

WORKDIR /app

# Install build dependencies and necessary libraries
RUN apk update && apk add --no-cache \
    build-base \
    libffi-dev \
    openblas-dev \
    ffmpeg \
    libjpeg-turbo-dev \
    libpng-dev

# Additional dependencies for handling X11 operations, if necessary
# RUN apk add --no-cache xvfb-run

# Install Python dependencies, including PyTorch and ultralytics
# PyTorch on Alpine may need to be compiled or you might need to find a compatible wheel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install torch==2.3.1 numpy pandas boto3 ultralytics

ENV PYTHONUNBUFFERED=1

COPY . /app