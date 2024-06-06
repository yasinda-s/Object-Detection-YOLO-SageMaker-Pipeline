FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --fix-missing build-essential

RUN apt-get install -y build-essential
RUN apt-get install -y libffi-dev
RUN apt-get install -y gfortran liblapack-dev libblas-dev
RUN apt-get install -y libsqlite3-dev libgeos-dev
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    boto3 \
    sagemaker \
    ultralytics

ENV PYTHONUNBUFFERED=1