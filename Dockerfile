FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    wget \
    curl \
    gnupg \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libz-dev \
    libgl1 \
    libatlas-base-dev \
    libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y edgetpu-compiler

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .

ENTRYPOINT ["bash", "-c", "python train_model.py && edgetpu_compiler /app/build/model.tflite -o /app/build"]
