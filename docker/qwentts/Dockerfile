# Dockerized QwenTTS one-sample voice cloning environment.
# Build from repo root:
#   docker build -t wakeword-forge-qwentts:latest docker/qwentts

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install qwen-tts soundfile numpy

COPY qwentts_clone_one.py /workspace/qwentts_clone_one.py

CMD ["python3", "/workspace/qwentts_clone_one.py", "--help"]
