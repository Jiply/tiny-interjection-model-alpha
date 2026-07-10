FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && chmod +x nebius/job_entrypoint.sh nebius/train_qwen_lora.sh

ENTRYPOINT ["nebius/job_entrypoint.sh"]
