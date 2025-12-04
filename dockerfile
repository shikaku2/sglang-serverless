FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Make python3 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache

WORKDIR /workspace

# ---- Python deps ----
# 1) Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# 2) Install CUDA-enabled PyTorch (cu121)
RUN pip3 install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) Install sglang + flashinfer
RUN pip3 install --no-cache-dir \
    --prefer-binary \
    --no-build-isolation \
    "sglang==0.5.6" \
    "flashinfer-python"

# 4) Common LLM deps + RunPod SDK
RUN pip3 install --no-cache-dir \
    "huggingface_hub" \
    "transformers" \
    "requests" \
    "sentencepiece" \
    "runpod"

# Copy handler (serverless entrypoint)
COPY handler.py /workspace/handler.py

# Default command for RunPod Serverless
CMD ["python", "-u", "handler.py"]
