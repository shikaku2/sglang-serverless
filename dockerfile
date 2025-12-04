# Use the official SGLang GPU image.
# This gives you the latest SGLang + a CUDA 12.x runtime (currently 12.6),
# with sgl_kernel + flashinfer already wired up.
FROM lmsysorg/sglang:latest

# Use RunPod's shared volume for HF cache so model downloads
# are reused across cold starts.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRUST_REMOTE_CODE=true

# We'll keep our own code under /workspace
WORKDIR /workspace

# Install only what we actually need on top of SGLang:
# - runpod: for serverless handler
# - requests: to talk to the local SGLang HTTP server
RUN pip install --no-cache-dir \
    runpod \
    requests

# Copy your RunPod serverless handler
COPY handler.py /workspace/handler.py

# Default command for RunPod Serverless
CMD ["python3", "-u", "handler.py"]

