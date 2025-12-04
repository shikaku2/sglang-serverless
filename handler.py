#!/usr/bin/env python
import os
import time
import traceback

import runpod
import torch
from sglang import Runtime, ChatCompletion

# ----------------------------
# Globals
# ----------------------------

runtime = None          # SGLang runtime (model) – reused across jobs
GPU_VRAM_GIB = None     # float or None
AUTO_MAX_TOKENS_CAP = None  # int or None
MODEL_PATH = None       # the actual HF model we loaded


# ----------------------------
# Hardware / scaling helpers
# ----------------------------

def _init_hardware_info():
    """
    Detect GPU VRAM once and compute an automatic max_tokens cap.

    NOTE: This is deliberately conservative. The VRAM → tokens mapping
    is a heuristic, not an exact formula.
    """
    global GPU_VRAM_GIB, AUTO_MAX_TOKENS_CAP

    if GPU_VRAM_GIB is not None and AUTO_MAX_TOKENS_CAP is not None:
        return

    vram_gib = 0.0
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gib = props.total_memory / (1024 ** 3)
    except Exception:
        # If anything goes weird, treat as "no GPU info"
        vram_gib = 0.0

    GPU_VRAM_GIB = vram_gib

    # Heuristic caps tuned for a ~24B quantized model:
    # - <24GB: 1k tokens cap (barely safe)
    # - 24–40GB: 2k tokens
    # - 40–60GB (your 48GB case): 4k tokens
    # - 60–90GB (your 80GB case): 8k tokens
    # - >90GB: 16k tokens
    if vram_gib <= 0:
        cap = 512
    elif vram_gib < 24:
        cap = 1024
    elif vram_gib < 40:
        cap = 2048
    elif vram_gib < 60:
        cap = 4096
    elif vram_gib < 90:
        cap = 8192
    else:
        cap = 16384

    AUTO_MAX_TOKENS_CAP = cap


def _sanitize_sampling_params(event: dict):
    """
    Validate and clamp user sampling params against our auto cap.

    Supports both OpenAI-style `max_tokens` and `max_completion_tokens`.

    - If user asks for too many tokens, we clamp to AUTO_MAX_TOKENS_CAP.
    - If user sends garbage types, we fall back to sane defaults.
    """
    _init_hardware_info()

    # If hardware detection failed for some reason, still have a default cap
    auto_cap = AUTO_MAX_TOKENS_CAP or 2048

    # --- max_tokens / max_completion_tokens ---
    default_max = min(2048, auto_cap)  # default, but never above auto_cap
    raw_requested = event.get(
        "max_tokens",
        event.get("max_completion_tokens", default_max)
    )

    try:
        requested = int(raw_requested)
    except (TypeError, ValueError):
        requested = default_max

    if requested <= 0:
        requested = default_max

    # Clamp to [1, auto_cap]
    safe_max = max(1, min(requested, auto_cap))

    # --- temperature ---
    raw_temp = event.get("temperature", 0.7)
    try:
        temperature = float(raw_temp)
    except (TypeError, ValueError):
        temperature = 0.7
    # clamp to [0, 2]
    if temperature < 0.0:
        temperature = 0.0
    elif temperature > 2.0:
        temperature = 2.0

    # --- top_p ---
    raw_top_p = event.get("top_p", 0.95)
    try:
        top_p = float(raw_top_p)
    except (TypeError, ValueError):
        top_p = 0.95
    # clamp to (0, 1]
    if top_p <= 0.0:
        top_p = 0.01
    elif top_p > 1.0:
        top_p = 1.0

    return {
        "max_tokens": safe_max,
        "max_tokens_requested": requested,
        "max_tokens_cap": auto_cap,
        "temperature": temperature,
        "top_p": top_p,
    }


def _build_messages(event: dict):
    """
    Build an OpenAI-style `messages` array from the incoming payload.

    Supports two modes:

    1) OpenAI chat-style:

        {
          "model": "...",          # ignored for now
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
          ],
          ...
        }

    2) Legacy simple payload:

        {
          "prompt": "...",
          "system_prompt": "...",   # optional
          ...
        }
    """
    # --- Mode 1: OpenAI-style messages ---
    raw_messages = event.get("messages")
    if isinstance(raw_messages, list) and len(raw_messages) > 0:
        messages = []
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            messages.append({"role": role, "content": content})

        if messages:
            return messages

    # --- Mode 2: fallback to {system_prompt, prompt} ---
    system_prompt = event.get("system_prompt") or event.get("system")
    prompt = (
        event.get("prompt")
        or event.get("input")
        or event.get("text")
    )

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "No valid 'messages' array and no non-empty 'prompt' found in input."
        )

    messages = []
    if isinstance(system_prompt, str) and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# ----------------------------
# Model loading
# ----------------------------

def load_runtime():
    """
    Lazily initialize the SGLang runtime for your HF model.
    """
    global runtime, MODEL_PATH
    if runtime is not None:
        return runtime

    _init_hardware_info()

    model_path = os.getenv(
        "MODEL_PATH",
        "DavidAU/Llama3.2-24B-A3B-II-Dark-Champion-INSTRUCT-Heretic-Abliterated-Uncensored",
    )
    MODEL_PATH = model_path

    # If the HF repo is gated/private, you'll need HUGGINGFACE_HUB_TOKEN
    # or HF_TOKEN set in the environment.

    runtime = Runtime(
        model=model_path,
        mem_fraction_static=0.60,
        mem_fraction_dynamic=0.30,
        disable_torch_compile=True,
        disable_cuda_graph=True,
        tokenizer_mode="auto",
        dtype="bfloat16",  # change to "float16" if bf16 is a problem
    )

    return runtime


# ----------------------------
# Core generation logic
# ----------------------------

def generate_from_event(event: dict) -> dict:
    """
    Core logic: takes an `event` dict (your input payload)
    and returns a structured response.

    `event` is treated as:
      - an OpenAI /v1/chat/completions body (preferred), OR
      - your legacy {prompt, system_prompt, ...} format.
    """
    rt = load_runtime()

    try:
        messages = _build_messages(event)
    except ValueError as ve:
        return {"error": str(ve)}

    sampling = _sanitize_sampling_params(event)

    out = ChatCompletion.create(
        runtime=rt,
        messages=messages,
        max_tokens=sampling["max_tokens"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        stream=False,
    )

    try:
        # Assume SGLang returns OpenAI-style dict
        content = out["choices"][0]["message"]["content"]
    except Exception as exc:
        # If SGLang ever changes format, still return *something*
        return {
            "error": f"Unexpected SGLang output format: {exc}",
            "raw": str(out),
        }

    requested_model = event.get("model") or MODEL_PATH

    return {
        # Simple text for quick consumption
        "response": content,

        # Try to surface SGLang's own metadata if present
        "usage": out.get("usage", None) if isinstance(out, dict) else None,
        "model": out.get("model", requested_model) if isinstance(out, dict) else requested_model,

        # Our own metadata and safety clamps
        "sampling": {
            "max_tokens_used": sampling["max_tokens"],
            "max_tokens_requested": sampling["max_tokens_requested"],
            "max_tokens_cap": sampling["max_tokens_cap"],
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
        },
        "hardware": {
            "gpu_vram_gib": GPU_VRAM_GIB,
        },
    }


# ----------------------------
# RunPod entrypoint
# ----------------------------

def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Expects either:

    A) OpenAI-style body (inside job["input"]):

        {
          "id": "...",                 # added by RunPod
          "input": {
            "model": "any-string",     # optional, just echoed back
            "messages": [
              {"role": "system", "content": "..."},
              {"role": "user",   "content": "..."}
            ],
            "max_tokens": 4096,        # optional
            "temperature": 0.7,        # optional
            "top_p": 0.95,             # optional
            "stream": false            # ignored; we always run non-streaming
          }
        }

    OR the legacy payload (inside job["input"]):

        {
          "prompt": "....",
          "system_prompt": "...",      # optional
          "max_tokens": 4096,          # optional
          "temperature": 0.7,          # optional
          "top_p": 0.95                # optional
        }
    """
    try:
        event = job.get("input", {}) or {}
        if not isinstance(event, dict):
            return {"error": "job['input'] must be a JSON object/dict."}

        return generate_from_event(event)

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    # RunPod's serverless wrapper
    runpod.serverless.start({"handler": handler})

