#!/usr/bin/env python
import os
import time
import subprocess
import traceback

import runpod
import torch
import requests

# ----------------------------
# Globals
# ----------------------------

GPU_VRAM_GIB = None          # float or None
AUTO_MAX_TOKENS_CAP = None   # int or None
MODEL_PATH = None            # HF repo or path
SGLANG_PORT = int(os.getenv("SGLANG_PORT", "30000"))
SGLANG_PROC = None           # subprocess.Popen for sglang.launch_server


# ----------------------------
# Hardware / scaling helpers
# ----------------------------

def _init_hardware_info():
    """
    Detect GPU VRAM once and compute an automatic max_tokens cap.

    Heuristic, tuned for ~24B quantized model on 48–80GB cards.
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
        vram_gib = 0.0

    GPU_VRAM_GIB = vram_gib

    if vram_gib <= 0:
        cap = 512
    elif vram_gib < 24:
        cap = 1024
    elif vram_gib < 40:
        cap = 2048
    elif vram_gib < 60:
        cap = 4096       # 48 GB case
    elif vram_gib < 90:
        cap = 8192       # 80 GB case
    else:
        cap = 16384

    AUTO_MAX_TOKENS_CAP = cap


def _sanitize_sampling_params(event: dict):
    """
    Validate and clamp user sampling params against our auto cap.

    Supports both OpenAI-style `max_tokens` and `max_completion_tokens`.
    """
    _init_hardware_info()

    auto_cap = AUTO_MAX_TOKENS_CAP or 2048

    # --- max_tokens / max_completion_tokens ---
    default_max = min(2048, auto_cap)
    raw_requested = event.get(
        "max_tokens",
        event.get("max_completion_tokens", default_max),
    )

    try:
        requested = int(raw_requested)
    except (TypeError, ValueError):
        requested = default_max

    if requested <= 0:
        requested = default_max

    safe_max = max(1, min(requested, auto_cap))

    # --- temperature ---
    raw_temp = event.get("temperature", 0.7)
    try:
        temperature = float(raw_temp)
    except (TypeError, ValueError):
        temperature = 0.7
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

    Mode A: full OpenAI body:
        { "model": "...", "messages": [ {role, content}, ... ], ... }

    Mode B: legacy:
        { "prompt": "...", "system_prompt"/"system": "...", ... }
    """
    # --- Mode A: OpenAI-style messages ---
    raw_messages = event.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
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

    # --- Mode B: simple prompt ---
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
# SGLang server management
# ----------------------------

def _ensure_sglang_server():
    """
    Start `python -m sglang.launch_server` once per container and wait
    for /v1/chat/completions to be ready.
    """
    global SGLANG_PROC, MODEL_PATH

    if MODEL_PATH is None:
        MODEL_PATH = os.getenv(
            "MODEL_PATH",
            "DavidAU/Llama3.2-24B-A3B-II-Dark-Champion-INSTRUCT-Heretic-Abliterated-Uncensored",
        )

    # If already running and healthy, just return
    if SGLANG_PROC is not None and SGLANG_PROC.poll() is None:
        try:
            r = requests.get(
                f"http://127.0.0.1:{SGLANG_PORT}/health",
                timeout=1.0,
            )
            if r.status_code == 200:
                return
        except Exception:
            # fall through and restart
            pass

    # (Re)start server
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
        # Keep these conservative; tweak if you want
        "--mem-fraction-static",
        "0.60",
        "--disable-cuda-graph",
    ]

    # In serverless, we *must not* block forever here, but we do need to
    # load the model once. Cold start will be chunky, then it's warm.
    SGLANG_PROC = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for health endpoint
    deadline = time.time() + 600  # 10 min max for initial model load
    last_err = None

    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://127.0.0.1:{SGLANG_PORT}/health",
                timeout=2.0,
            )
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(2)

    raise RuntimeError(f"SGLang server failed to become healthy: {last_err}")


# ----------------------------
# Core generation logic
# ----------------------------

def generate_from_event(event: dict) -> dict:
    """
    Take `event` (OpenAI-style or legacy) and run it via
    SGLang's OpenAI-compatible /v1/chat/completions HTTP API.
    """
    _init_hardware_info()
    _ensure_sglang_server()

    try:
        messages = _build_messages(event)
    except ValueError as ve:
        return {"error": str(ve)}

    sampling = _sanitize_sampling_params(event)

    requested_model = event.get("model") or MODEL_PATH

    payload = {
        "model": requested_model,
        "messages": messages,
        "max_tokens": sampling["max_tokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "stream": False,
    }

    url = f"http://127.0.0.1:{SGLANG_PORT}/v1/chat/completions"

    try:
        resp = requests.post(url, json=payload, timeout=900)
    except Exception as e:
        return {
            "error": f"Error contacting SGLang server: {e}",
            "endpoint": url,
        }

    if not resp.ok:
        # Bubble up SGLang's error body if present
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {
            "error": f"SGLang HTTP {resp.status_code}",
            "body": body,
        }

    try:
        out = resp.json()
    except Exception as e:
        return {
            "error": f"Failed to parse SGLang response JSON: {e}",
            "raw": resp.text,
        }

    # Standard OpenAI-style ChatCompletion
    try:
        content = out["choices"][0]["message"]["content"]
    except Exception as exc:
        return {
            "error": f"Unexpected SGLang output format: {exc}",
            "raw": out,
        }

    return {
        "response": content,
        "usage": out.get("usage", None),
        "model": out.get("model", requested_model),
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
        "raw": out,
    }


# ----------------------------
# RunPod entrypoint
# ----------------------------

def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Expects job["input"] to be either:

    A) OpenAI-style body:
        {
          "model": "whatever",   # optional, defaults to MODEL_PATH
          "messages": [...],
          "max_tokens": ...,
          "temperature": ...,
          "top_p": ...,
        }

    OR B) legacy body:
        {
          "prompt": "...",
          "system_prompt": "...",  # optional
          "max_tokens": ...,
          "temperature": ...,
          "top_p": ...
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
    runpod.serverless.start({"handler": handler})

