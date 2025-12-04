#!/usr/bin/env python
import os
import time
import subprocess
import traceback
import logging
import json

import runpod
import torch
import requests

# ----------------------------
# Logging setup
# ----------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[handler] %(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("handler")

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
            log.info(
                "Detected GPU: %s, CC=%d.%d, VRAM=%.2f GiB",
                props.name,
                props.major,
                props.minor,
                vram_gib,
            )
        else:
            log.warning("torch.cuda.is_available() is False; assuming no GPU.")
    except Exception as e:
        log.exception("Failed to query CUDA device properties: %s", e)
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
    log.info("Auto max_tokens cap set to %d based on %.2f GiB VRAM", cap, vram_gib)


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

    log.info(
        "Sampling params -> requested: %s, using max_tokens=%d (cap=%d), temp=%.3f, top_p=%.3f",
        raw_requested,
        safe_max,
        auto_cap,
        temperature,
        top_p,
    )

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
            log.info("Using OpenAI-style messages with %d entries", len(messages))
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

    preview = prompt[:120].replace("\n", " ")
    log.info("Using legacy prompt mode, prompt preview: %r...", preview)

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
    log.info("Using MODEL_PATH=%s", MODEL_PATH)

    # If already running and healthy, just return
    if SGLANG_PROC is not None and SGLANG_PROC.poll() is None:
        try:
            r = requests.get(
                f"http://127.0.0.1:{SGLANG_PORT}/health",
                timeout=1.0,
            )
            if r.status_code == 200:
                return
            else:
                log.warning("Existing SGLang server unhealthy, status=%s", r.status_code)
        except Exception as e:
            log.warning("Existing SGLang server health check failed: %s", e)

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
        "--mem-fraction-static",
        "0.60",
        "--disable-cuda-graph",
    ]

    log.info("Starting SGLang server: %s", " ".join(cmd))

    # In serverless, we must not block forever here, but we do need to
    # load the model once. Cold start will be chunky, then it's warm.
    SGLANG_PROC = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        # inherit stdout/stderr so we see SGLang logs in RunPod output
    )

    log.info("SGLang server launched with PID %s", SGLANG_PROC.pid)

    # Quick check if it died immediately
    time.sleep(3)
    if SGLANG_PROC.poll() is not None:
        code = SGLANG_PROC.returncode
        log.error("SGLang server exited immediately with code %s", code)
        raise RuntimeError(f"SGLang server exited early with code {code}. Check logs above.")

    # Wait for health endpoint
    deadline = time.time() + 600  # 10 min max for initial model load
    last_err = None
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(
                f"http://127.0.0.1:{SGLANG_PORT}/health",
                timeout=2.0,
            )
            if r.status_code == 200:
                log.info("SGLang server is healthy on attempt %d", attempt)
                return
            last_err = f"HTTP {r.status_code}"
            if attempt % 10 == 0:
                log.warning("Health check attempt %d: %s", attempt, last_err)
        except Exception as e:
            last_err = repr(e)
            if attempt in (1, 5, 10, 20, 30):
                log.warning("Health check attempt %d failed: %s", attempt, last_err)
        time.sleep(2)

    # If we get here, it never became healthy
    log.error("SGLang server failed to become healthy: %s", last_err)
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
        log.warning("Bad input event: %s", ve)
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
    log.info(
        "Calling SGLang /v1/chat/completions on %s with model=%s, n_messages=%d",
        url,
        requested_model,
        len(messages),
    )

    try:
        resp = requests.post(url, json=payload, timeout=900)
    except Exception as e:
        log.exception("Error contacting SGLang server: %s", e)
        return {
            "error": f"Error contacting SGLang server: {e}",
            "endpoint": url,
        }

    if not resp.ok:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        log.error("SGLang HTTP %s error: %s", resp.status_code, body)
        return {
            "error": f"SGLang HTTP {resp.status_code}",
            "body": body,
        }

    try:
        out = resp.json()
    except Exception as e:
        log.exception("Failed to parse SGLang response JSON: %s", e)
        return {
            "error": f"Failed to parse SGLang response JSON: {e}",
            "raw": resp.text,
        }

    try:
        content = out["choices"][0]["message"]["content"]
    except Exception as exc:
        log.error("Unexpected SGLang output format: %s; raw=%r", exc, out)
        return {
            "error": f"Unexpected SGLang output format: {exc}",
            "raw": out,
        }

    log.info("Generation succeeded, returning response of length %d", len(content))

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
    job_id = job.get("id") or job.get("requestId")
    log.info("Received job id=%s keys=%s", job_id, list(job.keys()))

    try:
        event = job.get("input", {}) or {}
        if not isinstance(event, dict):
            log.error("job['input'] is not a dict: %r", type(event))
            return {"error": "job['input'] must be a JSON object/dict."}

        # Log a small debug summary
        debug_summary = {
            k: event.get(k)
            for k in ("model", "max_tokens", "max_completion_tokens", "temperature", "top_p")
            if k in event
        }
        log.info("Event summary: %s", json.dumps(debug_summary))

        return generate_from_event(event)

    except Exception as e:
        log.exception("Unhandled exception in handler: %s", e)
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    log.info("Starting RunPod serverless worker main loop.")
    runpod.serverless.start({"handler": handler})

