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


def log(msg: str):
    print(f"[handler] {msg}", flush=True)


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
    except Exception as e:
        log(f"_init_hardware_info: torch.cuda check failed: {e}")
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
    log(f"_init_hardware_info: vram_gib={vram_gib:.2f}, auto_cap={AUTO_MAX_TOKENS_CAP}")


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

    sampling = {
        "max_tokens": safe_max,
        "max_tokens_requested": requested,
        "max_tokens_cap": auto_cap,
        "temperature": temperature,
        "top_p": top_p,
    }
    log(f"_sanitize_sampling_params: {sampling}")
    return sampling


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
            log(f"_build_messages: using OpenAI-style messages, count={len(messages)}")
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

    log(f"_build_messages: built legacy messages, count={len(messages)}")
    return messages


# ----------------------------
# SGLang server management
# ----------------------------

def _ensure_sglang_server():
    """
    Start `python -m sglang.launch_server` once per container and wait
    for /health to be ready.
    """
    global SGLANG_PROC, MODEL_PATH

    if MODEL_PATH is None:
        MODEL_PATH = os.getenv(
            "MODEL_PATH",
            "DavidAU/Llama3.2-24B-A3B-II-Dark-Champion-INSTRUCT-Heretic-Abliterated-Uncensored",
        )

    log(f"_ensure_sglang_server: MODEL_PATH={MODEL_PATH}, PORT={SGLANG_PORT}")

    # If already running and healthy, just return
    if SGLANG_PROC is not None and SGLANG_PROC.poll() is None:
        try:
            r = requests.get(
                f"http://127.0.0.1:{SGLANG_PORT}/health",
                timeout=1.0,
            )
            if r.status_code == 200:
                log("_ensure_sglang_server: existing SGLang healthy, reusing.")
                return
            else:
                log(f"_ensure_sglang_server: existing /health={r.status_code}, restarting.")
        except Exception as e:
            log(f"_ensure_sglang_server: existing SGLang health check failed: {e}, restarting.")

    cmd = [
        "python3",
        "-u",
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
        "--trust-remote-code",
    ]

    log(f"_ensure_sglang_server: launching: {' '.join(cmd)}")

    SGLANG_PROC = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
    )

    log("_ensure_sglang_server: waiting for /health ...")

    wait_timeout = int(os.getenv("SGLANG_WAIT_TIMEOUT", "1800"))
    deadline = time.time() + wait_timeout
    last_err = None

    while time.time() < deadline:
        ret = SGLANG_PROC.poll()
        if ret is not None:
            raise RuntimeError(f"SGLang server exited early with code {ret}. Check logs above.")

        try:
            r = requests.get(
                f"http://127.0.0.1:{SGLANG_PORT}/health",
                timeout=5.0,
            )
            log(f"_ensure_sglang_server: /health status={r.status_code}")
            if r.status_code == 200:
                log("_ensure_sglang_server: SGLang server is healthy.")
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
            log(f"_ensure_sglang_server: waiting for server... {last_err}")

        time.sleep(5)

    raise RuntimeError(f"SGLang server failed to become healthy within {wait_timeout}s: {last_err}")


# ----------------------------
# Debug handler (no model load)
# ----------------------------

def handle_debug(event: dict) -> dict:
    """
    Lightweight debug path. Does NOT start SGLang.
    Use this to verify env, GPU, and cache wiring.
    """
    log("handle_debug: debug flag detected, returning diagnostics.")
    _init_hardware_info()

    cache_root = os.getenv("HF_HOME", "/runpod-volume/huggingface-cache")
    cache_exists = os.path.exists(cache_root)
    cache_sample = []
    if cache_exists:
        try:
            cache_sample = sorted(os.listdir(cache_root))[:5]
        except Exception as e:
            cache_sample = [f"<error listing cache dir: {e}>"]

    return {
        "debug": True,
        "env": {
            "MODEL_PATH": os.getenv("MODEL_PATH", MODEL_PATH),
            "HF_HOME": os.getenv("HF_HOME"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
            "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
            "SGLANG_PORT": SGLANG_PORT,
        },
        "hardware": {
            "gpu_vram_gib": GPU_VRAM_GIB,
            "auto_max_tokens_cap": AUTO_MAX_TOKENS_CAP,
        },
        "cache": {
            "root": cache_root,
            "exists": cache_exists,
            "sample_entries": cache_sample,
        },
        "echo_input_keys": sorted(event.keys()),
    }


# ----------------------------
# Core generation logic
# ----------------------------

def generate_from_event(event: dict) -> dict:
    """
    Take `event` (OpenAI-style or legacy) and run it via
    SGLang's OpenAI-compatible /v1/chat/completions HTTP API.
    """
    log(f"generate_from_event: start, keys={list(event.keys())}")
    _init_hardware_info()
    _ensure_sglang_server()

    try:
        messages = _build_messages(event)
    except ValueError as ve:
        log(f"generate_from_event: message build error: {ve}")
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
    log(f"generate_from_event: POST {url} with max_tokens={sampling['max_tokens']}")

    try:
        resp = requests.post(url, json=payload, timeout=900)
    except Exception as e:
        log(f"generate_from_event: HTTP error contacting SGLang: {e}")
        return {
            "error": f"Error contacting SGLang server: {e}",
            "endpoint": url,
        }

    if not resp.ok:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        log(f"generate_from_event: SGLang HTTP {resp.status_code}, body={body}")
        return {
            "error": f"SGLang HTTP {resp.status_code}",
            "body": body,
        }

    try:
        out = resp.json()
    except Exception as e:
        log(f"generate_from_event: failed to parse JSON: {e}")
        return {
            "error": f"Failed to parse SGLang response JSON: {e}",
            "raw": resp.text,
        }

    try:
        content = out["choices"][0]["message"]["content"]
    except Exception as exc:
        log(f"generate_from_event: unexpected SGLang format: {exc}")
        return {
            "error": f"Unexpected SGLang output format: {exc}",
            "raw": out,
        }

    log("generate_from_event: success, returning response.")
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

    OR C) debug:
        {
          "debug": true
        }
    """
    try:
        log(f"handler: received job keys={list(job.keys())}")
        event = job.get("input", {}) or {}
        if not isinstance(event, dict):
            log("handler: job['input'] is not a dict.")
            return {"error": "job['input'] must be a JSON object/dict."}

        log(f"handler: input keys={list(event.keys())}")

        if event.get("debug") or event.get("action") == "debug":
            return handle_debug(event)

        return generate_from_event(event)

    except Exception as e:
        log(f"handler: exception: {e}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    log("__main__: starting RunPod serverless worker.")
    runpod.serverless.start({"handler": handler})
