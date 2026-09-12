"""MiniMax-H3 on ComfyUI — API gateway.

Fronts a local ComfyUI (127.0.0.1:8188) and re-exposes the *same* JSON API the
SGLang deployment used, so the metaso-style web UI works unchanged:

    GET  /api/health          gateway + ComfyUI + GPU
    GET  /api/variant         single hybrid model -> no partition switching
    POST /api/generate        {prompt, short_edge, aspect_ratio, duration_seconds,
                               num_inference_steps, conditions[], seed} -> {"id"}
    GET  /api/status/{id}     job status / progress
    GET  /api/video/{id}      finished MP4 (Range-aware, ?download=1 = attachment)
    GET  /api/tasks           recent jobs
    POST /api/upload          image -> ComfyUI/input, returns {"uri": "file://..."}

The workflow replicated here is "低分辨率一采样 -> 3D latent 放大 -> 二采样"
(the community Latent-upscaler recipe) on the hybrid fl2va+ref2va int8 DiT with
the 4-step Turbo LoRA.

Run: uvicorn comfy_gateway:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from typing import Any

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

COMFY = os.environ.get("COMFY_BASE", "http://127.0.0.1:8188")
COMFY_ROOT = os.environ.get("COMFY_ROOT", "/content/ComfyUI")
COMFY_INPUT = os.path.join(COMFY_ROOT, "input")
UPLOAD_DIR = os.environ.get("H3_UPLOAD_DIR", "/content/h3comfy/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(COMFY_INPUT, exist_ok=True)

DIFFUSION_MODEL = os.environ.get(
    "H3_UNET", "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors")
TEXT_ENCODER = os.environ.get(
    "H3_CLIP", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
VIDEO_VAE = os.environ.get("H3_VIDEO_VAE", "minimax_h3_video_vae_fp16.safetensors")
AUDIO_VAE = os.environ.get("H3_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors")
TURBO_LORA = os.environ.get(
    "H3_LORA", "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors")
LORA_STRENGTH = float(os.environ.get("H3_LORA_STRENGTH", "0.75"))
UPSCALER = os.environ.get("H3_UPSCALER", "minimax_h3_latent_upscaler_3d_fp16.safetensors")
UPSCALE_SCALE = float(os.environ.get("H3_UPSCALE_SCALE", "1.5"))
SHIFT_VIDEO = float(os.environ.get("H3_SHIFT_VIDEO", "6"))   # 768p Turbo values
SHIFT_AUDIO = float(os.environ.get("H3_SHIFT_AUDIO", "3"))
FPS = 24

app = FastAPI(title="MiniMax-H3 ComfyUI gateway")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                   expose_headers=["*"])

# ------------------------------------------------------------------ job bookkeeping
JOBS: dict[str, dict[str, Any]] = {}
PROGRESS: dict[str, tuple[int, int]] = {}
CLIENT_ID = uuid.uuid4().hex


def _ws_listen() -> None:
    """Track per-prompt step progress from ComfyUI's websocket."""
    try:
        from websockets.sync.client import connect
    except Exception:  # noqa: BLE001
        return
    while True:
        try:
            with connect(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}",
                         open_timeout=10, close_timeout=5) as ws:
                for raw in ws:
                    try:
                        m = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        continue
                    d = m.get("data") or {}
                    if m.get("type") == "progress":
                        pid = d.get("prompt_id")
                        if pid:
                            PROGRESS[pid] = (int(d.get("value") or 0), int(d.get("max") or 1))
                    elif m.get("type") in ("execution_start",):
                        pid = d.get("prompt_id")
                        if pid:
                            PROGRESS[pid] = (0, 1)
        except Exception:  # noqa: BLE001
            time.sleep(3)


threading.Thread(target=_ws_listen, daemon=True).start()


def _frames_for(seconds: float) -> int:
    """H3 uses 17n+5 frame buckets at 24 fps."""
    want = max(5, int(round(seconds * FPS)))
    n = max(0, int(round((want - 5) / 17.0)))
    return 17 * n + 5


def _build_workflow(prompt: str, width: int, height: int, frames: int,
                    steps: int, seed: int, ref_files: list[str]) -> dict[str, Any]:
    """The Latent-dual-sampling graph: low-res sample -> 3D latent upscale -> 2nd sample."""
    first = max(1, steps // 2)
    wf: dict[str, Any] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": TURBO_LORA,
                         "strength_model": LORA_STRENGTH}},
        "3": {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["2", 0], "shift_video": SHIFT_VIDEO,
                         "shift_audio": SHIFT_AUDIO}},
        "4": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "7": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0],
                         "prompt": prompt, "width": width, "height": height,
                         "length": frames, "ref_image_size": "match"}},
        "8": {"class_type": "BasicGuider",
              "inputs": {"model": ["3", 0], "conditioning": ["7", 0]}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["3", 0], "scheduler": "simple", "steps": steps,
                         "denoise": 1.0}},
        "10": {"class_type": "H3SigmaRefiner",
               "inputs": {"sigmas": ["9", 0], "extra_steps": 1, "start_at_sigma": 0.7,
                          "end_at_sigma": 0.0, "spacing": "cosine"}},
        "11": {"class_type": "SplitSigmas",
               "inputs": {"sigmas": ["10", 0], "step": first}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        # ---- stage 1: low resolution ----
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["13", 0], "guider": ["8", 0], "sampler": ["12", 0],
                          "sigmas": ["11", 0], "latent_image": ["7", 1]}},
        "15": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["14", 0]}},
        # ---- latent upscale (video only; audio rides through) ----
        "16": {"class_type": "MinimaxH3LatentUpscaler3D",
               "inputs": {"latent": ["15", 0], "model_name": UPSCALER,
                          "mode": {"mode": "scale by multiplier", "scale": UPSCALE_SCALE},
                          "align": 32, "enable_temporal_chunking": True,
                          "force_unload": True, "device": "cuda", "precision": "fp16"}},
        "17": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["16", 0], "audio_latent": ["15", 1]}},
        # ---- stage 2: full resolution ----
        "18": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["13", 0], "guider": ["8", 0], "sampler": ["12", 0],
                          "sigmas": ["11", 1], "latent_image": ["17", 0]}},
        "19": {"class_type": "VAEDecode",
               "inputs": {"samples": ["18", 0], "vae": ["5", 0]}},
        "20": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["18", 0], "vae": ["6", 0]}},
        "21": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["19", 0], "audio": ["20", 0], "frame_rate": FPS,
                          "loop_count": 0, "filename_prefix": "h3/h3",
                          "format": "video/h264-mp4", "pingpong": False,
                          "save_output": True}},
    }
    for i, name in enumerate(ref_files[:9]):
        nid = str(40 + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["7"]["inputs"][f"ref_image_{i}"] = [nid, 0]
    return wf


# ---------------------------------------------------------------------- endpoints
def _comfy_up() -> bool:
    try:
        r = requests.get(f"{COMFY}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _gpu() -> str:
    try:
        return subprocess.run(
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader", shell=True, capture_output=True, text=True,
            timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


@app.get("/api/health")
def health() -> JSONResponse:
    up = _comfy_up()
    return JSONResponse({
        "gateway": "ok", "backend": "comfyui",
        "model": "MiniMax-H3 (hybrid fl2va+ref2va, int8 + Turbo LoRA)",
        "sglang": {"reachable": up, "status_code": 200 if up else 503},
        "variant": "hybrid", "variant_state": "ready", "variant_target": None,
        "gpu": _gpu(), "time": int(time.time()),
    })


@app.get("/api/variant")
def get_variant() -> JSONResponse:
    # the hybrid checkpoint serves text/keyframe/reference jobs from one load
    return JSONResponse({"current": "hybrid", "state": "ready", "target": None,
                         "since": 0, "note": "hybrid model: no partition switching"})


def _clean_conditions(raw: Any) -> list[dict[str, Any]]:
    out = []
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        uri = str(c.get("uri") or "")
        if not uri.startswith("file://"):
            continue
        role = str(c.get("role") or "reference").lower()
        if role not in ("reference", "keyframe"):
            continue
        out.append({"uri": uri, "role": role, "frame_index": c.get("frame_index")})
    return out


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    if not _comfy_up():
        return JSONResponse({"error": "ComfyUI is not running"}, status_code=503)

    try:
        short_edge = int(body.get("short_edge") or 768)
    except Exception:  # noqa: BLE001
        short_edge = 768
    aspect = str(body.get("aspect_ratio") or "16:9")
    ratio = {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0,
             "21:9": 21 / 9}.get(aspect, 16 / 9)
    if short_edge >= 1344:
        ratio = 1.0
    if ratio >= 1:
        height, width = short_edge, int(round(short_edge * ratio))
    else:
        width, height = short_edge, int(round(short_edge / ratio))
    width = max(64, (width // 32) * 32)
    height = max(64, (height // 32) * 32)

    try:
        duration = float(body.get("duration_seconds") or 4.0)
    except Exception:  # noqa: BLE001
        duration = 4.0
    duration = max(4.0, min(duration, 15.0))
    frames = _frames_for(duration)

    try:
        steps = int(body.get("num_inference_steps") or 8)
    except Exception:  # noqa: BLE001
        steps = 8
    steps = max(2, min(steps, 50))

    seed = body.get("seed")
    try:
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
    except Exception:  # noqa: BLE001
        seed = random.randint(0, 2**31 - 1)

    # reference images: accept the WebUI's file:// URIs and copy them into input/
    ref_files: list[str] = []
    for c in _clean_conditions(body.get("conditions")):
        if c["role"] != "reference":
            continue
        src = c["uri"][len("file://"):]
        if not os.path.exists(src):
            continue
        name = "h3ref_" + os.path.basename(src)
        try:
            shutil.copyfile(src, os.path.join(COMFY_INPUT, name))
            ref_files.append(name)
        except OSError:
            pass

    wf = _build_workflow(prompt, width, height, frames, steps, seed, ref_files)
    try:
        r = requests.post(f"{COMFY}/prompt",
                          json={"prompt": wf, "client_id": CLIENT_ID}, timeout=120)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"ComfyUI submit failed: {e!r}"}, status_code=502)
    if r.status_code >= 400:
        return JSONResponse({"error": "ComfyUI rejected the workflow",
                             "detail": r.text[:1500]}, status_code=r.status_code)
    pid = r.json().get("prompt_id")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"prompt_id": pid, "created": time.time(), "params": {
        "width": width, "height": height, "frames": frames, "steps": steps,
        "first_pass": max(1, steps // 2), "seed": seed, "duration": duration,
        "refs": len(ref_files), "aspect": aspect}}
    return JSONResponse({"id": job_id, "object": "video", "status": "queued",
                         "progress": 0, "created_at": int(time.time()),
                         "seconds": str(round(frames / FPS, 3)),
                         "size": f"{width}x{height}",
                         "_submitted": {"task": "hybrid", "variant": "hybrid",
                                        "seed": seed, "steps": steps,
                                        "short_edge": short_edge, "duration": duration,
                                        "aspect": aspect,
                                        "references": len(ref_files), "keyframes": 0}})


@app.get("/api/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    pid = job["prompt_id"]
    try:
        hist = requests.get(f"{COMFY}/history/{pid}", timeout=30).json()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": repr(e)}, status_code=502)
    el = time.time() - job["created"]
    if pid in hist:
        entry = hist[pid]
        st = (entry.get("status") or {})
        if st.get("status_str") == "error" or not st.get("completed", True):
            msgs = json.dumps(st.get("messages") or [])[:800]
            return JSONResponse({"id": job_id, "status": "failed", "progress": 0,
                                 "error": {"message": msgs}})
        files = []
        for node_out in (entry.get("outputs") or {}).values():
            for k in ("gifs", "videos", "images"):
                for f in (node_out.get(k) or []):
                    files.append(f)
        if not files:
            return JSONResponse({"id": job_id, "status": "failed", "progress": 100,
                                 "error": {"message": "no output file produced"}})
        f = files[-1]
        job["file"] = f
        return JSONResponse({"id": job_id, "status": "completed", "progress": 100,
                             "inference_time_s": el, "file_path": f.get("filename"),
                             "peak_memory_mb": None})
    value, total = PROGRESS.get(pid, (0, 1))
    pct = int(95 * value / total) if total else 0
    return JSONResponse({"id": job_id, "status": "running" if value else "queued",
                         "progress": pct, "step": value, "total_steps": total,
                         "elapsed_s": round(el, 1)})


@app.get("/api/tasks")
def tasks() -> JSONResponse:
    data = [{"id": k, "status": "completed" if v.get("file") else "running",
             "created_at": int(v["created"])} for k, v in JOBS.items()]
    return JSONResponse({"data": data[-20:]})


@app.get("/api/video/{job_id}")
def video(job_id: str, request: Request, download: int = 0) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job or "file" not in job:
        return JSONResponse({"error": "not ready"}, status_code=404)  # type: ignore[return-value]
    f = job["file"]
    params = {"filename": f.get("filename"), "type": f.get("type", "output"),
              "subfolder": f.get("subfolder", "")}
    fwd = {}
    rng = request.headers.get("range")
    if rng:
        fwd["Range"] = rng
    up = requests.get(f"{COMFY}/view", params=params, headers=fwd, stream=True,
                      timeout=600)

    def it():
        for chunk in up.iter_content(chunk_size=1 << 20):
            if chunk:
                yield chunk

    headers = {"Accept-Ranges": up.headers.get("accept-ranges", "bytes"),
               "Content-Disposition":
                   f'{"attachment" if download else "inline"}; '
                   f'filename="minimax-h3-{job_id[:8]}.mp4"'}
    for h in ("Content-Length", "Content-Range"):
        if up.headers.get(h):
            headers[h] = up.headers[h]
    return StreamingResponse(it(), media_type=up.headers.get("content-type", "video/mp4"),
                             headers=headers, status_code=up.status_code)


_SAFE = re.compile(r"[^A-Za-z0-9._-]")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    name = _SAFE.sub("_", os.path.basename(file.filename or "img.png"))
    ext = os.path.splitext(name)[1].lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return JSONResponse({"error": f"unsupported type {ext}"}, status_code=400)
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        return JSONResponse({"error": "image too large (max 20MB)"}, status_code=400)
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(dst, "wb") as fh:
        fh.write(data)
    return JSONResponse({"uri": "file://" + dst, "bytes": len(data)})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    ok = _comfy_up()
    return (f"<!doctype html><meta charset=utf-8><title>MiniMax-H3 ComfyUI</title>"
            f"<body style='font:14px/1.6 system-ui;background:#0d0d10;color:#e8e8ee;"
            f"padding:40px'><h2 style='color:#5b8cff'>MiniMax-H3 · ComfyUI gateway</h2>"
            f"<p>comfyui: <b>{'up' if ok else 'down'}</b> @ {COMFY}</p>"
            f"<p>gpu: {_gpu()}</p><p>jobs: {len(JOBS)}</p>"
            f"<p>API: /api/generate · /api/status/&lt;id&gt; · /api/video/&lt;id&gt; · "
            f"/api/upload</p></body>")
