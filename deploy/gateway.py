"""Minimax-H3 L4 gateway.

Fronts the local sglang MiniMax-H3 server (127.0.0.1:30010) and exposes a small
CORS-enabled JSON API plus image upload, so a statically hosted web UI (GitHub
Pages) can drive it through a cloudflared tunnel.

MiniMax-H3 ships two mutually exclusive checkpoint partitions: `fl2va` serves
text-to-video and keyframes, `ref2va` serves reference-image conditioning. Only
one can be resident on a single 24 GB card, so the gateway tracks which one is
loaded and can swap them on demand (a few minutes: the model is reloaded).

Endpoints
  GET  /                     tiny status page
  GET  /api/health           gateway + sglang + GPU snapshot
  GET  /api/variant          which partition is loaded / switching state
  POST /api/variant          {"variant": "fl2va"|"ref2va"} request a swap
  POST /api/generate         submit a video job   -> {"id": ...} | {"switching": true}
  GET  /api/status/{vid}     job status/progress
  GET  /api/video/{vid}      stream finished mp4
  GET  /api/tasks            recent jobs
  POST /api/upload           multipart image -> {"uri": "file:///..."}

Run: uvicorn gateway:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import uuid
from typing import Any

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

SGLANG_BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:30010")
MODEL_ID = os.environ.get("H3_MODEL_ID", "MiniMaxAI/MiniMax-H3")
UPLOAD_DIR = os.environ.get("H3_UPLOAD_DIR", "/content/h3/uploads")
VARIANT_FILE = os.environ.get("H3_VARIANT_FILE", "/content/h3/variant.json")
SWITCH_SCRIPT = os.environ.get("H3_SWITCH_SCRIPT", "/content/h3/switch_variant.sh")
VARIANTS = ("fl2va", "ref2va")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MiniMax-H3 L4 gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ---------------------------------------------------------------- state helpers
def _variant_state() -> dict[str, Any]:
    try:
        with open(VARIANT_FILE) as f:
            st = json.load(f)
    except Exception:  # noqa: BLE001
        return {"current": None, "state": "unknown", "target": None}
    if st.get("state") == "switching" and time.time() - float(st.get("since", 0)) > 1800:
        st["state"] = "error"
    return st


def _start_switch(variant: str) -> None:
    """Launch the reload detached; it rewrites variant.json as it progresses."""
    try:
        with open(VARIANT_FILE, "w") as f:
            json.dump({"current": None, "state": "switching", "target": variant,
                       "since": time.time()}, f)
    except OSError:
        pass
    try:
        subprocess.Popen(["bash", SWITCH_SCRIPT, variant],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception:  # noqa: BLE001
        pass


def _sglang_health() -> dict[str, Any]:
    try:
        r = requests.get(f"{SGLANG_BASE}/health", timeout=5)
        return {"reachable": True, "status_code": r.status_code}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "error": repr(e)}


def _gpu() -> str:
    try:
        return subprocess.run(
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader",
            shell=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> JSONResponse:
    st = _variant_state()
    return JSONResponse({
        "gateway": "ok",
        "model": MODEL_ID,
        "sglang": _sglang_health(),
        "variant": st.get("current"),
        "variant_state": st.get("state"),
        "variant_target": st.get("target"),
        "gpu": _gpu(),
        "time": int(time.time()),
    })


@app.get("/api/variant")
def get_variant() -> JSONResponse:
    return JSONResponse(_variant_state())


@app.post("/api/variant")
async def set_variant(request: Request) -> JSONResponse:
    body = await request.json()
    want = str(body.get("variant") or "").strip().lower()
    if want not in VARIANTS:
        return JSONResponse({"error": f"variant must be one of {VARIANTS}"},
                            status_code=400)
    st = _variant_state()
    if st.get("state") == "switching":
        return JSONResponse({"switching": True, "target": st.get("target"),
                             "message": f"already switching to {st.get('target')}"},
                            status_code=202)
    if st.get("current") == want:
        return JSONResponse({"switching": False, "variant": want})
    _start_switch(want)
    return JSONResponse({"switching": True, "target": want,
                         "message": f"reloading {want} (a few minutes)"},
                        status_code=202)


def _clean_conditions(raw: Any) -> list[dict[str, Any]]:
    """Keep only server-local reference/keyframe images we wrote via /api/upload."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        uri = str(c.get("uri") or "")
        if not uri.startswith("file://" + UPLOAD_DIR):
            continue
        role = str(c.get("role") or "reference").lower()
        if role not in ("reference", "keyframe"):
            continue
        item: dict[str, Any] = {"type": "image", "uri": uri, "role": role}
        if role == "keyframe":
            try:
                item["frame_index"] = int(c.get("frame_index"))
            except Exception:  # noqa: BLE001
                continue
        out.append(item)
    return out


def _variant_for(conditions: list[dict[str, Any]]) -> str:
    # reference images need the ref2va partition; keyframes/text use fl2va
    return "ref2va" if any(c["role"] == "reference" for c in conditions) else "fl2va"


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    conditions = _clean_conditions(body.get("conditions"))
    want = _variant_for(conditions)

    # the two partitions are mutually exclusive on one GPU: swap first if needed
    st = _variant_state()
    if st.get("state") == "switching":
        return JSONResponse({"switching": True, "target": st.get("target"),
                             "message": f"正在切换到 {st.get('target')} 分区…"},
                            status_code=202)
    if st.get("current") != want:
        _start_switch(want)
        return JSONResponse({"switching": True, "target": want,
                             "message": f"正在切换到 {want} 分区（需重新加载模型，约 3–5 分钟）…"},
                            status_code=202)

    if want == "ref2va":
        task = "ref2va"
    else:
        task = "fl2va" if conditions else "t2va"

    try:
        short_edge = int(body.get("short_edge") or 768)
    except Exception:  # noqa: BLE001
        short_edge = 768
    short_edge = max(256, min(short_edge, 1344))

    try:
        duration = float(body.get("duration_seconds") or 4.0)
    except Exception:  # noqa: BLE001
        duration = 4.0
    duration = max(4.0, min(duration, 15.0))

    try:
        steps = int(body.get("num_inference_steps") or 8)
    except Exception:  # noqa: BLE001
        steps = 8
    steps = max(1, min(steps, 50))

    aspect = str(body.get("aspect_ratio") or "16:9")
    if any(c["role"] == "keyframe" for c in conditions):
        # keyframes define the geometry
        aspect = "auto"

    seed = body.get("seed")
    try:
        seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    except Exception:  # noqa: BLE001
        seed = random.randint(1, 2**31 - 1)

    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "prompt": prompt,
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": short_edge,
            "aspect_ratio": aspect,
            "duration_seconds": duration,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps,
        "flow_shift": float(body.get("flow_shift") or 12.0),
        "audio_flow_shift": float(body.get("audio_flow_shift") or 3.0),
        "seed": seed,
    }
    quality = body.get("quality")
    if quality:
        payload["quality"] = quality

    try:
        r = requests.post(f"{SGLANG_BASE}/v1/videos", json=payload, timeout=120)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"submit failed: {e!r}"}, status_code=502)
    if r.status_code >= 400:
        return JSONResponse({"error": "sglang rejected the request",
                             "detail": r.text[:800]}, status_code=r.status_code)
    data = r.json()
    data["_submitted"] = {"task": task, "variant": want, "seed": seed, "steps": steps,
                          "short_edge": short_edge, "duration": duration,
                          "aspect": aspect,
                          "references": sum(1 for c in conditions if c["role"] == "reference"),
                          "keyframes": sum(1 for c in conditions if c["role"] == "keyframe")}
    return JSONResponse(data)


@app.get("/api/status/{vid}")
def status(vid: str) -> JSONResponse:
    try:
        r = requests.get(f"{SGLANG_BASE}/v1/videos/{vid}", timeout=30)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": repr(e)}, status_code=502)
    if r.status_code >= 400:
        return JSONResponse({"error": r.text[:500]}, status_code=r.status_code)
    return JSONResponse(r.json())


@app.get("/api/tasks")
def tasks() -> JSONResponse:
    try:
        r = requests.get(f"{SGLANG_BASE}/v1/videos", timeout=30)
        if r.status_code < 400:
            return JSONResponse(r.json())
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse({"data": []})


@app.get("/api/video/{vid}")
def video(vid: str) -> StreamingResponse:
    r = requests.get(f"{SGLANG_BASE}/v1/videos/{vid}/content",
                     stream=True, timeout=600)
    ctype = r.headers.get("content-type", "video/mp4")

    def it():
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                yield chunk

    headers = {"Content-Disposition": f'inline; filename="{vid}.mp4"'}
    return StreamingResponse(it(), media_type=ctype, headers=headers,
                             status_code=r.status_code)


_SAFE = re.compile(r"[^A-Za-z0-9._-]")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    name = _SAFE.sub("_", os.path.basename(file.filename or "img.png"))
    ext = os.path.splitext(name)[1].lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return JSONResponse({"error": f"unsupported type {ext}"}, status_code=400)
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        return JSONResponse({"error": "image too large (max 20MB)"}, status_code=400)
    with open(dst, "wb") as f:
        f.write(data)
    return JSONResponse({"uri": "file://" + dst, "bytes": len(data)})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    h = _sglang_health()
    st = _variant_state()
    return f"""<!doctype html><meta charset=utf-8><title>MiniMax-H3 L4</title>
<body style="font:14px/1.6 system-ui;background:#0d0d10;color:#e8e8ee;padding:40px">
<h2 style="color:#5b8cff">MiniMax-H3 · L4 gateway</h2>
<p>sglang: <b>{'up' if h.get('reachable') else 'down'}</b> {h}</p>
<p>partition: <b>{st.get('current') or '-'}</b> ({st.get('state')})</p>
<p>gpu: {_gpu()}</p>
<p>API: <code>/api/generate</code>, <code>/api/status/&lt;id&gt;</code>,
<code>/api/video/&lt;id&gt;</code>, <code>/api/upload</code>,
<code>/api/variant</code></p>
</body>"""
