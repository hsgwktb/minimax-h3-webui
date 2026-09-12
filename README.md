# MiniMax-H3 on a single Colab L4 (24 GB) — quantized, with web UI

Deploys **MiniMax-H3** (joint video+audio generation) from
[RH-RunningHub/MiniMax-H3-MultiGPU-Lightning](https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning)
on **one 24 GB L4**, using **4-bit GGUF weights** + SGLang layerwise offload
(weights streamed from host RAM / disk, never fully resident in VRAM).

The front end is this repo's `index.html` (a replica of the metaso.cn/minimax-h3
UI). It is hosted on GitHub Pages and talks to the Colab backend through a
cloudflared quick tunnel.

```
GitHub Pages (index.html)  ──HTTPS──▶  cloudflared tunnel
                                          └─▶ gateway.py (FastAPI :8000)
                                                └─▶ sglang serve (:30010)  →  L4
```

## Measured result (Colab Pro, L4 24 GB, 52 GB host RAM)

| Metric | Value |
|---|---|
| Output | 1344×768 (768P), 4.46 s, 107 frames, H.264 + AAC |
| Inference time | ~193 s (4 steps) |
| **Peak VRAM** | **11.3 GB / 24 GB** |
| Weights | GGUF Q4_K_M: DiT 18.8 GB, text encoder 18.2 GB |

`/health` returns 200 in ~7 min from launch (2–3 min of that is loading;
first request warms the JIT kernels).

## 1. Colab runtime

Colab Pro, **L4 GPU + High-RAM** (host RAM is the real constraint: the offload
pool needs ~35 GB, and 12 GB standard RAM is not enough).

## 2. Install

```bash
python3.12 -m venv /content/h3/venv            # or: uv venv --python 3.12
uv pip install --python /content/h3/venv/bin/python -q wheel_stub
uv pip install --python /content/h3/venv/bin/python torch==2.13.0
git clone --depth 1 https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning /content/h3/repo
cd /content/h3/repo/sglang/python
SGLANG_BUILD_RUST_EXTS=none uv pip install --python /content/h3/venv/bin/python \
  --no-build-isolation -e ".[diffusion]"
```

`wheel_stub` is required because `cuda-tile==1.6.0rc5` is an sdist and we build
with `--no-build-isolation`.

## 3. Quantized weights (GGUF, 4-bit)

```bash
export HF_HOME=/content/h3/hf HF_HUB_DISABLE_XET=1
hf download leejet/MiniMax-H3-GGUF \
  minimax_h3_fl2va-Q4_K_M.gguf qwen3vl_32b_minimax_h3-Q4_K_M.gguf
```

## 4. Minimal local model root (important)

`sglang` calls `snapshot_download` on `--model-path` and would otherwise pull
the **whole FL2VA partition (~135 GB: 67 GB DiT + 51 GB text encoder BF16)**,
even though both are overridden by GGUF. Instead build a local root holding
only metadata + VAEs (~11 GB) and point `--model-path` at it:

```bash
hf download MiniMaxAI/MiniMax-H3 --include 'model_index.json' 'FL2VA/**' \
  --local-dir /content/h3/MiniMax-H3
# drop the weights we override with GGUF
rm -f /content/h3/MiniMax-H3/FL2VA/transformer/*.safetensors* \
      /content/h3/MiniMax-H3/FL2VA/text_encoder/*.safetensors*
# the root model_index.json (modular pipeline) declares root-level components
ln -s FL2VA/transformer   /content/h3/MiniMax-H3/transformer
ln -s FL2VA/text_encoder  /content/h3/MiniMax-H3/text_encoder
ln -s FL2VA/video_vae     /content/h3/MiniMax-H3/vae
ln -s FL2VA/audio_vae     /content/h3/MiniMax-H3/audio_vae
ln -s FL2VA/tokenizer     /content/h3/MiniMax-H3/tokenizer
ln -s FL2VA/processor     /content/h3/MiniMax-H3/processor
mkdir -p /content/h3/MiniMax-H3/scheduler /content/h3/MiniMax-H3/audio_scheduler
```

## 5. GGUF vision-tower loader patch (required)

The Qwen3-VL GGUF stores the vision `patch_embed.proj` conv kernel flattened
(`(out*in, kt, kh, kw)`), which fails the shape assert. In
`.../runtime/models/encoders/minimax_h3_qwen3vl.py`, replace

```python
weight_loader(param, loaded_weight.to(param.dtype))
```

with a reshape when the element count matches:

```python
_lw = loaded_weight.to(param.dtype)
if tuple(_lw.shape) != tuple(param.shape) and _lw.numel() == param.numel():
    _lw = _lw.reshape(param.shape)
weight_loader(param, _lw)
```

## 6. Serve

```bash
export PATH=/content/h3/venv/bin:$PATH          # JIT kernels need ninja
export HF_HOME=/content/h3/hf HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sglang serve \
  --model-path /content/h3/MiniMax-H3 --model-id MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va --num-gpus 1 \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder \
  --layerwise-resident-layers video_vae=36 \
  --attention-backend fa \
  --component-weights-paths.transformer leejet/MiniMax-H3-GGUF/minimax_h3_fl2va-Q4_K_M.gguf \
  --component-weights-paths.text_encoder leejet/MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --host 127.0.0.1 --port 30010
```

Offload decisions it prints at startup (measured): transformer pins 36/50 blocks
(12.18 GiB host), text encoder 0/50 pinned (16.6 GiB pageable), VAE resident.

## 7. Gateway + tunnel

```bash
uvicorn gateway:app --host 127.0.0.1 --port 8000 &
/content/h3/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

Then write the printed `https://*.trycloudflare.com` into this repo's
`api.json` (`{"base": "..."}`). The page reads `api.json` on load, so a new
tunnel URL only needs that one file updated.

## Partitions: 文/图生视频 vs 多参考

MiniMax-H3 ships two **mutually exclusive** checkpoint partitions and one
instance serves only one of them:

| UI tab | `--model-variant` | `task` | Conditioning |
| --- | --- | --- | --- |
| 文/图生视频 | `fl2va` | `t2va` / `fl2va` | text, first/last keyframe |
| 多参考 | `ref2va` | `ref2va` | up to 9 reference images |

`Ref2VA/model_index.json` declares `_minimax_h3.tasks = ["ref2va"]`, so a
reference request cannot be served by an fl2va instance. On a single 24 GB card
both partitions cannot be resident at once (each needs ~35 GB of host memory for
offload plus its own VRAM working set), so the deployment **swaps** them.

How the swap works:

- `serve_variant.sh <fl2va|ref2va>` serves one partition; the GGUF weight and
  `--model-variant` are chosen from the argument, and `--model-path` stays the
  shared local root (`--model-variant` makes SGLang resolve `FL2VA/` or
  `Ref2VA/` inside it).
- `switch_variant.sh <variant>` kills the current server, starts the other, and
  polls `/health` until it returns 200. It writes
  `/content/h3/variant.json` = `{current, state: switching|ready|error, target}`
  as it goes. **Measured swap cost: 376 s (~6.3 min)** for `fl2va → ref2va`
  (load + warmup).
- The gateway reads that file and reloads *lazily on demand*: a request whose
  conditions include `role: "reference"` needs `ref2va`, so `/api/generate`
  returns `{"switching": true, "target": "ref2va"}` (HTTP 202) instead of
  submitting. The web UI shows the switch progress, waits on `/api/variant`,
  then re-submits the same job automatically. After the swap the partition stays
  resident, so subsequent reference jobs run immediately.
- `GET /api/variant` reports the state; `POST /api/variant {"variant": "..."}`
  requests a swap explicitly.

Only the first reference job pays the reload cost.

## API (gateway)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | gateway + sglang + GPU |
| POST | `/api/generate` | `{prompt, short_edge, aspect_ratio, duration_seconds, num_inference_steps, conditions[], enable_cache_dit, cache_dit_params?}` |
| GET | `/api/status/{id}` | job status/progress |
| GET | `/api/video/{id}` | finished MP4; Range-aware, and `?download=1` returns `Content-Disposition: attachment` |
| POST | `/api/upload` | image → `file://` URI for keyframe conditioning |

## Caveats

- **Tunnel URL is ephemeral** and is the only secret protecting the backend —
  anyone with it can use the GPU. Rotate/stop when done.
- `api.json` is public on this Pages site; it therefore publishes the tunnel URL.
- Generation is slow: ~48 s/step at 1344×768 on one L4 (streamed weights).
- The loader patch and `/content/h3` live only inside the Colab VM — a runtime
  restart requires re-running sections 2–6.
