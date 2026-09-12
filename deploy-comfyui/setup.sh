#!/bin/bash
# MiniMax-H3 on ComfyUI — one-shot setup for Colab (L4 24GB + High-RAM).
#
# Replicates the community "Latent 双采样 / 显存终极优化" recipe:
#   stage-1 sample at low res -> 3D latent upscale (x1.5) -> stage-2 sample
# on the hybrid fl2va+ref2va int8 DiT with the 4-step Turbo LoRA.
#
# NOTE on fidelity: the workflow this is based on also used four node types
# (MiniMaxLowVRAMAttention, MiniMaxChunkFeedForward, ReservedVRAMSetter,
# MiniMaxH3MemoryEfficientSageAttentionPatch) that are NOT in the public node
# packs -- so they cannot be reproduced. ComfyUI's own dynamic VRAM loading
# plus the upscaler's `force_unload` cover the VRAM side instead.
set -euo pipefail

CUI="${CUI:-/content/ComfyUI}"
H3C="${H3C:-/content/h3comfy}"
STAGE=/content/staging
export HF_HUB_DISABLE_XET=1
export HF_HOME=/content/hf

echo "== [1/6] ComfyUI =="
[ -d "$CUI/.git" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$CUI"
python3 -m pip install -q -r "$CUI/requirements.txt"

echo "== [2/6] custom nodes =="
mkdir -p "$CUI/custom_nodes"
clone() { [ -d "$CUI/custom_nodes/$1" ] || git clone --depth 1 "$2" "$CUI/custom_nodes/$1"; }
clone Comfyui_Minimax_h3_latent_Upscaler https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
clone ComfyUI-YCNodes-MiniMax-H3        https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3
clone ComfyUI-VideoHelperSuite          https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
python3 -m pip install -q -r "$CUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"

echo "== [3/6] gateway deps =="
python3 -m pip install -q fastapi uvicorn websockets requests python-multipart

echo "== [4/6] weights (~57GB) =="
mkdir -p "$STAGE"
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
STAGE = "/content/staging"
MODELS = "/content/ComfyUI/models"
JOBS = [
    # the exact hybrid checkpoint the workflow references (20.97GB, not the 34GB full int8)
    ("smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models",
     "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors", "diffusion_models"),
    ("Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", "vae"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
    ("lightx2v/Minimax-h3-Turbo",
     "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors", "loras"),
    ("LBH-123-AI/Minimax_h3_latent_Upscaler",
     "minimax_h3_latent_upscaler_3d_fp16.safetensors", "latent_upscale_models"),
]
for repo, rfile, kind in JOBS:
    p = hf_hub_download(repo, rfile, local_dir=STAGE)
    d = os.path.join(MODELS, kind)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(rfile))
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(p, dst)          # symlink so staging stays the single copy
    print("OK", os.path.basename(rfile), "%.2f GB" % (os.path.getsize(p) / 1e9))
PY

echo "== [5/6] gateway =="
mkdir -p "$H3C"
curl -fsSL https://raw.githubusercontent.com/hsgwktb/minimax-h3-webui/main/deploy-comfyui/comfy_gateway.py \
     -o "$H3C/comfy_gateway.py"

echo "== [6/6] start ComfyUI (headless) =="
cat > /content/run_comfy.sh <<'BASH'
#!/bin/bash
cd /content/ComfyUI || exit 9
export HF_HOME=/content/hf
exec python3 main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch
BASH
chmod +x /content/run_comfy.sh
pkill -f 'main.py --listen' 2>/dev/null || true
sleep 2
nohup bash /content/run_comfy.sh > /content/comfy.log 2>&1 &
for _ in $(seq 1 40); do
  sleep 4
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8188/)" = "200" ] && break
done

nohup python3 -m uvicorn comfy_gateway:app --host 127.0.0.1 --port 8000 \
  --app-dir "$H3C" > "$H3C/gateway.log" 2>&1 &

echo
echo "ComfyUI  : http://127.0.0.1:8188"
echo "gateway  : http://127.0.0.1:8000/api/health"
echo "next     : cloudflared tunnel --url http://127.0.0.1:8000"
