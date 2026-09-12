#!/bin/bash
# Serve one MiniMax-H3 checkpoint partition on the single L4.
# usage: serve_variant.sh fl2va|ref2va
set -e
VARIANT="${1:-fl2va}"

export PATH=/content/h3/venv/bin:$PATH
export HF_HOME=/content/h3/hf
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_CACHE_DIT_ENABLED=false

case "$VARIANT" in
  fl2va)  GGUF=minimax_h3_fl2va-Q4_K_M.gguf ;;
  ref2va) GGUF=minimax_h3_ref2va-Q4_K_M.gguf ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

cd /content/h3
exec /content/h3/venv/bin/sglang serve \
  --model-path /content/h3/MiniMax-H3 \
  --model-id MiniMaxAI/MiniMax-H3 \
  --model-variant "$VARIANT" \
  --num-gpus 1 \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder \
  --layerwise-resident-layers video_vae=36 \
  --attention-backend fa \
  --component-weights-paths.transformer "leejet/MiniMax-H3-GGUF/${GGUF}" \
  --component-weights-paths.text_encoder leejet/MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --host 127.0.0.1 --port 30010
