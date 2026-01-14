#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

# Put EVERYTHING on the big disk
export HF_HOME=/workspace/work/cache/huggingface
export HF_HUB_CACHE=/workspace/work/cache/huggingface/hub
export HF_MODULES_CACHE=/workspace/work/cache/huggingface/modules
export TRANSFORMERS_CACHE=/workspace/work/cache/transformers
export HF_DATASETS_CACHE=/workspace/work/cache/datasets

export XDG_CACHE_HOME=/workspace/work/cache
export MPLCONFIGDIR=/workspace/work/cache/matplotlib
export TMPDIR=/workspace/work/cache/tmp
export VLLM_CACHE_DIR=/workspace/work/cache/vllm

# (optional) helps many libs stop using /home/unsloth
export HOME=/workspace/work

mkdir -p "$HF_HUB_CACHE" "$HF_MODULES_CACHE" "$TRANSFORMERS_CACHE" \
         "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"

vllm serve unsloth/Qwen3-VL-8B-Instruct \
  --max-model-len 8192 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --limit-mm-per-prompt '{"image":1}' \
  --allowed-local-media-path /workspace/work \
  --disable-mm-preprocessor-cache \
  --max-num-seqs 1 
