#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CONFIG
# ============================================================

MERGED_DIR="./merged"

# 🔁 MODELS TO RUN (ORDER MATTERS)
MODELS=(
  "qwen3vl_peft_generator"
  "vl_qwen3_8b"
  "llava-med-vlm-8b-new"
)

VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"

INFERENCE_SCRIPT="merge_llava.py"

# Initial input
CURRENT_INPUT="200_sample.jsonl"

# Output directory
OUT_DIR="data_new/chain_outputs"
mkdir -p "${OUT_DIR}"

# ============================================================
# FUNCTIONS
# ============================================================

start_vllm () {
  local model_name="$1"
  local model_path="${MERGED_DIR}/${model_name}"

  echo "🚀 Starting vLLM for ${model_name}"

  if [[ "${model_name}" == llava-med-vlm-8b* ]]; then
    echo "🖼️  LLaVA model detected → using vision-safe config"

    vllm serve "${model_path}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --trust-remote-code \
      --quantization bitsandbytes \
      --max-model-len 5000 \
      --gpu-memory-utilization 0.90 \
      > "vllm_${model_name}.log" 2>&1 &

  else
    echo "🧠 Text / LoRA model detected → using high-throughput config"

    vllm serve "${model_path}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --enable-lora \
      --quantization bitsandbytes \
      --max-loras 4 \
      --max-lora-rank 64 \
      --max-model-len 5000 \
      --gpu-memory-utilization 0.97 \
      > "vllm_${model_name}.log" 2>&1 &
  fi

  VLLM_PID=$!
  export VLLM_PID
}

wait_for_vllm () {
  echo "⏳ Waiting for vLLM to become healthy..."
  until curl -sf "${VLLM_BASE_URL}/v1/models" > /dev/null; do
    sleep 2
  done
  echo "✅ vLLM ready."
}

stop_vllm () {
  echo "🛑 Stopping vLLM (PID ${VLLM_PID})"
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}

# ============================================================
# MAIN LOOP
# ============================================================

export VLLM_BASE_URL

for idx in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$idx]}"
  STEP=$((idx + 1))
  OUTPUT_JSONL="${OUT_DIR}/step_${STEP}_${MODEL_NAME}.jsonl"

  echo "============================================================"
  echo "▶ STEP ${STEP}: ${MODEL_NAME}"
  echo "============================================================"

  start_vllm "${MODEL_NAME}"
  wait_for_vllm

  echo "▶ Running inference:"
  echo "   Input : ${CURRENT_INPUT}"
  echo "   Output: ${OUTPUT_JSONL}"

  python "${INFERENCE_SCRIPT}" \
    --input_jsonl "${CURRENT_INPUT}" \
    --output_jsonl "${OUTPUT_JSONL}"

  stop_vllm

  # 🔁 Chain output → next input
  CURRENT_INPUT="${OUTPUT_JSONL}"
done

echo "🎉 ALL MODELS COMPLETED"
echo "📁 Final output: ${CURRENT_INPUT}"
