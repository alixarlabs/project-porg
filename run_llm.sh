#!/bin/bash
# Run the vLLM inference server standalone
#
# Usage:
#   ./run_llm.sh                           # Default: Qwen2.5-7B
#   ./run_llm.sh --model Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq
#
# This script runs vLLM directly for development/testing.
# For production, use docker-compose.yml instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default model (smaller for faster testing)
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"

echo "========================================"
echo "Starting vLLM Server"
echo "========================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "========================================"

docker run --rm --runtime nvidia --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e CUDA_VISIBLE_DEVICES=0 \
  nvcr.io/nvidia/tritonserver:25.08-vllm-python-py3 \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    "$@"
