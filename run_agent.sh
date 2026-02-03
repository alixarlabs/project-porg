#!/bin/bash
# Run the Agent API server
#
# Usage:
#   ./run_agent.sh                    # Connect to localhost:8000
#   VLLM_URL=http://host:8000 ./run_agent.sh
#
# The agent API is lightweight and can run on CPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8080}"

echo "========================================"
echo "Starting Agent API"
echo "========================================"
echo "vLLM Backend: $VLLM_URL"
echo "Model: $MODEL"
echo "API Port: $PORT"
echo "========================================"

docker run --rm --network host \
  -v "$SCRIPT_DIR/llm_service:/app" \
  -e VLLM_BASE_URL="$VLLM_URL" \
  -e MODEL_NAME="$MODEL" \
  -w /app \
  python:3.12-slim \
  sh -c "pip install -q fastapi uvicorn openai pydantic && python agent_api.py --port $PORT"
