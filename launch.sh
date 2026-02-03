#!/bin/bash
# Project Porg - Full Stack Launch Script
#
# Clears memory cache (requires sudo) and starts all services:
#   - vLLM inference server (Qwen2.5-3B-Instruct)
#   - Agent API (camera control interpreter)
#   - Voice Web UI (ASR + LLM pipeline)
#
# Usage: ./launch.sh [--no-cache-clear] [--device N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
CLEAR_CACHE=true
AUDIO_DEVICE=36

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache-clear)
            CLEAR_CACHE=false
            shift
            ;;
        --device|-d)
            AUDIO_DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./launch.sh [--no-cache-clear] [--device N]"
            exit 1
            ;;
    esac
done

echo ""
echo "=================================================="
echo "  Project Porg - Voice-Controlled Camera System"
echo "=================================================="
echo ""

# Step 1: Clear memory cache
if [ "$CLEAR_CACHE" = true ]; then
    echo "[1/4] Clearing memory cache (requires sudo)..."
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    sleep 1
    FREE_MEM=$(free -g | awk '/^Mem:/{print $7}')
    echo "      Available memory: ${FREE_MEM}G"
else
    echo "[1/4] Skipping cache clear (--no-cache-clear)"
fi

# Step 2: Stop any existing containers
echo "[2/4] Stopping existing containers..."
docker stop porg-vllm porg-agent 2>/dev/null || true
docker rm porg-vllm porg-agent 2>/dev/null || true
# Stop any running voice containers
docker ps -q --filter "ancestor=porg-voice:r38-cu130" | xargs -r docker stop 2>/dev/null || true

# Step 3: Start vLLM and Agent via docker compose
echo "[3/4] Starting vLLM and Agent API..."
docker compose up -d --build

# Wait for vLLM to be healthy
echo "      Waiting for vLLM to load model..."
TIMEOUT=300
ELAPSED=0
while ! curl -s http://localhost:8889/health > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "      ERROR: vLLM failed to start within ${TIMEOUT}s"
        docker logs porg-vllm --tail 20
        exit 1
    fi
    printf "      ... %ds\r" $ELAPSED
done
echo "      vLLM ready (${ELAPSED}s)                    "

# Wait for Agent to be healthy
echo "      Waiting for Agent API..."
ELAPSED=0
while ! curl -s http://localhost:8887/health > /dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge 60 ]; then
        echo "      ERROR: Agent API failed to start"
        docker logs porg-agent --tail 20
        exit 1
    fi
done
echo "      Agent API ready"

# Step 4: Start Voice Web UI
echo "[4/4] Starting Voice Web UI..."
docker run -d --rm --runtime nvidia --network host \
    --name porg-voice \
    --device /dev/snd \
    -v "$SCRIPT_DIR:/workspace" \
    -v "$HOME/.cache:/root/.cache" \
    -w /workspace \
    porg-voice:r38-cu130 \
    python3 web_voice.py --device "$AUDIO_DEVICE" --port 8888

# Wait for voice service
echo "      Waiting for Voice service to initialize..."
sleep 5

echo ""
echo "=================================================="
echo "  All services started successfully!"
echo "=================================================="
echo ""
echo "  Web UI:      http://localhost:8888"
echo "  Agent API:   http://localhost:8887"
echo "  vLLM API:    http://localhost:8889"
echo ""
echo "  Voice commands:"
echo "    - 'move up/down/left/right'"
echo "    - 'zoom in/out'"
echo "    - 'focus closer/farther'"
echo "    - 'stop'"
echo ""
echo "  To view logs:"
echo "    docker logs -f porg-voice"
echo "    docker logs -f porg-agent"
echo "    docker logs -f porg-vllm"
echo ""
echo "  To stop all services:"
echo "    ./stop.sh"
echo "=================================================="
