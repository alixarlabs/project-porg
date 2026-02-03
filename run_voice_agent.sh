#!/bin/bash
# Run the full voice agent pipeline (ASR + LLM)
#
# Prerequisites:
#   1. Start vLLM: ./run_llm.sh
#   2. Start Agent API: ./run_agent.sh
#   3. Then run this script
#
# Usage: ./run_voice_agent.sh [--device N] [--agent-url URL]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to webcam device 36 if not specified
DEVICE_ARG=""
if [[ ! "$*" =~ "--device" ]] && [[ ! "$*" =~ "-d" ]]; then
    DEVICE_ARG="--device 36"
fi

# Default agent URL
AGENT_ARG=""
if [[ ! "$*" =~ "--agent-url" ]] && [[ ! "$*" =~ "-a" ]]; then
    AGENT_ARG="--agent-url http://localhost:8887"
fi

docker run -it --rm --runtime nvidia --network host \
  --device /dev/snd \
  -v "$SCRIPT_DIR:/workspace" \
  -v "$HOME/.cache:/root/.cache" \
  -w /workspace \
  porg-voice:r38-cu130 \
  python3 voice_agent.py $DEVICE_ARG $AGENT_ARG "$@"
