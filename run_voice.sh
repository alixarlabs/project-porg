#!/bin/bash
# Run the voice input pipeline in Docker
# Usage: ./run_voice.sh [--device N] [other args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to webcam device 36 if not specified
DEVICE_ARG=""
if [[ ! "$*" =~ "--device" ]] && [[ ! "$*" =~ "-d" ]]; then
    DEVICE_ARG="--device 36"
fi

docker run -it --rm --runtime nvidia --network host \
  --device /dev/snd \
  -v "$SCRIPT_DIR:/workspace" \
  -v "$HOME/.cache:/root/.cache" \
  -w /workspace \
  porg-voice:r38-cu130 \
  python3 voice_input.py $DEVICE_ARG "$@"
