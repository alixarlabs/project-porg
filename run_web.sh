#!/bin/bash
# Run the web-based voice transcription in Docker
# Usage: ./run_web.sh [--device N] [--port P] [other args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to webcam device 36 if not specified
DEVICE_ARG=""
if [[ ! "$*" =~ "--device" ]] && [[ ! "$*" =~ "-d" ]]; then
    DEVICE_ARG="--device 36"
fi

# Default port 8888
PORT_ARG=""
if [[ ! "$*" =~ "--port" ]] && [[ ! "$*" =~ "-p" ]]; then
    PORT_ARG="--port 8888"
fi

docker run -it --rm --runtime nvidia --network host \
  --device /dev/snd \
  -v "$SCRIPT_DIR:/workspace" \
  -v "$HOME/.cache:/root/.cache" \
  -w /workspace \
  porg-voice:r38-cu130 \
  python3 web_voice.py $DEVICE_ARG $PORT_ARG "$@"
