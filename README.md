# Project Porg

Real-time voice transcription using NVIDIA Parakeet ASR on Jetson AGX Thor. Features streaming speech-to-text with a web interface for live monitoring.

## Features

- **Real-time transcription** - See partial results as you speak
- **Voice Activity Detection** - Silero VAD for accurate speech detection
- **Web Interface** - Browser-based UI showing live transcription
- **Optimized for Jetson** - Runs on Jetson AGX Thor with CUDA 13.0

## Quick Start

### Prerequisites

- NVIDIA Jetson AGX Thor (L4T R38.3.0 / JetPack 7.1)
- Docker with NVIDIA runtime
- USB microphone or webcam with mic

### Build the Container

```bash
# Build the voice agent container
docker build -t voice-agent:r38-cu130 -f Dockerfile.voice .
```

### Run with Web Interface

```bash
./run_web.sh
```

Open http://localhost:8888 in your browser. Speak into your microphone and watch transcriptions appear in real-time.

### Run CLI Only

```bash
./run_voice.sh
```

### Transcribe a File

```bash
docker run --rm --runtime nvidia \
  -v "$(pwd):/workspace" \
  -v "$HOME/.cache:/root/.cache" \
  -w /workspace \
  voice-agent:r38-cu130 \
  python3 transcribe_file.py audio.wav
```

## Configuration

### Audio Device

List available devices:
```bash
docker run --rm --device /dev/snd voice-agent:r38-cu130 \
  python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Specify a device:
```bash
./run_web.sh --device 2
```

### Port

Change the web interface port:
```bash
./run_web.sh --port 9000
```

## Project Structure

```
.
├── Dockerfile.voice      # Container with ASR, VAD, and web dependencies
├── Dockerfile.parakeet   # Minimal ASR-only container
├── web_voice.py          # Web interface with WebSocket streaming
├── voice_input.py        # CLI streaming transcription
├── transcribe_file.py    # File transcription utility
├── test_audio_input.py   # Audio device testing
├── run_web.sh            # Launch web interface
└── run_voice.sh          # Launch CLI interface
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│  Silero VAD │────▶│  Parakeet   │
│  (16kHz)    │     │  (Speech    │     │  ASR        │
└─────────────┘     │   Detect)   │     │  (CTC 0.6B) │
                    └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐             │
                    │  WebSocket  │◀────────────┘
                    │  Broadcast  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Browser   │
                    │   (Web UI)  │
                    └─────────────┘
```

## Supported Models

| Model | Type | Notes |
|-------|------|-------|
| `nvidia/parakeet-ctc-0.6b` | CTC | Default, fast, non-autoregressive |
| `nvidia/parakeet-rnnt-0.6b` | RNN-T | Streaming capable |
| `nvidia/parakeet-tdt-0.6b-v2` | TDT | Better accuracy |

Use `--model` to switch:
```bash
./run_web.sh --model nvidia/parakeet-rnnt-0.6b
```

## Hardware Environment

| Component | Version |
|-----------|---------|
| Device | NVIDIA Jetson AGX Thor |
| L4T | R38.3.0 |
| JetPack | 7.1 |
| CUDA | 13.0 |
| Python | 3.12 |

## Container Details

Built on `nemo:r38.3.arm64-sbsa-cu130-24.04-numba` with:
- PyTorch 2.10
- NeMo Toolkit (ASR)
- Silero VAD
- FastAPI + WebSockets
- sounddevice + soundfile

**Note**: Uses `lhotse==1.29.0` for NeMo compatibility (newer versions have breaking changes).

## Development Notes

### Building the Base Container

If you need to rebuild the base NeMo container:

```bash
git clone --depth 1 https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers
./build.sh nemo --skip-tests=all
```

### Known Issues

- First run downloads the Parakeet model (~1.2GB) to `~/.cache/huggingface/`
- TDT models may have Lhotse compatibility issues; CTC model is most reliable
- Silero VAD requires exactly 512 samples per chunk at 16kHz

## Roadmap

- [ ] TTS (text-to-speech) integration
- [ ] Local LLM integration for voice agent
- [ ] Wake word detection
- [ ] Multi-language support

## License

MIT

## References

- [NVIDIA Parakeet Models](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers)
