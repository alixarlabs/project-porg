# Project Porg

Voice-controlled surgical camera system using NVIDIA Parakeet ASR and Qwen LLM on Jetson AGX Thor. Interprets natural language commands and converts them to structured JSON for camera PTZ control, integrating with [project-jango](../project-jango) for endoscope manipulation.

## Features

- **Real-time ASR** - Parakeet CTC 0.6B with streaming partial results
- **Voice Activity Detection** - Silero VAD for accurate speech segmentation
- **LLM Command Interpretation** - Qwen2.5-3B with native tool calling
- **Camera PTZ Control** - Move, zoom, focus, and action commands
- **Web Interface** - Browser-based UI showing transcription, LLM reasoning, and tool calls
- **Project-Jango Integration** - Outputs JSON commands compatible with headset control protocol

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Voice Input                              │
│                     (Microphone/XR Headset)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Parakeet ASR (porg-voice)                     │
│                   - Silero VAD speech detection                 │
│                   - Streaming transcription                     │
│                   - Web UI on :8888                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Text
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Agent API (porg-agent :8887)                  │
│                   - Camera tool definitions                     │
│                   - Conversation context                        │
│                   - Jango JSON format output                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   vLLM (porg-vllm :8889)                        │
│                   - Qwen2.5-3B-Instruct                         │
│                   - Hermes tool calling                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Tool Calls
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Project-Jango Control System                  │
│                   - UDP JSON on port 9000                       │
│                   - ESP32 camera controller                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### One-Command Launch

```bash
# Start everything (clears memory cache, requires sudo)
./launch.sh

# Skip cache clearing
./launch.sh --no-cache-clear

# Use different audio device
./launch.sh --device 42
```

### Stop All Services

```bash
./stop.sh
```

### Manual Startup

```bash
# 1. Start LLM stack
docker compose up -d

# 2. Start voice interface
./run_web.sh

# Open http://localhost:8888
```

## Voice Commands

| Command | Action |
|---------|--------|
| "move up/down/left/right" | Pan/tilt camera |
| "move up and left" | Diagonal movement |
| "zoom in" / "zoom out" | Optical zoom |
| "focus closer" / "focus farther" | Adjust focus |
| "stop" | Stop all movement |
| "switch to gamepad" | Change control mode |
| "switch to manual" | Switch to GUI control |

## Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Voice Web UI | porg-voice | 8888 | ASR + LLM web interface |
| Agent API | porg-agent | 8887 | Command interpreter |
| vLLM | porg-vllm | 8889 | LLM inference server |

## API Reference

### Agent API (port 8887)

#### POST /command
Process a voice command and return tool calls.

```bash
curl -X POST http://localhost:8887/command \
  -H "Content-Type: application/json" \
  -d '{"text": "zoom in and move left"}'
```

Response:
```json
{
  "tool_calls": [
    {"id": "call_1", "name": "camera_zoom", "arguments": {"direction": "in"}},
    {"id": "call_2", "name": "camera_move", "arguments": {"direction": "left"}}
  ],
  "message": null,
  "processing_time_ms": 85.2
}
```

#### POST /command/execute
Returns tool calls in project-jango JSON format:

```json
{
  "jango_commands": [
    {"type": "zoom_focus", "zoom": "in"},
    {"type": "direction", "direction": "left"}
  ]
}
```

#### GET /tools
List all available camera controls.

#### GET /health
Health check with model info.

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `camera_move` | Pan/tilt camera | `direction`: up, down, left, right, up_left, up_right, down_left, down_right |
| `camera_zoom` | Optical zoom | `direction`: in, out |
| `camera_focus` | Adjust focus | `direction`: in (closer), out (farther) |
| `camera_stop` | Stop all movement | (none) |
| `trigger_action` | Instrument actions | `action`: 0-6 |
| `change_mode` | Switch control source | `mode`: headset, gamepad, tool, gui, test_pattern |

## Project-Jango Integration

Project Porg outputs commands compatible with project-jango's headset control protocol. The `/command/execute` endpoint returns `jango_commands` ready for UDP transmission to port 9000:

```python
import socket
import json

# Get command from Porg
response = requests.post("http://localhost:8887/command/execute",
                         json={"text": "zoom in"})

# Send to Jango
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for cmd in response.json()["jango_commands"]:
    sock.sendto(json.dumps(cmd).encode(), ("localhost", 9000))
```

## Project Structure

```
project-porg/
├── launch.sh                 # One-command startup (clears cache + all services)
├── stop.sh                   # Stop all services
├── docker-compose.yml        # vLLM + Agent orchestration
├── llm_service/
│   ├── agent_api.py          # Camera control interpreter
│   └── Dockerfile
├── web_voice.py              # Web UI with ASR + LLM display
├── voice_agent.py            # CLI voice pipeline
├── Dockerfile.voice          # ASR container
├── run_web.sh                # Start web interface
├── run_voice_agent.sh        # Start CLI voice agent
└── test_agent.py             # API test client
```

## Configuration

### Model Selection

Default is Qwen2.5-3B-Instruct for fast inference. Edit `docker-compose.yml` for larger models:

| Model | Size | Latency | Memory |
|-------|------|---------|--------|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~50-100ms | ~6GB |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~100-200ms | ~14GB |
| `Qwen/Qwen2.5-72B-Instruct-AWQ` | 72B | ~500-1000ms | ~40GB |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://vllm:8889/v1` | vLLM server URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Model to use |
| `MAX_TOKENS` | `256` | Max response tokens |

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Device | NVIDIA Jetson AGX Thor |
| Memory | 128GB unified |
| L4T | R38.3.0+ |
| CUDA | 13.0 |

## Performance

| Component | Latency |
|-----------|---------|
| VAD + ASR (Parakeet) | ~100-500ms |
| LLM (Qwen 3B) | ~50-100ms |
| **End-to-end** | **~500ms - 1s** |

## Troubleshooting

### Out of Memory
```bash
# Clear system cache before launch
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### vLLM Won't Start
Check available memory with `tegrastats` - need ~30GB free for 3B model with comfortable margin.

### No Audio Input
List devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
Set device: `./launch.sh --device <N>`

## Roadmap

- [x] ASR with streaming transcription
- [x] Web interface with LLM reasoning display
- [x] Camera PTZ tool calling
- [x] Project-jango JSON output format
- [x] One-command launch script
- [ ] Direct UDP integration with project-jango
- [ ] Wake word detection
- [ ] TTS for voice responses

## License

MIT

## References

- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [vLLM](https://docs.vllm.ai/)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Project Jango](../project-jango) - Surgical camera control system
