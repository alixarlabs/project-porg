# Project Porg

Voice-controlled surgical assistant using NVIDIA Parakeet ASR and Qwen LLM on Jetson AGX Thor. Interprets natural language commands and converts them to structured tool calls for medical device control.

## Features

- **Real-time ASR** - Parakeet CTC with streaming partial results
- **Voice Activity Detection** - Silero VAD for accurate speech detection
- **LLM Command Interpretation** - Qwen2.5 with native tool calling
- **Multi-source Input** - Supports microphone, XR headsets, or any text source
- **Web Interface** - Browser-based UI for transcription monitoring
- **REST + WebSocket API** - Easy integration with device gateways

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Input Sources             │
                    │  ┌─────────┐  ┌─────────┐  ┌─────┐ │
                    │  │   Mic   │  │   XR    │  │ API │ │
                    │  │ (ASR)   │  │ Headset │  │     │ │
                    │  └────┬────┘  └────┬────┘  └──┬──┘ │
                    └───────┼────────────┼─────────┼─────┘
                            │            │         │
                            ▼            ▼         ▼
┌─────────────┐      ┌─────────────────────────────────────┐
│  Parakeet   │      │           Agent API (:8080)         │
│  ASR        │─────▶│  - Receives text commands           │
│  (voice)    │      │  - Manages tool definitions         │
└─────────────┘      │  - Conversation context             │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │           vLLM (:8000)               │
                     │  - Qwen2.5 with tool calling         │
                     │  - OpenAI-compatible API             │
                     │  - Hermes-style function parsing     │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │         Structured Tool Calls        │
                     │  {                                   │
                     │    "name": "adjust_surgical_light",  │
                     │    "arguments": {"intensity": 80}    │
                     │  }                                   │
                     └──────────────────┬──────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
               ┌──────────┐      ┌──────────┐      ┌──────────┐
               │  Lights  │      │  Table   │      │ Displays │
               └──────────┘      └──────────┘      └──────────┘
```

## Quick Start

### 1. Start the LLM Stack

```bash
# Option A: Using docker-compose (recommended)
docker compose up -d

# Option B: Manual startup
./run_llm.sh &      # Terminal 1: vLLM server
./run_agent.sh &    # Terminal 2: Agent API
```

### 2. Test with CLI

```bash
# Interactive testing
python test_agent.py

# Single command
python test_agent.py "dim the lights to 50 percent"
```

### 3. Run Voice Agent

```bash
# Full voice pipeline (requires LLM stack running)
./run_voice_agent.sh
```

### 4. Web Interface (ASR only)

```bash
./run_web.sh
# Open http://localhost:8888
```

## API Reference

### Agent API (port 8080)

#### POST /command
Process a voice command and return tool calls.

```bash
curl -X POST http://localhost:8080/command \
  -H "Content-Type: application/json" \
  -d '{"text": "raise the table 10 centimeters", "session_id": "or1"}'
```

Response:
```json
{
  "tool_calls": [
    {
      "id": "call_123",
      "name": "adjust_operating_table",
      "arguments": {"height_change_cm": 10}
    }
  ],
  "message": null,
  "needs_confirmation": false,
  "processing_time_ms": 145.2
}
```

#### GET /tools
List all available device controls.

#### GET /health
Health check with model info.

#### WebSocket /ws/{session_id}
Real-time bidirectional command processing.

### Available Tools

| Tool | Description |
|------|-------------|
| `adjust_surgical_light` | Control intensity and position |
| `adjust_operating_table` | Height, tilt, lateral tilt |
| `control_room_environment` | Ambient light, temperature, music |
| `control_display` | Route video sources to displays |
| `request_assistance` | Alert team members |
| `control_insufflator` | CO2 pressure and flow |
| `start_recording` | Surgical video recording |

## Project Structure

```
.
├── docker-compose.yml        # Full stack orchestration
├── llm_service/
│   ├── agent_api.py          # Command interpreter API
│   ├── Dockerfile            # Agent container
│   └── requirements.txt
├── voice_agent.py            # Integrated ASR → LLM pipeline
├── voice_input.py            # CLI ASR only
├── web_voice.py              # Web UI for ASR
├── Dockerfile.voice          # ASR container
├── run_llm.sh                # Start vLLM server
├── run_agent.sh              # Start agent API
├── run_voice_agent.sh        # Full voice pipeline
├── run_web.sh                # Web transcription UI
└── test_agent.py             # API test client
```

## Configuration

### Models

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Fast | Development/testing |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | Medium | Production (balanced) |
| `Qwen/Qwen2.5-72B-Instruct-AWQ` | 72B | Slower | Production (best quality) |

Change model in `docker-compose.yml` or via environment:
```bash
MODEL=Qwen/Qwen2.5-32B-Instruct ./run_llm.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model to use |
| `MAX_TOKENS` | `512` | Max response tokens |

## Adding Custom Tools

Edit `llm_service/agent_api.py` and add to `SURGICAL_TOOLS`:

```python
{
    "type": "function",
    "function": {
        "name": "your_device_control",
        "description": "What this device does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "integer", "description": "..."},
                "param2": {"type": "string", "enum": ["a", "b", "c"]}
            },
            "required": ["param1"]
        }
    }
}
```

## XR Headset Integration

The Agent API accepts commands from any source. For XR headsets:

```python
import requests

# From your XR app's speech recognition
transcribed_text = xr_speech_to_text()

# Send to agent
response = requests.post(
    "http://jetson-thor:8080/command",
    json={"text": transcribed_text, "session_id": "headset_1"}
)

# Execute tool calls
for tool_call in response.json()["tool_calls"]:
    execute_device_command(tool_call)
```

Or use WebSocket for real-time:
```javascript
const ws = new WebSocket("ws://jetson-thor:8080/ws/headset_1");
ws.send(JSON.stringify({text: "more light please"}));
ws.onmessage = (e) => handleToolCalls(JSON.parse(e.data));
```

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Device | NVIDIA Jetson AGX Thor |
| Memory | 128GB unified (for 70B+ models) |
| L4T | R38.3.0 |
| CUDA | 13.0 |

## Performance

| Component | Latency |
|-----------|---------|
| ASR (Parakeet CTC) | ~100-500ms |
| LLM (Qwen 7B) | ~100-200ms |
| LLM (Qwen 72B AWQ) | ~500-1000ms |
| **End-to-end** | **~1-2 seconds** |

## Roadmap

- [x] ASR with streaming transcription
- [x] Web interface for monitoring
- [x] LLM with tool calling
- [x] REST + WebSocket API
- [ ] TTS for voice responses
- [ ] Device gateway integration
- [ ] Wake word detection
- [ ] Multi-language support

## License

MIT

## References

- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [vLLM](https://docs.vllm.ai/)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers)
