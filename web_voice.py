#!/usr/bin/env python3
"""
Web-based voice transcription with real-time display and LLM integration.

Combines the voice input pipeline with a FastAPI web server
that streams transcription updates via WebSocket, then sends
final transcriptions to the Agent API for command interpretation.
"""

import asyncio
import json
import numpy as np
import threading
import queue
import time
import httpx
from collections import deque
from dataclasses import dataclass, asdict
from typing import Set, Optional
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Audio settings
SAMPLE_RATE = 16000
# Silero VAD requires exactly 512 samples at 16kHz (32ms)
CHUNK_SIZE = 512
TRANSCRIBE_INTERVAL = 0.5
MIN_SPEECH_DURATION = 0.3
SILENCE_THRESHOLD = 0.5

# Agent API settings
AGENT_URL = "http://localhost:8887"

app = FastAPI()

# Global state for WebSocket connections
connected_clients: Set[WebSocket] = set()
main_event_loop: asyncio.AbstractEventLoop = None


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Project Porg - Voice Command</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
            font-weight: bold;
        }
        .status.connected {
            background: #0f3d3e;
            color: #00ff88;
        }
        .status.disconnected {
            background: #3d0f0f;
            color: #ff4444;
        }
        .status.listening {
            background: #0f3d3e;
            color: #00ff88;
            animation: pulse 2s infinite;
        }
        .status.speaking {
            background: #3d3d0f;
            color: #ffff00;
            animation: pulse 0.5s infinite;
        }
        .status.processing {
            background: #0f2d3d;
            color: #00aaff;
            animation: pulse 0.8s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .current-box {
            background: #16213e;
            border: 2px solid #00d4ff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            min-height: 80px;
        }
        .current-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .current-text {
            font-size: 24px;
            line-height: 1.4;
            color: #fff;
        }
        .current-text.partial {
            color: #aaa;
            font-style: italic;
        }

        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 768px) {
            .panels {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }
        .panel-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 12px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .panel-label .icon {
            font-size: 16px;
        }

        .llm-response {
            border: 1px solid #4a5568;
        }
        .llm-message {
            font-size: 16px;
            line-height: 1.6;
            color: #e2e8f0;
            white-space: pre-wrap;
        }
        .llm-message.empty {
            color: #555;
            font-style: italic;
        }

        .tool-calls {
            border: 1px solid #48bb78;
        }
        .tool-call-item {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 3px solid #48bb78;
        }
        .tool-call-item:last-child {
            margin-bottom: 0;
        }
        .tool-name {
            font-weight: bold;
            color: #48bb78;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .tool-args {
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            font-size: 12px;
            background: #0d1117;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            color: #79c0ff;
        }
        .no-tools {
            color: #555;
            font-style: italic;
        }

        .history {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }
        .history-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 12px;
            text-transform: uppercase;
        }
        .history-item {
            padding: 12px;
            margin-bottom: 8px;
            background: #1a1a2e;
            border-radius: 8px;
            border-left: 3px solid #00d4ff;
        }
        .history-item .text {
            font-size: 16px;
            margin-bottom: 4px;
        }
        .history-item .meta {
            font-size: 12px;
            color: #666;
        }
        .history-item .tools-summary {
            font-size: 12px;
            color: #48bb78;
            margin-top: 4px;
        }
        .empty {
            color: #555;
            font-style: italic;
        }

        .processing-time {
            font-size: 11px;
            color: #666;
            margin-top: 8px;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Project Porg</h1>
        <div id="status" class="status disconnected">Connecting...</div>

        <div class="current-box">
            <div class="current-label">Current Speech</div>
            <div id="current" class="current-text empty">Waiting for speech...</div>
        </div>

        <div class="panels">
            <div class="panel llm-response">
                <div class="panel-label"><span class="icon">🧠</span> LLM Response</div>
                <div id="llm-message" class="llm-message empty">Waiting for command...</div>
                <div id="processing-time" class="processing-time"></div>
            </div>

            <div class="panel tool-calls">
                <div class="panel-label"><span class="icon">🔧</span> Tool Calls</div>
                <div id="tool-calls" class="no-tools">No tool calls yet</div>
            </div>
        </div>

        <div class="history">
            <div class="history-label">History</div>
            <div id="history"></div>
        </div>
    </div>

    <script>
        const statusEl = document.getElementById('status');
        const currentEl = document.getElementById('current');
        const llmMessageEl = document.getElementById('llm-message');
        const toolCallsEl = document.getElementById('tool-calls');
        const processingTimeEl = document.getElementById('processing-time');
        const historyEl = document.getElementById('history');

        let ws;
        let reconnectTimeout;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                statusEl.textContent = 'Listening...';
                statusEl.className = 'status listening';
            };

            ws.onclose = () => {
                statusEl.textContent = 'Disconnected - Reconnecting...';
                statusEl.className = 'status disconnected';
                reconnectTimeout = setTimeout(connect, 2000);
            };

            ws.onerror = () => {
                ws.close();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'partial') {
                    statusEl.textContent = 'Speaking...';
                    statusEl.className = 'status speaking';
                    currentEl.textContent = data.text;
                    currentEl.className = 'current-text partial';
                } else if (data.type === 'final') {
                    statusEl.textContent = 'Processing...';
                    statusEl.className = 'status processing';
                    currentEl.textContent = data.text;
                    currentEl.className = 'current-text';
                } else if (data.type === 'agent_response') {
                    statusEl.textContent = 'Listening...';
                    statusEl.className = 'status listening';
                    currentEl.textContent = 'Waiting for speech...';
                    currentEl.className = 'current-text empty';

                    // Update LLM message
                    if (data.message) {
                        llmMessageEl.textContent = data.message;
                        llmMessageEl.className = 'llm-message';
                    } else {
                        llmMessageEl.textContent = '(No text response)';
                        llmMessageEl.className = 'llm-message empty';
                    }

                    // Update processing time
                    if (data.processing_time_ms) {
                        processingTimeEl.textContent = `Processed in ${data.processing_time_ms.toFixed(0)}ms`;
                    }

                    // Update tool calls
                    if (data.tool_calls && data.tool_calls.length > 0) {
                        toolCallsEl.innerHTML = data.tool_calls.map(tc => `
                            <div class="tool-call-item">
                                <div class="tool-name">${escapeHtml(tc.name)}</div>
                                <div class="tool-args">${escapeHtml(JSON.stringify(tc.arguments, null, 2))}</div>
                            </div>
                        `).join('');
                    } else {
                        toolCallsEl.innerHTML = '<div class="no-tools">No tool calls</div>';
                    }

                    // Add to history
                    const toolsSummary = data.tool_calls && data.tool_calls.length > 0
                        ? data.tool_calls.map(tc => tc.name).join(', ')
                        : 'no actions';

                    const item = document.createElement('div');
                    item.className = 'history-item';
                    item.innerHTML = `
                        <div class="text">${escapeHtml(data.original_text)}</div>
                        <div class="meta">${data.duration.toFixed(1)}s speech · ${data.processing_time_ms.toFixed(0)}ms processing</div>
                        <div class="tools-summary">→ ${escapeHtml(toolsSummary)}</div>
                    `;
                    historyEl.insertBefore(item, historyEl.firstChild);

                    // Keep only last 20 items
                    while (historyEl.children.length > 20) {
                        historyEl.removeChild(historyEl.lastChild);
                    }
                } else if (data.type === 'agent_error') {
                    statusEl.textContent = 'Listening...';
                    statusEl.className = 'status listening';
                    currentEl.textContent = 'Waiting for speech...';
                    currentEl.className = 'current-text empty';

                    llmMessageEl.textContent = `Error: ${data.error}`;
                    llmMessageEl.className = 'llm-message';
                    toolCallsEl.innerHTML = '<div class="no-tools">Error occurred</div>';
                }
            };
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        connect();
    </script>
</body>
</html>
"""


class SileroVAD:
    """Voice Activity Detection using Silero VAD"""

    def __init__(self):
        import torch
        self.torch = torch

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.model = model
        self.model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_tensor = self.torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()
        return speech_prob > 0.5

    def reset(self):
        self.model.reset_states()


class StreamingTranscriber:
    """Handles continuous transcription with Parakeet"""

    def __init__(self, model_name: str = 'nvidia/parakeet-ctc-0.6b'):
        import torch
        import nemo.collections.asr as nemo_asr

        print(f"Loading ASR model: {model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model = self.model.to('cuda')
        self.model.eval()
        self.torch = torch
        print("ASR model loaded and ready")

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return ""

        import soundfile as sf
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, SAMPLE_RATE)

        try:
            with self.torch.no_grad():
                result = self.model.transcribe([temp_path])
                return result[0].text if hasattr(result[0], 'text') else str(result[0])
        finally:
            os.unlink(temp_path)


class WebVoicePipeline:
    """Voice pipeline that broadcasts to WebSocket clients and calls Agent API"""

    def __init__(self, broadcast_func, agent_url: str, model_name: str = 'nvidia/parakeet-ctc-0.6b',
                 input_device: int = None):
        self.broadcast = broadcast_func
        self.agent_url = agent_url
        self.input_device = input_device

        print("Initializing VAD...")
        self.vad = SileroVAD()

        print("Initializing ASR...")
        self.transcriber = StreamingTranscriber(model_name)

        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * 30))
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.last_transcribe_time = 0

        self.audio_queue = queue.Queue()
        self.running = False

        # HTTP client for agent API
        self.http_client = httpx.Client(timeout=30.0)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        audio_chunk = indata[:, 0].copy()
        self.audio_queue.put(audio_chunk)

    def _process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            current_time = time.time()
            has_speech = self.vad.is_speech(audio_chunk)

            if has_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.audio_buffer.clear()
                    print("\n[Speech detected]")

                self.last_speech_time = current_time
                self.audio_buffer.extend(audio_chunk)

                if current_time - self.last_transcribe_time >= TRANSCRIBE_INTERVAL:
                    self._do_partial_transcription()
                    self.last_transcribe_time = current_time

            elif self.is_speaking:
                self.audio_buffer.extend(audio_chunk)

                silence_duration = current_time - self.last_speech_time
                if silence_duration >= SILENCE_THRESHOLD:
                    self._do_final_transcription()
                    self.is_speaking = False
                    self.audio_buffer.clear()
                    self.vad.reset()

    def _do_partial_transcription(self):
        if len(self.audio_buffer) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return

        audio = np.array(self.audio_buffer)
        text = self.transcriber.transcribe(audio)

        if text:
            print(f"[Partial] {text}")
            self.broadcast({"type": "partial", "text": text})

    def _do_final_transcription(self):
        if len(self.audio_buffer) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return

        audio = np.array(self.audio_buffer)
        duration = len(audio) / SAMPLE_RATE
        text = self.transcriber.transcribe(audio)

        if text:
            print(f"[Final] {text} ({duration:.1f}s)")
            self.broadcast({"type": "final", "text": text, "duration": duration})

            # Call Agent API
            self._call_agent(text, duration)

    def _call_agent(self, text: str, duration: float):
        """Send transcribed text to Agent API for command interpretation"""
        try:
            print(f"[Agent] Sending to {self.agent_url}/command...")
            response = self.http_client.post(
                f"{self.agent_url}/command",
                json={"text": text}
            )
            response.raise_for_status()

            data = response.json()
            print(f"[Agent] Response: {len(data.get('tool_calls', []))} tool calls")

            # Broadcast agent response
            self.broadcast({
                "type": "agent_response",
                "original_text": text,
                "duration": duration,
                "message": data.get("message"),
                "tool_calls": data.get("tool_calls", []),
                "needs_confirmation": data.get("needs_confirmation", False),
                "processing_time_ms": data.get("processing_time_ms", 0)
            })

        except Exception as e:
            print(f"[Agent] Error: {e}")
            self.broadcast({
                "type": "agent_error",
                "original_text": text,
                "duration": duration,
                "error": str(e)
            })

    def start(self):
        import sounddevice as sd

        print(f"\nStarting audio capture (device: {self.input_device})...")
        print(f"Agent API: {self.agent_url}")
        self.running = True

        processing_thread = threading.Thread(target=self._process_audio)
        processing_thread.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
            device=self.input_device
        ):
            while self.running:
                time.sleep(0.1)

        processing_thread.join()

    def stop(self):
        self.running = False
        self.http_client.close()


# Global pipeline reference
pipeline = None


@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()


@app.get("/")
async def get():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected ({len(connected_clients)} total)")

    try:
        while True:
            # Keep connection alive, wait for messages
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send ping to keep alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected ({len(connected_clients)} total)")


def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not connected_clients or not main_event_loop:
        return

    for client in list(connected_clients):
        try:
            asyncio.run_coroutine_threadsafe(
                client.send_json(message),
                main_event_loop
            )
        except Exception:
            connected_clients.discard(client)


def run_pipeline(device: int, model: str, agent_url: str):
    """Run the voice pipeline in a separate thread"""
    global pipeline
    pipeline = WebVoicePipeline(
        broadcast_func=broadcast_message,
        agent_url=agent_url,
        model_name=model,
        input_device=device
    )
    pipeline.start()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Web-based voice transcription with LLM')
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--model', '-m', type=str, default='nvidia/parakeet-ctc-0.6b',
                        help='Parakeet model name')
    parser.add_argument('--port', '-p', type=int, default=8888,
                        help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Web server host')
    parser.add_argument('--agent-url', '-a', type=str, default='http://localhost:8887',
                        help='Agent API URL')
    args = parser.parse_args()

    global AGENT_URL
    AGENT_URL = args.agent_url

    print(f"\n{'='*50}")
    print("Project Porg - Voice Command Interface")
    print(f"{'='*50}")
    print(f"Web UI: http://localhost:{args.port}")
    print(f"Agent API: {args.agent_url}")
    print(f"{'='*50}\n")

    # Start voice pipeline in background thread
    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(args.device, args.model, args.agent_url),
        daemon=True
    )
    pipeline_thread.start()

    # Run web server
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
