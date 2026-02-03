#!/usr/bin/env python3
"""
Voice-Controlled Surgical Agent

End-to-end pipeline: Microphone → ASR → LLM → Tool Calls

Combines:
- Parakeet ASR for speech-to-text
- Silero VAD for voice activity detection
- Agent API for command interpretation

Usage:
    python voice_agent.py --agent-url http://localhost:8080
"""

import argparse
import json
import numpy as np
import threading
import queue
import time
import requests
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable
import warnings
warnings.filterwarnings("ignore")

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requires 512 samples at 16kHz
TRANSCRIBE_INTERVAL = 0.5
MIN_SPEECH_DURATION = 0.3
SILENCE_THRESHOLD = 0.8  # Slightly longer for command completion


@dataclass
class AgentResponse:
    """Response from the agent API"""
    tool_calls: list
    message: Optional[str]
    needs_confirmation: bool
    processing_time_ms: float


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
    """Handles transcription with Parakeet"""

    def __init__(self, model_name: str = 'nvidia/parakeet-ctc-0.6b'):
        import torch
        import nemo.collections.asr as nemo_asr

        print(f"Loading ASR model: {model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model = self.model.to('cuda')
        self.model.eval()
        self.torch = torch
        print("ASR model loaded")

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


class AgentClient:
    """Client for the Agent API"""

    def __init__(self, base_url: str, session_id: str = "voice"):
        self.base_url = base_url.rstrip('/')
        self.session_id = session_id

    def send_command(self, text: str) -> AgentResponse:
        """Send a command to the agent API"""
        response = requests.post(
            f"{self.base_url}/command",
            json={
                "text": text,
                "session_id": self.session_id
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        return AgentResponse(
            tool_calls=data.get("tool_calls", []),
            message=data.get("message"),
            needs_confirmation=data.get("needs_confirmation", False),
            processing_time_ms=data.get("processing_time_ms", 0)
        )

    def health_check(self) -> dict:
        """Check agent health"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()


class VoiceAgentPipeline:
    """
    Complete voice-to-action pipeline.

    Listens for speech, transcribes it, sends to agent, displays results.
    """

    def __init__(
        self,
        agent_url: str,
        asr_model: str = 'nvidia/parakeet-ctc-0.6b',
        input_device: Optional[int] = None,
        on_tool_call: Optional[Callable] = None
    ):
        self.agent = AgentClient(agent_url)
        self.input_device = input_device
        self.on_tool_call = on_tool_call

        print("\n" + "=" * 50)
        print("Initializing Voice Agent Pipeline")
        print("=" * 50)

        # Check agent health
        print("Checking agent connection...")
        try:
            health = self.agent.health_check()
            print(f"  Agent: {health['status']}")
            print(f"  Model: {health['model']}")
            print(f"  Tools: {health['tools_available']}")
        except Exception as e:
            print(f"  ⚠️  Agent not reachable: {e}")
            print("  Make sure to start the agent: ./run_agent.sh")
            raise

        print("\nInitializing VAD...")
        self.vad = SileroVAD()

        print("Initializing ASR...")
        self.transcriber = StreamingTranscriber(asr_model)

        # Audio state
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * 30))
        self.is_speaking = False
        self.last_speech_time = 0
        self.last_transcribe_time = 0
        self.current_partial = ""

        # Threading
        self.audio_queue = queue.Queue()
        self.running = False

        print("=" * 50 + "\n")

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
                    self.audio_buffer.clear()
                    self.current_partial = ""
                    print("\n🎤 Listening...", end="", flush=True)

                self.last_speech_time = current_time
                self.audio_buffer.extend(audio_chunk)

                # Periodic transcription for feedback
                if current_time - self.last_transcribe_time >= TRANSCRIBE_INTERVAL:
                    self._do_partial_transcription()
                    self.last_transcribe_time = current_time

            elif self.is_speaking:
                self.audio_buffer.extend(audio_chunk)

                silence_duration = current_time - self.last_speech_time
                if silence_duration >= SILENCE_THRESHOLD:
                    self._process_command()
                    self.is_speaking = False
                    self.audio_buffer.clear()
                    self.vad.reset()

    def _do_partial_transcription(self):
        if len(self.audio_buffer) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return

        audio = np.array(self.audio_buffer)
        text = self.transcriber.transcribe(audio)

        if text and text != self.current_partial:
            self.current_partial = text
            # Clear line and show partial
            print(f"\r🎤 {text:<60}", end="", flush=True)

    def _process_command(self):
        """Transcribe final audio and send to agent"""
        if len(self.audio_buffer) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            print("\r" + " " * 70 + "\r", end="")
            return

        audio = np.array(self.audio_buffer)
        text = self.transcriber.transcribe(audio)

        if not text:
            print("\r" + " " * 70 + "\r", end="")
            return

        # Show final transcription
        print(f"\r📝 \"{text}\"" + " " * 30)

        # Send to agent
        print("   🤖 Processing...", end="", flush=True)

        try:
            response = self.agent.send_command(text)
            print(f"\r   ✅ Response ({response.processing_time_ms:.0f}ms)" + " " * 20)

            # Display tool calls
            if response.tool_calls:
                for tc in response.tool_calls:
                    print(f"\n   📟 {tc['name']}:")
                    for k, v in tc['arguments'].items():
                        print(f"      {k}: {v}")

                    # Callback for actual execution
                    if self.on_tool_call:
                        self.on_tool_call(tc)

            # Display message if any
            if response.message:
                print(f"\n   💬 {response.message}")

            # Confirmation warning
            if response.needs_confirmation:
                print("\n   ⚠️  This action requires confirmation!")

            print()

        except Exception as e:
            print(f"\r   ❌ Error: {e}" + " " * 30)

    def start(self):
        """Start the voice agent pipeline"""
        import sounddevice as sd

        print("=" * 50)
        print("🎙️  Voice Agent Active")
        print("=" * 50)
        print("Speak commands like:")
        print('  "More light"')
        print('  "Raise the table 10 centimeters"')
        print('  "Show the CT scan on the main display"')
        print("\nPress Ctrl+C to stop")
        print("=" * 50 + "\n")

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
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopping...")

        self.running = False
        processing_thread.join()
        print("Voice agent stopped.")


def main():
    parser = argparse.ArgumentParser(description="Voice-controlled surgical agent")
    parser.add_argument('--agent-url', '-a', default='http://localhost:8080',
                        help='Agent API URL')
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--model', '-m', default='nvidia/parakeet-ctc-0.6b',
                        help='ASR model name')
    parser.add_argument('--list-devices', '-l', action='store_true',
                        help='List audio devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("\n=== Audio Devices ===\n")
        print(sd.query_devices())
        return

    # Example tool call handler (would connect to real devices)
    def handle_tool_call(tool_call):
        """Handle tool calls - connect to actual devices here"""
        # In production, this would call device APIs
        # For now, just log
        pass

    pipeline = VoiceAgentPipeline(
        agent_url=args.agent_url,
        asr_model=args.model,
        input_device=args.device,
        on_tool_call=handle_tool_call
    )

    pipeline.start()


if __name__ == "__main__":
    main()
