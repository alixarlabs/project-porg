#!/usr/bin/env python3
"""
Real-time streaming voice input with continuous transcription.

Architecture:
- Continuous audio capture via sounddevice
- Silero VAD for speech detection and end-of-utterance
- Incremental transcription updates while speaking
- Final transcription when speech ends

Usage:
    python voice_input.py
"""

import numpy as np
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional
import warnings
warnings.filterwarnings("ignore")

# Audio settings
SAMPLE_RATE = 16000  # Parakeet expects 16kHz
# Silero VAD requires exactly 512 samples at 16kHz (32ms)
CHUNK_SIZE = 512
TRANSCRIBE_INTERVAL = 0.5  # Transcribe every 500ms while speaking
MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to transcribe
SILENCE_THRESHOLD = 0.5  # Seconds of silence to end utterance


@dataclass
class TranscriptionResult:
    """Holds transcription results"""
    text: str
    is_final: bool
    duration: float


class SileroVAD:
    """Voice Activity Detection using Silero VAD"""

    def __init__(self):
        import torch
        self.torch = torch

        # Load Silero VAD model
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.model = model
        self.get_speech_timestamps = utils[0]

        # Reset state
        self.model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech"""
        # Convert to tensor
        audio_tensor = self.torch.from_numpy(audio_chunk).float()

        # Get speech probability
        speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

        return speech_prob > 0.5

    def reset(self):
        """Reset VAD state"""
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
        """Transcribe audio array"""
        if len(audio) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return ""

        # Save to temp file (NeMo's transcribe expects file paths)
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


class VoiceInputPipeline:
    """
    Main pipeline for real-time voice input with streaming transcription.

    Provides continuous transcription updates while speaking,
    with a final result when the user stops.
    """

    def __init__(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[TranscriptionResult], None]] = None,
        model_name: str = 'nvidia/parakeet-ctc-0.6b',
        input_device: Optional[int] = None
    ):
        """
        Initialize the voice input pipeline.

        Args:
            on_partial: Callback for partial transcription updates
            on_final: Callback for final transcription when speech ends
            model_name: Parakeet model to use
            input_device: Audio input device index (None for default)
        """
        self.on_partial = on_partial or (lambda x: None)
        self.on_final = on_final or (lambda x: None)
        self.input_device = input_device

        # Initialize components
        print("Initializing VAD...")
        self.vad = SileroVAD()

        print("Initializing ASR...")
        self.transcriber = StreamingTranscriber(model_name)

        # Audio buffer and state
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * 30))  # Max 30s
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.last_transcribe_time = 0

        # Threading
        self.audio_queue = queue.Queue()
        self.running = False
        self.processing_thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice audio stream"""
        if status:
            print(f"Audio status: {status}")

        # Add audio to processing queue
        audio_chunk = indata[:, 0].copy()  # Mono
        self.audio_queue.put(audio_chunk)

    def _process_audio(self):
        """Process audio chunks from queue"""
        import sounddevice as sd

        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            current_time = time.time()

            # Check for speech
            has_speech = self.vad.is_speech(audio_chunk)

            if has_speech:
                if not self.is_speaking:
                    # Speech started
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.audio_buffer.clear()
                    print("\n🎤 Speech detected...")

                self.last_speech_time = current_time

                # Add to buffer
                self.audio_buffer.extend(audio_chunk)

                # Periodic transcription while speaking
                if current_time - self.last_transcribe_time >= TRANSCRIBE_INTERVAL:
                    self._do_partial_transcription()
                    self.last_transcribe_time = current_time

            elif self.is_speaking:
                # Still add audio during short silences
                self.audio_buffer.extend(audio_chunk)

                # Check for end of utterance
                silence_duration = current_time - self.last_speech_time
                if silence_duration >= SILENCE_THRESHOLD:
                    # Speech ended - do final transcription
                    self._do_final_transcription()

                    # Reset state
                    self.is_speaking = False
                    self.audio_buffer.clear()
                    self.vad.reset()

    def _do_partial_transcription(self):
        """Perform partial transcription of current buffer"""
        if len(self.audio_buffer) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return

        audio = np.array(self.audio_buffer)
        text = self.transcriber.transcribe(audio)

        if text:
            self.on_partial(text)

    def _do_final_transcription(self):
        """Perform final transcription when speech ends"""
        if len(self.audio_buffer) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return

        audio = np.array(self.audio_buffer)
        duration = len(audio) / SAMPLE_RATE

        text = self.transcriber.transcribe(audio)

        if text:
            result = TranscriptionResult(
                text=text,
                is_final=True,
                duration=duration
            )
            self.on_final(result)

    def start(self):
        """Start the voice input pipeline"""
        import sounddevice as sd

        print("\n" + "="*50)
        print("🎙️  Voice Input Pipeline Started")
        print("="*50)
        print("Speak into your microphone...")
        print("Press Ctrl+C to stop\n")

        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()

        # Start audio stream
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

        self.stop()

    def stop(self):
        """Stop the voice input pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        print("Pipeline stopped.")


def main():
    """Demo of the voice input pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time voice transcription')
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--model', '-m', type=str, default='nvidia/parakeet-ctc-0.6b',
                        help='Parakeet model name')
    parser.add_argument('--list-devices', '-l', action='store_true',
                        help='List audio devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("\n=== Audio Devices ===\n")
        print(sd.query_devices())
        return

    # Callbacks for transcription results
    def on_partial(text: str):
        # Clear line and print partial result
        print(f"\r📝 {text:<80}", end="", flush=True)

    def on_final(result: TranscriptionResult):
        # Print final result on new line
        print(f"\r✅ {result.text:<80}")
        print(f"   (Duration: {result.duration:.1f}s)")
        print()

    # Create and start pipeline
    pipeline = VoiceInputPipeline(
        on_partial=on_partial,
        on_final=on_final,
        model_name=args.model,
        input_device=args.device
    )

    pipeline.start()


if __name__ == "__main__":
    main()
