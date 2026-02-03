#!/usr/bin/env python3
"""Transcribe an audio file with Parakeet"""
import sys
import nemo.collections.asr as nemo_asr

filename = sys.argv[1] if len(sys.argv) > 1 else 'test_audio.wav'
print(f'Loading Parakeet model...')
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-ctc-0.6b')
model = model.to('cuda')
print(f'Transcribing: {filename}')
result = model.transcribe([filename])
print(f'\n=== Transcription ===\n{result[0].text}\n')
