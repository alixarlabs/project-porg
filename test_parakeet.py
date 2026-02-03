#!/usr/bin/env python3
import nemo.collections.asr as nemo_asr

print('Loading Parakeet CTC model...')
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-ctc-0.6b')
asr_model = asr_model.to('cuda')

print('Transcribing test audio...')
output = asr_model.transcribe(['test_audio.wav'])
print(f'\n=== Transcription ===\n{output[0].text}\n')
