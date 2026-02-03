#!/usr/bin/env python3
"""
Test audio input devices and record a sample.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import sys

def list_devices():
    """List all audio devices"""
    print("\n=== Audio Devices ===\n")
    print(sd.query_devices())
    print("\nDefault input device:", sd.default.device[0])
    print("Default output device:", sd.default.device[1])

def test_recording(device=None, duration=5, filename="test_recording.wav"):
    """Record a test sample"""
    sample_rate = 16000

    print(f"\n=== Recording Test ===")
    print(f"Device: {device if device is not None else 'default'}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print("\nRecording in 3...")
    import time
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("🎤 Recording NOW - speak!")

    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            device=device
        )
        sd.wait()

        print("✅ Recording complete!")

        # Save to file
        sf.write(filename, recording, sample_rate)
        print(f"Saved to: {filename}")

        # Check audio levels
        max_level = np.max(np.abs(recording))
        rms_level = np.sqrt(np.mean(recording**2))
        print(f"\nAudio levels:")
        print(f"  Max: {max_level:.4f}")
        print(f"  RMS: {rms_level:.4f}")

        if max_level < 0.01:
            print("⚠️  Warning: Audio level very low - check microphone")
        elif max_level > 0.95:
            print("⚠️  Warning: Audio may be clipping")
        else:
            print("✅ Audio levels look good!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    list_devices()

    # Check for command line device argument
    device = None
    if len(sys.argv) > 1:
        try:
            device = int(sys.argv[1])
        except ValueError:
            device = sys.argv[1]  # Could be device name

    print("\n" + "="*50)
    response = input("Would you like to test recording? [Y/n]: ").strip().lower()
    if response != 'n':
        test_recording(device=device)

if __name__ == "__main__":
    main()
