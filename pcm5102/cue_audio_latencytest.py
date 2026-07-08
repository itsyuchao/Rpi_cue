#!/usr/bin/env python3
"""
audio_latency_test.py
======================
Barebones cold-start audio latency probe. No haptics, no CLI args.

Sets 'low' latency preset, reports the negotiated stream latency at SR,
then plays a short tone on each Enter press, timing sd.play()+sd.wait()
against the theoretical playback duration. Compare the first press
(cold) to later presses (warm) to see how much startup overhead exists.

Ctrl+C to quit.
"""

import time

import numpy as np
import sounddevice as sd

SR = 48000
TONE_FREQ = 440.0
TONE_DUR_S = 0.15
PAD_S = 0.05


def perf_counter_raw() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def audio_init():
    sd.default.samplerate = SR
    sd.default.channels   = 1
    sd.default.dtype      = 'float32'
    sd.default.latency    = 'high'

    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            sd.default.device = i
            print(f"Audio: [{i}] {dev['name']}")
            break
    else:
        print("WARNING: HiFiBerry not found — using system default")

    probe = sd.OutputStream(samplerate=SR, channels=1, dtype='float32')
    probe.start()
    print(f"Negotiated output latency: {probe.latency * 1000:.2f} ms (at {SR} Hz)")
    probe.stop()
    probe.close()


def build_tone() -> np.ndarray:
    n = int(round(TONE_DUR_S * SR))
    t = np.arange(n, dtype=np.float64) / SR
    y = np.sin(2 * np.pi * TONE_FREQ * t).astype(np.float32)
    pad = np.zeros(int(round(PAD_S * SR)), dtype=np.float32)
    return np.concatenate([pad, y])


def play_and_time(payload: np.ndarray):
    t0 = perf_counter_raw()
    sd.play(payload, samplerate=SR, blocksize=128)
    sd.wait()
    elapsed = perf_counter_raw() - t0

    expected = len(payload) / SR
    overhead_ms = (elapsed - expected) * 1000
    print(f"  call={elapsed:.4f}s  expected={expected:.4f}s  overhead≈{overhead_ms:.1f}ms")


def main():
    audio_init()
    tone = build_tone()
    print("\nPress Enter to play a tone, Ctrl+C to quit.\n")
    try:
        i = 0
        while True:
            input(f">> [{i}] Enter: ")
            play_and_time(tone)
            i += 1
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == '__main__':
    main()
