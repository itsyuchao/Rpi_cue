#!/usr/bin/env python3
"""
loopback_latency_test.py
=========================
Measures REAL audio onset latency: plays a tone burst on the HifiBerry DAC
while simultaneously recording on the USB mic, then finds when the tone
actually arrives in the recording vs. when it was supposed to start.

This replaces the earlier call-duration-only test (which only measured
sd.play()+sd.wait() total time, not actual sound onset) with a genuine
loopback measurement.

IMPORTANT — acoustic vs. cable loopback:
  If the mic just sits near the speaker/DAC output, the measured latency
  includes real acoustic travel time (~2.9 ms per meter) plus ambient
  noise margin — not just electronic/driver latency. For the cleanest
  number, wire the HifiBerry's analog output directly into the mic's
  input (a cable loopback) instead of relying on air. Acoustic pickup is
  fine for a rough number; cable loopback is what you want if the ~ms
  precision matters.

Method:
  1. Build a payload: silence pad, then a short tone burst with a fast
     attack (sharp onset = easy to detect), then trailing silence.
  2. sd.playrec() starts playback (DAC) and recording (mic) together in
     one call, so both streams share the same t0 reference.
  3. Find the first sample in the recording where energy rises above the
     noise floor — that's the measured/actual onset time.
  4. measured_latency = actual_onset_time - intended_onset_time (where
     the tone starts in the payload). This is the real cold/warm delay,
     not just call duration.

Usage: run repeatedly (each Enter press = one measurement). Compare the
first press (cold) to later ones (warm).
"""

import time

import numpy as np
import sounddevice as sd

SR           = 44100          # common safe rate for both devices; adjust if
                               # sd.query_devices() shows a different shared rate
PAD_S        = 0.5            # silence before the tone (must exceed cold-start
                               # latency you're trying to measure, or it clips)
TONE_FREQ    = 1000.0
TONE_DUR_S   = 0.1
TAIL_S       = 0.5            # silence after, for margin
ATTACK_MS    = 1              # fast attack = sharp, detectable onset

DAC_NAME_MATCH = 'hifiberry'
MIC_NAME_MATCH = 'usb'


sd.default.latency = 'low'

def perf_counter_raw() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def find_devices():
    dac_idx = mic_idx = None
    for i, dev in enumerate(sd.query_devices()):
        name = dev['name'].lower()
        if dac_idx is None and DAC_NAME_MATCH in name and dev['max_output_channels'] > 0:
            dac_idx = i
        if mic_idx is None and MIC_NAME_MATCH in name and dev['max_input_channels'] > 0:
            mic_idx = i
    if dac_idx is None or mic_idx is None:
        print(sd.query_devices())
        raise RuntimeError(
            f"Could not find both devices (dac={dac_idx}, mic={mic_idx}). "
            "Check the printed device list above and hardcode indices if needed."
        )
    print(f"DAC device: [{dac_idx}] {sd.query_devices(dac_idx)['name']}")
    print(f"Mic device: [{mic_idx}] {sd.query_devices(mic_idx)['name']}")
    return dac_idx, mic_idx


def build_payload():
    pad  = np.zeros(int(round(PAD_S * SR)), dtype=np.float32)
    tail = np.zeros(int(round(TAIL_S * SR)), dtype=np.float32)

    n = int(round(TONE_DUR_S * SR))
    t = np.arange(n, dtype=np.float64) / SR
    tone = np.sin(2 * np.pi * TONE_FREQ * t).astype(np.float32)
    atk = max(1, int(ATTACK_MS * 1e-3 * SR))
    env = np.ones(n, dtype=np.float32)
    env[:atk] = np.linspace(0, 1, atk)
    tone *= env

    payload = np.concatenate([pad, tone, tail])
    intended_onset_s = PAD_S
    return payload, intended_onset_s


def detect_onset(recording: np.ndarray, threshold_k: float = 6.0) -> float | None:
    """Return the time (s) of the first energy rise above the noise floor,
    or None if nothing crossed threshold. Noise floor is estimated from the
    first 80% of PAD_S (assumed silent before the tone arrives)."""
    rec = recording.flatten()
    noise_window = int(0.8 * PAD_S * SR)
    noise = rec[:noise_window]
    floor = np.abs(noise).mean()
    spread = np.abs(noise).std()
    threshold = floor + threshold_k * spread

    envelope = np.abs(rec)
    above = np.where(envelope > threshold)[0]
    if len(above) == 0:
        return None
    return above[0] / SR


def run_once(dac_idx: int, mic_idx: int, payload: np.ndarray, intended_onset_s: float):
    t0 = perf_counter_raw()
    recording = sd.playrec(
        payload, samplerate=SR, channels=1,
        device=(mic_idx, dac_idx), dtype='float32',
	latency=0.02,blocksize=256,
    )
    sd.wait()
    call_elapsed = perf_counter_raw() - t0

    actual_onset_s = detect_onset(recording)
    if actual_onset_s is None:
        print(f"  call={call_elapsed:.4f}s  [no onset detected above noise floor — "
              f"check mic gain/position, or lower threshold_k]")
        return

    measured_latency_ms = (actual_onset_s - intended_onset_s) * 1000
    print(f"  call={call_elapsed:.4f}s  "
          f"intended_onset={intended_onset_s:.4f}s  "
          f"actual_onset={actual_onset_s:.4f}s  "
          f"measured_latency≈{measured_latency_ms:.1f}ms")


def main():
    dac_idx, mic_idx = find_devices()
    payload, intended_onset_s = build_payload()

    print(f"\nSR={SR}Hz  pad={PAD_S}s  tone={TONE_DUR_S}s  tail={TAIL_S}s")
    print("Press Enter to run one loopback measurement, Ctrl+C to quit.\n")
    try:
        i = 0
        while True:
            input(f">> [{i}] Enter: ")
            run_once(dac_idx, mic_idx, payload, intended_onset_s)
            i += 1
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == '__main__':
    main()
