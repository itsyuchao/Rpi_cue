#!/usr/bin/env python3
"""
cue_demo.py
===========
Barebones presentation demo of the regular cue train.

Prompts for 'a' (auditory) or 'v' (vibration), plays a 4-cycle regular train
at --freq (default 0.83 Hz), then prompts again. Ctrl+C exits.

No logging, no UDP, no templates — just the train.
"""

import argparse
import time

import numpy as np
import sounddevice as sd

# ── Audio synthesis (mirrors cue_experiment.py) ──────────────────────────────
SR         = 48000
F_LOW      = 440.0
F_HIGH     = 659.3
AMPLITUDE  = 1
ATTACK_MS  = 5
RELEASE_MS = 20
HARMONICS  = [(1, 1.00), (2, 0.30), (3, 0.15), (4, 0.10)]
TONE_DURATION = 0.15

# Haptic effects (DRV2605 library)
DEFAULT_EFFECT1 = 1    # Strong Click → "low" beat
DEFAULT_EFFECT2 = 14   # Strong Buzz   → "high" beat

# Haptic (graceful fallback)
try:
    import board
    import busio
    import adafruit_drv2605
    HAPTIC_AVAILABLE = True
except ImportError:
    HAPTIC_AVAILABLE = False
    print("INFO: adafruit_drv2605 not found — haptic output disabled")

_drv = None


def perf_counter_raw() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


# ── Audio ────────────────────────────────────────────────────────────────────

def audio_init():
    sd.default.samplerate = SR
    sd.default.channels   = 1
    sd.default.dtype      = 'float32'
    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            sd.default.device = i
            print(f"Audio: [{i}] {dev['name']}")
            return
    print("WARNING: HiFiBerry not found — using system default")


def _silence(dur_s: float) -> np.ndarray:
    return np.zeros(max(1, int(round(dur_s * SR))), dtype=np.float32)


def _synth_tone(freq: float, dur_s: float) -> np.ndarray:
    n = max(1, int(round(dur_s * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    y = sum(amp * np.sin(2 * np.pi * k * freq * t) for k, amp in HARMONICS)
    atk = max(1, int(ATTACK_MS  * 1e-3 * SR))
    rel = max(1, int(RELEASE_MS * 1e-3 * SR))
    env = np.ones(n, dtype=np.float64)
    env[:atk]  = np.linspace(0, 1, atk)
    if rel < n:
        env[-rel:] = np.linspace(1, 0, rel)
    y *= env
    peak = np.max(np.abs(y))
    if peak > 0:
        y = AMPLITUDE * y / peak
    return y.astype(np.float32)


def build_regular_audio_cycles(n_cycles: int, freq_hz: float) -> np.ndarray:
    """Build exactly n_cycles of low/high alternation at freq_hz.
    One cycle = low + gap + high + gap, period = 1/freq_hz."""
    half   = 0.5 / freq_hz
    tone_d = TONE_DURATION
    gap_d  = max(0.0, half - tone_d)
    cycle  = np.concatenate([
        _synth_tone(F_LOW, tone_d), _silence(gap_d),
        _synth_tone(F_HIGH, tone_d), _silence(gap_d),
    ])
    return np.tile(cycle, n_cycles)


def play_audio(waveform: np.ndarray):
    pad = _silence(0.05)
    sd.play(np.concatenate([pad, waveform]), samplerate=SR, blocksize=128)
    sd.wait()


# ── Haptic ───────────────────────────────────────────────────────────────────

def haptic_init():
    global _drv
    if not HAPTIC_AVAILABLE:
        return
    try:
        i2c  = busio.I2C(board.SCL, board.SDA)
        _drv = adafruit_drv2605.DRV2605(i2c)
        _drv.use_LRM()
        print("Haptic: DRV2605L initialised (LRA)")
    except Exception as e:
        print(f"WARNING: DRV2605 init failed — {e}")
        _drv = None


def build_regular_haptic_cycles(n_cycles: int, freq_hz: float,
                                 effect1: int = DEFAULT_EFFECT1,
                                 effect2: int = DEFAULT_EFFECT2) -> list:
    """List of (time_s, effect_id) for n_cycles at freq_hz."""
    half = 0.5 / freq_hz
    events = []
    t = 0.0
    for _ in range(n_cycles):
        events.append((t, effect1))
        t += half
        events.append((t, effect2))
        t += half
    return events


def play_haptic(events: list, total_s: float):
    if _drv is None or not events:
        return
    t0 = perf_counter_raw()
    for rel_t, effect_id in events:
        wait = rel_t - (perf_counter_raw() - t0)
        if wait > 0.002:
            time.sleep(wait - 0.001)
        while perf_counter_raw() - t0 < rel_t:
            pass
        _drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        _drv.play()
    _drv.stop()
    remaining = total_s - (perf_counter_raw() - t0)
    if remaining > 0:
        time.sleep(remaining)


# ── Main loop ────────────────────────────────────────────────────────────────

N_CYCLES = 4


def main():
    args = parse_args()
    freq = args.freq
    period = 1.0 / freq
    total_s = N_CYCLES * period

    audio_init()
    haptic_init()

    # Precompute once
    wav_train     = build_regular_audio_cycles(N_CYCLES, freq)
    haptic_events = build_regular_haptic_cycles(N_CYCLES, freq)

    print(f"\nDemo: {N_CYCLES} cycles @ {freq} Hz "
          f"(period {period:.3f}s, total {total_s:.2f}s)")
    print("Enter 'a' for auditory, 'v' for vibration. Ctrl+C to quit.\n")

    try:
        while True:
            try:
                k = input(">> a/v: ").strip().lower()
            except EOFError:
                break
            if k == 'a':
                play_audio(wav_train)
            elif k == 'v':
                if _drv is None:
                    print("  [HAPTIC] No driver — skipping")
                else:
                    play_haptic(haptic_events, total_s)
            else:
                print("  (ignored — type 'a' or 'v')")
    except KeyboardInterrupt:
        print("\nExiting.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Barebones 4-cycle cue demo (auditory or vibration).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--freq', type=float, default=0.83,
                   help="Regular-cue rate (Hz).")
    return p.parse_args()


if __name__ == '__main__':
    main()
