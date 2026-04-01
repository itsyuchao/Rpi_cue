#!/usr/bin/env python3
"""
cue_experiment.py
=================
Rhythmic cueing experiment controller for Raspberry Pi 4.
Delivers sound (PCM5102A via I2S) and/or vibration (DRV2605L LRA via I2C)
in structured 3-sub-block trials.

Hardware
--------
  Sound : PCM5102A breakout → I2S (BCLK=GPIO18, LRCLK=GPIO19, DIN=GPIO21)
          /boot/config.txt: dtoverlay=hifiberry-dac
  Haptic: DRV2605L → I2C (SDA/SCL), LRA motor on Motor+/Motor-

Install
-------
  pip install numpy sounddevice scipy \
              adafruit-blinka adafruit-circuitpython-drv2605

Quick start
-----------
  python3 cue_experiment.py                        # defaults
  python3 cue_experiment.py --randomize
  python3 cue_experiment.py --firstblock vibration --blocknum 10
  python3 cue_experiment.py --simtest              # run simultaneous test only

API
---
  generate_cue(blocknum=20, blocktime=15, preptime=2, freq=1,
               randomize=False, firstblock='sound')
"""

import argparse
import random
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# ── Optional haptic driver (graceful degradation on dev machines) ─────────────
try:
    import board
    import busio
    import adafruit_drv2605
    HAPTIC_AVAILABLE = True
except ImportError:
    HAPTIC_AVAILABLE = False
    print("INFO: adafruit_drv2605 not found — haptic output disabled")

# ─────────────────────────────────────────────────────────────────────────────
# Audio constants
# ─────────────────────────────────────────────────────────────────────────────
SR         = 48000   # Hz — native rate for PCM5102A (no resampling in PipeWire)
F_LOW      = 440.0    # A4
F_HIGH     = 659.3    # E5 (perfect fifth above A4)
AMPLITUDE  = 0.80
ATTACK_MS  = 5
RELEASE_MS = 20

# Harmonic recipe — bright but pleasant timbre (same as twotone_precomputed.py)
HARMONICS = [
    (1, 1.00),   # fundamental
    (2, 0.30),   # octave
    (3, 0.15),   # fifth above octave
    (4, 0.10),   # two octaves
]

TONE_DUTY = 0.3     # tone fills 30 % of each half-period; rest is silence

# Arrhythmic ISI bounds (seconds between successive pair onsets)
ARHYTHM_ISI_MIN = 0.35
ARHYTHM_ISI_MAX = 1.20

# ─────────────────────────────────────────────────────────────────────────────
# Haptic defaults (DRV2605L built-in library effects)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_EFFECT1 = 1    # Strong Click  — maps to "low" beat (A4)
DEFAULT_EFFECT2 = 47   # Sharp Tick    — maps to "high" beat (E5)

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Audio helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_hifiberry():
    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            return i
    return None


def _setup_audio():
    sd.default.samplerate = SR
    sd.default.channels   = 1
    sd.default.dtype      = 'float32'
    idx = _find_hifiberry()
    if idx is not None:
        sd.default.device = idx
        print(f"Audio: [{idx}] {sd.query_devices(idx)['name']}")
    else:
        print("WARNING: HiFiBerry DAC not found — using system default audio")


def load_wav(path: str) -> np.ndarray:
    """Load a WAV file, convert to float32 mono at SR."""
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        n_out = int(len(data) * SR / sr)
        indices = np.round(np.linspace(0, len(data) - 1, n_out)).astype(int)
        data = data[indices]
    return data.astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(max(1, int(round(duration_s * SR))), dtype=np.float32)


def _pad_startup(audio: np.ndarray, ms: float = 50) -> np.ndarray:
    """Prepend brief silence to avoid I2S startup click."""
    return np.concatenate([_silence(ms / 1000), audio])


def _synth_tone(freq: float, duration_s: float) -> np.ndarray:
    """Synthesize a single tone with harmonics, attack, and release."""
    n = max(1, int(round(duration_s * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    y = sum(amp * np.sin(2.0 * np.pi * k * freq * t) for k, amp in HARMONICS)

    atk = max(1, int(ATTACK_MS  * 1e-3 * SR))
    rel = max(1, int(RELEASE_MS * 1e-3 * SR))
    env = np.ones(n, dtype=np.float64)
    env[:atk]  = np.linspace(0.0, 1.0, atk)
    if rel < n:
        env[-rel:] = np.linspace(1.0, 0.0, rel)
    y *= env

    peak = np.max(np.abs(y))
    if peak > 0:
        y = AMPLITUDE * y / peak
    return y.astype(np.float32)


def _play_audio(waveform: np.ndarray, blocking: bool = True):
    """Fire a precomputed waveform through sounddevice."""
    sd.play(_pad_startup(waveform), samplerate=SR, blocksize=128)
    if blocking:
        sd.wait()


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Sound waveform builders  (all fully precomputed)
# ══════════════════════════════════════════════════════════════════════════════

def build_regular_sound(blocktime_s: float, freq_hz: float) -> np.ndarray:
    """
    Regular alternating A4 / E5 train at freq_hz Hz for blocktime_s seconds.
    Each half-period: tone (TONE_DUTY) + silence (1 - TONE_DUTY).
    Structure mirrors twotone_precomputed.py.
    """
    half_period_s = 0.5 / freq_hz
    tone_dur_s    = half_period_s * TONE_DUTY
    gap_dur_s     = half_period_s - tone_dur_s

    tone_lo = _synth_tone(F_LOW,  tone_dur_s)
    tone_hi = _synth_tone(F_HIGH, tone_dur_s)
    gap     = _silence(gap_dur_s)

    cycle = np.concatenate([tone_lo, gap, tone_hi, gap])
    n_cyc = max(1, int(np.ceil(blocktime_s / (len(cycle) / SR))))
    train = np.tile(cycle, n_cyc)
    return train[:int(round(blocktime_s * SR))]


def build_arrhythmic_sound(blocktime_s: float, freq_hz: float,
                            arhythmtime_s: float) -> np.ndarray:
    """
    Arrhythmic A4/E5 pairs for the first `arhythmtime_s` seconds of the
    block; silence for the remaining (blocktime_s - arhythmtime_s) seconds.

    Pair structure: A4 tone → intra-pair gap → E5 tone  (same durations as
    regular train), placed at random inter-onset intervals drawn from
    U[ARHYTHM_ISI_MIN, ARHYTHM_ISI_MAX].
    """
    arhythmtime_s = min(arhythmtime_s, blocktime_s)

    half_period_s = 0.5 / freq_hz
    tone_dur_s    = half_period_s * TONE_DUTY
    gap_dur_s     = half_period_s - tone_dur_s

    tone_lo = _synth_tone(F_LOW,  tone_dur_s)
    tone_hi = _synth_tone(F_HIGH, tone_dur_s)
    intra   = _silence(gap_dur_s)
    pair    = np.concatenate([tone_lo, intra, tone_hi])   # one low+high burst
    pair_n  = len(pair)

    total_n    = int(round(blocktime_s    * SR))
    arrhythm_n = int(round(arhythmtime_s * SR))
    buf = np.zeros(total_n, dtype=np.float32)

    t = 0.0
    while True:
        isi   = random.uniform(ARHYTHM_ISI_MIN, ARHYTHM_ISI_MAX)
        onset = t + isi
        end_s = onset + pair_n / SR
        if end_s > arhythmtime_s:
            break
        start = int(round(onset * SR))
        buf[start : start + pair_n] += pair
        t = onset + pair_n / SR

    np.clip(buf, -1.0, 1.0, out=buf)
    return buf


def build_silent_sound(blocktime_s: float) -> np.ndarray:
    return _silence(blocktime_s)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Haptic schedule builders  (precomputed event lists)
# ══════════════════════════════════════════════════════════════════════════════
# A haptic schedule is a list of (rel_time_s, effect_id) tuples.
# play_haptic() fires them at the correct absolute time using perf_counter.

def build_regular_haptic(blocktime_s: float, freq_hz: float,
                          effect1: int = DEFAULT_EFFECT1,
                          effect2: int = DEFAULT_EFFECT2) -> list:
    """
    Alternating effect1 / effect2 at freq_hz Hz for blocktime_s seconds.
    effect1 fires on the "low" beat, effect2 on the "high" beat,
    matching the A4/E5 alternation in build_regular_sound().
    """
    half_period = 0.5 / freq_hz
    events = []
    t = 0.0
    while t < blocktime_s - 1e-9:
        events.append((t, effect1))
        t += half_period
        if t < blocktime_s - 1e-9:
            events.append((t, effect2))
            t += half_period
    return events


def build_arrhythmic_haptic(blocktime_s: float, freq_hz: float,
                              arhythmtime_s: float,
                              effect1: int = DEFAULT_EFFECT1,
                              effect2: int = DEFAULT_EFFECT2) -> list:
    """
    Arrhythmic effect1/effect2 pairs for the first `arhythmtime_s` seconds.
    Pair structure: effect1 then effect2 spaced by half_period — mirrors
    build_arrhythmic_sound() so sound and haptic have the same temporal skeleton.
    """
    arhythmtime_s = min(arhythmtime_s, blocktime_s)
    half_period   = 0.5 / freq_hz
    events        = []

    t = 0.0
    while True:
        isi   = random.uniform(ARHYTHM_ISI_MIN, ARHYTHM_ISI_MAX)
        onset = t + isi
        if onset >= arhythmtime_s:
            break
        events.append((onset, effect1))
        t2 = onset + half_period
        if t2 < arhythmtime_s:
            events.append((t2, effect2))
        t = onset + half_period
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Haptic playback engine
# ══════════════════════════════════════════════════════════════════════════════

def play_haptic(drv, events: list):
    """
    Execute a precomputed haptic schedule with sub-millisecond accuracy.
    Uses sleep + spin-wait: sleep until ~1 ms before the target, then busy-
    wait for the last slice so the OS scheduler doesn't cause late firing.
    """
    if not events:
        return
    t0 = time.perf_counter()
    for rel_t, effect_id in events:
        now  = time.perf_counter() - t0
        wait = rel_t - now
        if wait > 0.002:
            time.sleep(wait - 0.001)               # sleep most of the gap
        while time.perf_counter() - t0 < rel_t:   # spin the last ~1 ms
            pass
        drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        drv.play()
    drv.stop()


def play_haptic_block(drv, events: list, total_s: float):
    """Play haptic schedule then wait until total_s has elapsed from start."""
    t0 = time.perf_counter()
    play_haptic(drv, events)
    remaining = total_s - (time.perf_counter() - t0)
    if remaining > 0:
        time.sleep(remaining)


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Simultaneous sound + haptic
# ══════════════════════════════════════════════════════════════════════════════

def play_simultaneous(drv, waveform: np.ndarray,
                       haptic_events: list, total_s: float):
    """
    Play audio waveform and haptic schedule at the same time using two threads.
    Both threads are started as close together as possible; the function
    returns only after both have finished (or total_s has elapsed).
    """
    barrier = threading.Barrier(2)

    def _audio_thread():
        barrier.wait()
        _play_audio(waveform, blocking=True)

    def _haptic_thread():
        barrier.wait()
        play_haptic_block(drv, haptic_events, total_s)

    t_audio  = threading.Thread(target=_audio_thread,  daemon=True)
    t_haptic = threading.Thread(target=_haptic_thread, daemon=True)
    t_audio.start()
    t_haptic.start()
    t_audio.join()
    t_haptic.join()


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — DRV2605 singleton
# ══════════════════════════════════════════════════════════════════════════════

_drv_instance = None

def _get_drv():
    global _drv_instance
    if _drv_instance is None:
        if not HAPTIC_AVAILABLE:
            raise RuntimeError("adafruit_drv2605 not installed")
        i2c = busio.I2C(board.SCL, board.SDA)
        _drv_instance = adafruit_drv2605.DRV2605(i2c)
        _drv_instance.use_LRM()   # LRA (Linear Resonance Motor) mode
        _drv_instance.library = 6 # Library 6: LRA effects
        print("Haptic: DRV2605L initialised (LRA mode, library 6)")
    return _drv_instance


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Trial block generators (public API)
# ══════════════════════════════════════════════════════════════════════════════

def generate_cue_sound(blocktime: float = 15, freq: float = 1,
                        arhythmtime: float = 5):
    """
    One sound trial — three sequential sub-blocks:

      1. Arrhythmic A4/E5 pairs for the first `arhythmtime` s,
         silence for the remainder  (total = blocktime s)
      2. Regular alternating A4/E5 at `freq` Hz  (blocktime s)
      3. Silence                                  (blocktime s)

    All three waveforms are precomputed before playback begins to minimise
    Python-level jitter during delivery.
    """
    print("  [SOUND] Precomputing waveforms...", end=' ', flush=True)
    w_arrhythm = build_arrhythmic_sound(blocktime, freq, arhythmtime)
    w_regular  = build_regular_sound(blocktime, freq)
    w_silent   = build_silent_sound(blocktime)
    print("done")

    print(f"  [SOUND] Sub-block 1: arrhythmic ({arhythmtime:.0f}s active / {blocktime:.0f}s total)")
    _play_audio(w_arrhythm)

    print(f"  [SOUND] Sub-block 2: regular {freq}Hz for {blocktime:.0f}s")
    _play_audio(w_regular)

    print(f"  [SOUND] Sub-block 3: silence {blocktime:.0f}s")
    _play_audio(w_silent)


def generate_cue_vibration(blocktime: float = 15, freq: float = 1,
                            effect1: int = DEFAULT_EFFECT1,
                            effect2: int = DEFAULT_EFFECT2,
                            arhythmtime: float = 5):
    """
    One haptic trial — three sequential sub-blocks mirroring generate_cue_sound():

      1. Arrhythmic effect1/effect2 pairs for the first `arhythmtime` s  (blocktime s total)
      2. Regular alternating effect1/effect2 at `freq` Hz                 (blocktime s)
      3. No vibration                                                      (blocktime s)

    Haptic schedules are precomputed as (time, effect_id) lists, then fired
    via a spin-wait loop for precise timing — analogous to precomputing the
    audio waveform.
    """
    if not HAPTIC_AVAILABLE:
        print("  [HAPTIC] Driver not available — waiting silently")
        time.sleep(blocktime * 3)
        return

    drv = _get_drv()

    print("  [HAPTIC] Precomputing schedules...", end=' ', flush=True)
    sched_arrhythm = build_arrhythmic_haptic(blocktime, freq, arhythmtime,
                                              effect1, effect2)
    sched_regular  = build_regular_haptic(blocktime, freq, effect1, effect2)
    print(f"done ({len(sched_arrhythm)} arrhythmic events, "
          f"{len(sched_regular)} regular events)")

    print(f"  [HAPTIC] Sub-block 1: arrhythmic ({arhythmtime:.0f}s active / {blocktime:.0f}s total)")
    play_haptic_block(drv, sched_arrhythm, blocktime)

    print(f"  [HAPTIC] Sub-block 2: regular {freq}Hz, "
          f"effect{effect1}/{effect2} for {blocktime:.0f}s")
    play_haptic_block(drv, sched_regular, blocktime)

    print(f"  [HAPTIC] Sub-block 3: no vibration {blocktime:.0f}s")
    time.sleep(blocktime)


def test_simultaneous(blocktime: float = 15, freq: float = 1,
                       effect1: int = DEFAULT_EFFECT1,
                       effect2: int = DEFAULT_EFFECT2):
    """
    Simultaneous delivery test: regular sound + regular haptic at the same
    time for one blocktime. Used to verify hardware synchrony.
    """
    print("\n=== Simultaneous delivery test ===")
    print("  Precomputing...", end=' ', flush=True)
    waveform = build_regular_sound(blocktime, freq)
    if HAPTIC_AVAILABLE:
        drv    = _get_drv()
        events = build_regular_haptic(blocktime, freq, effect1, effect2)
        print("done")
        print(f"  Playing sound + haptic simultaneously for {blocktime:.0f}s")
        play_simultaneous(drv, waveform, events, blocktime)
    else:
        print("done (haptic unavailable — sound only)")
        _play_audio(waveform)
    print("  Simultaneous test complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Section 8 — Main experiment API
# ══════════════════════════════════════════════════════════════════════════════

def _play_word(words: dict, name: str):
    if name in words:
        _play_audio(words[name])
    else:
        print(f"  [WARN] '{name}.wav' not loaded — skipping")


def _wait_for_y() -> bool:
    """Block until user presses 'y' (continue) or 'q' (quit)."""
    while True:
        try:
            key = input("  >> Press y + ENTER to continue, q to quit: ").strip().lower()
        except EOFError:
            return False
        if key == 'y':
            return True
        if key == 'q':
            print("  Experiment stopped by user.")
            return False
        print("  (enter 'y' or 'q')")


def generate_cue(
    blocknum:    int   = 20,
    blocktime:   float = 15,
    preptime:    float = 2,
    freq:        float = 1,
    randomize:   bool  = False,
    firstblock:  str   = 'sound',
    effect1:     int   = DEFAULT_EFFECT1,
    effect2:     int   = DEFAULT_EFFECT2,
    arhythmtime: float = 5,
    run_simtest: bool  = False,
):
    """
    Main experiment entry point.

    Parameters
    ----------
    blocknum     : number of trials per modality (or total trials if randomize=True)
    blocktime    : duration of each sub-block in seconds (3 sub-blocks per trial)
    preptime     : pause between 'ready.wav' and 'walk.wav', ± 0.5 s jitter
    freq         : cue frequency in Hz (beats per second within each sub-block)
    randomize    : if True,  assign modality randomly for each of blocknum trials;
                   if False, run firstblock × blocknum then other modality × blocknum
    firstblock   : 'sound' or 'vibration' (used when randomize=False)
    effect1      : DRV2605 built-in effect number for the "low" beat
    effect2      : DRV2605 built-in effect number for the "high" beat
    arhythmtime  : arrhythmic window at the start of sub-block 1 (seconds)
    run_simtest  : if True, run a simultaneous delivery test before the experiment

    Trial structure (per trial)
    ---------------------------
      Sub-block 1 — blocktime s: arrhythmic pairs, active for first arhythmtime s
      Sub-block 2 — blocktime s: regular rhythmic at freq Hz
      Sub-block 3 — blocktime s: silence / no vibration
      → play 'rest.wav', wait for 'y' before next trial
    """
    if firstblock not in ('sound', 'vibration'):
        raise ValueError(f"firstblock must be 'sound' or 'vibration', got {firstblock!r}")

    _setup_audio()

    # Pre-load instruction WAV files
    WAV_FILES = ['ready', 'walk', 'rest']
    words = {}
    for name in WAV_FILES:
        try:
            words[name] = load_wav(f'{name}.wav')
            print(f"Loaded {name}.wav  ({len(words[name])/SR:.2f}s)")
        except FileNotFoundError:
            print(f"WARNING: {name}.wav not found — will skip")
        except Exception as exc:
            print(f"WARNING: could not load {name}.wav — {exc}")

    # Optional simultaneous test
    if run_simtest:
        test_simultaneous(blocktime, freq, effect1, effect2)
        if not _wait_for_y():
            return

    # Build trial modality sequence
    if randomize:
        modalities = [random.choice(['sound', 'vibration']) for _ in range(blocknum)]
        mode_desc  = f"randomized ({blocknum} trials)"
    else:
        other      = 'vibration' if firstblock == 'sound' else 'sound'
        modalities = [firstblock] * blocknum + [other] * blocknum
        mode_desc  = f"{firstblock} × {blocknum}, then {other} × {blocknum}"

    total_trials   = len(modalities)
    trial_dur_s    = blocktime * 3
    experiment_min = total_trials * trial_dur_s / 60

    print(f"\n{'='*60}")
    print(f"Experiment start")
    print(f"  Modality order : {mode_desc}")
    print(f"  Trials         : {total_trials}")
    print(f"  Sub-block time : {blocktime}s  ×3 = {trial_dur_s}s per trial")
    print(f"  Cue frequency  : {freq} Hz")
    print(f"  Arrhythm window: {arhythmtime}s")
    print(f"  Effects        : #{effect1} (low) / #{effect2} (high)")
    print(f"  Est. duration  : ~{experiment_min:.1f} min (excl. rest periods)")
    print(f"{'='*60}")

    # ── Preparation phase ────────────────────────────────────────────────────
    print("\n[PREP] Playing 'ready'...")
    _play_word(words, 'ready')

    jitter   = random.uniform(-0.5, 0.5)
    pause    = max(0.5, preptime + jitter)
    print(f"[PREP] Waiting {pause:.2f}s (preptime={preptime}s, jitter={jitter:+.2f}s)")
    time.sleep(pause)

    print("[PREP] Playing 'walk'...")
    _play_word(words, 'walk')

    # ── Trial loop ───────────────────────────────────────────────────────────
    for idx, modality in enumerate(modalities):
        print(f"\n{'─'*50}")
        print(f"Trial {idx + 1}/{total_trials}  [{modality.upper()}]")
        print(f"{'─'*50}")
        t_trial = time.perf_counter()

        if modality == 'sound':
            generate_cue_sound(blocktime=blocktime, freq=freq,
                               arhythmtime=arhythmtime)
        else:
            generate_cue_vibration(blocktime=blocktime, freq=freq,
                                   effect1=effect1, effect2=effect2,
                                   arhythmtime=arhythmtime)

        elapsed = time.perf_counter() - t_trial
        print(f"  Trial {idx + 1} complete ({elapsed:.1f}s)")

        _play_word(words, 'rest')

        if idx < total_trials - 1:
            if not _wait_for_y():
                return

    print(f"\n{'='*60}")
    print("Experiment complete. Well done!")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 — CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Rhythmic cueing experiment (sound + haptic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--blocknum',    type=int,   default=20,
                   help='Trials per modality (or total if --randomize)')
    p.add_argument('--blocktime',   type=float, default=15,
                   help='Duration of each sub-block (s)')
    p.add_argument('--preptime',    type=float, default=2,
                   help='Pause between ready/walk cues (s, ±0.5s jitter)')
    p.add_argument('--freq',        type=float, default=1.0,
                   help='Cue frequency within each block (Hz)')
    p.add_argument('--randomize',   action='store_true',
                   help='Randomise modality assignment across trials')
    p.add_argument('--firstblock',  type=str,   default='sound',
                   choices=['sound', 'vibration'],
                   help='First modality when not randomising')
    p.add_argument('--effect1',     type=int,   default=DEFAULT_EFFECT1,
                   help='DRV2605 effect # for "low" beat')
    p.add_argument('--effect2',     type=int,   default=DEFAULT_EFFECT2,
                   help='DRV2605 effect # for "high" beat')
    p.add_argument('--arhythmtime', type=float, default=5,
                   help='Arrhythmic active window in sub-block 1 (s)')
    p.add_argument('--simtest',     action='store_true',
                   help='Run simultaneous sound+haptic test before experiment')
    p.add_argument('--simtest-only', action='store_true',
                   help='Run simultaneous test only, then exit')
    p.add_argument('--list-devices', action='store_true',
                   help='List audio devices and exit')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    if args.simtest_only:
        _setup_audio()
        test_simultaneous(args.blocktime, args.freq, args.effect1, args.effect2)
        sys.exit(0)

    generate_cue(
        blocknum    = args.blocknum,
        blocktime   = args.blocktime,
        preptime    = args.preptime,
        freq        = args.freq,
        randomize   = args.randomize,
        firstblock  = args.firstblock,
        effect1     = args.effect1,
        effect2     = args.effect2,
        arhythmtime = args.arhythmtime,
        run_simtest = args.simtest,
    )
