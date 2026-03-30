#!/usr/bin/env python3
"""
twotone_precomputed.py
======================
Precomputed two-tone alternating beep train for Raspberry Pi 4
Output: I2S via PCM5102A DAC (Adafruit breakout) → 3.5mm → portable speaker

The entire waveform is built in memory as one contiguous NumPy array,
then handed to sounddevice in a single stream. The I2S hardware clock
controls all timing — no Python sleep loops, no jitter.

Setup:
  1. PCM5102A wired to Pi 4 GPIO (BCLK=GPIO18, LRCLK=GPIO19, DIN=GPIO21)
  2. /boot/config.txt: dtoverlay=hifiberry-dac  (comment out dtparam=audio=on)
  3. 3.5mm cable from PCM5102A jack to portable speaker aux in

Install:
  pip install numpy sounddevice --break-system-packages

Usage:
  python3 twotone_precomputed.py
  python3 twotone_precomputed.py --train     (train mode: ON/OFF blocks)
"""

import argparse
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# ─────────────────────────────────────────────────────────────
# PCM5102A via hifiberry-dac overlay: 48kHz is the native rate.
# 44.1kHz also works but 48k avoids any resampling in PipeWire.
# The Adafruit PCM5102A breakout auto-detects from BCLK/LRCLK.
# ─────────────────────────────────────────────────────────────
SR = 48000

# Two pitches — A4 and E5, musically a fifth apart
F_LOW  = 440.0
F_HIGH = 660.0

# Harmonic recipe: fundamental + overtones for a bright but pleasant timbre
# (harmonic_number, relative_amplitude)
HARMONICS = [
    (1, 1.00),   # fundamental
    (2, 0.30),   # octave
    (3, 0.15),   # fifth above octave
    (4, 0.10),   # two octaves
]

# Envelope: short attack avoids click, longer release sounds natural
ATTACK_MS  = 5
RELEASE_MS = 20
AMPLITUDE  = 0.80
def find_hifiberry():
    """Auto-detect the PCM5102A DAC by name."""
    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            return i
    return None

def load_word(path, target_sr=48000):
    """Load a WAV, convert to float32 mono, resample to target SR."""
    sr, data = wav.read(path)
    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample if needed (Piper outputs 22050Hz by default)
    if sr != target_sr:
        indices = np.round(np.linspace(0, len(data) - 1,
                    int(len(data) * target_sr / sr))).astype(int)
        data = data[indices]
    return data

def synth_tone(freq, duration_s):
    """Synthesize a single tone burst with harmonics and envelope."""
    n = int(round(duration_s * SR))
    if n < 1:
        return np.zeros(1, dtype=np.float32)

    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n, dtype=np.float64)

    for k, amp in HARMONICS:
        y += amp * np.sin(2.0 * np.pi * k * freq * t)

    # Attack / release envelope
    atk = max(1, int(ATTACK_MS * 0.001 * SR))
    rel = max(1, int(RELEASE_MS * 0.001 * SR))
    env = np.ones(n, dtype=np.float64)
    env[:atk] = np.linspace(0.0, 1.0, atk)
    if rel < n:
        env[-rel:] = np.linspace(1.0, 0.0, rel)

    y *= env

    # Normalize and scale
    peak = np.max(np.abs(y))
    if peak > 0:
        y = AMPLITUDE * y / peak

    return y.astype(np.float32)


def build_continuous(seq_hz, tone_ms, total_sec):
    """
    Build a continuous alternating low/high beep train.

    seq_hz:    how many beep pairs per second (e.g. 1.0 = one low+high per sec)
    tone_ms:   duration of each individual beep in ms
    total_sec: total sequence length
    """
    half_period_s = 0.5 / seq_hz
    tone_s = min(tone_ms / 1000.0, half_period_s)
    gap_s = max(0.0, half_period_s - tone_s)

    tone_lo = synth_tone(F_LOW, tone_s)
    tone_hi = synth_tone(F_HIGH, tone_s)
    silence = np.zeros(int(round(gap_s * SR)), dtype=np.float32)

    # One full cycle: low + gap + high + gap
    cycle = np.concatenate([tone_lo, silence, tone_hi, silence])
    cycle_dur = len(cycle) / SR

    n_cycles = max(1, int(np.ceil(total_sec / cycle_dur)))
    train = np.tile(cycle, n_cycles)

    # Trim to exact requested duration
    total_samples = int(round(total_sec * SR))
    train = train[:total_samples]

    return train, cycle_dur


def build_trains(seq_hz, tone_ms, on_sec, n_trains):
    """
    Build train mode: alternating beeps for on_sec, then silence for on_sec.
    Repeat n_trains times.

    Returns the full waveform and the cycle duration for info.
    """
    half_period_s = 0.5 / seq_hz
    tone_s = min(tone_ms / 1000.0, half_period_s)
    gap_s = max(0.0, half_period_s - tone_s)

    tone_lo = synth_tone(F_LOW, tone_s)
    tone_hi = synth_tone(F_HIGH, tone_s)
    silence_gap = np.zeros(int(round(gap_s * SR)), dtype=np.float32)

    # One beep cycle
    cycle = np.concatenate([tone_lo, silence_gap, tone_hi, silence_gap])
    cycle_dur = len(cycle) / SR

    # ON block: fill on_sec with beep cycles
    on_samples = int(round(on_sec * SR))
    n_cycles = max(1, int(np.ceil(on_sec / cycle_dur)))
    on_block = np.tile(cycle, n_cycles)[:on_samples]

    # OFF block: silence for on_sec
    off_block = np.zeros(on_samples, dtype=np.float32)

    # One train = ON + OFF
    one_train = np.concatenate([on_block, off_block])

    # Stack all trains
    full = np.tile(one_train, n_trains)

    return full, cycle_dur

def pad_silence(audio, ms=100, sr=SR):
    """Prepend silence so stream startup doesn't clip the first sound."""
    pad = np.zeros(int(ms * 0.001 * sr), dtype=np.float32)
    return np.concatenate([pad, audio])

def main():
    parser = argparse.ArgumentParser(
        description="Precomputed two-tone beep train for PCM5102A on RPi4"
    )
    parser.add_argument('--train', action='store_true',
                        help='Train mode: ON/OFF blocks')
    parser.add_argument('--freq', type=float, default=1.0,
                        help='Sequence frequency in Hz')
    parser.add_argument('--tone-ms', type=float, default=150,
                        help='Tone duration in ms')
    parser.add_argument('--duration', type=float, default=30,
                        help='Total duration in seconds (continuous mode)')
    parser.add_argument('--on-sec', type=float, default=15,
                        help='ON duration per train in seconds (train mode)')
    parser.add_argument('--n-trains', type=int, default=3,
                        help='Number of trains (train mode)')
    parser.add_argument('--device', type=str, default=None,
                        help='Audio device name or index')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Interactive prompts for anything not passed as args
    train_mode = args.train
    if not any([args.freq, args.tone_ms, args.duration, args.on_sec]):
        train_mode = input("Train mode? [y/N]: ").strip().lower().startswith("y")

    if train_mode:
        on_sec = args.on_sec or float(
            input("ON (train) duration in seconds: ").strip())
        n_trains = args.n_trains or int(
            input("Number of trains: ").strip())
    else:
        total_sec = args.duration or float(
            input("Total sequence duration (seconds): ").strip())

    seq_hz = args.freq or float(
        input("Sequence frequency (Hz) (e.g. 1.0): ").strip())
    tone_ms = args.tone_ms or float(
        input("Tone length per beep (ms) (e.g. 150): ").strip())

    # Configure sounddevice
    sd.default.samplerate = SR
    sd.default.channels = 1
    sd.default.dtype = 'float32'
    if args.device:
        sd.default.device = args.device
    else:
        hifi_idx = find_hifiberry()
        if hifi_idx is not None:
            sd.default.device = hifi_idx
            print(f"Auto-detected DAC: [{hifi_idx}] {sd.query_devices(hifi_idx)['name']}")
        else:
            print("WARNING: HiFiBerry DAC not found, using system default")

    # Preload at startup
    WORDS = {
        'ready': load_word('ready.wav'),
        'set':   load_word('set.wav'),
        'walk':  load_word('walk.wav'),
        'stop':  load_word('stop.wav'),
        'rest':  load_word('rest.wav'),
	'pause': load_word('pause.wav')
    }

    # Play with zero startup latency — same as your tones
    for k, v in WORDS.items(): 
        sd.play(pad_silence(WORDS[k], sr=SR), samplerate=SR, blocksize=128)
        sd.wait()

    # Build the waveform
    print(f"\nSample rate: {SR} Hz")
    print(f"Low tone:    {F_LOW} Hz")
    print(f"High tone:   {F_HIGH} Hz")
    print(f"Seq freq:    {seq_hz} Hz")
    print(f"Tone length: {tone_ms} ms")

    if train_mode:
        waveform, cycle_dur = build_trains(seq_hz, tone_ms, on_sec, n_trains)
        total_dur = len(waveform) / SR
        print(f"ON/OFF:      {on_sec}s / {on_sec}s × {n_trains} trains")
    else:
        waveform, cycle_dur = build_continuous(seq_hz, tone_ms, total_sec)
        total_dur = len(waveform) / SR

    print(f"Cycle:       {cycle_dur*1000:.1f} ms")
    print(f"Total:       {total_dur:.2f}s ({len(waveform)} samples)")
    print(f"Peak:        {np.max(np.abs(waveform)):.3f}")

    input("\nPress ENTER to play...")

    # Play the entire precomputed buffer in one shot.
    # blocksize=128 matches our low-latency ALSA config.
    # The I2S clock handles all timing from here.
    print("Playing...")
    sd.play(pad_silence(waveform, sr=SR), samplerate=SR, blocksize=128)
    sd.wait()
    print("Done.")


if __name__ == '__main__':
    main()
