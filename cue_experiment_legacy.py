#!/usr/bin/env python3
"""
cue_experiment.py
=================
Rhythmic cueing experiment controller for Raspberry Pi 4.

Hardware
--------
  Sound  : PCM5102A breakout → I2S (BCLK=GPIO18, LRCLK=GPIO19, DIN=GPIO21)
  Haptic : DRV2605L → I2C 0x5a, LRA motor
  LCD    : 16×2 I2C LCD → I2C 0x27 (lcd_i2c driver)
  Keypad : 4×4 membrane → GPIO 23,24,25,8 (rows) / 7,1,12,16 (cols)

Trial structure (4 sub-blocks × blocktime s each)
--------------------------------------------------
  'ready.wav' → pause → 'ignore.wav' →
  Block 1 — Arrhythmic  : random beats from 20 pre-generated templates
  Block 2 — Silent      : no cue
  Block 3 — Regular     : steady alternating cue at freq Hz
  Block 4 — Silent      : no cue
  → 'rest.wav', wait for continue

Install
-------
  pip install numpy sounddevice scipy \
              adafruit-blinka adafruit-circuitpython-drv2605
  # lcd_i2c and gpiozero come with Raspberry Pi OS
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Hardware imports (graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════

# Haptic
try:
    import board
    import busio
    import adafruit_drv2605
    HAPTIC_AVAILABLE = True
except ImportError:
    HAPTIC_AVAILABLE = False
    print("INFO: adafruit_drv2605 not found — haptic output disabled")

# LCD
try:
    from lcd_i2c import LCD_I2C
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False
    print("INFO: lcd_i2c not found — LCD output disabled")

# Keypad
try:
    from gpiozero import DigitalOutputDevice, Button
    KEYPAD_AVAILABLE = True
except ImportError:
    KEYPAD_AVAILABLE = False
    print("INFO: gpiozero not found — keypad input disabled")


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Constants
# ══════════════════════════════════════════════════════════════════════════════

# Audio synthesis
SR         = 48000
F_LOW      = 440.0      # A4
F_HIGH     = 659.3      # E5
AMPLITUDE  = 1
ATTACK_MS  = 5
RELEASE_MS = 20
HARMONICS  = [(1, 1.00), (2, 0.30), (3, 0.15), (4, 0.10)]
TONE_DURATION = 0.15

# Arrhythmic timing — both inter-pair AND intra-pair randomised
ARHYTHM_INTER_MIN   = 0.1
ARHYTHM_INTER_MAX   = 0.9
ARHYTHM_INTRA_MIN = 0.1
ARHYTHM_INTRA_MAX = 0.9 

# Templates
NUM_TEMPLATES = 20
TEMPLATE_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'templates.csv'
)

# Haptic effects (DRV2605 built-in library)
DEFAULT_EFFECT1 = 1    # Strong Click → "low" beat
DEFAULT_EFFECT2 = 14   # Strong Buzz   → "high" beat

# Keypad GPIO (left→right on ribbon: 23,24,25,8,7,1,12,16)
KEYPAD_ROWS_PINS = [23, 24, 25, 8]
KEYPAD_COLS_PINS = [7, 1, 12, 16]
KEYPAD_KEYS = [
    "1", "2", "3", "A",
    "4", "5", "6", "B",
    "7", "8", "9", "C",
    "*", "0", "#", "D",
]

# LCD
LCD_I2C_ADDR = 0x27
LCD_COLS     = 16
LCD_ROWS     = 2


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — LCD driver
# ══════════════════════════════════════════════════════════════════════════════

_lcd = None


def lcd_init():
    """Initialise the 16×2 I2C LCD."""
    global _lcd
    if not LCD_AVAILABLE:
        return
    try:
        _lcd = LCD_I2C(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS)
        _lcd.backlight.on()
        _lcd.clear()
        print(f"LCD: 16x2 at I2C 0x{LCD_I2C_ADDR:02x}")
    except Exception as e:
        print(f"WARNING: LCD init failed — {e}")
        _lcd = None


def lcd_show(line1: str = "", line2: str = ""):
    """Write two lines to LCD. Falls back to terminal if no LCD."""
    if _lcd is None:
        print(f"  LCD| {line1[:LCD_COLS]:<{LCD_COLS}}")
        print(f"  LCD| {line2[:LCD_COLS]:<{LCD_COLS}}")
        return
    try:
        _lcd.clear()
        _lcd.cursor.setPos(0, 0)
        _lcd.write_text(line1[:LCD_COLS].ljust(LCD_COLS))
        _lcd.cursor.setPos(1, 0)
        _lcd.write_text(line2[:LCD_COLS].ljust(LCD_COLS))
    except Exception as e:
        print(f"  LCD write error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 — Keypad driver
# ══════════════════════════════════════════════════════════════════════════════

_kp_rows = None
_kp_cols = None


def keypad_init():
    """Initialise keypad GPIO pins."""
    global _kp_rows, _kp_cols
    if not KEYPAD_AVAILABLE:
        return
    if _kp_rows is not None:
        return
    _kp_rows = [DigitalOutputDevice(pin) for pin in KEYPAD_ROWS_PINS]
    _kp_cols = [Button(pin, pull_up=False) for pin in KEYPAD_COLS_PINS]
    print("Keypad: 4x4 matrix initialised")


def keypad_scan() -> str | None:
    """Single scan. Returns key char or None."""
    if _kp_rows is None:
        return None
    for i, row in enumerate(_kp_rows):
        row.on()
        for j, col in enumerate(_kp_cols):
            if col.is_pressed:
                row.off()
                return KEYPAD_KEYS[i * 4 + j]
        row.off()
    return None


def keypad_wait(timeout: float = None) -> str | None:
    """Block until a key is pressed and released. Debounced."""
    last = None
    start = time.monotonic()
    while True:
        key = keypad_scan()
        if key and key != last:
            time.sleep(0.05)
            if keypad_scan() == key:
                while keypad_scan() == key:
                    time.sleep(0.02)
                return key
        last = key
        time.sleep(0.03)
        if timeout and (time.monotonic() - start) > timeout:
            return None


def keypad_close():
    """Release keypad GPIO."""
    global _kp_rows, _kp_cols
    if _kp_rows:
        for r in _kp_rows:
            r.close()
        for c in _kp_cols:
            c.close()
    _kp_rows = None
    _kp_cols = None


def keypad_input_string(prompt: str = "Enter ID:",
                        max_len: int = 12) -> str:
    """
    Collect a string via keypad + LCD.
    0-9, A-D = character.  * = backspace.  # = confirm.
    """
    buf = ""
    lcd_show(prompt, buf + "_")
    while True:
        key = keypad_wait()
        if key is None:
            continue
        if key == '#':
            if buf:
                return buf
        elif key == '*':
            buf = buf[:-1]
        else:
            if len(buf) < max_len:
                buf += key
        lcd_show(prompt, buf + ("_" if len(buf) < max_len else ""))


def keypad_choice(line1: str, options: list[str]) -> int:
    """
    LCD menu — A=up, B/D=down, #=confirm.
    Returns index of selected option.
    """
    idx = 0
    lcd_show(line1, f"> {options[idx]}")
    while True:
        key = keypad_wait()
        if key is None:
            continue
        if key == 'A' and idx > 0:
            idx -= 1
        elif key in ('B', 'D') and idx < len(options) - 1:
            idx += 1
        elif key == '#':
            return idx
        lcd_show(line1, f"> {options[idx]}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 5 — Audio engine
# ══════════════════════════════════════════════════════════════════════════════

def audio_init():
    """Find HiFiBerry DAC and configure sounddevice."""
    sd.default.samplerate = SR
    sd.default.channels   = 1
    sd.default.dtype      = 'float32'
    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            sd.default.device = i
            print(f"Audio: [{i}] {dev['name']}")
            return
    print("WARNING: HiFiBerry not found — using system default")


def load_wav(path: str) -> np.ndarray:
    """Load WAV → float32 mono at SR."""
    sr_in, data = wav.read(f"audio/{path}")
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_in != SR:
        n_out   = int(len(data) * SR / sr_in)
        indices = np.round(np.linspace(0, len(data) - 1, n_out)).astype(int)
        data    = data[indices]
    return data.astype(np.float32)


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


def _pad_startup(audio: np.ndarray, ms: float = 50) -> np.ndarray:
    return np.concatenate([_silence(ms / 1000), audio])


def play_audio(waveform: np.ndarray, blocking: bool = True):
    """Play a precomputed waveform with padded start for buffering."""
    sd.play(_pad_startup(waveform), samplerate=SR, blocksize=128)
    if blocking:
        sd.wait()


def play_word(words: dict, name: str):
    """Play a named WAV from the preloaded dict."""
    if name in words:
        play_audio(words[name])
    else:
        print(f"  [WARN] '{name}.wav' not loaded — skipping")


# ══════════════════════════════════════════════════════════════════════════════
# Module 6 — Sound waveform builders
# ══════════════════════════════════════════════════════════════════════════════

def build_regular_sound(blocktime_s: float, freq_hz: float) -> np.ndarray:
    """Regular alternating A4/E5 at freq_hz for blocktime_s."""
    half   = 0.5 / freq_hz
    tone_d = TONE_DURATION
    gap_d  = half - tone_d
    cycle  = np.concatenate([
        _synth_tone(F_LOW, tone_d), _silence(gap_d),
        _synth_tone(F_HIGH, tone_d), _silence(gap_d),
    ])
    n_cyc = max(1, int(np.ceil(blocktime_s / (len(cycle) / SR))))
    train = np.tile(cycle, n_cyc)
    return train[:int(round(blocktime_s * SR))]


def build_arrhythmic_sound_from_template(template: list,
                                          blocktime_s: float) -> np.ndarray:
    """
    Build arrhythmic waveform from a template.
    Template: list of (onset_s, intra_gap_s) tuples.
    """
    total_n       = int(round(blocktime_s * SR))
    buf           = np.zeros(total_n, dtype=np.float32)
    tone_dur = TONE_DURATION

    for onset, intra_gap in template:
        tone_lo  = _synth_tone(F_LOW, tone_dur)
        lo_start = int(round(onset * SR))
        lo_end   = lo_start + len(tone_lo)
        if lo_end <= total_n:
            buf[lo_start:lo_end] += tone_lo

        hi_onset = onset + tone_dur + intra_gap
        tone_hi  = _synth_tone(F_HIGH, tone_dur)
        hi_start = int(round(hi_onset * SR))
        hi_end   = hi_start + len(tone_hi)
        if hi_end <= total_n:
            buf[hi_start:hi_end] += tone_hi

    np.clip(buf, -1.0, 1.0, out=buf)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# Module 7 — Arrhythmic template loader
# ══════════════════════════════════════════════════════════════════════════════

def load_arrhythmic_templates(path: str,
                              expected_n: int,
                              blocktime_s: float) -> list[list]:
    """Load templates from CSV file with tolerant parsing."""
    del blocktime_s  # kept for call compatibility
    templates = [[] for _ in range(expected_n)]
    rows_by_template = {i: [] for i in range(expected_n)}

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tpl_idx = int(row.get('template_index', ''))
                pair_idx = int(row.get('pair_index', ''))
                onset = float(row.get('onset_s', ''))
                intra_gap = float(row.get('intra_gap_s', ''))
            except (TypeError, ValueError):
                continue

            if 0 <= tpl_idx < expected_n:
                rows_by_template[tpl_idx].append((pair_idx, onset, intra_gap))

    for tpl_idx in range(expected_n):
        rows = sorted(rows_by_template[tpl_idx], key=lambda x: x[0])
        templates[tpl_idx] = [
            (round(onset, 4), round(intra_gap, 4))
            for _, onset, intra_gap in rows
        ]

    return templates


def template_to_haptic_events(template: list,
                               effect1: int = DEFAULT_EFFECT1,
                               effect2: int = DEFAULT_EFFECT2) -> list:
    """Convert arrhythmic template → haptic event list [(time, effect_id)]."""
    tone_dur = TONE_DURATION
    events = []
    for onset, intra_gap in template:
        events.append((onset, effect1))
        events.append((onset + tone_dur + intra_gap, effect2))
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Module 8 — Haptic engine
# ══════════════════════════════════════════════════════════════════════════════

_drv = None


def haptic_init():
    """Initialise DRV2605L."""
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


def play_haptic(drv, events: list):
    """Fire haptic schedule with spin-wait precision."""
    if not events or drv is None:
        return
    t0 = time.perf_counter()
    for rel_t, effect_id in events:
        now  = time.perf_counter() - t0
        wait = rel_t - now
        if wait > 0.002:
            time.sleep(wait - 0.001)
        while time.perf_counter() - t0 < rel_t:
            pass
        drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        drv.play()
    drv.stop()


def play_haptic_block(drv, events: list, total_s: float):
    """Play haptic schedule then wait until total_s elapsed."""
    t0 = time.perf_counter()
    play_haptic(drv, events)
    remaining = total_s - (time.perf_counter() - t0)
    if remaining > 0:
        time.sleep(remaining)


def play_haptic_event(drv, effect_id: int): 
    drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
    drv.play()
    time.sleep(1)
    drv.stop()


def build_regular_haptic(blocktime_s: float, freq_hz: float,
                          effect1: int = DEFAULT_EFFECT1,
                          effect2: int = DEFAULT_EFFECT2) -> list:
    """Alternating effect1/effect2 at freq_hz for blocktime_s."""
    half   = 0.5 / freq_hz
    events = []
    t      = 0.0
    while t < blocktime_s - 1e-9:
        events.append((t, effect1))
        t += half
        if t < blocktime_s - 1e-9:
            events.append((t, effect2))
            t += half
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Module 10 — CSV logger
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentLogger:
    """Writes block-level data to a timestamped CSV."""

    HEADER = [
        'participant_id',
        'trial_num',
        'block_start_time_utc',
        'block_type',
        'template_number',
        'block_number_in_trial',
        'sync_rating',
    ]

    def __init__(self, participant_id: str, output_dir: str = 'log'):
        self.pid = participant_id
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(
            output_dir, f"cue_log_{participant_id}_{ts}.csv"
        )
        with open(self.filename, 'w', newline='') as f:
            csv.writer(f).writerow(self.HEADER)
            f.flush()
            os.fsync(f.fileno())
        print(f"Logger: {self.filename}")

    def _append_row(self, row: list):
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def log_block(self, trial_num: int, block_type: str,
                  template_number: int | str, block_number_in_trial: int,
                  sync_rating: int | str = '',
                  block_start_time_utc: str | None = None):
        if block_start_time_utc is None:
            block_start_time_utc = datetime.now(timezone.utc).isoformat(
                timespec='microseconds'
            )
        row = [
            self.pid,
            trial_num,
            block_start_time_utc,
            block_type,
            template_number,
            block_number_in_trial,
            sync_rating,
        ]
        self._append_row(row)


# ══════════════════════════════════════════════════════════════════════════════
# Module 11 — Trial runners
# ══════════════════════════════════════════════════════════════════════════════

def run_trial_sound(blocktime: float, freq: float, template: list,
                     words: dict, logger: ExperimentLogger,
                     trial_num: int, template_index: int, attend_high: bool=True):
    """
    SOUND trial — 4 blocks:
      'ready' → pause → 'walk' →
      Block 1: arrhythmic → Block 2: silent →
      Block 3: regular    → Block 4: silent →
      'ratesync'
    """
    print("    Precomputing waveforms...", flush=True)
    w_arrhythm = build_arrhythmic_sound_from_template(template, blocktime)
    w_regular  = build_regular_sound(blocktime, freq)
    w_silent   = _silence(blocktime)

    lcd_show("SOUND", "Get ready...")
    play_word(words, 'ready')
    time.sleep(0.5)
    play_word(words, 'pause')
    jitter = random.uniform(-0.5, 0.5)
    time.sleep(max(0.5, 2.0 + jitter))
    play_word(words, 'ignore')

    logger.log_block(trial_num, 'sound', template_index, 1)
    lcd_show("SOUND 1/4", "Arrhythmic")
    print(f"    Block 1: arrhythmic ({len(template)} pairs)")
    play_audio(w_arrhythm)

    logger.log_block(trial_num, 'sound', '', 2)
    lcd_show("SOUND 2/4", "Silent")
    print(f"    Block 2: silent")
    play_audio(w_silent)

    play_word(words, 'attendhigh' if attend_high else 'attendlow')
    time.sleep(0.5)
    play_audio(_synth_tone(F_HIGH if attend_high else F_LOW, 0.5))
    time.sleep(2)

    logger.log_block(trial_num, 'sound', '', 3)
    lcd_show("SOUND 3/4", "Regular")
    print(f"    Block 3: regular {freq}Hz")
    play_audio(w_regular)

    block4_start_time_utc = datetime.now(timezone.utc).isoformat(
        timespec='microseconds'
    )
    lcd_show("SOUND 4/4", "Silent")
    print(f"    Block 4: silent")
    play_audio(w_silent)

    play_word(words, 'ratesync')
    sync_rating = get_sync_rating()
    logger.log_block(
        trial_num,
        'sound',
        '',
        4,
        sync_rating=sync_rating,
        block_start_time_utc=block4_start_time_utc,
    )


def run_trial_vibration(blocktime: float, freq: float, template: list,
                         words: dict, logger: ExperimentLogger,
                         trial_num: int, template_index: int,
                         effect1: int = DEFAULT_EFFECT1,
                         effect2: int = DEFAULT_EFFECT2, 
                         attend_high: bool=True):
    """
    VIBRATION trial — 4 blocks:
      'ready' → pause → 'walk' →
      Block 1: arrhythmic haptic → Block 2: silent →
      Block 3: regular haptic    → Block 4: silent →
      'ratesync'
    """
    drv = _drv
    if drv is None:
        print("    [HAPTIC] No driver — waiting silently")
        time.sleep(blocktime * 4)
        return

    print("    Precomputing haptic schedules...", flush=True)
    h_arrhythm = template_to_haptic_events(template, effect1, effect2)
    h_regular  = build_regular_haptic(blocktime, freq, effect1, effect2)

    lcd_show("VIBRATION", "Get ready...")
    play_word(words, 'ready')
    time.sleep(0.5)
    play_word(words, 'pause')
    jitter = random.uniform(-0.5, 0.5)
    time.sleep(max(0.5, 2.0 + jitter))
    play_word(words, 'ignore')

    logger.log_block(trial_num, 'vib', template_index, 1)
    lcd_show("VIBR 1/4", "Arrhythmic")
    print(f"    Block 1: arrhythmic ({len(h_arrhythm)} events)")
    play_haptic_block(drv, h_arrhythm, blocktime)

    logger.log_block(trial_num, 'vib', '', 2)
    lcd_show("VIBR 2/4", "Silent")
    print(f"    Block 2: no vibration")
    time.sleep(blocktime)

    play_word(words, 'attendclick' if attend_high else 'attendbuzz')
    time.sleep(1)
    play_haptic_event(drv, effect1 if attend_high else effect2)
    time.sleep(2)

    logger.log_block(trial_num, 'vib', '', 3)
    lcd_show("VIBR 3/4", "Regular")
    print(f"    Block 3: regular {freq}Hz")
    play_haptic_block(drv, h_regular, blocktime)

    block4_start_time_utc = datetime.now(timezone.utc).isoformat(
        timespec='microseconds'
    )
    lcd_show("VIBR 4/4", "Silent")
    print(f"    Block 4: no vibration")
    time.sleep(blocktime)

    play_word(words, 'ratesync')
    sync_rating = get_sync_rating()
    logger.log_block(
        trial_num,
        'vib',
        '',
        4,
        sync_rating=sync_rating,
        block_start_time_utc=block4_start_time_utc,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Module 12 — User input helpers (keypad or terminal fallback)
# ══════════════════════════════════════════════════════════════════════════════

def get_participant_id() -> str:
    if KEYPAD_AVAILABLE and _kp_rows is not None:
        lcd_show("Participant ID:", "Key in + # OK")
        return keypad_input_string("Participant ID:")
    else:
        return input("Enter participant ID: ").strip()


def wait_for_continue() -> bool:
    """# = continue, * = quit."""
    lcd_show("Press # to", "continue  *=quit")
    if KEYPAD_AVAILABLE and _kp_rows is not None:
        while True:
            key = keypad_wait(timeout=300)
            if key == '#':
                return True
            if key == '*':
                return False
    else:
        while True:
            try:
                k = input("  >> y=continue, q=quit: ").strip().lower()
            except EOFError:
                return False
            if k == 'y':
                return True
            if k == 'q':
                return False


def get_sync_rating() -> int:
    """Collect a 1-5 sync rating after ratesync."""
    if KEYPAD_AVAILABLE and _kp_rows is not None:
        lcd_show("Sync rating", "1-5")
        while True:
            key = keypad_wait()
            if key in ('1', '2', '3', '4', '5'):
                return int(key)
    else:
        while True:
            try:
                val = input("  >> Sync rating (1-5): ").strip()
            except EOFError:
                return 0
            if val in {'1', '2', '3', '4', '5'}:
                return int(val)


def get_resume_start_index(blocknum: int, total_trials: int,
                           has_two_blocks: bool) -> int:
    """Prompt resume position and return 0-based trial index to start from."""
    if KEYPAD_AVAILABLE and _kp_rows is not None:
        lcd_show("Resume session?", "1=yes 0=no")
        while True:
            key = keypad_wait()
            if key == '0':
                return 0
            if key == '1':
                break

        block = 'a'
        if has_two_blocks:
            lcd_show("Which block", "A or B")
            while True:
                key = keypad_wait()
                if key in ('A', 'B'):
                    block = key.lower()
                    break

        trial_str = keypad_input_string("Trial number:", max_len=4)
        try:
            trial_num = int(trial_str)
        except ValueError:
            return 0
    else:
        ans = input("Resume previous session? (1/0): ").strip()
        if ans != '1':
            return 0

        block = 'a'
        if has_two_blocks:
            block = input("Which block (a/b): ").strip().lower()

        try:
            trial_num = int(input("Which trial number: ").strip())
        except ValueError:
            return 0

    if trial_num < 1 or trial_num > blocknum:
        return 0
    if block not in ('a', 'b'):
        return 0

    start_idx = (trial_num - 1) if block == 'a' else (blocknum + trial_num - 1)
    if start_idx < 0 or start_idx >= total_trials:
        return 0
    return start_idx


def build_trial_plan(blocknum: int,
                     randomize: bool,
                     firstblock: str,
                     participant_id: str) -> tuple[list[str], list[bool], str]:
    """Build modality and attention schedules from one participant-seeded RNG."""
    rng = random.Random(f"participant::{participant_id}")

    if randomize:
        modalities = [rng.choice(['sound', 'vibration']) for _ in range(blocknum)]
        mode_desc = f"randomized-seeded ({blocknum} trials)"
    else:
        first = firstblock
        other = 'vibration' if first == 'sound' else 'sound'
        modalities = [first] * blocknum + [other] * blocknum
        mode_desc = f"{first} x{blocknum} then {other} x{blocknum}"

    total_trials = len(modalities)
    n_high = total_trials // 2
    n_low = total_trials // 2
    if total_trials % 2 == 1:
        if rng.choice([True, False]):
            n_high += 1
        else:
            n_low += 1

    attention_schedule = [True] * n_high + [False] * n_low
    rng.shuffle(attention_schedule)
    return modalities, attention_schedule, mode_desc


# ══════════════════════════════════════════════════════════════════════════════
# Module 13 — Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Init hardware ────────────────────────────────────────────────────
    audio_init()
    lcd_init()
    keypad_init()
    haptic_init()

    if args.list_devices:
        print(sd.query_devices())
        keypad_close()
        return

    # ── Participant ID ───────────────────────────────────────────────────
    lcd_show("== CUE EXPT ==", "Starting...")
    time.sleep(1)

    participant_id = get_participant_id()
    lcd_show(f"ID: {participant_id}", "Confirmed")
    print(f"\nParticipant: {participant_id}")
    time.sleep(1)

    # ── Parameters ───────────────────────────────────────────────────────
    blocknum  = args.blocknum
    blocktime = args.blocktime
    freq      = args.freq
    effect1   = args.effect1
    effect2   = args.effect2

    # ── Load templates from fixed CSV ─────────────────────────────────────
    print(f"\nLoading {NUM_TEMPLATES} arrhythmic templates from "
          f"{TEMPLATE_CSV_PATH} ({blocktime}s each)...")
    try:
        templates = load_arrhythmic_templates(
            TEMPLATE_CSV_PATH,
            NUM_TEMPLATES,
            blocktime,
        )
    except FileNotFoundError:
        print("ERROR: templates.csv not found.")
        print("Run: python3 generate_templates.py --blocktime "
              f"{blocktime} --num-templates {NUM_TEMPLATES}")
        keypad_close()
        return
    except Exception as e:
        print(f"ERROR: failed to load templates.csv — {e}")
        print("Regenerate with: python3 generate_templates.py")
        keypad_close()
        return
    for i, tpl in enumerate(templates):
        if tpl:
            print(f"  Template {i:2d}: {len(tpl)} pairs, "
                  f"span {tpl[-1][0]:.2f}s")
        else:
            print(f"  Template {i:2d}: empty")

    # No shuffling order for now since templates are already randomised
    template_order = list(range(NUM_TEMPLATES))
    # random.shuffle(template_order)

    # ── Build modality sequence ──────────────────────────────────────────
    modalities, attention_schedule, mode_desc = build_trial_plan(
        blocknum,
        args.randomize,
        args.firstblock,
        participant_id,
    )

    total_trials = len(modalities)
    has_two_blocks = not args.randomize
    start_idx = get_resume_start_index(blocknum, total_trials, has_two_blocks)
    if start_idx > 0:
        print(f"Resuming from trial {start_idx + 1}/{total_trials}")

    # ── Logger ───────────────────────────────────────────────────────────
    logger = ExperimentLogger(participant_id)

    # ── Pre-load WAV cues ────────────────────────────────────────────────
    words = {}
    for name in ['ready', 'pause', 'ignore', 'attendhigh', 'attendlow', 'attendclick', 'attendbuzz', 'ratesync']:
        try:
            words[name] = load_wav(f'{name}.wav')
            print(f"Loaded {name}.wav ({len(words[name])/SR:.2f}s)")
        except FileNotFoundError:
            print(f"WARNING: {name}.wav not found")
        except Exception as e:
            print(f"WARNING: {name}.wav — {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    trial_dur   = blocktime * 4
    est_minutes = total_trials * trial_dur / 60

    print(f"\n{'='*60}")
    print(f"  Participant   : {participant_id}")
    print(f"  Modalities    : {mode_desc}")
    print(f"  Trials        : {total_trials}")
    print(f"  Block time    : {blocktime}s x 4 = {trial_dur}s / trial")
    print(f"  Block order   : arrhythmic > silent > regular > silent")
    print(f"  Frequency     : {freq} Hz")
    print(f"  Templates     : {NUM_TEMPLATES} (shuffled, no repeat)")
    print(f"  Est. duration : ~{est_minutes:.1f} min (excl. rests)")
    print(f"  Log file      : {logger.filename}")
    print(f"{'='*60}")

    lcd_show(f"{total_trials} trials", f"~{est_minutes:.0f} min total")
    time.sleep(2)

    # ── Trial loop ───────────────────────────────────────────────────────
    interrupted = False
    try:
        for idx in range(start_idx, total_trials):
            modality  = modalities[idx]
            tpl_idx   = template_order[idx % NUM_TEMPLATES]
            template  = templates[tpl_idx]
            trial_num = idx + 1

            print(f"\n{'─'*50}")
            print(f"  Trial {trial_num}/{total_trials}  "
                  f"[{modality.upper()}]  template #{tpl_idx}")
            print(f"{'─'*50}")

            lcd_show(f"Trial {trial_num}/{total_trials}",
                     f"{modality} T#{tpl_idx}")

            t0 = time.perf_counter()

            if modality == 'sound':
                run_trial_sound(blocktime, freq, template, words,
                                logger, trial_num, tpl_idx,
                                attend_high=attention_schedule[idx])
            elif modality == 'vibration':
                run_trial_vibration(blocktime, freq, template, words,
                                    logger, trial_num, tpl_idx,
                                    effect1, effect2,
                                    attend_high=attention_schedule[idx])
            else:
                print(f"  [ERROR] Unknown modality '{modality}' — skipping trial")

            elapsed = time.perf_counter() - t0

            print(f"  Trial {trial_num} done ({elapsed:.1f}s)")
            lcd_show("Rest", "# next  * quit")

            if idx < total_trials - 1:
                if not wait_for_continue():
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user (Ctrl+C). Logged rows were flushed to disk.")

    # ── Done ─────────────────────────────────────────────────────────────
    lcd_show("Experiment", "Interrupted" if interrupted else "Complete!")
    print(f"\n{'='*60}")
    if interrupted:
        print(f"Experiment interrupted — {logger.filename}")
    else:
        print(f"Experiment complete — {logger.filename}")
    print(f"{'='*60}")
    time.sleep(3)
    lcd_show("", "")
    keypad_close()


# ══════════════════════════════════════════════════════════════════════════════
# Module 14 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Rhythmic cueing experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--blocknum',    type=int,   default=20)
    p.add_argument('--blocktime',   type=float, default=15)
    p.add_argument('--freq',        type=float, default=1.0)
    p.add_argument('--randomize',   action='store_true')
    p.add_argument('--firstblock',  type=str,   default='sound',
                   choices=['sound', 'vibration'])
    p.add_argument('--effect1',     type=int,   default=DEFAULT_EFFECT1)
    p.add_argument('--effect2',     type=int,   default=DEFAULT_EFFECT2)
    p.add_argument('--list-devices', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    main()
