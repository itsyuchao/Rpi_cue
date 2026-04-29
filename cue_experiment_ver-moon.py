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

Trial structure (5 sub-blocks × blocktime s each)
-------------------------------------------------
  'ready.wav' → pause → 'ignore.wav' → 5 blocks → 'ratesync.wav'

  Cue-block order may be counter-balanced across trials if --randomize-arr-reg is specified:

    block_order='arr_first'  : silent → arrhythmic → silent → regular    → silent
    block_order='reg_first'  : silent → regular    → silent → arrhythmic → silent

  The delivered regular-cue rate is counter-balanced between
  freq*(1+freq_jitter_ratio) and freq*(1-freq_jitter_ratio). With the
  defaults (--blocknum 20, --freq 1.0, --freq-jitter-ratio 0.1) that is
  10 trials at 1.1 Hz and 10 trials at 0.9 Hz per modality, fully crossed
  with the two block orders (5 trials per cell).

  The attention cue ('attendhigh'/'attendlow' or 'attendclick'/'attendbuzz')
  always plays immediately before the regular block.

  After the final silent block, 'ratesync.wav' plays and the participant
  gives a 1-5 keypad rating.

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

import socket as _socket
import threading as _threading

_marker_sock: _socket.socket | None = None
_marker_addr: tuple | None = None

def marker_init(host: str, port: int = 5005):
    global _marker_sock, _marker_addr
    _marker_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _marker_addr = (host, port)

def send_marker(label: str):
    if _marker_sock and _marker_addr:
        try:
            _marker_sock.sendto(label.encode(), _marker_addr)
        except OSError:
            pass


# Ping responder — answers latency pings from fetch.py with a modular-addition
# echo so the logger can RTT-tag every relayed marker. Counterpart lives in
# fetch.py UDPMarkerThread._measure_rtt.
_UDP_PING_PORT = 5006
_PING_K        = 42
_PING_MOD      = 65536

_ping_sock: _socket.socket | None = None
_ping_thread: _threading.Thread | None = None


def _ping_responder_loop():
    global _ping_sock
    while True:
        try:
            data, addr = _ping_sock.recvfrom(128)
        except OSError:
            return
        try:
            parts = data.decode("ascii", errors="ignore").strip().split(":", 2)
            if len(parts) < 3 or parts[0] != "PING":
                continue
            nonce = int(parts[1])
            marker_id = parts[2]
            result = (nonce + _PING_K) % _PING_MOD
            pong = f"PONG:{result}:{marker_id}".encode("ascii")
            _ping_sock.sendto(pong, addr)
        except Exception:
            continue


def ping_responder_init(port: int = _UDP_PING_PORT):
    global _ping_sock, _ping_thread
    if _ping_thread is not None:
        return
    _ping_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _ping_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    _ping_sock.bind(("0.0.0.0", port))
    _ping_thread = _threading.Thread(target=_ping_responder_loop, daemon=True)
    _ping_thread.start()
    print(f"Ping responder: :{port}")


logger_ip = "192.168.50.5"

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
ARHYTHM_INTER_MIN = 0.1
ARHYTHM_INTER_MAX = 0.9
ARHYTHM_INTRA_MIN = 0.1
ARHYTHM_INTRA_MAX = 0.9

# Regular-cue frequency jitter — the delivered rate is
#   freq * (1 ± FREQ_JITTER_RATIO_DEFAULT)
# counter-balanced across trials (see build_trial_plan).
FREQ_JITTER_RATIO_DEFAULT = 0.1

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
    """Writes one CSV row per sub-block to a timestamped file.

    Columns with limited scope are left blank when not applicable:
      - template_index    : only for arrhythmic blocks
      - freq_hz           : only for regular blocks (delivered frequency)
      - freq_jitter_sign  : only for regular blocks (+1 / -1)
      - sync_rating       : only for the final silent block of each trial
    """

    HEADER = [
        'participant_id',
        'trial_num',
        'block_start_time_utc',
        'modality',
        'block_position',
        'block_subtype',
        'block_order',
        'template_index',
        'freq_hz',
        'freq_jitter_sign',
        'attend_high',
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

    def log_block(self, *, trial_num: int, modality: str,
                  block_position: int, block_subtype: str,
                  block_order: str, attend_high: bool,
                  template_index: int | str = '',
                  freq_hz: float | str = '',
                  freq_jitter_sign: int | str = '',
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
            modality,
            block_position,
            block_subtype,
            block_order,
            template_index,
            freq_hz,
            freq_jitter_sign,
            attend_high,
            sync_rating,
        ]
        self._append_row(row)


# ══════════════════════════════════════════════════════════════════════════════
# Module 11 — Stimulus precomputation + unified trial runner
# ══════════════════════════════════════════════════════════════════════════════

def precompute_stimuli(templates: list, blocktime: float,
                       freq: float, freq_jitter_ratio: float,
                       effect1: int, effect2: int
                       ) -> tuple[dict, dict, float, float]:
    """Build every waveform and haptic schedule that trials will need.

    Returns (audio, haptic, freq_low, freq_high). During trials, the runner
    only selects a precomputed object — no synthesis happens in the hot path.
    """
    freq_low  = freq * (1.0 - freq_jitter_ratio)
    freq_high = freq * (1.0 + freq_jitter_ratio)

    audio = {
        'silent':       _silence(blocktime),
        'regular_low':  build_regular_sound(blocktime, freq_low),
        'regular_high': build_regular_sound(blocktime, freq_high),
        'arrhythmic': [
            build_arrhythmic_sound_from_template(tpl, blocktime)
            for tpl in templates
        ],
    }
    haptic = {
        'regular_low':  build_regular_haptic(blocktime, freq_low,  effect1, effect2),
        'regular_high': build_regular_haptic(blocktime, freq_high, effect1, effect2),
        'arrhythmic': [
            template_to_haptic_events(tpl, effect1, effect2)
            for tpl in templates
        ],
    }
    return audio, haptic, freq_low, freq_high


def run_trial(*,
              modality: str,
              trial_num: int,
              blocktime: float,
              freq_hz: float,
              freq_jitter_sign: int,
              template: list,
              template_index: int,
              block_order: str,
              attend_high: bool,
              stim_audio: dict,
              stim_haptic: dict,
              words: dict,
              logger: ExperimentLogger,
              effect1: int,
              effect2: int):
    """Run one trial for either modality.

    Preamble : 'ready' → pause → 'ignore'
    Blocks   : 5 sub-blocks, each `blocktime` seconds long. Position 0, 2, 4
               are always silent. Positions 1 and 3 hold the cue blocks in
               the order given by `block_order`:
                 'arr_first' → arrhythmic at 1, regular at 3
                 'reg_first' → regular    at 1, arrhythmic at 3
    Attention: plays immediately before the regular block.
    Rating   : after the final silent block, 'ratesync' plays and a 1-5
               keypad rating is collected and logged.
    """
    is_sound     = (modality == 'sound')
    mod_marker   = 'a' if is_sound else 'v'
    lcd_banner   = 'SOUND' if is_sound else 'VIBRATION'
    lcd_short    = 'SOUND' if is_sound else 'VIBR'
    freq_key     = 'regular_high' if freq_jitter_sign > 0 else 'regular_low'

    # ── Pick stimuli ─────────────────────────────────────────────────────
    if is_sound:
        w_silent   = stim_audio['silent']
        w_arrhythm = stim_audio['arrhythmic'][template_index]
        w_regular  = stim_audio[freq_key]
    else:
        if _drv is None:
            print("    [HAPTIC] No driver — skipping vibration trial")
            time.sleep(blocktime * 5)
            return
        h_arrhythm = stim_haptic['arrhythmic'][template_index]
        h_regular  = stim_haptic[freq_key]

    # ── Block layout for this trial ──────────────────────────────────────
    if block_order == 'arr_first':
        subtype_by_position = {0: 'silent', 1: 'arrhythmic', 2: 'silent',
                               3: 'regular', 4: 'silent'}
    else:  # 'reg_first'
        subtype_by_position = {0: 'silent', 1: 'regular', 2: 'silent',
                               3: 'arrhythmic', 4: 'silent'}

    if is_sound:
        attend_word = 'attendhigh'  if attend_high else 'attendlow'
    else:
        attend_word = 'attendclick' if attend_high else 'attendbuzz'

    # ── Preamble ─────────────────────────────────────────────────────────
    lcd_show(lcd_banner, "Get ready...")
    play_word(words, 'ready')
    time.sleep(0.5)
    play_word(words, 'pause')
    time.sleep(max(0.5, 2.0 + random.uniform(-0.5, 0.5)))
    play_word(words, 'ignore')

    # ── Run the 5 blocks ─────────────────────────────────────────────────
    n_pairs = len(template)
    final_silent_start_utc = None

    for position in range(5):
        subtype = subtype_by_position[position]

        # Attention cue immediately before the regular block (either position)
        if subtype == 'regular':
            play_word(words, attend_word)
            time.sleep(0.5)
            if is_sound:
                play_audio(_synth_tone(
                    F_HIGH if attend_high else F_LOW, 0.5
                ))
            else:
                play_haptic_event(_drv, effect1 if attend_high else effect2)
            time.sleep(2)

        block_start_utc = datetime.now(timezone.utc).isoformat(
            timespec='microseconds'
        )
        marker = f"{mod_marker}_{subtype[:3]}_p{position}_t{trial_num}"

        # Per-block metadata for the logger
        if subtype == 'silent':
            tpl_log       = ''
            freq_log      = ''
            freq_sign_log = ''
            desc          = 'silent'
        elif subtype == 'arrhythmic':
            tpl_log       = template_index
            freq_log      = ''
            freq_sign_log = ''
            desc          = f"arrhythmic ({n_pairs} pairs)"
        else:  # 'regular'
            tpl_log       = ''
            freq_log      = f"{freq_hz:.6f}"
            freq_sign_log = freq_jitter_sign
            desc          = f"regular {freq_hz:.3f}Hz"

        lcd_show(f"{lcd_short} {position}/4", subtype.capitalize())
        print(f"    Block {position}: {desc}")

        # The final silent block is logged AFTER its sync rating is collected
        # (so the rating lands on the same row). Every other block is logged now.
        if position < 4:
            logger.log_block(
                trial_num=trial_num,
                modality=modality,
                block_position=position,
                block_subtype=subtype,
                block_order=block_order,
                attend_high=attend_high,
                template_index=tpl_log,
                freq_hz=freq_log,
                freq_jitter_sign=freq_sign_log,
                block_start_time_utc=block_start_utc,
            )
        else:
            final_silent_start_utc = block_start_utc
        send_marker(marker)

        # Play the stimulus
        if subtype == 'silent':
            if is_sound:
                play_audio(w_silent)
            else:
                time.sleep(blocktime)
        elif subtype == 'arrhythmic':
            if is_sound:
                play_audio(w_arrhythm)
            else:
                play_haptic_block(_drv, h_arrhythm, blocktime)
        else:  # 'regular'
            if is_sound:
                play_audio(w_regular)
            else:
                play_haptic_block(_drv, h_regular, blocktime)

    # ── Sync rating for the final silent block ───────────────────────────
    play_word(words, 'ratesync')
    sync_rating = get_sync_rating()
    send_marker(f"{mod_marker}_rate_t{trial_num}")
    logger.log_block(
        trial_num=trial_num,
        modality=modality,
        block_position=4,
        block_subtype='silent',
        block_order=block_order,
        attend_high=attend_high,
        sync_rating=sync_rating,
        block_start_time_utc=final_silent_start_utc,
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
                     participant_id: str,
                     randomize_arr_reg: bool = False,
                     ) -> tuple[list[str], list[bool],
                                list[int], list[str], str]:
    """Build all per-trial schedules from one participant-seeded RNG.

    Returns (modalities, attend_high, freq_signs, block_orders, description).

    Within each contiguous modality chunk of `blocknum` trials (or across the
    whole run when `randomize=True`):
      - freq_sign (±1) is always counter-balanced 50/50 and shuffled.
      - block_order is fixed to 'arr_first' unless `randomize_arr_reg=True`,
        in which case 'arr_first'/'reg_first' is fully crossed with freq_sign
        (2×2 design, 5 trials per cell at --blocknum 20).
      - attend_high is balanced 50/50 and shuffled.
    """
    rng = random.Random(f"participant::{participant_id}")

    def counterbalance_chunk(n: int) -> tuple[list[int], list[str], list[bool]]:
        conditions: list[tuple[int, str]] = []
        if randomize_arr_reg:
            # 2×2 crossed: freq_sign × block_order
            per_cell = n // 4
            for sign in (+1, -1):
                for order in ('arr_first', 'reg_first'):
                    conditions.extend([(sign, order)] * per_cell)
            for _ in range(n - len(conditions)):
                conditions.append((
                    rng.choice([+1, -1]),
                    rng.choice(['arr_first', 'reg_first']),
                ))
        else:
            # Only freq_sign counter-balanced; every trial is 'arr_first'
            per_cell = n // 2
            for sign in (+1, -1):
                conditions.extend([(sign, 'arr_first')] * per_cell)
            for _ in range(n - len(conditions)):
                conditions.append((rng.choice([+1, -1]), 'arr_first'))
        rng.shuffle(conditions)
        signs  = [c[0] for c in conditions]
        orders = [c[1] for c in conditions]

        n_high = n // 2 + (1 if (n % 2 and rng.random() < 0.5) else 0)
        attend = [True] * n_high + [False] * (n - n_high)
        rng.shuffle(attend)
        return signs, orders, attend

    if randomize:
        modalities = [rng.choice(['sound', 'vibration']) for _ in range(blocknum)]
        signs, orders, attend = counterbalance_chunk(blocknum)
        mode_desc = f"randomized-seeded ({blocknum} trials)"
    else:
        first = firstblock
        other = 'vibration' if first == 'sound' else 'sound'
        modalities = [first] * blocknum + [other] * blocknum
        s1, o1, a1 = counterbalance_chunk(blocknum)
        s2, o2, a2 = counterbalance_chunk(blocknum)
        signs  = s1 + s2
        orders = o1 + o2
        attend = a1 + a2
        mode_desc = f"{first} x{blocknum} then {other} x{blocknum}"

    return modalities, attend, signs, orders, mode_desc


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

    marker_init(logger_ip)   # logger Pi's IP
    ping_responder_init()    # answers fetch.py's latency pings

    # ── Participant ID ───────────────────────────────────────────────────
    lcd_show("== CUE EXPT ==", "Starting...")
    time.sleep(1)

    participant_id = get_participant_id()
    lcd_show(f"ID: {participant_id}", "Confirmed")
    print(f"\nParticipant: {participant_id}")
    time.sleep(1)

    # ── Parameters ───────────────────────────────────────────────────────
    blocknum          = args.blocknum
    blocktime         = args.blocktime
    freq              = args.freq
    freq_jitter_ratio = args.freq_jitter_ratio
    effect1           = args.effect1
    effect2           = args.effect2

    # ── Load templates from fixed CSV ────────────────────────────────────
    print(f"\nLoading {NUM_TEMPLATES} arrhythmic templates from "
          f"{TEMPLATE_CSV_PATH} ({blocktime}s each)...")
    try:
        templates = load_arrhythmic_templates(
            TEMPLATE_CSV_PATH, NUM_TEMPLATES, blocktime,
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
            print(f"  Template {i:2d}: {len(tpl):2d} pairs, "
                  f"span {tpl[-1][0]:.2f}s")
        else:
            print(f"  Template {i:2d}: empty")

    # ── Precompute every stimulus up front (low-latency trials) ──────────
    print("\nPrecomputing audio & haptic stimuli...", flush=True)
    t_pre = time.perf_counter()
    stim_audio, stim_haptic, freq_low, freq_high = precompute_stimuli(
        templates, blocktime, freq, freq_jitter_ratio, effect1, effect2,
    )
    print(f"  Done in {time.perf_counter() - t_pre:.2f}s "
          f"(regular rates: {freq_low:.3f} / {freq_high:.3f} Hz)")

    # ── Build per-trial schedules ────────────────────────────────────────
    modalities, attention_schedule, freq_signs, block_orders, mode_desc = (
        build_trial_plan(blocknum, args.randomize, args.firstblock,
                         participant_id,
                         randomize_arr_reg=args.randomize_arr_reg)
    )

    total_trials   = len(modalities)
    has_two_blocks = not args.randomize
    start_idx      = get_resume_start_index(blocknum, total_trials, has_two_blocks)
    if start_idx > 0:
        print(f"Resuming from trial {start_idx + 1}/{total_trials}")

    # ── Logger ───────────────────────────────────────────────────────────
    logger = ExperimentLogger(participant_id)

    # ── Pre-load WAV cues ────────────────────────────────────────────────
    words = {}
    for name in ['ready', 'pause', 'ignore',
                 'attendhigh', 'attendlow',
                 'attendclick', 'attendbuzz',
                 'ratesync']:
        try:
            words[name] = load_wav(f'{name}.wav')
            print(f"Loaded {name}.wav ({len(words[name])/SR:.2f}s)")
        except FileNotFoundError:
            print(f"WARNING: {name}.wav not found")
        except Exception as e:
            print(f"WARNING: {name}.wav — {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    trial_dur   = blocktime * 5
    est_minutes = total_trials * trial_dur / 60

    print(f"\n{'='*60}")
    print(f"  Participant       : {participant_id}")
    print(f"  Modalities        : {mode_desc}")
    print(f"  Trials            : {total_trials}")
    print(f"  Block time        : {blocktime}s x 5 = {trial_dur}s / trial")
    if args.randomize_arr_reg:
        print(f"  Block order       : counter-balanced (arr_first / reg_first)")
    else:
        print(f"  Block order       : fixed arr_first "
              f"(silent→arr→silent→reg→silent)")
    print(f"  Regular freq      : {freq} Hz ± {freq_jitter_ratio*100:.1f}% "
          f"→ {freq_low:.3f} / {freq_high:.3f} Hz")
    print(f"  Templates         : {NUM_TEMPLATES}")
    print(f"  Est. duration     : ~{est_minutes:.1f} min (excl. rests)")
    print(f"  Log file          : {logger.filename}")
    print(f"{'='*60}")

    lcd_show(f"{total_trials} trials", f"~{est_minutes:.0f} min total")
    time.sleep(2)

    # ── Trial loop ───────────────────────────────────────────────────────
    interrupted = False
    try:
        for idx in range(start_idx, total_trials):
            modality       = modalities[idx]
            tpl_idx        = idx % NUM_TEMPLATES
            template       = templates[tpl_idx]
            trial_num      = idx + 1
            freq_sign      = freq_signs[idx]
            block_order    = block_orders[idx]
            attend_high    = attention_schedule[idx]
            freq_delivered = freq_high if freq_sign > 0 else freq_low

            print(f"\n{'─'*50}")
            print(f"  Trial {trial_num}/{total_trials}  "
                  f"[{modality.upper()}]  T#{tpl_idx}")
            print(f"    order={block_order}  "
                  f"freq={freq_delivered:.3f}Hz ({'+' if freq_sign > 0 else '-'})  "
                  f"attend={'high' if attend_high else 'low'}")
            print(f"{'─'*50}")

            lcd_show(f"Trial {trial_num}/{total_trials}",
                     f"{modality[:4]} T#{tpl_idx}")

            t0 = time.perf_counter()
            if modality in ('sound', 'vibration'):
                run_trial(
                    modality=modality,
                    trial_num=trial_num,
                    blocktime=blocktime,
                    freq_hz=freq_delivered,
                    freq_jitter_sign=freq_sign,
                    template=template,
                    template_index=tpl_idx,
                    block_order=block_order,
                    attend_high=attend_high,
                    stim_audio=stim_audio,
                    stim_haptic=stim_haptic,
                    words=words,
                    logger=logger,
                    effect1=effect1,
                    effect2=effect2,
                )
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
    p.add_argument('--blocknum',    type=int,   default=20,
                   help="Trials per modality (or total trials if --randomize)")
    p.add_argument('--blocktime',   type=float, default=15,
                   help="Duration of each sub-block (s)")
    p.add_argument('--freq',        type=float, default=1.0,
                   help="Nominal regular-cue rate (Hz). Delivered rate is "
                        "counter-balanced to freq*(1 ± freq_jitter_ratio).")
    p.add_argument('--freq-jitter-ratio', type=float,
                   default=FREQ_JITTER_RATIO_DEFAULT,
                   help="Fractional jitter applied to --freq on each trial "
                        "(+r on half the trials, -r on the other half)")
    p.add_argument('--randomize',   action='store_true',
                   help="Randomize modality per trial instead of blocking")
    p.add_argument('--randomize-arr-reg', action='store_true',
                   help="Counter-balance within-trial block order "
                        "(arr_first / reg_first). Off by default — every "
                        "trial runs silent→arr→silent→reg→silent.")
    p.add_argument('--firstblock',  type=str,   default='sound',
                   choices=['sound', 'vibration'],
                   help="Modality of the first block when not --randomize")
    p.add_argument('--effect1',     type=int,   default=DEFAULT_EFFECT1,
                   help="DRV2605 effect id for the 'low/click' haptic tone")
    p.add_argument('--effect2',     type=int,   default=DEFAULT_EFFECT2,
                   help="DRV2605 effect id for the 'high/buzz' haptic tone")
    p.add_argument('--list-devices', action='store_true',
                   help="Print sounddevice devices and exit")
    return p.parse_args()


if __name__ == '__main__':
    main()
