#!/usr/bin/env python3
"""
cue_experiment.py
=================
Rhythmic cueing experiment controller for Raspberry Pi 3.

Hardware
--------
  Audio  : PCM5102A breakout → I2S (BCLK=GPIO18, LRCLK=GPIO19, DIN=GPIO21)
  Haptic : DRV2605L → I2C 0x5a, LRA motor

Operator console
----------------
  All operator I/O (subject ID, focus leg, resume prompt, ready-cue
  confirmation, between-trial continue, sync rating) goes through the
  controlling terminal (PiTFT). An RF-dongle USB keyboard plugs into Pi.

Pre-experiment
--------------
  After subject ID + focus leg ('left'/'right') are entered, one
  'ready{cw,ccw}.wav' cue plays and the operator confirms with 'y' before
  trial 1. Rotation direction is counter-balanced against PID parity:
      odd  PID + left → readyccw    odd  PID + right → readycw
      even PID + left → readycw     even PID + right → readyccw
  The focused leg is appended to the CSV log filename
  (cue_log_pid-{pid}_focusleg-{leg}_{ts}.csv).

Trial structure (1 baseline silent + 4 cue sub-blocks + 1 rate sub-block)
-------------------------------------------------------------------------
  'ignore.wav' → pause → 'go.wav' → 5 stim blocks → rate block

    silent(base) → arrhythmic → silent → regular → silent → rate

  Position 0 (the baseline silent block) is --baseblocktime seconds long;
  positions 1–4 (arrhythmic / silent / regular / silent) are each
  --cueblocktime seconds long. The rate block runs as long as it takes
  the operator to type the rating.

  The delivered regular-cue period is counter-balanced per modality between
  dur_s-dur_s_jitter and dur_s+dur_s_jitter. With the
  defaults (--blocknum 8, --dur-s 1.2, --dur-s-jitter 0.15) that is
  8 trials at 1.05 s and 8 trials at 1.35 s across modality, with a
  balanced attend-high / attend-low split.

  The attention cue ('attend{leg}high'/'attend{leg}low' or
  'attend{leg}click'/'attend{leg}buzz') always plays immediately before
  the regular block, where {leg} is the per-session focused leg.

  In the rate sub-block 'stop.wav' plays (logged as a 'gostop' row), then
  a 3±1 s pause, then 'ratesync.wav' — all BEFORE the block's ping volley
  and marker. WAVs loaded from disk are operator/participant instructions,
  not cues, so the block-5 marker/timestamps bracket only the rating
  collection. The operator enters a 0-9 sync rating at the terminal; the
  rating lands in the rate row's sync_rating column. In headless mode the
  rate ping and marker still emit, but the ratesync prompt and rating
  collection are skipped.

Per-block network protocol
--------------------------
  For each sub-block the order is **always**:
    1. Three ping-pong rounds against rpi-fetch (warms the network path,
       logged as a 'ping' row in this Pi's cue_log).
    2. Capture block-start recv_time_iso + recv_perf_s and emit the marker
       UDP packet on the now-warm path.
    3. Begin stimulation.

  Instruction WAVs (ignore/go/attend*/stop/ratesync) always play before
  step 1, so the marker and block timestamps bracket only the actual
  stimulation (or rating) window. All waveforms are pre-padded with 50 ms
  of startup silence at build/load time; audio onsets therefore lag the
  block timestamp by that fixed 50 ms plus device latency
  (sd.default.latency = 'low').

  Putting the volley before the marker emit gives the marker packet the
  lowest-latency, highest-reliability slot in the sequence. Column names
  (recv_time_iso, recv_perf_s) match the fetch CSV so cross-Pi alignment
  works on a single pair of column names.

Install
-------
  pip install numpy sounddevice scipy \
              adafruit-circuitpython-drv2605
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

# Marker UDP + three-round latency ping — rpi-stim initiates, rpi-fetch responds.
# Each block runs ping_volley() (three ping-pongs, logged as a 'ping' row) and
# then send_marker_packet() to emit the marker. The volley warms the network
# path so the marker UDP arrives with the lowest possible latency.

_UDP_PING_PORT   = 5006
_PING_K          = 42
_PING_MOD        = 65536
_PING_TIMEOUT_S  = 0.5
_PING_TOLERANCE_MS = 5.0   # if 3rd RTT exceeds this, keep pinging
_PING_EXTRA_MAX    = 10    # max extra pings before giving up

_marker_sock: _socket.socket | None = None
_marker_addr: tuple | None = None
_ping_sock: _socket.socket | None = None
_ping_addr: tuple | None = None
_ping_nonce: int = 0
_logger = None  # ExperimentLogger, set by marker_init


# trial_dict = {0: "high", 1: "low", 2: "click", 3: "buzz"}
# trial_order = [0, 1, 2, 3, 1, 3, 0, 2, 2, 0, 3, 1, 2, 3, 0, 1, 3, 2, 1, 0]


def perf_counter_raw() -> float:
    """Monotonic seconds from the SoC oscillator, NOT NTP-slewed.

    time.perf_counter() on Linux is clock_gettime(CLOCK_MONOTONIC), which is
    rate-disciplined by NTP — its tick rate tracks the upstream consensus
    rather than the local hardware. CLOCK_MONOTONIC_RAW reads the same
    underlying timekeeper but bypasses NTP slewing, so cross-device elapsed
    time reflects the actual hardware oscillator skew between Pis.
    """
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def marker_init(host: str, port: int = 5005,
                ping_port: int = _UDP_PING_PORT,
                logger=None):
    """Open marker + ping sockets to `host`; ping rows are appended to `logger`."""
    global _marker_sock, _marker_addr, _ping_sock, _ping_addr, _logger
    _marker_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _marker_addr = (host, port)
    _ping_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _ping_sock.settimeout(_PING_TIMEOUT_S)
    _ping_addr = (host, ping_port)
    _logger = logger


def _drain_ping_sock() -> None:
    """Discard any leftover pong packets from earlier rounds."""
    if _ping_sock is None:
        return
    _ping_sock.setblocking(False)
    try:
        while True:
            _ping_sock.recvfrom(128)
    except (BlockingIOError, OSError):
        pass
    finally:
        _ping_sock.settimeout(_PING_TIMEOUT_S)


def _one_ping(label: str) -> str:
    """One ping-pong round. Returns RTT in ms as a string, or an error tag."""
    global _ping_nonce
    if _ping_sock is None or _ping_addr is None:
        return "noinit"
    _ping_nonce = (_ping_nonce + 1) % _PING_MOD
    nonce    = _ping_nonce
    expected = (nonce + _PING_K) % _PING_MOD
    payload  = f"PING:{nonce}:{label}".encode("ascii")

    _drain_ping_sock()
    t0 = perf_counter_raw()
    try:
        _ping_sock.sendto(payload, _ping_addr)
        data, _ = _ping_sock.recvfrom(128)
        t1 = perf_counter_raw()
    except _socket.timeout:
        return "timeout"
    except OSError as e:
        return f"err:{e.errno}"

    parts = data.decode("ascii", errors="ignore").strip().split(":", 2)
    if len(parts) < 3 or parts[0] != "PONG":
        return "bad"
    try:
        got = int(parts[1])
    except ValueError:
        return "bad"
    if got != expected or parts[2] != label:
        return "mismatch"
    return f"{(t1 - t0) * 1000.0:.3f}"


def ping_volley(label: str, trial_num: int | str = ''):
    """Run three ping-pong rounds against rpi-fetch and log the RTTs.

    This is the first step of every block — it warms the network path
    so the subsequent send_marker_packet() emits onto a hot route. The
    'ping' row is timestamped at the start of the volley; the matching
    'block' row (logged separately) carries the post-volley timestamps.
    """
    if _ping_sock is None or _ping_addr is None:
        return
    ping_iso  = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    ping_perf = perf_counter_raw()
    rtts = [_one_ping(label) for _ in range(3)]

    # If the 3rd RTT is over tolerance (or a non-numeric error tag), keep
    # pinging until one comes back under tolerance, or _PING_EXTRA_MAX extra
    # pings have fired. Marker emit happens unconditionally in the caller.
    rtts_additional: list[str] = []
    def _under_tol(r: str) -> bool:
        try:
            return float(r) <= _PING_TOLERANCE_MS
        except (TypeError, ValueError):
            return False
    if not _under_tol(rtts[-1]):
        for _ in range(_PING_EXTRA_MAX):
            r = _one_ping(label)
            rtts_additional.append(r)
            if _under_tol(r):
                break

    if _logger is not None:
        _logger.log_ping(
            marker_label=label, trial_num=trial_num,
            recv_time_iso=ping_iso, recv_perf_s=ping_perf, rtts=rtts,
            rtts_additional=rtts_additional,
        )


def send_marker_packet(label: str):
    """Emit a single marker UDP packet. Caller must have run ping_volley()
    immediately beforehand so the network path is warm."""
    if _marker_sock is None or _marker_addr is None:
        return
    try:
        _marker_sock.sendto(label.encode(), _marker_addr)
    except OSError:
        pass


logger_ip = "192.168.50.15"
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

# Regular-cue period jitter — the delivered period is
#   dur_s ± DUR_S_JITTER_DEFAULT  (seconds)
# counter-balanced across trials (see build_trial_plan).
DUR_S_JITTER_DEFAULT = 0.08

# Templates
NUM_TEMPLATES = 20
TEMPLATE_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'templates.csv'
)

# Haptic effects (DRV2605 built-in library)
DEFAULT_EFFECT1 = 1    # Strong Click → "low" beat
DEFAULT_EFFECT2 = 14   # Strong Buzz   → "high" beat


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — Audio engine
# ══════════════════════════════════════════════════════════════════════════════

def audio_init():
    """Find HiFiBerry DAC and configure sounddevice."""
    sd.default.samplerate = SR
    sd.default.channels   = 1
    sd.default.dtype      = 'float32'
    #sd.default.latency    = 'low'
    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            sd.default.device = i
            print(f"Audio: [{i}] {dev['name']}")
            return
    print("WARNING: HiFiBerry not found — using system default")


def _pad_startup(audio: np.ndarray, ms: float = 50) -> np.ndarray:
    """Prepend startup silence. Applied once at build/load time so play_audio
    does no copying in the hot path; audio onsets lag block timestamps by
    this fixed pad plus device latency."""
    return np.concatenate([_silence(ms / 1000), audio])


def load_wav(path: str) -> np.ndarray:
    """Load WAV → float32 mono at SR, pre-padded with startup silence."""
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
    return _pad_startup(data.astype(np.float32))


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


def play_audio(waveform: np.ndarray, blocking: bool = True):
    """Play a waveform already padded at build/load time (_pad_startup)."""
    sd.play(waveform, samplerate=SR, blocksize=128)
    if blocking:
        sd.wait()


def play_word(words: dict, name: str):
    """Play a named WAV from the preloaded dict."""
    if name in words:
        play_audio(words[name])
    else:
        print(f"  [WARN] '{name}.wav' not loaded — skipping")


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 — Audio waveform builders
# ══════════════════════════════════════════════════════════════════════════════

def build_regular_audio(blocktime_s: float, dur_s: float) -> np.ndarray:
    """Regular alternating A4/E5 with cycle period dur_s for blocktime_s."""
    half   = dur_s / 2
    tone_d = TONE_DURATION
    gap_d  = half - tone_d
    cycle  = np.concatenate([
        _synth_tone(F_LOW, tone_d), _silence(gap_d),
        _synth_tone(F_HIGH, tone_d), _silence(gap_d),
    ])
    n_cyc = max(1, int(np.ceil(blocktime_s / (len(cycle) / SR))))
    train = np.tile(cycle, n_cyc)
    return train[:int(round(blocktime_s * SR))]


def build_arrhythmic_audio_from_template(template: list,
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
# Module 5 — Arrhythmic template loader
# ══════════════════════════════════════════════════════════════════════════════

def load_arrhythmic_templates(path: str,
                              expected_n: int) -> list[list]:
    """Load templates from CSV file with tolerant parsing."""
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
                               blocktime_s: float,
                               effect1: int = DEFAULT_EFFECT1,
                               effect2: int = DEFAULT_EFFECT2) -> list:
    """Convert arrhythmic template → haptic event list [(time, effect_id)].

    Drops any event whose TONE_DURATION span would extend past blocktime_s,
    mirroring build_arrhythmic_audio_from_template's sample-grid truncation
    so audio and haptic emit the identical set of events for a given
    (template, blocktime). Filtering happens here at precompute — not inside
    play_haptic — because live time-checks during playback would add latency
    in the hot path.
    """
    tone_dur = TONE_DURATION
    total_n  = int(round(blocktime_s * SR))
    tone_n   = max(1, int(round(tone_dur * SR)))
    events = []
    for onset, intra_gap in template:
        lo_start = int(round(onset * SR))
        if lo_start + tone_n <= total_n:
            events.append((onset, effect1))

        hi_onset = onset + tone_dur + intra_gap
        hi_start = int(round(hi_onset * SR))
        if hi_start + tone_n <= total_n:
            events.append((hi_onset, effect2))
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Module 6 — Haptic engine
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
    t0 = perf_counter_raw()
    for rel_t, effect_id in events:
        now  = perf_counter_raw() - t0
        wait = rel_t - now
        if wait > 0.002:
            time.sleep(wait - 0.001)
        while perf_counter_raw() - t0 < rel_t:
            pass
        drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        drv.play()
    drv.stop()


def play_haptic_block(drv, events: list, total_s: float):
    """Play haptic schedule then wait until total_s elapsed."""
    t0 = perf_counter_raw()
    play_haptic(drv, events)
    remaining = total_s - (perf_counter_raw() - t0)
    if remaining > 0:
        time.sleep(remaining)


def play_haptic_event(drv, effect_id: int):
    drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
    drv.play()
    time.sleep(1)
    drv.stop()


def build_regular_haptic(blocktime_s: float, dur_s: float,
                          effect1: int = DEFAULT_EFFECT1,
                          effect2: int = DEFAULT_EFFECT2) -> list:
    """Alternating effect1/effect2 with cycle period dur_s for blocktime_s."""
    half   = dur_s / 2
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
# Module 7 — CSV logger
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentLogger:
    """Writes one CSV row per sub-block (+ one per ping volley) to a
    timestamped file.

    Columns with limited scope are left blank when not applicable:
      - template_index    : only for arrhythmic blocks
      - dur_s             : only for regular blocks (delivered cycle period, s)
      - dur_jitter_sign   : only for regular blocks (+1 / -1)
      - sync_rating       : only for the rate block of each trial

    marker_label encodes modality / block_subtype / block_position / trial_num
    ('{a|v}_{sil|arr|reg|rate}_p{pos}_t{trial}'), so those fields are not
    stored as their own columns. A block row carries the same marker_label
    as the ping row that immediately precedes it.
    """

    # Unified schema: one file per Pi, rows distinguished by `event_type`.
    # Time columns are the same names as in the rpi-fetch CSV so cross-Pi
    # alignment uses one pair of names:
    #   recv_time_iso : host wall clock (timezone-aware ISO8601 microseconds)
    #   recv_perf_s   : this Pi's clock_gettime(CLOCK_MONOTONIC_RAW) in
    #                   seconds (NOT perf_counter / CLOCK_MONOTONIC, which
    #                   is NTP-slewed; see perf_counter_raw above)
    #
    #   block rows: recv_time_iso/recv_perf_s = block start (captured AFTER
    #               ping volley); marker_label matches the preceding ping
    #               row; rtt*_ms empty.
    #   ping  rows: recv_time_iso/recv_perf_s = first-ping time;
    #               block-metadata columns empty except trial_num; rtt*_ms
    #               hold the three RTTs (ms) or an error tag ('timeout',
    #               'bad', 'mismatch', 'err:<n>').
    #
    # modality, block_position, and block_subtype are not stored as
    # separate columns — they are encoded in marker_label
    # ('{a|v}_{sil|arr|reg|rate}_p{pos}_t{trial}'). Block order is fixed
    # (arrhythmic always first), so it is not stored either.
    HEADER = [
        'event_type',
        'participant_id',
        'trial_num',
        'recv_time_iso',
        'recv_perf_s',
        'template_index',
        'dur_s',
        'dur_jitter_sign',
        'attend_high',
        'sync_rating',
        'marker_label',
        'rtt1_ms',
        'rtt2_ms',
        'rtt3_ms',
        'rtt_additional',
        'gostop_pause_s',
    ]

    def __init__(self, participant_id: str, output_dir: str = 'log',
                 focus_leg: str | None = None):
        self.pid = participant_id
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"focusleg-{focus_leg}_" if focus_leg else ''
        self.filename = os.path.join(
            output_dir, f"cue_log_pid-{participant_id}_{suffix}{ts}.csv"
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

    def log_block(self, *, trial_num: int, marker_label: str,
                  attend_high: bool,
                  template_index: int | str = '',
                  dur_s: float | str = '',
                  dur_jitter_sign: int | str = '',
                  sync_rating: int | str = '',
                  recv_time_iso: str | None = None,
                  recv_perf_s: float | None = None):
        if recv_time_iso is None:
            recv_time_iso = datetime.now(timezone.utc).isoformat(
                timespec='microseconds'
            )
        if recv_perf_s is None:
            recv_perf_s = perf_counter_raw()
        row = [
            'block',
            self.pid,
            trial_num,
            recv_time_iso,
            f"{recv_perf_s:.6f}",
            template_index,
            dur_s,
            dur_jitter_sign,
            attend_high,
            sync_rating,
            marker_label,
            '', '', '', '',   # rtt1_ms, rtt2_ms, rtt3_ms, rtt_additional
            '',               # gostop_pause_s
        ]
        self._append_row(row)

    def log_ping(self, *, marker_label: str, trial_num: int | str,
                 recv_time_iso: str, recv_perf_s: float, rtts: list[str],
                 rtts_additional: list[str] | None = None):
        """Append one 'ping' row (three-round latency probe for `marker_label`).
        Block-metadata columns are left blank; `rtts` is a 3-element list of
        RTT strings already formatted in ms (or an error tag). Any RTTs from
        the tolerance-driven extension loop are stored as a single
        slash-separated string in `rtt_additional` (empty if none)."""
        r1 = rtts[0] if len(rtts) > 0 else ''
        r2 = rtts[1] if len(rtts) > 1 else ''
        r3 = rtts[2] if len(rtts) > 2 else ''
        r_add = '/'.join(rtts_additional) if rtts_additional else ''
        row = [
            'ping',
            self.pid,
            trial_num,
            recv_time_iso,
            f"{recv_perf_s:.6f}",
            '', '', '', '', '',  # template_index, dur_s, dur_jitter_sign, attend_high, sync_rating
            marker_label,
            r1, r2, r3, r_add,
            '',                  # gostop_pause_s
        ]
        self._append_row(row)

    def log_gostop(self, *, trial_num: int, marker_label: str,
                   recv_time_iso: str, recv_perf_s: float,
                   gostop_pause_s: float):
        """Append one 'gostop' row marking the go.wav / stop.wav play moment.

        Logged for block 0 (go.wav, with the pre-go pause) and block 5
        (stop.wav, with the post-stop pause before ratesync). recv_time_iso /
        recv_perf_s are captured immediately before the play call; the row
        itself is appended after playback so the CSV fsync never sits
        between timestamp capture and wav onset.
        """
        row = [
            'gostop',
            self.pid,
            trial_num,
            recv_time_iso,
            f"{recv_perf_s:.6f}",
            '', '', '', '', '',  # template_index, dur_s, dur_jitter_sign, attend_high, sync_rating
            marker_label,
            '', '', '', '',      # rtt1_ms, rtt2_ms, rtt3_ms, rtt_additional
            f"{gostop_pause_s:.6f}",
        ]
        self._append_row(row)


# ══════════════════════════════════════════════════════════════════════════════
# Module 8 — Stimulus precomputation + unified trial runner
# ══════════════════════════════════════════════════════════════════════════════

def precompute_stimuli(templates: list,
                       cueblocktime: float,
                       dur_s: float, dur_s_jitter: float,
                       effect1: int, effect2: int
                       ) -> tuple[dict, dict, float, float]:
    """Build every waveform and haptic schedule that trials will need.

    Cue blocks (arrhythmic, regular) are precomputed at `cueblocktime`.
    Silent blocks are just `time.sleep(...)` in the runner — no waveform
    needed for either modality.

    Returns (audio, haptic, dur_low, dur_high). During trials, the runner
    only selects a precomputed object — no synthesis, padding, or copying
    happens in the hot path. Every audio buffer (including the attention
    tones) is pre-padded here with the 50 ms startup silence.
    """
    dur_low  = dur_s - dur_s_jitter
    dur_high = dur_s + dur_s_jitter

    audio = {
        'regular_low':  _pad_startup(build_regular_audio(cueblocktime, dur_low)),
        'regular_high': _pad_startup(build_regular_audio(cueblocktime, dur_high)),
        'arrhythmic': [
            _pad_startup(build_arrhythmic_audio_from_template(tpl, cueblocktime))
            for tpl in templates
        ],
        'attend_tone_high': _pad_startup(_synth_tone(F_HIGH, 0.5)),
        'attend_tone_low':  _pad_startup(_synth_tone(F_LOW,  0.5)),
    }
    haptic = {
        'regular_low':  build_regular_haptic(cueblocktime, dur_low,  effect1, effect2),
        'regular_high': build_regular_haptic(cueblocktime, dur_high, effect1, effect2),
        'arrhythmic': [
            template_to_haptic_events(tpl, cueblocktime, effect1, effect2)
            for tpl in templates
        ],
    }
    return audio, haptic, dur_low, dur_high


def run_trial(*,
              modality: str,
              trial_num: int,
              baseblocktime: float,
              cueblocktime: float,
              dur_s: float,
              dur_jitter_sign: int,
              template: list,
              template_index: int,
              attend_high: bool,
              leg: str,
              stim_audio: dict,
              stim_haptic: dict,
              words: dict,
              logger: ExperimentLogger,
              effect1: int,
              effect2: int,
              headless: bool = False):
    """Run one trial for either modality.

    Preamble : 'ignore' → pause → 'go'
    Blocks   : 6 sub-blocks, fixed layout:
                 p0=silent (baseblocktime), p1=arrhythmic (cueblocktime),
                 p2=silent (cueblocktime),  p3=regular   (cueblocktime),
                 p4=silent (cueblocktime),  p5=rate.
               Position 5 is the rate block: 'stop' (gostop row) → 3±1 s
               pause → 'ratesync' all play before the volley/marker, then
               the operator enters a 0-9 sync rating at the terminal
               (skipped in headless mode); the rating lands in the rate
               row's sync_rating column.
    Per-block: instruction WAVs (if any) → ping_volley() → capture
               recv_time_iso + recv_perf_s → send_marker_packet() → play
               stimulus → log_block. The pings warm the network path so
               the marker UDP arrives with the lowest possible latency.
    Attention: plays immediately before the regular block (before its
               volley/marker).
    """
    is_audio   = (modality == 'audio')
    mod_marker = 'a' if is_audio else 'v'
    dur_key    = 'regular_high' if dur_jitter_sign > 0 else 'regular_low'

    # ── Pick stimuli ─────────────────────────────────────────────────────
    if is_audio:
        w_arrhythm = stim_audio['arrhythmic'][template_index]
        w_regular  = stim_audio[dur_key]
    else:
        if _drv is None:
            print("    [HAPTIC] No driver — skipping vibration trial")
            time.sleep(baseblocktime + cueblocktime * 4)
            return
        h_arrhythm = stim_haptic['arrhythmic'][template_index]
        h_regular  = stim_haptic[dur_key]

    # ── Block layout for this trial ──────────────────────────────────────
    # Fixed order (arrhythmic always first). Subtype tokens match the
    # marker_label encoding: '{a|v}_{sil|arr|reg|rate}_p{position}_t{trial}'.
    subtype_by_position = {0: 'sil', 1: 'arr', 2: 'sil',
                           3: 'reg', 4: 'sil', 5: 'rate'}

    if is_audio:
        attend_word = f'attend{leg}high'  if attend_high else f'attend{leg}low'
    else:
        attend_word = f'attend{leg}click' if attend_high else f'attend{leg}buzz'

    # ── Preamble ─────────────────────────────────────────────────────────
    # Pre-go pause is generated once so the same value can be both slept and
    # logged in the 'gostop' row alongside the go.wav play timestamp. The
    # row is appended AFTER the play so the CSV fsync never sits between
    # timestamp capture and go.wav onset.
    play_word(words, 'ignore')
    pre_go_pause = 3.0 + random.uniform(-1, 1)
    time.sleep(pre_go_pause)
    go_recv_time_iso = datetime.now(timezone.utc).isoformat(
        timespec='microseconds'
    )
    go_recv_perf_s = perf_counter_raw()
    play_word(words, 'go')
    logger.log_gostop(
        trial_num=trial_num,
        marker_label=f"{mod_marker}_go_p0_t{trial_num}",
        recv_time_iso=go_recv_time_iso,
        recv_perf_s=go_recv_perf_s,
        gostop_pause_s=pre_go_pause,
    )

    # ── Run the 6 sub-blocks ─────────────────────────────────────────────
    n_pairs = len(template)

    for position in range(6):
        subtype = subtype_by_position[position]

        # Attention instruction immediately before the regular block
        if subtype == 'reg':
            play_word(words, attend_word)
            time.sleep(0.5)
            if is_audio:
                play_audio(stim_audio['attend_tone_high' if attend_high
                                      else 'attend_tone_low'])
            else:
                play_haptic_event(_drv, effect1 if attend_high else effect2)
            time.sleep(2)

        # Rate-block instructions play BEFORE the volley/marker: stop.wav
        # (logged as a 'gostop' row, appended after the play so the fsync
        # never sits between timestamp capture and wav onset), a 3±1 s
        # pause, then the ratesync prompt. Loaded WAVs are instructions,
        # not cues — the block-5 marker/timestamps bracket only the rating
        # collection.
        if subtype == 'rate':
            stop_recv_time_iso = datetime.now(timezone.utc).isoformat(
                timespec='microseconds'
            )
            stop_recv_perf_s = perf_counter_raw()
            post_stop_pause = 3.0 + random.uniform(-1, 1)
            play_word(words, 'stop')
            time.sleep(post_stop_pause)
            logger.log_gostop(
                trial_num=trial_num,
                marker_label=f"{mod_marker}_stop_p5_t{trial_num}",
                recv_time_iso=stop_recv_time_iso,
                recv_perf_s=stop_recv_perf_s,
                gostop_pause_s=post_stop_pause,
            )
            if not headless:
                play_word(words, 'ratesync')

        marker = f"{mod_marker}_{subtype}_p{position}_t{trial_num}"

        # Per-block metadata for the logger
        if subtype == 'sil':
            tpl_log = dur_log = dur_sign_log = ''
            desc = 'silent'
        elif subtype == 'arr':
            tpl_log = template_index
            dur_log = dur_sign_log = ''
            desc = f"arrhythmic ({n_pairs} pairs)"
        elif subtype == 'reg':
            tpl_log = ''
            dur_log = f"{dur_s:.6f}"
            dur_sign_log = dur_jitter_sign
            desc = f"regular {dur_s:.3f}s"
        else:  # 'rate'
            tpl_log = dur_log = dur_sign_log = ''
            desc = 'rate'

        print(f"    Block {position}: {desc}")

        # 1. Ping volley FIRST — warms the network path before the marker.
        ping_volley(marker, trial_num=trial_num)

        # 2. Capture block-start timestamps and emit the marker packet on
        #    the now-warm path. recv_time_iso/recv_perf_s reflect the actual
        #    block trigger moment, not when the pre-block volley started.
        block_recv_time_iso = datetime.now(timezone.utc).isoformat(
            timespec='microseconds'
        )
        block_recv_perf_s = perf_counter_raw()
        send_marker_packet(marker)

        # 3. Play the stimulus. For 'rate' the "stimulus" is the rating
        #    collection (skipped in headless mode); its instruction WAVs
        #    already played before the volley.
        sync_rating = ''
        if subtype == 'sil':
            time.sleep(baseblocktime if position == 0 else cueblocktime)
        elif subtype == 'arr':
            if is_audio:
                play_audio(w_arrhythm)
            else:
                play_haptic_block(_drv, h_arrhythm, cueblocktime)
        elif subtype == 'reg':
            if is_audio:
                play_audio(w_regular)
            else:
                play_haptic_block(_drv, h_regular, cueblocktime)
        else:  # 'rate'
            # Instructions (stop / pause / ratesync) already played before
            # the volley — the block window is just the rating collection.
            if not headless:
                sync_rating = get_sync_rating()

        # 4. Log the block row. sync_rating is '' except for the rate block.
        logger.log_block(
            trial_num=trial_num,
            marker_label=marker,
            attend_high=attend_high,
            template_index=tpl_log,
            dur_s=dur_log,
            dur_jitter_sign=dur_sign_log,
            sync_rating=sync_rating,
            recv_time_iso=block_recv_time_iso,
            recv_perf_s=block_recv_perf_s,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 9 — Operator terminal input
# ══════════════════════════════════════════════════════════════════════════════
#
# All operator I/O is plain stdin via the controlling terminal (RF-dongle
# USB keyboard). No on-Pi keypad or LCD/OLED.

def get_participant_id() -> str:
    return input("Enter participant ID: ").strip()


def get_focus_leg() -> str:
    """Prompt for the per-session focused leg ('left' or 'right')."""
    while True:
        try:
            v = input("Focus leg (left/right): ").strip().lower()
        except EOFError:
            return 'left'
        if v in ('left', 'l'):
            return 'left'
        if v in ('right', 'r'):
            return 'right'


def _ready_direction(pid: str, leg: str) -> str:
    """Return 'cw' or 'ccw' for the pre-trial ready cue.

    Counterbalances rotation direction across subjects:
      odd  PID + left  → ccw    odd  PID + right → cw
      even PID + left  → cw     even PID + right → ccw
    Non-numeric `pid` (e.g. 'headless') is treated as even.
    """
    digits = ''.join(c for c in pid if c.isdigit())
    pid_odd = bool(int(digits) % 2) if digits else False
    return 'ccw' if (pid_odd == (leg == 'left')) else 'cw'


def wait_for_continue() -> bool:
    """y/Enter = continue, q = quit."""
    while True:
        try:
            k = input("  >> y=continue, q=quit: ").strip().lower()
        except EOFError:
            return False
        if k in ('', 'y'):
            return True
        if k == 'q':
            return False


def get_sync_rating() -> int:
    """Collect a 0-9 sync rating after ratesync."""
    while True:
        try:
            val = input("  >> Sync rating (0-9): ").strip()
        except EOFError:
            return 0
        if val in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
            return int(val)


def get_resume_start_index(total_trials: int) -> int:
    """Prompt resume position and return 0-based trial index to start from.

    With the always-randomized schedule the run is a single interleaved
    sequence (no a/b blocks), and the (pid, leg) seed in build_trial_plan
    makes 'trial N' the same trial every restart.
    """
    ans = input("Resume previous session? (1/0): ").strip()
    if ans != '1':
        return 0
    try:
        trial_num = int(input("Which trial number: ").strip())
    except ValueError:
        return 0
    if trial_num < 1 or trial_num > total_trials:
        return 0
    return trial_num - 1


def build_trial_plan(blocknum: int,
                     participant_id: str,
                     leg: str,
                     ) -> tuple[list[str], list[bool], list[int]]:
    """Build a balanced, randomized per-trial schedule.

    Returns (modalities, attend_high, dur_signs).

    The schedule is deterministic in (participant_id, leg): restarting with
    the same PID + leg reproduces the exact same trial sequence, so
    resume-from-trial-N is well-defined even though modalities/dur/attend
    are interleaved.

    Total = 2 * blocknum trials, balanced across the 8 cells of
    (modality × dur_sign × attend_high). Block order is fixed
    (arrhythmic always first) so it is not part of the schedule.
    Any remainder when total is not a multiple of 8 is padded with
    random cells from the same (pid, leg) stream.
    """
    rng = random.Random(f"participant::{participant_id}::leg::{leg}")

    total = 2 * blocknum
    cells = [(m, s, a) for m in ('audio', 'vibration')
                       for s in (+1, -1)
                       for a in (True, False)]
    per_cell = total // len(cells)
    conditions: list[tuple[str, int, bool]] = []
    for cell in cells:
        conditions.extend([cell] * per_cell)
    for _ in range(total - len(conditions)):
        conditions.append(rng.choice(cells))
    rng.shuffle(conditions)

    modalities = [c[0] for c in conditions]
    signs      = [c[1] for c in conditions]
    attend     = [c[2] for c in conditions]

    return modalities, attend, signs


# ══════════════════════════════════════════════════════════════════════════════
# Module 10 — Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Init hardware ────────────────────────────────────────────────────
    audio_init()
    haptic_init()

    if args.list_devices:
        print(sd.query_devices())
        return

    # ── Participant ID + focus leg ───────────────────────────────────────
    if args.headless:
        participant_id = f"headless"
        leg            = 'left'
    else:
        participant_id = get_participant_id()
        leg            = get_focus_leg()
    ready_dir = _ready_direction(participant_id, leg)
    print(f"\nParticipant: {participant_id}  "
          f"(focus leg: {leg}, ready: {ready_dir})")

    # ── Parameters ───────────────────────────────────────────────────────
    blocknum          = args.blocknum
    baseblocktime     = args.baseblocktime
    cueblocktime      = args.cueblocktime
    dur_s             = args.dur_s
    dur_s_jitter      = args.dur_s_jitter
    effect1           = DEFAULT_EFFECT1
    effect2           = DEFAULT_EFFECT2

    # ── Load templates from fixed CSV ────────────────────────────────────
    print(f"\nLoading {NUM_TEMPLATES} arrhythmic templates from "
          f"{TEMPLATE_CSV_PATH}...")
    try:
        templates = load_arrhythmic_templates(
            TEMPLATE_CSV_PATH, NUM_TEMPLATES,
        )
    except FileNotFoundError:
        print("ERROR: templates.csv not found.")
        print("Run: python3 generate_templates.py --blocktime "
              f"{cueblocktime} --num-templates {NUM_TEMPLATES}")
        return
    except Exception as e:
        print(f"ERROR: failed to load templates.csv — {e}")
        print("Regenerate with: python3 generate_templates.py")
        return

    if any(len(tpl) == 0 for tpl in templates):
        print("ERROR: some templates are empty — check templates.csv")
        return
    else:
        print("  Loaded templates.")

    # ── Precompute every stimulus up front (low-latency trials) ──────────
    print("\nPrecomputing audio & haptic stimuli...", flush=True)
    t_pre = perf_counter_raw()
    stim_audio, stim_haptic, dur_low, dur_high = precompute_stimuli(
        templates, cueblocktime, dur_s, dur_s_jitter, effect1, effect2,
    )
    print(f"  Done in {perf_counter_raw() - t_pre:.2f}s "
          f"(regular periods: {dur_low:.3f} / {dur_high:.3f} s)")

    # ── Build per-trial schedules ────────────────────────────────────────
    modalities, attention_schedule, dur_signs = (
        build_trial_plan(blocknum, participant_id, leg)
    )

    total_trials = len(modalities)
    if args.headless:
        start_idx = 0
    else:
        start_idx = get_resume_start_index(total_trials)
        if start_idx > 0:
            print(f"Resuming from trial {start_idx + 1}/{total_trials}")

    # ── Logger ───────────────────────────────────────────────────────────
    logger = ExperimentLogger(participant_id, focus_leg=leg)
    marker_init(logger_ip, logger=logger)

    # ── Pre-load WAV cues ────────────────────────────────────────────────
    words = {}
    loaded_names = []
    for name in ['ignore', 'go', 'stop',
                 f'attend{leg}high',  f'attend{leg}low',
                 f'attend{leg}click', f'attend{leg}buzz',
                 'ratesync', 'readyccw', 'readycw']:
        try:
            words[name] = load_wav(f'{name}.wav')
            loaded_names.append(name)
        except FileNotFoundError:
            print(f"WARNING: {name}.wav not found")
        except Exception as e:
            print(f"WARNING: {name}.wav — {e}")

    if loaded_names:
        print("  Loaded .wav files: " + ", ".join(loaded_names))

    # ── Summary ──────────────────────────────────────────────────────────
    # Per-trial overhead beyond the stim blocks: preamble (ignore + nominal
    # 3 s pause + go), attention sequence (word + 0.5 s + ~1 s cue + 2 s),
    # rate-block instructions (stop + nominal 3 s pause + ratesync).
    # Operator time (rating entry, between-trial continue) is not counted.
    def _wav_s(name: str) -> float:
        return len(words[name]) / SR if name in words else 0.0

    attend_s = max(_wav_s(f'attend{leg}{sfx}')
                   for sfx in ('high', 'low', 'click', 'buzz'))
    trial_overhead = (
        _wav_s('ignore') + 3.0 + _wav_s('go')
        + attend_s + 0.5 + 1.0 + 2.0
        + _wav_s('stop') + 3.0 + _wav_s('ratesync')
    )
    trial_dur   = baseblocktime + cueblocktime * 4 + trial_overhead
    est_minutes = total_trials * trial_dur / 60

    print(f"\n{'='*20}")
    print(f"  Participant   : {participant_id}")
    print(f"  Trials    : {total_trials}")
    print(f"  Regular period: {dur_s} s ± {dur_s_jitter} s ")
    print(f"  Est. duration : ~{est_minutes:.1f} min (+ operator time)")
    print(f"{'='*20}")

    # ── Pre-experiment ready cue ─────────────────────────────────────────
    ready_word = f'ready{ready_dir}'
    print(f"  Ready cue         : {ready_word}")
    if not args.headless:
        play_word(words, ready_word)
        if not wait_for_continue():
            return

    time.sleep(2)

    # ── Trial loop ───────────────────────────────────────────────────────
    interrupted = False
    try:
        for idx in range(start_idx, total_trials):
            modality       = modalities[idx]
            tpl_idx        = idx % NUM_TEMPLATES
            template       = templates[tpl_idx]
            trial_num      = idx + 1
            dur_sign       = dur_signs[idx]
            attend_high    = attention_schedule[idx]
            dur_delivered  = dur_high if dur_sign > 0 else dur_low

            print(f"\n{'─'*20}")
            print(f"T#{trial_num}/{total_trials}  "
                  f"[{modality.upper()}] "
                  f"dur={dur_delivered:.3f}s ({'+' if dur_sign > 0 else '-'})  "
                  f"attn={'high' if attend_high else 'low'}")
            print(f"{'─'*20}")

            t0 = perf_counter_raw()
            if modality in ('audio', 'vibration'):
                run_trial(
                    modality=modality,
                    trial_num=trial_num,
                    baseblocktime=baseblocktime,
                    cueblocktime=cueblocktime,
                    dur_s=dur_delivered,
                    dur_jitter_sign=dur_sign,
                    template=template,
                    template_index=tpl_idx,
                    attend_high=attend_high,
                    leg=leg,
                    stim_audio=stim_audio,
                    stim_haptic=stim_haptic,
                    words=words,
                    logger=logger,
                    effect1=effect1,
                    effect2=effect2,
                    headless=args.headless,
                )
            else:
                print(f"  [ERROR] Unknown modality '{modality}' — skipping trial")
            elapsed = perf_counter_raw() - t0

            print(f"  Trial {trial_num} done ({elapsed:.1f}s)")

            if idx < total_trials - 1 and not args.headless:
                if not wait_for_continue():
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user (Ctrl+C). Logged rows were flushed to disk.")

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\n{'='*20}")
    if interrupted:
        print(f"Experiment interrupted — {logger.filename}")
    else:
        print(f"Experiment complete — {logger.filename}")
    print(f"{'='*20}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 11 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Rhythmic cueing experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--blocknum',    type=int,   default=8,
                   help="Trials per modality (total = 2 × blocknum, balanced-"
                        "randomized across modality × dur_sign × attend_high)")
    p.add_argument('--baseblocktime', type=float, default=10,
                   help="Duration of the baseline silent block at position 0 (s)")
    p.add_argument('--cueblocktime',  type=float, default=20,
                   help="Duration of each cue / post-cue silent sub-block "
                        "at positions 1-4 (s). Arrhythmic templates must be "
                        "generated for this duration.")
    p.add_argument('--dur-s',       type=float, default=1.2,
                   help="Nominal regular-cue cycle period (s). Delivered "
                        "period is counter-balanced to dur_s ± dur_s_jitter.")
    p.add_argument('--dur-s-jitter', type=float,
                   default=DUR_S_JITTER_DEFAULT,
                   help="Absolute jitter (s) applied to --dur-s on each trial "
                        "(+j on half the trials, -j on the other half)")
    p.add_argument('--list-devices', action='store_true',
                   help="Print sounddevice devices and exit")
    p.add_argument('--headless',    action='store_true',
                   help="Run with no operator input: auto-generate "
                        "participant ID, start at trial 1 (no resume prompt), "
                        "skip the ratesync + sync rating block and the "
                        "between-trial continue prompt. For unattended "
                        "hardware-load / heating tests.")
    return p.parse_args()


if __name__ == '__main__':
    main()
