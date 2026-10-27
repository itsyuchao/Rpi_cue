#!/usr/bin/env python3
"""
cue_sit.py
==========
Sit-and-count familiarization controller for Raspberry Pi 3.

Hardware
--------
  Audio  : PCM5102A breakout → I2S (BCLK=GPIO18, LRCLK=GPIO19, DIN=GPIO21)
  Haptic : DRV2605L → I2C 0x5a, LRA motor

Operator console
----------------
  All operator I/O (subject ID, start confirmation, between-trial continue,
  counted number) goes through the controlling terminal (PiTFT). An
  RF-dongle USB keyboard plugs into Pi.

Session structure (4 fixed trials, seated)
------------------------------------------
  Trial order is fixed — audio first, then vibration, slow before fast:

    t1  audio      slow (+1)  'sitcountlow'    sample: A4 tone
    t2  audio      fast (-1)  'sitcounthigh'   sample: E5 tone
    t3  vibration  slow (+1)  'sitcountshort'  sample: effect1 (click)
    t4  vibration  fast (-1)  'sitcountlong'   sample: effect2 (buzz)

  Each trial: instruction WAV → 0.5 s → sample → 2 s → 3 sub-blocks

    regular (--regblocktime) → silent (--silblocktime) → count

  Slow/fast periods are dur_s+dur_s_jitter / dur_s-dur_s_jitter; dur_s is
  the period of one low/high tone (or click/buzz) pair. With the defaults
  (--dur-s 1.0, --dur-s-jitter 0.08) that is 1.08 s / 0.92 s, i.e. ~28 /
  ~33 pairs in the 30 s regular block.

  The count sub-block mirrors cue_simple's rate block without stop.wav /
  ratesync.wav: ping volley → marker → the operator enters the 0-50 number
  the participant counted, which lands in the count row's count column.

Per-block network protocol
--------------------------
  Identical to cue_simple.py: ping_volley() → capture block-start
  recv_time_iso + recv_perf_s → send_marker_packet() → stimulus → log_block.
  Instruction WAVs and samples play before the volley, so the marker and
  block timestamps bracket only the stimulation (or count) window. All
  waveforms are pre-padded with 50 ms of startup silence at build/load time.

Install
-------
  pip install numpy sounddevice scipy \
              adafruit-circuitpython-drv2605
"""

import argparse
import csv
import os
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
DUR_S_DEFAULT        = 1.0
DUR_S_JITTER_DEFAULT = 0.08

# Block durations (s)
REG_BLOCKTIME = 30
SIL_BLOCKTIME = 5

# Haptic effects (DRV2605 built-in library)
DEFAULT_EFFECT1 = 1    # Strong Click → "low" beat
DEFAULT_EFFECT2 = 14   # Strong Buzz   → "high" beat

# Fixed trial plan: (modality, tempo, target). target selects both the
# instruction WAV ('sitcount{target}') and the sample that follows it.
TRIAL_PLAN = [
    ('audio',     +1, 'low'),
    ('audio',     -1, 'high'),
    ('vibration', +1, 'short'),
    ('vibration', -1, 'long'),
]

COUNT_MAX = 50


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


# ══════════════════════════════════════════════════════════════════════════════
# Module 5 — Haptic engine
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
# Module 6 — CSV logger
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentLogger:
    """Writes one CSV row per sub-block (+ one per ping volley) to a
    timestamped file.

    Columns with limited scope are left blank when not applicable:
      - dur_s   : only for regular blocks (delivered cycle period, s)
      - tempo   : only for regular blocks (+1 = dur_s+jitter slower,
                  -1 = dur_s-jitter faster)
      - target  : only for regular blocks ('low' / 'high' / 'short' / 'long')
      - count   : only for the count block of each trial

    marker_label encodes modality / block_subtype / block_position / trial_num
    ('{a|v}_{reg|sil|cnt}_p{pos}_t{trial}'), so those fields are not
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
    HEADER = [
        'event_type',
        'participant_id',
        'trial_num',
        'recv_time_iso',
        'recv_perf_s',
        'dur_s',
        'tempo',
        'target',
        'count',
        'marker_label',
        'rtt1_ms',
        'rtt2_ms',
        'rtt3_ms',
        'rtt_additional',
    ]

    def __init__(self, participant_id: str, output_dir: str = 'log'):
        self.pid = participant_id
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(
            output_dir, f"cuesit_log_pid-{participant_id}_{ts}.csv"
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
                  dur_s: float | str = '',
                  tempo: int | str = '',
                  target: str = '',
                  count: int | str = '',
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
            dur_s,
            tempo,
            target,
            count,
            marker_label,
            '', '', '', '',   # rtt1_ms, rtt2_ms, rtt3_ms, rtt_additional
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
            '', '', '', '',      # dur_s, tempo, target, count
            marker_label,
            r1, r2, r3, r_add,
        ]
        self._append_row(row)


# ══════════════════════════════════════════════════════════════════════════════
# Module 7 — Stimulus precomputation + trial runner
# ══════════════════════════════════════════════════════════════════════════════

def precompute_stimuli(regblocktime: float,
                       dur_s: float, dur_s_jitter: float,
                       effect1: int, effect2: int
                       ) -> tuple[dict, dict, float, float]:
    """Build every waveform and haptic schedule that trials will need.

    Returns (audio, haptic, dur_low, dur_high). During trials, the runner
    only selects a precomputed object — no synthesis, padding, or copying
    happens in the hot path. Every audio buffer (including the sample
    tones) is pre-padded here with the 50 ms startup silence.
    """
    dur_low  = dur_s - dur_s_jitter
    dur_high = dur_s + dur_s_jitter

    audio = {
        'regular_low':  _pad_startup(build_regular_audio(regblocktime, dur_low)),
        'regular_high': _pad_startup(build_regular_audio(regblocktime, dur_high)),
        'sample_tone_high': _pad_startup(_synth_tone(F_HIGH, 0.5)),
        'sample_tone_low':  _pad_startup(_synth_tone(F_LOW,  0.5)),
    }
    haptic = {
        'regular_low':  build_regular_haptic(regblocktime, dur_low,  effect1, effect2),
        'regular_high': build_regular_haptic(regblocktime, dur_high, effect1, effect2),
    }
    return audio, haptic, dur_low, dur_high


def run_trial(*,
              modality: str,
              trial_num: int,
              regblocktime: float,
              silblocktime: float,
              dur_s: float,
              tempo: int,
              target: str,
              stim_audio: dict,
              stim_haptic: dict,
              words: dict,
              logger: ExperimentLogger,
              effect1: int,
              effect2: int):
    """Run one sit-and-count trial for either modality.

    Preamble : 'sitcount{target}' → 0.5 s → sample → 2 s (same timing as
               the attention sequence in cue_experiment). Sample is the
               A4 / E5 tone for low / high, effect1 / effect2 for
               short / long.
    Blocks   : 3 sub-blocks, fixed layout:
                 p0=regular (regblocktime), p1=silent (silblocktime),
                 p2=count.
               The count block is cue_simple's rate block without the
               stop / ratesync WAVs: the operator enters the 0-50 number
               the participant counted; it lands in the count column.
    Per-block: ping_volley() → capture recv_time_iso + recv_perf_s →
               send_marker_packet() → play stimulus → log_block.
    """
    is_audio   = (modality == 'audio')
    mod_marker = 'a' if is_audio else 'v'
    dur_key    = 'regular_high' if tempo > 0 else 'regular_low'

    # ── Pick stimuli ─────────────────────────────────────────────────────
    if is_audio:
        w_regular  = stim_audio[dur_key]
    else:
        if _drv is None:
            print("    [HAPTIC] No driver — skipping vibration trial")
            time.sleep(regblocktime + silblocktime)
            return
        h_regular  = stim_haptic[dur_key]

    # Subtype tokens match the marker_label encoding:
    # '{a|v}_{reg|sil|cnt}_p{position}_t{trial}'.
    subtype_by_position = {0: 'reg', 1: 'sil', 2: 'cnt'}

    # ── Preamble: instruction + sample ───────────────────────────────────
    play_word(words, f'sitcount{target}')
    time.sleep(0.5)
    if is_audio:
        play_audio(stim_audio['sample_tone_high' if target == 'high'
                              else 'sample_tone_low'])
    else:
        play_haptic_event(_drv, effect1 if target == 'short' else effect2)
    time.sleep(2)

    # ── Run the 3 sub-blocks ─────────────────────────────────────────────
    for position in range(3):
        subtype = subtype_by_position[position]

        if subtype == 'cnt':
            print("\n  Input counted number")

        marker = f"{mod_marker}_{subtype}_p{position}_t{trial_num}"

        # Per-block metadata for the logger
        if subtype == 'reg':
            dur_log = f"{dur_s:.6f}"
            tempo_log = tempo
            target_log = target
            desc = f"regular {dur_s:.3f}s ({target})"
        else:
            dur_log = tempo_log = target_log = ''
            desc = 'silent' if subtype == 'sil' else 'count'

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

        # 3. Play the stimulus. For 'cnt' the "stimulus" is the count entry.
        count = ''
        if subtype == 'reg':
            if is_audio:
                play_audio(w_regular)
            else:
                play_haptic_block(_drv, h_regular, regblocktime)
        elif subtype == 'sil':
            time.sleep(silblocktime)
        else:  # 'cnt'
            count = get_count()

        # 4. Log the block row. count is '' except for the count block.
        logger.log_block(
            trial_num=trial_num,
            marker_label=marker,
            dur_s=dur_log,
            tempo=tempo_log,
            target=target_log,
            count=count,
            recv_time_iso=block_recv_time_iso,
            recv_perf_s=block_recv_perf_s,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 8 — Operator terminal input
# ══════════════════════════════════════════════════════════════════════════════
#
# All operator I/O is plain stdin via the controlling terminal (RF-dongle
# USB keyboard). No on-Pi keypad or LCD/OLED.

def get_participant_id() -> str:
    return input("Enter participant ID: ").strip()


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


def get_count() -> int:
    """Collect the 0-50 number the participant counted."""
    while True:
        try:
            val = input(f"  >> Counted number (0-{COUNT_MAX}): ").strip()
        except EOFError:
            return 0
        if val.isdigit() and 0 <= int(val) <= COUNT_MAX:
            return int(val)


# ══════════════════════════════════════════════════════════════════════════════
# Module 9 — Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Init hardware ────────────────────────────────────────────────────
    audio_init()
    haptic_init()

    if args.list_devices:
        print(sd.query_devices())
        return

    # ── Participant ID ───────────────────────────────────────────────────
    participant_id = get_participant_id()
    print(f"\nParticipant: {participant_id}")

    # ── Parameters ───────────────────────────────────────────────────────
    regblocktime      = REG_BLOCKTIME
    silblocktime      = SIL_BLOCKTIME
    dur_s             = args.dur_s
    dur_s_jitter      = args.dur_s_jitter
    effect1           = DEFAULT_EFFECT1
    effect2           = DEFAULT_EFFECT2

    # ── Precompute every stimulus up front (low-latency trials) ──────────
    print("\nPrecomputing audio & haptic stimuli...", flush=True)
    t_pre = perf_counter_raw()
    stim_audio, stim_haptic, dur_low, dur_high = precompute_stimuli(
        regblocktime, dur_s, dur_s_jitter, effect1, effect2,
    )
    print(f"  Done in {perf_counter_raw() - t_pre:.2f}s "
          f"(regular periods: {dur_low:.3f} / {dur_high:.3f} s)")

    # ── Logger ───────────────────────────────────────────────────────────
    logger = ExperimentLogger(participant_id)
    marker_init(logger_ip, logger=logger)

    # ── Pre-load WAV cues ────────────────────────────────────────────────
    words = {}
    loaded_names = []
    for name in [f'sitcount{target}' for _, _, target in TRIAL_PLAN]:
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
    total_trials = len(TRIAL_PLAN)
    print(f"\n{'='*20}")
    print(f"  Participant   : {participant_id}")
    print(f"  Trials        : {total_trials}")
    print(f"  Regular period: {dur_s} s ± {dur_s_jitter} s ")
    print(f"  Blocks        : regular {regblocktime} s → silent {silblocktime} s")
    print(f"{'='*20}")

    if not wait_for_continue():
        return

    time.sleep(2)

    # ── Trial loop ───────────────────────────────────────────────────────
    interrupted = False
    try:
        for idx, (modality, tempo, target) in enumerate(TRIAL_PLAN):
            trial_num      = idx + 1
            dur_delivered  = dur_high if tempo > 0 else dur_low

            print(f"\n{'─'*20}")
            print(f"  Trial {trial_num}/{total_trials}: {modality} "
                  f"{'slow' if tempo > 0 else 'fast'} ({target})")
            print(f"{'─'*20}")

            t0 = perf_counter_raw()
            run_trial(
                modality=modality,
                trial_num=trial_num,
                regblocktime=regblocktime,
                silblocktime=silblocktime,
                dur_s=dur_delivered,
                tempo=tempo,
                target=target,
                stim_audio=stim_audio,
                stim_haptic=stim_haptic,
                words=words,
                logger=logger,
                effect1=effect1,
                effect2=effect2,
            )
            elapsed = perf_counter_raw() - t0

            print(f"  Trial {trial_num} done ({elapsed:.1f}s)")

            if idx < total_trials - 1:
                if not wait_for_continue():
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user (Ctrl+C). Logged rows were flushed to disk.")

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\n{'='*20}")
    if interrupted:
        print(f"Session interrupted — {logger.filename}")
    else:
        print(f"Session complete — {logger.filename}")
    print(f"{'='*20}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 10 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Sit-and-count cue familiarization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dur-s',       type=float, default=DUR_S_DEFAULT,
                   help="Nominal regular-cue cycle period (s), one low/high "
                        "pair. Slow trials use dur_s+jitter, fast trials "
                        "dur_s-jitter.")
    p.add_argument('--dur-s-jitter', type=float,
                   default=DUR_S_JITTER_DEFAULT,
                   help="Absolute jitter (s) applied to --dur-s")
    p.add_argument('--list-devices', action='store_true',
                   help="Print sounddevice devices and exit")
    return p.parse_args()


if __name__ == '__main__':
    main()
