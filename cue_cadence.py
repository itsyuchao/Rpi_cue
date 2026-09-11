#!/usr/bin/env python3
"""
cue_cadence.py
==============
Continuous-walking cadence-sweep cueing controller for Raspberry Pi 3.

Hardware
--------
  Audio  : PCM5102A breakout → I2S (BCLK=GPIO18, LRCLK=GPIO19, DIN=GPIO21)
  Haptic : DRV2605L → I2C 0x5a, LRA motor

Operator console
----------------
  All operator I/O (subject ID, loop direction, modality, resume prompt,
  ready-cue confirmation) goes through the controlling terminal (PiTFT).
  An RF-dongle USB keyboard plugs into Pi.

Pre-experiment
--------------
  Subject ID, loop direction ('cw'/'ccw'), and modality ('a'=audio /
  'v'=vibration) are entered by the operator (counter-balance loop
  direction and modality across subjects when assigning sessions). The
  matching 'ready{cw,ccw}.wav' cue plays and the operator confirms with
  'y'; a single 'go.wav' then marks the start of the session (logged as
  one 'gostop' row). The loop direction is appended to the CSV log
  filename (cuecadence_log_pid-{pid}_loop-{dir}_{ts}.csv).

Trial structure (continuous walking — 2 sub-blocks per trial, no preamble)
---------------------------------------------------------------------------
    regular (p1) → silent (p2)

  Trials run back-to-back with no per-trial 'ignore'/'go', no baseline
  silence (p0), and no rating sub-block (p3) — those positions from the
  base protocol are not used here. Position 1 (regular cue) is
  --regblocktime seconds long (default 20 s); position 2 (silent) is
  --silblocktime seconds long (default 10 s, down from the base
  protocol's 20 s).

  Each trial's regular-cue period (dur_s) is drawn from a randomized
  cadence-sweep schedule (build_cadence_plan): every value in
  --dur-s-list (default 0.8-1.5 s in 0.1 s steps) appears --reps times
  (default 2), shuffled deterministically by participant_id so
  resume-from-trial-N reproduces the same sequence. With the defaults
  that is 16 trials — each of the 8 cadence periods delivered twice, in
  random order. Unlike cue_simple.py/cue_experiment.py, modality is a
  single operator choice for the whole session, not a per-trial factor,
  and there is no attend/focusleg cue.

Per-block network protocol
--------------------------
  For each sub-block the order is **always**:
    1. Three ping-pong rounds against rpi-fetch (warms the network path,
       logged as a 'ping' row in this Pi's cuecadence_log).
    2. Capture block-start recv_time_iso + recv_perf_s and emit the marker
       UDP packet on the now-warm path.
    3. Begin stimulation.

  All waveforms are pre-padded with 50 ms of startup silence at
  build/load time; audio onsets therefore lag the block timestamp by
  that fixed 50 ms plus device latency (sd.default.latency = 'low').

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

# Cadence sweep — default regular-cue periods (s) and repeats per value.
# Each value in DEFAULT_DUR_S_LIST appears DEFAULT_DUR_REPS times in the
# randomized per-trial schedule (see build_cadence_plan).
DEFAULT_DUR_S_LIST = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DEFAULT_DUR_REPS = 2

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
      - template_index    : unused (arrhythmic condition removed)
      - dur_s             : only for regular blocks (delivered cycle period, s)
      - tempo             : unused (no jitter — dur_s is swept directly)
      - attend            : unused (no attend cue in the cadence sweep)
      - focusleg          : unused (no focus-leg cue in the cadence sweep)
      - sync_rating       : unused (no rating sub-block in the cadence sweep)

    marker_label encodes modality / block_subtype / block_position / trial_num
    ('{a|v}_{reg|sil}_p{pos}_t{trial}'), so those fields are not
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
    # ('{a|v}_{reg|sil}_p{pos}_t{trial}'). Block order is fixed,
    # so it is not stored either.
    HEADER = [
        'event_type',
        'participant_id',
        'trial_num',
        'recv_time_iso',
        'recv_perf_s',
        'template_index',
        'dur_s',
        'tempo',
        'attend',
        'focusleg',
        'sync_rating',
        'marker_label',
        'rtt1_ms',
        'rtt2_ms',
        'rtt3_ms',
        'rtt_additional',
        'gostop_pause_s',
    ]

    def __init__(self, participant_id: str, output_dir: str = 'log',
                 loop_dir: str | None = None):
        self.pid = participant_id
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"loop-{loop_dir}_" if loop_dir else ''
        self.filename = os.path.join(
            output_dir, f"cuecadence_log_pid-{participant_id}_{suffix}{ts}.csv"
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
                  attend: str = '',
                  focusleg: str = '',
                  template_index: int | str = '',
                  dur_s: float | str = '',
                  tempo: int | str = '',
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
            tempo,
            attend,
            focusleg,
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
            '', '', '', '', '', '',  # template_index, dur_s, tempo, attend, focusleg, sync_rating
            marker_label,
            r1, r2, r3, r_add,
            '',                  # gostop_pause_s
        ]
        self._append_row(row)

    def log_gostop(self, *, trial_num: int, marker_label: str,
                   recv_time_iso: str, recv_perf_s: float,
                   gostop_pause_s: float):
        """Append one 'gostop' row marking the session-start go.wav play
        moment (logged once, before trial 1 — there is no per-trial
        preamble or rate block in the cadence sweep). recv_time_iso /
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
            '', '', '', '', '', '',  # template_index, dur_s, tempo, attend, focusleg, sync_rating
            marker_label,
            '', '', '', '',      # rtt1_ms, rtt2_ms, rtt3_ms, rtt_additional
            f"{gostop_pause_s:.6f}",
        ]
        self._append_row(row)


# ══════════════════════════════════════════════════════════════════════════════
# Module 7 — Stimulus precomputation + unified trial runner
# ══════════════════════════════════════════════════════════════════════════════

def precompute_stimuli(regblocktime: float, dur_s_list: list[float],
                       effect1: int, effect2: int
                       ) -> tuple[dict, dict]:
    """Build every regular-cue waveform/haptic schedule the cadence sweep
    will need, one per distinct value in `dur_s_list`.

    The silent sub-block is just `time.sleep(...)` in the runner — no
    waveform needed for either modality. Returns (audio, haptic), each
    keyed by the dur_s value. During trials the runner only selects a
    precomputed object — no synthesis, padding, or copying happens in the
    hot path.
    """
    distinct = set(dur_s_list)
    audio = {
        dur_s: _pad_startup(build_regular_audio(regblocktime, dur_s))
        for dur_s in distinct
    }
    haptic = {
        dur_s: build_regular_haptic(regblocktime, dur_s, effect1, effect2)
        for dur_s in distinct
    }
    return audio, haptic


def run_trial(*,
              modality: str,
              trial_num: int,
              regblocktime: float,
              silblocktime: float,
              dur_s: float,
              stim_audio: dict,
              stim_haptic: dict,
              logger: ExperimentLogger):
    """Run one cadence-sweep trial: a regular-cue block at `dur_s`
    (position 1) followed by a short silent gap (position 2). Trials run
    back-to-back for continuous walking — no preamble, attention cue, or
    rating (positions 0 and 3 from the base protocol are not used here).

    Per-block: ping_volley() → capture recv_time_iso + recv_perf_s →
    send_marker_packet() → play stimulus → log_block. The pings warm the
    network path so the marker UDP arrives with the lowest possible
    latency.
    """
    is_audio   = (modality == 'audio')
    mod_marker = 'a' if is_audio else 'v'

    if is_audio:
        w_regular = stim_audio[dur_s]
    else:
        if _drv is None:
            print("    [HAPTIC] No driver — skipping vibration trial")
            time.sleep(regblocktime + silblocktime)
            return
        h_regular = stim_haptic[dur_s]

    for position, subtype in ((1, 'reg'), (2, 'sil')):
        marker = f"{mod_marker}_{subtype}_p{position}_t{trial_num}"
        desc = f"regular {dur_s:.3f}s" if subtype == 'reg' else 'silent'
        print(f"    Block {position}: {desc}")

        # 1. Ping volley FIRST — warms the network path before the marker.
        ping_volley(marker, trial_num=trial_num)

        # 2. Capture block-start timestamps and emit the marker packet on
        #    the now-warm path.
        block_recv_time_iso = datetime.now(timezone.utc).isoformat(
            timespec='microseconds'
        )
        block_recv_perf_s = perf_counter_raw()
        send_marker_packet(marker)

        # 3. Play the stimulus.
        if subtype == 'reg':
            if is_audio:
                play_audio(w_regular)
            else:
                play_haptic_block(_drv, h_regular, regblocktime)
            dur_log = f"{dur_s:.6f}"
        else:
            time.sleep(silblocktime)
            dur_log = ''

        # 4. Log the block row.
        logger.log_block(
            trial_num=trial_num,
            marker_label=marker,
            dur_s=dur_log,
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


def get_loop_dir() -> str:
    """Prompt for the session loop direction ('cw' or 'ccw'), used for the
    pre-trial ready cue and the log filename. Counter-balance it across
    subjects when assigning sessions."""
    while True:
        try:
            v = input("Loop direction (cw/ccw): ").strip().lower()
        except EOFError:
            return 'cw'
        if v in ('cw', 'ccw'):
            return v


def get_modality() -> str:
    """Prompt for the session modality ('a'=audio / 'v'=vibration), fixed
    for the whole session (unlike cue_simple.py's per-trial modality
    schedule). Counter-balance it across subjects when assigning sessions."""
    while True:
        try:
            v = input("Modality (a=audio / v=vibration): ").strip().lower()
        except EOFError:
            return 'audio'
        if v in ('a', 'audio'):
            return 'audio'
        if v in ('v', 'vibration'):
            return 'vibration'


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


def get_resume_start_index(total_trials: int) -> int:
    """Prompt resume position and return 0-based trial index to start from.

    The pid seed in build_cadence_plan makes 'trial N' the same trial
    every restart.
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


def build_cadence_plan(dur_s_list: list[float], reps: int,
                       participant_id: str) -> list[float]:
    """Build a randomized cadence-sweep schedule: every value in
    `dur_s_list` repeated `reps` times, shuffled.

    The schedule is deterministic in participant_id: restarting with the
    same PID reproduces the exact same trial sequence, so
    resume-from-trial-N is well-defined.
    """
    rng = random.Random(f"participant::{participant_id}")
    trials = list(dur_s_list) * reps
    rng.shuffle(trials)
    return trials


# ══════════════════════════════════════════════════════════════════════════════
# Module 9 — Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Init hardware ────────────────────────────────────────────────────
    audio_init()
    haptic_init()

    if args.list_devices:
        print(sd.query_devices())
        return

    # ── Participant ID + loop direction + modality ───────────────────────
    if args.headless:
        participant_id = "headless"
        loop_dir       = 'cw'
        modality       = 'audio'
    else:
        participant_id = get_participant_id()
        loop_dir       = get_loop_dir()
        modality       = get_modality()
    print(f"\nParticipant: {participant_id}  (loop: {loop_dir}, modality: {modality})")

    # ── Parameters ───────────────────────────────────────────────────────
    regblocktime = args.regblocktime
    silblocktime = args.silblocktime
    dur_s_list   = args.dur_s_list
    reps         = args.reps
    effect1      = DEFAULT_EFFECT1
    effect2      = DEFAULT_EFFECT2

    # ── Precompute every stimulus up front (low-latency trials) ──────────
    print("\nPrecomputing audio & haptic stimuli...", flush=True)
    t_pre = perf_counter_raw()
    stim_audio, stim_haptic = precompute_stimuli(
        regblocktime, dur_s_list, effect1, effect2,
    )
    print(f"  Done in {perf_counter_raw() - t_pre:.2f}s "
          f"(cadence periods: {', '.join(f'{d:.2f}' for d in dur_s_list)} s)")

    # ── Build cadence-sweep schedule ──────────────────────────────────────
    dur_sequence = build_cadence_plan(dur_s_list, reps, participant_id)
    total_trials = len(dur_sequence)

    print("Schedule: " + ' '.join(f'{d:.2f}' for d in dur_sequence))

    if args.headless:
        start_idx = 0
    else:
        start_idx = get_resume_start_index(total_trials)
        if start_idx > 0:
            print(f"Resuming from trial {start_idx + 1}/{total_trials}")

    # ── Logger ───────────────────────────────────────────────────────────
    logger = ExperimentLogger(participant_id, loop_dir=loop_dir)
    marker_init(logger_ip, logger=logger)

    # ── Pre-load WAV cues ────────────────────────────────────────────────
    words = {}
    loaded_names = []
    for name in ['go', 'readyccw', 'readycw']:
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
    def _wav_s(name: str) -> float:
        return len(words[name]) / SR if name in words else 0.0

    session_pause  = 2.0  # fixed pause between ready-cue confirmation and 'go'
    trial_dur      = regblocktime + silblocktime
    est_minutes    = (session_pause + _wav_s('go')
                      + total_trials * trial_dur) / 60

    print(f"\n{'='*20}")
    print(f"  Participant   : {participant_id}")
    print(f"  Modality      : {modality}")
    print(f"  Trials        : {total_trials}")
    print(f"  Cadence sweep : {dur_s_list} s × {reps} reps")
    print(f"  Est. duration : ~{est_minutes:.1f} min")
    print(f"{'='*20}")

    # ── Pre-experiment ready cue ─────────────────────────────────────────
    ready_word = f'ready{loop_dir}'
    print(f"  Ready cue         : {ready_word}")
    if not args.headless:
        play_word(words, ready_word)
        if not wait_for_continue():
            return

    # ── Session-start 'go' — no per-trial preamble, continuous walking ───
    mod_marker = 'a' if modality == 'audio' else 'v'
    time.sleep(session_pause)
    go_recv_time_iso = datetime.now(timezone.utc).isoformat(
        timespec='microseconds'
    )
    go_recv_perf_s = perf_counter_raw()
    play_word(words, 'go')
    logger.log_gostop(
        trial_num=0,
        marker_label=f"{mod_marker}_go_p0_t0",
        recv_time_iso=go_recv_time_iso,
        recv_perf_s=go_recv_perf_s,
        gostop_pause_s=session_pause,
    )

    # ── Trial loop — continuous, no between-trial prompts ────────────────
    interrupted = False
    try:
        for idx in range(start_idx, total_trials):
            trial_num = idx + 1
            dur_s     = dur_sequence[idx]

            print(f"\n{'─'*20}")
            print(f"Trial {trial_num}/{total_trials}  dur_s={dur_s:.2f}s")
            print(f"{'─'*20}")

            t0 = perf_counter_raw()
            run_trial(
                modality=modality,
                trial_num=trial_num,
                regblocktime=regblocktime,
                silblocktime=silblocktime,
                dur_s=dur_s,
                stim_audio=stim_audio,
                stim_haptic=stim_haptic,
                logger=logger,
            )
            elapsed = perf_counter_raw() - t0

            print(f"  Trial {trial_num} done ({elapsed:.1f}s)")
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
# Module 10 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Continuous-walking cadence-sweep cueing experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dur-s-list',  type=float, nargs='+',
                   default=list(DEFAULT_DUR_S_LIST),
                   help="Regular-cue cycle periods (s) to sweep")
    p.add_argument('--reps',        type=int,   default=DEFAULT_DUR_REPS,
                   help="Number of times each --dur-s-list value appears "
                        "(randomized order, deterministic per participant ID)")
    p.add_argument('--regblocktime', type=float, default=20,
                   help="Duration of the regular-cue sub-block (position 1, s)")
    p.add_argument('--silblocktime', type=float, default=10,
                   help="Duration of the post-cue silent sub-block (position 2, s)")
    p.add_argument('--list-devices', action='store_true',
                   help="Print sounddevice devices and exit")
    p.add_argument('--headless',    action='store_true',
                   help="Run with no operator input: auto-generate "
                        "participant ID, default modality to audio, start "
                        "at trial 1 (no resume prompt), skip the ready-cue "
                        "confirmation. For unattended hardware-load / "
                        "heating tests.")
    return p.parse_args()


if __name__ == '__main__':
    main()
