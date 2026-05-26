#!/usr/bin/env python3
"""
cue_vib_timing.py
=================
Bench DRV2605 vibration-cue scheduling accuracy under different wait
schemes. Per event, the chosen scheme waits for the scheduled time,
then drv.play() fires; the script records (scheduled, fire, post-fire)
and prints jitter + I/O latency percentiles.

Usage:
  python3 cue_vib_timing.py --duration 30 --freq 0.83 --scheme hybrid
  sudo python3 cue_vib_timing.py --scheme spin --rt
  python3 cue_vib_timing.py --scheme nanosleep --csv out.csv

Schemes:
  hybrid    — exact cue_experiment.py:486-498 implementation
              (time.sleep(wait - 0.001) if wait > 0.002, then spin)
  spin      — pure spin on perf_counter_raw, no sleep
  sleep     — pure time.sleep, no spin compensation (reveals raw sleep jitter)
  nanosleep — clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME) via libc;
              usually the most precise kernel-side waiter on Linux

drv.stop() is NEVER called — it truncates the final buzz because the
library does not expose each built-in effect's duration.
"""

import argparse
import csv
import ctypes
import os
import statistics
import time

try:
    import board
    import busio
    import adafruit_drv2605
    HAPTIC_AVAILABLE = True
except ImportError:
    HAPTIC_AVAILABLE = False

DEFAULT_EFFECT1 = 1     # Strong Click
DEFAULT_EFFECT2 = 14    # Strong Buzz

_drv = None


def perf_counter_raw() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def haptic_init():
    global _drv
    if not HAPTIC_AVAILABLE:
        print("ERROR: adafruit_drv2605 not installed")
        return
    i2c = busio.I2C(board.SCL, board.SDA)
    _drv = adafruit_drv2605.DRV2605(i2c)
    _drv.use_LRM()
    print("Haptic: DRV2605L (LRA)")


def enable_rt(prio: int = 80):
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
        print(f"SCHED_FIFO prio={prio} enabled")
    except PermissionError:
        print("WARNING: SCHED_FIFO requires root — running SCHED_OTHER")
    except Exception as e:
        print(f"WARNING: SCHED_FIFO setup failed: {e}")


# ── Wait schemes ─────────────────────────────────────────────────────────────

def wait_hybrid(t0_raw: float, t0_mono: float, rel_t: float):
    # Exact copy of cue_experiment.py:486-498 wait body.
    now  = perf_counter_raw() - t0_raw
    wait = rel_t - now
    if wait > 0.002:
        time.sleep(wait - 0.001)
    while perf_counter_raw() - t0_raw < rel_t:
        pass


def wait_spin(t0_raw: float, t0_mono: float, rel_t: float):
    while perf_counter_raw() - t0_raw < rel_t:
        pass


def wait_sleep(t0_raw: float, t0_mono: float, rel_t: float):
    wait = rel_t - (perf_counter_raw() - t0_raw)
    if wait > 0:
        time.sleep(wait)


# clock_nanosleep abstime via libc — Python's stdlib doesn't expose it.
_libc = ctypes.CDLL('libc.so.6', use_errno=True)
_CLOCK_MONOTONIC = 1
_TIMER_ABSTIME   = 1


class _Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]


def wait_nanosleep(t0_raw: float, t0_mono: float, rel_t: float):
    # CLOCK_MONOTONIC (not _RAW — abstime nanosleep doesn't support RAW).
    target = t0_mono + rel_t
    sec  = int(target)
    nsec = int(round((target - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    ts = _Timespec(sec, nsec)
    while True:
        ret = _libc.clock_nanosleep(_CLOCK_MONOTONIC, _TIMER_ABSTIME,
                                    ctypes.byref(ts), None)
        if ret == 0 or ret != 4:  # 4 = EINTR; on EINTR retry, else give up
            return


SCHEMES = {
    'hybrid':    wait_hybrid,
    'spin':      wait_spin,
    'sleep':     wait_sleep,
    'nanosleep': wait_nanosleep,
}


# ── Schedule + run ───────────────────────────────────────────────────────────

def build_schedule(duration_s: float, freq_hz: float) -> list:
    """Alternating effect1/effect2 at half = 0.5/freq spacing — matches
    cue_experiment.py:519-532 build_regular_haptic."""
    half = 0.5 / freq_hz
    events = []
    t = 0.0
    i = 0
    eff = (DEFAULT_EFFECT1, DEFAULT_EFFECT2)
    while t < duration_s - 1e-9:
        events.append((t, eff[i % 2]))
        t += half
        i += 1
    return events


def run(scheme: str, events: list) -> tuple[list, list]:
    waiter   = SCHEMES[scheme]
    jitter   = []   # (fire wall-time) - (scheduled wall-time)
    io_lat   = []   # post-play - pre-play
    t0_raw   = perf_counter_raw()
    t0_mono  = time.clock_gettime(time.CLOCK_MONOTONIC)

    for rel_t, effect_id in events:
        waiter(t0_raw, t0_mono, rel_t)
        t_pre = perf_counter_raw()
        _drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        _drv.play()
        t_post = perf_counter_raw()
        jitter.append((t_pre - t0_raw) - rel_t)
        io_lat.append(t_post - t_pre)
    return jitter, io_lat


# ── Stats ────────────────────────────────────────────────────────────────────

def pct(xs_sorted: list, p: float) -> float:
    if not xs_sorted:
        return 0.0
    k = max(0, min(len(xs_sorted) - 1,
                   int(round(p / 100 * (len(xs_sorted) - 1)))))
    return xs_sorted[k]


def print_stats(name: str, xs: list):
    xs_us = sorted(x * 1e6 for x in xs)
    print(f"  {name}  (n={len(xs_us)})")
    print(f"    min   {xs_us[0]:+10.1f} µs")
    print(f"    p50   {pct(xs_us, 50):+10.1f} µs")
    print(f"    mean  {statistics.fmean(xs_us):+10.1f} µs")
    print(f"    p95   {pct(xs_us, 95):+10.1f} µs")
    print(f"    p99   {pct(xs_us, 99):+10.1f} µs")
    print(f"    max   {xs_us[-1]:+10.1f} µs")
    print(f"    stdev {statistics.stdev(xs_us) if len(xs_us) > 1 else 0:10.1f} µs")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.rt:
        enable_rt()

    haptic_init()
    if _drv is None:
        return

    events = build_schedule(args.duration, args.freq)
    n      = len(events)
    half_ms = 0.5 / args.freq * 1000
    print(f"\nScheme    : {args.scheme}")
    print(f"RT prio   : {'SCHED_FIFO 80' if args.rt else 'SCHED_OTHER'}")
    print(f"Freq      : {args.freq} Hz   (interval = {half_ms:.2f} ms)")
    print(f"Duration  : {args.duration} s  →  {n} events")
    print(f"Running...\n")

    jitter, io_lat = run(args.scheme, events)

    print(f"Done.\n")
    print_stats("schedule jitter (fire vs scheduled)", jitter)
    print()
    print_stats("I/O latency (drv.play() call)", io_lat)

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'scheduled_s', 'jitter_s', 'io_latency_s'])
            for i, ((rel_t, _), j, io) in enumerate(zip(events, jitter, io_lat)):
                w.writerow([i, f"{rel_t:.6f}", f"{j:.6f}", f"{io:.6f}"])
        print(f"\nCSV: {args.csv}")


def parse_args():
    p = argparse.ArgumentParser(
        description="DRV2605 vibration cue timing-accuracy bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--duration', type=float, default=30.0,
                   help="Total test duration (s)")
    p.add_argument('--freq',     type=float, default=0.83,
                   help="Cue rate (Hz); event interval = 0.5/freq")
    p.add_argument('--scheme',   choices=list(SCHEMES.keys()), default='hybrid',
                   help="Wait scheme. hybrid = exact cue_experiment.py impl.")
    p.add_argument('--rt',       action='store_true',
                   help="Enable SCHED_FIFO at priority 80 (needs sudo)")
    p.add_argument('--csv',      type=str, default=None,
                   help="Optional CSV output of per-event timing")
    return p.parse_args()


if __name__ == '__main__':
    main()
