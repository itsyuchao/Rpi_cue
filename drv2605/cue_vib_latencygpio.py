#!/usr/bin/env python3
"""
cue_vib_latencygpio.py
======================
Vibration counterpart of pcm5102/cue_audio_latencygpio.py. Fires a train
of 3 pulses, 0.5 s onset-to-onset, using the same DRV2605 setup,
precomputed event list and spin-wait scheduler as cue_experiment.py
(haptic_init / play_haptic). Vibration has no startup padding — events
start at t=0, as in the experiment.

Pin 29 (BCM GPIO5) mirrors each pulse command: HIGH right before the
sequence + GO writes, LOW right after drv.play() returns. Rising edge =
the moment software issues the pulse; everything after it (I2C, driver,
motor spin-up) is visible on the scope.

Scope setup:
  Ch1 -> pin 29 (3.3V logic), ground -> any Pi GND pin.
  Ch2 -> LRA motor terminals (DRV2605 OUT+/OUT- are bridged, not
         ground-referenced: differential probe or two channels A-B),
         or an accelerometer on the motor.
  Trigger on Ch1 rising edge, window >= 1.5 s for the whole train.

Keys (then Enter):  a = effect A,  b = effect B,  empty = repeat last.
Ctrl+C to quit.

Usage:
  python3 cue_vib_latencygpio.py [--effect-a 1] [--effect-b 14]
"""

import argparse
import time

import board
import busio
import adafruit_drv2605

TRIGGER_PIN_BCM = 5    # physical pin 29 — free in this setup
DEFAULT_EFFECT1 = 1    # Strong Click (cue_experiment.py)
DEFAULT_EFFECT2 = 14   # Strong Buzz  (cue_experiment.py)
TRAIN_N         = 3
TRAIN_IOI_S     = 0.5


def perf_counter_raw() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


# ── GPIO backend: try lgpio first, fall back to RPi.GPIO ────────────────────

class GpioTrigger:
    """Minimal high/low trigger on one pin, backend-agnostic."""

    def __init__(self, bcm_pin: int):
        self.pin = bcm_pin
        self.backend = None
        self._h = None
        self._gpio_mod = None

        try:
            import lgpio
            self._h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._h, self.pin, 0)
            self._lgpio = lgpio
            self.backend = 'lgpio'
        except Exception as e:
            print(f"INFO: lgpio unavailable ({e}) — falling back to RPi.GPIO")
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio_mod = GPIO
            self.backend = 'RPi.GPIO'

        print(f"GPIO trigger: BCM{self.pin} via {self.backend}")

    def high(self):
        if self.backend == 'lgpio':
            self._lgpio.gpio_write(self._h, self.pin, 1)
        else:
            self._gpio_mod.output(self.pin, self._gpio_mod.HIGH)

    def low(self):
        if self.backend == 'lgpio':
            self._lgpio.gpio_write(self._h, self.pin, 0)
        else:
            self._gpio_mod.output(self.pin, self._gpio_mod.LOW)

    def close(self):
        if self.backend == 'lgpio':
            self._lgpio.gpio_write(self._h, self.pin, 0)
            self._lgpio.gpiochip_close(self._h)
        else:
            self._gpio_mod.cleanup(self.pin)


# ── Haptic (same as cue_experiment.haptic_init / play_haptic) ───────────────

def haptic_init():
    i2c = busio.I2C(board.SCL, board.SDA)
    drv = adafruit_drv2605.DRV2605(i2c)
    drv.use_LRM()
    print("Haptic: DRV2605L initialised (LRA)")
    return drv


def build_train(effect_id: int) -> list:
    """Precomputed [(time, effect_id)] list, same format as the experiment."""
    return [(i * TRAIN_IOI_S, effect_id) for i in range(TRAIN_N)]


def play_haptic(drv, events: list, trigger: GpioTrigger):
    """cue_experiment.play_haptic + pin 29 mirroring each pulse command."""
    t0 = perf_counter_raw()
    for rel_t, effect_id in events:
        now  = perf_counter_raw() - t0
        wait = rel_t - now
        if wait > 0.002:
            time.sleep(wait - 0.001)
        while perf_counter_raw() - t0 < rel_t:
            pass
        trigger.high()
        drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        drv.play()
        trigger.low()
    drv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--effect-a', type=int, default=DEFAULT_EFFECT1)
    ap.add_argument('--effect-b', type=int, default=DEFAULT_EFFECT2)
    args = ap.parse_args()
    trains = {'a': build_train(args.effect_a), 'b': build_train(args.effect_b)}

    drv = haptic_init()
    trigger = GpioTrigger(TRIGGER_PIN_BCM)

    print(f"\nScope: Ch1 = pin 29 (BCM{TRIGGER_PIN_BCM}), Ch2 = LRA. "
          "Trigger on Ch1 rising edge.")
    print(f"a = effect {args.effect_a}, b = effect {args.effect_b} "
          f"({TRAIN_N} pulses, {TRAIN_IOI_S}s apart). "
          "Enter = repeat last, Ctrl+C to quit.\n")
    try:
        i, key = 0, 'a'
        while True:
            k = input(f">> [{i}] {key.upper()} (a/b/Enter): ").strip().lower()
            if k:
                if k not in trains:
                    continue
                key = k
            play_haptic(drv, trains[key], trigger)
            i += 1
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        trigger.close()


if __name__ == '__main__':
    main()
