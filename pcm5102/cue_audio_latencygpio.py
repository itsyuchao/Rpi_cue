#!/usr/bin/env python3
"""
gpio_audio_latency_test.py
============================
Drives a GPIO pin HIGH immediately before sd.play() (and back LOW after),
so an oscilloscope can trigger on that rising edge and measure the delay
to the actual audio signal on the 3.5mm jack — isolating software/DAC
latency, with no keyboard/USB/human variability in the loop.

Pin choice: BCM GPIO5 (physical pin 29). NOT GPIO21 — GPIO21 is physical
pin 40, and it's already claimed by I2S (DAC DOUT) in this setup, so it
can't double as a free trigger pin. GPIO5 is unused by I2S/I2C here.

Uses lgpio if available (direct memory-mapped register access, sub-µs
toggle overhead) and falls back to RPi.GPIO otherwise (which may go
through slower sysfs writes on some setups — fine for this purpose, but
worth knowing if you see unexpected jitter on the trigger edge itself).

Scope setup:
  Channel 1 -> trigger GPIO pin, ground -> any Pi GND pin. 3.3V logic,
              standard 1x probe is fine.
  Channel 2 -> 3.5mm jack tip/sleeve.
  Trigger source = Channel 1, rising edge.
  Measure time-delta from Channel 1 rise to first visible Channel 2 signal.

Ctrl+C to quit.
"""

import time

import numpy as np
import sounddevice as sd

SR         = 48000
TONE_FREQ  = 440.0
TONE_DUR_S = 0.05
PAD_S      = 0.05
INT_S      = 0.1

TRIGGER_PIN_BCM = 5   # physical pin 29 — free in this setup

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


# ── Audio ────────────────────────────────────────────────────────────────────

def audio_init():
    sd.default.samplerate = SR
    sd.default.channels   = 1
    sd.default.dtype      = 'float32'
    sd.default.latency = 'low'
    for i, dev in enumerate(sd.query_devices()):
        if 'hifiberry' in dev['name'].lower():
            sd.default.device = i
            print(f"Audio: [{i}] {dev['name']}")
            break
    else:
        print("WARNING: HiFiBerry not found — using system default")

    probe = sd.OutputStream(samplerate=SR, channels=1, dtype='float32')
    probe.start()
    print(f"Negotiated output latency: {probe.latency * 1000:.2f} ms (at {SR} Hz)")
    probe.stop()
    probe.close()


def build_tone() -> np.ndarray:
    n = int(round(TONE_DUR_S * SR))
    t = np.arange(n, dtype=np.float64) / SR
    y = np.sin(2 * np.pi * TONE_FREQ * t).astype(np.float32)
    pad = np.zeros(int(round(PAD_S * SR)), dtype=np.float32)
    inter = np.zeros(int(INT_S * SR), dtype=np.float32)
    return np.concatenate([pad, y, inter, y])


def play_and_time(payload: np.ndarray, trigger: GpioTrigger):
    trigger.high()
    t0 = perf_counter_raw()
    sd.play(payload, samplerate=SR, blocksize=128)
    time.sleep(0.5)
    elapsed = perf_counter_raw() - t0
    trigger.low()

    expected = len(payload) / SR
    overhead_ms = (elapsed - expected) * 1000
    print(f"  call={elapsed:.4f}s  expected={expected:.4f}s  overhead≈{overhead_ms:.1f}ms")


def main():
    audio_init()
    tone = build_tone()
    trigger = GpioTrigger(TRIGGER_PIN_BCM)

    print("\nScope: Ch1 = GPIO trigger (BCM{}), Ch2 = 3.5mm jack. "
          "Trigger on Ch1 rising edge.".format(TRIGGER_PIN_BCM))
    print("Press Enter to fire one trial, Ctrl+C to quit.\n")
    try:
        i = 0
        while True:
            input(f">> [{i}] Enter: ")
            play_and_time(tone, trigger)
            i += 1
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        trigger.close()


if __name__ == '__main__':
    main()
