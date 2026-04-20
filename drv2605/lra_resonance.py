#!/usr/bin/env python3
"""
DRV2605L LRA Resonance Frequency Finder
========================================
Three methods:
  1. Auto-calibration (let the chip find resonance automatically)
  2. Diagnostics mode (quick impedance/resonance check)
  3. Open-loop frequency sweep (drive at many frequencies, feel/measure peak)

Run on Raspberry Pi:
  pip3 install adafruit-blinka adafruit-circuitpython-drv2605
  python3 drv2605_lra_resonance.py

Wiring: Pi SDA/SCL -> DRV2605 SDA/SCL, 3.3V, GND, LRA motor on Motor+/Motor-
"""

import time
import board
import busio
import adafruit_drv2605

# ── Register addresses (from DRV2605L datasheet) ──────────────────────────
REG_MODE           = 0x01
REG_RTPIN          = 0x02
REG_LIBRARY        = 0x03
REG_GO             = 0x0C
REG_RATED_V        = 0x16
REG_CLAMP_V        = 0x17
REG_AUTOCAL_COMP   = 0x18
REG_AUTOCAL_BEMF   = 0x19
REG_FEEDBACK       = 0x1A
REG_CONTROL1       = 0x1B
REG_CONTROL2       = 0x1C
REG_CONTROL3       = 0x1D
REG_CONTROL4       = 0x1E
REG_CONTROL5       = 0x1F
REG_OL_LRA_PERIOD  = 0x20
REG_VBAT           = 0x21
REG_LRA_PERIOD     = 0x22   # Read-only: measured resonance period
REG_STATUS         = 0x00


# ── Low-level helpers ──────────────────────────────────────────────────────

def write_reg(drv, reg, val):
    """Write a single byte to a DRV2605 register."""
    drv._device.write(bytes([reg, val & 0xFF]))

def read_reg(drv, reg):
    """Read a single byte from a DRV2605 register."""
    buf = bytearray(1)
    drv._device.write_then_readinto(bytes([reg]), buf)
    return buf[0]

def period_to_freq(period_reg_val):
    """Convert LRA_RESONANCE_PERIOD register value to Hz.
    Each LSB of register 0x22 = 98.46 us.
    Frequency = 1 / (period_reg_val * 98.46e-6)
    """
    if period_reg_val == 0:
        return 0.0
    return 1.0 / (period_reg_val * 98.46e-6)

def freq_to_ol_period(freq_hz):
    """Convert target frequency (Hz) to OL_LRA_PERIOD register value.
    Register = 1 / (freq_hz * 98.46e-6).  7-bit field (0-127).
    """
    if freq_hz == 0:
        return 0
    val = round(1.0 / (freq_hz * 98.46e-6))
    return max(0, min(127, val))


# ══════════════════════════════════════════════════════════════════════════
# METHOD 1: Auto-Calibration
# ══════════════════════════════════════════════════════════════════════════

def auto_calibrate(drv):
    """
    Run the DRV2605L auto-calibration procedure for LRA.
    The chip determines compensation, back-EMF gain, and measures
    resonance frequency. (Datasheet section 8.5.6)
    """
    print("\n" + "=" * 60)
    print("METHOD 1: DRV2605 Auto-Calibration for LRA")
    print("=" * 60)

    # 1. Set LRA mode in FEEDBACK register (bit 7 = 1)
    fb = read_reg(drv, REG_FEEDBACK)
    fb = 0x80 | 0x14   # LRA | brake_factor=2x | loop_gain=medium
    write_reg(drv, REG_FEEDBACK, fb)
    print(f"  FEEDBACK (0x1A) = 0x{fb:02X}  [LRA mode]")

    # 2. Set rated voltage — adjust for YOUR LRA's datasheet value
    #    Formula: V_rated = register_val * 21.33e-3
    rated_v = 0x56   # ~1.8 Vrms — safe starting point
    write_reg(drv, REG_RATED_V, rated_v)
    print(f"  RATED_VOLTAGE (0x16) = 0x{rated_v:02X}  (~{rated_v * 21.33e-3:.2f} Vrms)")

    # 3. Set overdrive clamp voltage
    clamp_v = 0xA5   # ~3.5V
    write_reg(drv, REG_CLAMP_V, clamp_v)
    print(f"  CLAMP_VOLTAGE (0x17) = 0x{clamp_v:02X}  (~{clamp_v * 21.33e-3:.2f} V)")

    # 4. Set auto-cal time to 1000ms for best accuracy
    ctrl4 = read_reg(drv, REG_CONTROL4)
    ctrl4 = (ctrl4 & 0xFC) | 0x03   # bits 1:0 = 11 -> 1000ms
    write_reg(drv, REG_CONTROL4, ctrl4)
    print(f"  CONTROL4 (0x1E) = 0x{ctrl4:02X}  [cal time = 1000ms]")

    # 5. Set initial open-loop period guess (~160 Hz typical LRA)
    initial_freq = 160
    ol_period = freq_to_ol_period(initial_freq)
    write_reg(drv, REG_OL_LRA_PERIOD, ol_period)
    print(f"  OL_LRA_PERIOD (0x20) = 0x{ol_period:02X}  (guess ~{initial_freq} Hz)")

    # 6. Enter auto-calibration mode and GO
    write_reg(drv, REG_MODE, 0x07)
    write_reg(drv, REG_GO, 0x01)
    print("  Calibrating...")

    # 7. Wait for GO bit to clear
    for _ in range(30):
        time.sleep(0.1)
        if read_reg(drv, REG_GO) == 0:
            break
    else:
        print("  WARNING: Calibration timed out!")

    # 8. Check result
    status = read_reg(drv, REG_STATUS)
    diag_result = (status >> 3) & 0x01

    if diag_result == 0:
        print("  >>> Auto-calibration PASSED")
    else:
        print("  >>> Auto-calibration FAILED (check motor wiring / voltages)")

    # 9. Read calibration outputs
    comp = read_reg(drv, REG_AUTOCAL_COMP)
    bemf = read_reg(drv, REG_AUTOCAL_BEMF)
    fb_after = read_reg(drv, REG_FEEDBACK)
    bemf_gain = fb_after & 0x03
    print(f"  Compensation (0x18) = 0x{comp:02X}")
    print(f"  Back-EMF     (0x19) = 0x{bemf:02X}")
    print(f"  BEMF gain           = {bemf_gain}")

    # 10. Drive motor briefly to read resonance period register
    print("  Driving motor to read resonance register...")
    write_reg(drv, REG_MODE, 0x00)
    write_reg(drv, REG_LIBRARY, 0x06)
    drv.sequence[0] = adafruit_drv2605.Effect(1)
    drv.play()
    time.sleep(0.3)

    period = read_reg(drv, REG_LRA_PERIOD)
    freq = period_to_freq(period)
    drv.stop()

    print(f"\n  +--------------------------------------+")
    print(f"  |  LRA_RESONANCE_PERIOD (0x22) = {period:3d}   |")
    print(f"  |  Measured resonance  = {freq:6.1f} Hz     |")
    print(f"  +--------------------------------------+")

    write_reg(drv, REG_MODE, 0x00)
    return freq


# ══════════════════════════════════════════════════════════════════════════
# METHOD 2: Diagnostics Mode
# ══════════════════════════════════════════════════════════════════════════

def run_diagnostics(drv):
    """Quick diagnostics — checks motor connection and reads resonance."""
    print("\n" + "=" * 60)
    print("METHOD 2: Diagnostics Mode")
    print("=" * 60)

    fb = read_reg(drv, REG_FEEDBACK)
    fb |= 0x80
    write_reg(drv, REG_FEEDBACK, fb)

    write_reg(drv, REG_MODE, 0x06)
    write_reg(drv, REG_GO, 0x01)

    for _ in range(20):
        time.sleep(0.1)
        if read_reg(drv, REG_GO) == 0:
            break

    status = read_reg(drv, REG_STATUS)
    diag_result = (status >> 3) & 0x01
    if diag_result == 0:
        print("  >>> Diagnostics PASSED — motor OK")
    else:
        print("  >>> Diagnostics FAILED — check wiring")

    period = read_reg(drv, REG_LRA_PERIOD)
    freq = period_to_freq(period)
    print(f"  LRA_RESONANCE_PERIOD = {period}, freq ~ {freq:.1f} Hz")

    write_reg(drv, REG_MODE, 0x00)
    return freq


# ══════════════════════════════════════════════════════════════════════════
# METHOD 3: Open-Loop Frequency Sweep
# ══════════════════════════════════════════════════════════════════════════

def frequency_sweep(drv, start_hz=50, end_hz=300, step_hz=5, drive_time=0.4):
    """
    Drive the LRA in open-loop RTP mode at each frequency.
    Read back the resonance period register at each step.
    You'll FEEL the motor get strongest at resonance.
    """
    print("\n" + "=" * 60)
    print("METHOD 3: Open-Loop Frequency Sweep")
    print(f"  Range: {start_hz} - {end_hz} Hz, step {step_hz} Hz")
    print("  (feel for the strongest vibration!)")
    print("=" * 60)

    # LRA mode
    fb = read_reg(drv, REG_FEEDBACK)
    fb |= 0x80
    write_reg(drv, REG_FEEDBACK, fb)

    # Enable open-loop LRA (bit 0 of CONTROL3)
    ctrl3 = read_reg(drv, REG_CONTROL3)
    ctrl3 |= 0x01
    write_reg(drv, REG_CONTROL3, ctrl3)

    # RTP mode
    write_reg(drv, REG_MODE, 0x05)

    # Drive level (0-127 positive)
    write_reg(drv, REG_RTPIN, 100)

    results = []
    print(f"\n  {'Drive Hz':>10}  {'OL_Reg':>8}  {'LRA_Reg':>8}  {'Meas Hz':>10}")
    print(f"  {'---':>10}  {'---':>8}  {'---':>8}  {'---':>10}")

    freq = start_hz
    while freq <= end_hz:
        ol_period = freq_to_ol_period(freq)
        write_reg(drv, REG_OL_LRA_PERIOD, ol_period)

        time.sleep(drive_time)

        lra_period = read_reg(drv, REG_LRA_PERIOD)
        measured = period_to_freq(lra_period) if lra_period > 0 else 0

        results.append({
            'target': freq,
            'ol_period': ol_period,
            'lra_period': lra_period,
            'measured': measured,
        })

        print(f"  {freq:>10.1f}  {ol_period:>8d}  {lra_period:>8d}  {measured:>10.1f}")
        freq += step_hz

    # Stop
    write_reg(drv, REG_RTPIN, 0)
    write_reg(drv, REG_MODE, 0x00)

    print("\n  Sweep done. The frequency where vibration was strongest")
    print("  is your LRA resonance frequency.")
    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("DRV2605L LRA Resonance Frequency Finder")
    print("-" * 40)

    i2c = busio.I2C(board.SCL, board.SDA)
    drv = adafruit_drv2605.DRV2605(i2c)
    drv.use_LRM()
    print("Configured: LRA mode\n")

    try:
        # 1) Auto-calibration
        cal_freq = auto_calibrate(drv)
        time.sleep(1)

        # 2) Diagnostics
        diag_freq = run_diagnostics(drv)
        time.sleep(1)

        # 3) Open-loop sweep (narrow range around auto-cal result)
        if cal_freq > 0:
            sweep_lo = max(30, cal_freq - 80)
            sweep_hi = min(400, cal_freq + 80)
        else:
            sweep_lo, sweep_hi = 50, 300

        results = frequency_sweep(drv, start_hz=sweep_lo, end_hz=sweep_hi, step_hz=5)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Auto-cal resonance:   {cal_freq:.1f} Hz")
        print(f"  Diagnostics reading:  {diag_freq:.1f} Hz")
        print(f"  Sweep range tested:   {sweep_lo:.0f} - {sweep_hi:.0f} Hz")
        print(f"\n  >>> Your LRA resonance is ~{cal_freq:.0f} Hz")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nStopping motor...")
        write_reg(drv, REG_RTPIN, 0)
        write_reg(drv, REG_MODE, 0x00)
        drv.stop()
