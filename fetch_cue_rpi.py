#!/usr/bin/env python3
"""
TTL edge logger (hardened, simple defaults).

- Logs both rising (+) and falling (-) edges.
- Suppresses duplicate same-polarity edges per channel (A or B): +,+ or -,- are dropped until the opposite edge arrives.
- Maximum durability by default: every row is flushed and fdatasync/fsync'ed.
- Console shows real (host) ISO8601 timestamps with timezone offset.
- Clean shutdown on Ctrl+C / SIGTERM with guaranteed final flush.

CSV schema:
  channel,edge,stamp_ticks,recv_time_iso,recv_perf_s,label

`recv_time_iso` is the host wall clock (timezone-aware ISO8601) and
`recv_perf_s` is this Pi's local `clock_gettime(CLOCK_MONOTONIC_RAW)` in
seconds, captured next to it. RAW is used (not perf_counter / CLOCK_MONOTONIC)
so the value reflects the SoC oscillator without NTP rate-slewing — needed
for cross-Pi clock-skew measurement. The same column names are used in
cue_experiment.py's cue_log so downstream alignment code sees a single
naming convention across both Pis.
"""

import argparse
import csv
import os
import sys
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import serial
from serial.tools import list_ports

import socket
UDP_MARKER_PORT = 5005
UDP_PING_PORT   = 5006   # port this Pi listens on for latency pings from rpi-stim
PING_K          = 42     # modular-addition constant verifying the pong matches this round
PING_MOD        = 65536  # modulus for both nonce and (nonce+K)

BAUD = 115200
running = True  # flipped to False by signal handler to trigger shutdown


# ------------------------ Signals ------------------------

def signal_handler(sig, frame):
    global running
    if running:
        running = False
        print("Signal received, shutting down...", file=sys.stderr)

# Register early
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ------------------------ Helpers ------------------------

def perf_counter_raw() -> float:
    """Monotonic seconds from the SoC oscillator, NOT NTP-slewed.

    time.perf_counter() on Linux is clock_gettime(CLOCK_MONOTONIC), which is
    rate-disciplined by NTP — its tick rate tracks the upstream consensus
    rather than the local hardware. CLOCK_MONOTONIC_RAW reads the same
    underlying timekeeper but bypasses NTP slewing, so cross-device elapsed
    time reflects the actual hardware oscillator skew between Pis.
    """
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def auto_detect_port():
    """
    Best-effort auto-detect:
      - Prefer ports whose description mentions Arduino/Due.
      - On Linux/macOS fall back to /dev/ttyACM* or /dev/ttyUSB*.
      - On Windows/others, only auto-pick if exactly one candidate exists.
    """
    # Pass 1: look for Arduino/Due in description
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        if "arduino" in desc or "due" in desc:
            return p.device

    # Pass 2: platform heuristics
    candidates = []
    for p in list_ports.comports():
        if sys.platform.startswith(("linux", "darwin")):
            if p.device.startswith("/dev/ttyACM") or p.device.startswith("/dev/ttyUSB"):
                candidates.append(p.device)
        else:
            candidates.append(p.device)

    if len(candidates) == 1:
        return candidates[0]
    return None


def parse_line(line: str):
    """
    Parses lines like:
      A+ 0:123456789
      B- 0:987654321
      A 123                (legacy/simple)
    Returns (channel, edge, stamp) or None.
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    raw_label = parts[0]
    if not raw_label:
        return None
    channel = raw_label[0]
    if channel not in ("A", "B"):
        return None
    edge = raw_label[1] if len(raw_label) > 1 else ""
    if edge and edge not in ("+", "-"):
        edge = ""  # treat unknown suffix as legacy/no-edge label

    token = parts[1]
    try:
        if ":" in token:
            hi, lo = token.split(":", 1)
            stamp = (int(hi) << 32) | (int(lo) & 0xFFFFFFFF)
        else:
            stamp = int(token)
    except ValueError:
        return None
    return channel, edge, stamp


class AlternationFilter:
    """
    Enforces alternating edges per channel: +,-,+,-,...
    Drops duplicate same-polarity edges until the opposite arrives.
    """
    def __init__(self):
        self._last = {"A": None, "B": None}

    def accept(self, channel: str, edge: str) -> bool:
        if edge not in ("+", "-"):
            return True  # legacy/no-edge doesn't affect state
        last = self._last.get(channel)
        if last == edge:
            return False
        self._last[channel] = edge
        return True

class UDPMarkerThread(threading.Thread):
    """Listens for UDP markers from cue_experiment.py and enqueues them.

    Latency measurement now lives on the rpi-stim side: cue_experiment.py runs
    a three-round ping-pong before each marker and logs the RTTs in its own
    log file. This thread just records the marker receive time (ISO + perf).
    See PingResponderThread for the counterpart that answers those pings.
    """

    def __init__(self, queue: Queue, port: int = UDP_MARKER_PORT):
        super().__init__(daemon=True)
        self.queue = queue
        self.port = port

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", self.port))
        print(f"[udp] Marker listener on :{self.port}", file=sys.stderr)
        while running:
            try:
                data, _ = sock.recvfrom(64)
            except socket.timeout:
                continue
            label = data.decode("ascii", errors="ignore").strip()
            if not label:
                continue

            recv_time_iso = datetime.now().astimezone().isoformat(timespec="microseconds")
            recv_perf_s   = perf_counter_raw()
            try:
                self.queue.put_nowait(("M", "", 0, recv_time_iso, recv_perf_s, label))
            except Exception:
                print(f"[udp] queue full, dropped: {label}", file=sys.stderr)
            print(f"M\t {recv_time_iso}  [{label}]", flush=True)
        sock.close()


class PingResponderThread(threading.Thread):
    """Answers latency pings from rpi-stim (cue_experiment.py).

    Protocol:
      stim  → fetch : "PING:<nonce>:<marker_id>"
      fetch → stim  : "PONG:<(nonce+K) mod M>:<marker_id>"
    The (nonce+K) mod M check lets the initiator verify the pong belongs to
    the current round, not a leftover from a previous one.
    """

    def __init__(self, port: int = UDP_PING_PORT):
        super().__init__(daemon=True)
        self.port = port

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", self.port))
        print(f"[udp] Ping responder on :{self.port}", file=sys.stderr)
        while running:
            try:
                data, addr = sock.recvfrom(128)
            except socket.timeout:
                continue
            try:
                parts = data.decode("ascii", errors="ignore").strip().split(":", 2)
                if len(parts) < 3 or parts[0] != "PING":
                    continue
                nonce = int(parts[1])
                marker_id = parts[2]
                result = (nonce + PING_K) % PING_MOD
                sock.sendto(f"PONG:{result}:{marker_id}".encode("ascii"), addr)
            except Exception:
                continue
        sock.close()



# ------------------------ Writer Thread ------------------------

class WriterThread(threading.Thread):
    """
    Dedicated writer thread:
      - Batches rows collected from a queue.
      - Flushes and fdatasync/fsyncs on every flush (max durability).
      - Uses queue.task_done()/join() so shutdown is lossless.
    """
    def __init__(self, csvfile, writer, queue: Queue):
        super().__init__(daemon=False)
        self.csvfile = csvfile
        self.writer = writer
        self.queue = queue

    def _sync(self):
        fd = self.csvfile.fileno()
        if hasattr(os, "fdatasync"):
            os.fdatasync(fd)
        else:
            os.fsync(fd)

    def run(self):
        pending = []
        while True:
            got_item = False
            # While running, block briefly for new items; when stopping, drain immediately
            if running:
                try:
                    item = self.queue.get(timeout=0.1)
                    pending.append(item)
                    got_item = True
                except Empty:
                    pass
            else:
                try:
                    while True:
                        item = self.queue.get_nowait()
                        pending.append(item)
                        got_item = True
                except Empty:
                    pass

            # Flush policy: maximum safety → flush whenever we have anything
            # Tuple shape: (channel, edge, stamp_ticks, recv_time_iso, recv_perf_s, label)
            if pending:
                try:
                    self.writer.writerows(
                        ([c, e, s, ri, f"{rp:.6f}", l] for (c, e, s, ri, rp, l) in pending)
                    )
                    self.csvfile.flush()
                    self._sync()
                except Exception as e:
                    print(f"[writer] write/flush/sync error: {e}", file=sys.stderr)
                    # Don't task_done() yet; keep pending for retry
                else:
                    for _ in range(len(pending)):
                        self.queue.task_done()
                    pending.clear()

            # Exit when asked to stop and nothing is left
            if (not running) and (not pending) and self.queue.empty():
                break

            # Small sleep if nothing was processed to avoid busy-wait
            if not got_item:
                time.sleep(0.01)


# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description="TTL edge logger (durable, duplicate-filtered).")
    parser.add_argument("-p", "--port", help="Serial port (auto-detected if omitted)")
    parser.add_argument("--no-serial", action="store_true",
                        help="Skip Arduino/serial entirely. UDP marker listener "
                             "+ ping responder still run, so the CSV gets only "
                             "M rows. Use on a Pi without the TTL hardware to "
                             "test the network marker pipeline. Does NOT "
                             "auto-engage when serial is missing — opt in.")
    parser.add_argument("outfile", nargs="?", help="CSV output file (default timestamps_YYYYMMDD_HHMMSS.csv)")
    args = parser.parse_args()

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    outname = args.outfile or f"timestamps_{timestamp}.csv"
    out_path = Path(outname).expanduser().resolve()

    ser = None
    if args.no_serial:
        print(f"[main] --no-serial: skipping Arduino. UDP-only mode; "
              f"logging to {out_path}", file=sys.stderr)
    else:
        port = args.port or auto_detect_port()
        if not port:
            sys.exit("Error: could not auto-detect serial port; specify with -p "
                     "(or pass --no-serial for UDP-only operation).")

        print(f"[main] Opening serial port {port} @ {BAUD} baud; logging to {out_path}", file=sys.stderr)

        # Open serial port with retries (1s timeout → quick Ctrl+C)
        backoff = 1.0
        while running:
            try:
                ser = serial.Serial(port, BAUD, timeout=1)
                break
            except serial.SerialException as e:
                print(f"[main] Serial open failed: {e}; retrying in {backoff:.1f}s", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)

        if ser is None:
            sys.exit("Failed to open serial port.")

    queue = Queue(maxsize=10000)
    alt_filter = AlternationFilter()

    # Helper to sync the header immediately (so a new file is durable too)
    def _sync_file(fobj):
        fd = fobj.fileno()
        if hasattr(os, "fdatasync"):
            os.fdatasync(fd)
        else:
            os.fsync(fd)

    with out_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["channel", "edge", "stamp_ticks", "recv_time_iso", "recv_perf_s", "label"])
            csvfile.flush()
            _sync_file(csvfile)

        wt = WriterThread(csvfile=csvfile, writer=writer, queue=queue)
        wt.start()

        umt = UDPMarkerThread(queue=queue)
        umt.start()

        prt = PingResponderThread()
        prt.start()
        
        print("\r\r")

        try:
            # --no-serial branch: park the main thread; UDPMarkerThread +
            # PingResponderThread keep doing all the work.
            if ser is None:
                while running:
                    time.sleep(0.2)
            else:
                while running:
                    try:
                        raw = ser.readline()
                    except serial.SerialException as e:
                        print(f"[main] Serial error: {e}; attempting reconnect...", file=sys.stderr)
                        try:
                            ser.close()
                        except Exception:
                            pass
                        time.sleep(1)
                        try:
                            ser = serial.Serial(port, BAUD, timeout=1)
                        except Exception as e2:
                            print(f"[main] Reopen failed: {e2}", file=sys.stderr)
                            time.sleep(1)
                        continue

                    if not raw:
                        continue

                    try:
                        line = raw.decode("ascii", errors="ignore").strip()
                    except Exception:
                        continue
                    if not line:
                        continue

                    if line.startswith("#"):
                        print(f"[device] {line}", file=sys.stderr)
                        continue

                    parsed = parse_line(line)
                    if not parsed:
                        print(f"[main] Unparsed line: {line}", file=sys.stderr)
                        continue

                    channel, edge, stamp = parsed

                    # Suppress duplicate same-polarity edges per channel
                    if not alt_filter.accept(channel, edge):
                        # Uncomment for debugging:
                        # print(f"[filter] Suppressed duplicate edge {channel}{edge}", file=sys.stderr)
                        continue

                    recv_time_iso = datetime.now().astimezone().isoformat(timespec="microseconds")
                    recv_perf_s   = perf_counter_raw()

                    try:
                        queue.put_nowait((channel, edge, stamp, recv_time_iso, recv_perf_s, ""))
                    except Exception:
                        print(f"[main] Warning: queue full, dropping event {channel}{edge} {stamp}", file=sys.stderr)
                        continue

                    # Console echo: human time + edge label
                    print(f"{channel}{edge}\t {recv_time_iso}", flush=True)
                    if(edge == '-'):
                        print("\r", flush=True)

        finally:
            print("[main] Waiting for writer to flush...", file=sys.stderr)
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            # Wait for all queued items to be durably written
            try:
                queue.join()
            except Exception:
                pass
            wt.join()
            print("[main] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
