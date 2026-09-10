#!/usr/bin/env python3
"""Serial link to the Arduino head, with no ROS in the way.

Framing, CRC and the command encoders come from the shared gaze core
(astro_base.gaze.head_controller) so this file cannot drift from what the ROS
bridge sends. What is genuinely new here is reading the stream back: the bridge did
that inside a node, tangled up with publishers.

Two facts from arduino/astro_firmware/src/main.cpp shape this:

  * the firmware disables the motors after 500 ms with no packet, so a link that
    goes quiet silently stops the robot — hence the heartbeat;
  * ENCODER_TICKS carries the absolute head position, which is the one measurement
    that keeps bearings honest. Without it the stack assumes the head sits at zero
    while it physically turns, and tracking collapses to centre.
"""

import struct
import time
from typing import Iterable, List, Optional, Tuple

import core_path  # noqa: F401  (puts the shared core on the path)
from astro_base.gaze.head_controller import (  # noqa: E402
    MSG_ENCODER_TICKS,
    MSG_HEAD_CMD,
    MSG_HEARTBEAT,
    SOF1,
    SOF2,
    build_packet,
    crc8,
)

# The firmware's watchdog is 500 ms; leave most of that as margin.
HEARTBEAT_INTERVAL_S = 0.2

# 440 ticks over 170 degrees, verified in calibration_params.yaml.
TICKS_PER_DEG = 2.5882

Packet = Tuple[int, bytes]


def encode_head_cmd(angle_deg: float) -> bytes:
    """A HEAD_CMD packet. The firmware also treats it as a heartbeat."""
    return build_packet(MSG_HEAD_CMD, struct.pack("<f", float(angle_deg)))


def encode_heartbeat(seq: int) -> bytes:
    return build_packet(MSG_HEARTBEAT, struct.pack("<I", int(seq) & 0xFFFFFFFF))


def head_degrees_from_encoder_payload(payload: bytes) -> float:
    """ENCODER_TICKS is <int32 dl, int32 dr, int32 head_ticks, uint32 dt_us>."""
    _dl, _dr, head_ticks, _dt = struct.unpack("<iiiI", payload[:16])
    return round(head_ticks / TICKS_PER_DEG, 2)


def parse_packets(stream: bytes, return_remainder: bool = False):
    """Pulls whole, CRC-valid packets out of a byte stream.

    Opening a port mid-transmission drops you into the middle of a packet, so
    anything that does not frame correctly is skipped a byte at a time rather than
    derailing everything after it. A trailing partial packet is handed back so the
    caller can prepend it to the next read.
    """
    packets: List[Packet] = []
    i, n = 0, len(stream)

    while i < n:
        if stream[i] != SOF1:
            i += 1
            continue
        if i + 1 >= n:
            break
        if stream[i + 1] != SOF2:
            i += 1
            continue
        if i + 2 >= n:
            break

        length = stream[i + 2]
        end = i + 3 + length  # LEN counts msg_id + payload; CRC sits at `end`
        if end >= n:
            break

        body = stream[i + 2 : end]
        if crc8(body) != stream[end]:
            i += 1  # a false SOF pair inside data; resync from the next byte
            continue

        packets.append((stream[i + 3], bytes(stream[i + 4 : end])))
        i = end + 1

    if return_remainder:
        return packets, bytes(stream[i:])
    return packets


def open_port(device: str, baud: int = 115200, timeout: float = 0.0):
    """Opens the serial device, or returns None so the caller can run open-loop."""
    try:
        import serial
    except ImportError:
        print("⚠️  pyserial kurulu değil — kafa komutu gönderilmeyecek")
        return None
    try:
        return serial.Serial(device, baud, timeout=timeout)
    except Exception as exc:
        print(f"⚠️  {device} açılamadı ({exc}) — kafa komutu gönderilmeyecek")
        return None


class HeadLink:
    """Sends angles to the head and reads back where it actually is."""

    def __init__(self, port=None):
        self.port = port
        self._rx = bytearray()
        self._seq = 0
        self._last_beat: Optional[float] = None
        self.measured_angle_deg: float = 0.0
        # Distinguished from "measured 0.0" on purpose: assuming zero while the head
        # is elsewhere is what makes every bearing collapse to centre.
        self.has_feedback: bool = False

    @property
    def connected(self) -> bool:
        return self.port is not None

    def send_angle(self, angle_deg: float) -> None:
        if self.port is None:
            return
        self.port.write(encode_head_cmd(angle_deg))
        self._last_beat = time.monotonic()

    def tick(self, now: Optional[float] = None) -> None:
        """Keeps the firmware watchdog fed between commands."""
        if self.port is None:
            return
        now = time.monotonic() if now is None else now
        if self._last_beat is not None and (now - self._last_beat) < HEARTBEAT_INTERVAL_S:
            return
        self._seq += 1
        self.port.write(encode_heartbeat(self._seq))
        self._last_beat = now

    def poll(self) -> None:
        """Drains the port and keeps the most recent encoder angle."""
        if self.port is None:
            return
        waiting = getattr(self.port, "in_waiting", 0)
        if waiting:
            self._rx.extend(self.port.read(waiting))

        packets, remainder = parse_packets(bytes(self._rx), return_remainder=True)
        self._rx = bytearray(remainder)

        for msg_id, payload in packets:
            if msg_id == MSG_ENCODER_TICKS and len(payload) >= 16:
                self.measured_angle_deg = head_degrees_from_encoder_payload(payload)
                self.has_feedback = True

    def close(self) -> None:
        if self.port is not None:
            try:
                self.port.close()
            except Exception:
                pass
