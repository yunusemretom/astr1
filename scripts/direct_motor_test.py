#!/usr/bin/env python3
"""ASTRO Robot — Direct Head Actuator & Low-Level Serial Diagnostic Tool.

Directly communicates with Arduino Mega 2560 via binary serial protocol to:
  1. Measure exact raw encoder ticks before, during, and after movement
  2. Test head angle commands directly without ROS 2 middle layers
  3. Inspect real-time MCU diagnostic flags (STALL, LIMIT, WATCHDOG)
  4. Verify encoder counting polarity and minimum movement torque
"""

import argparse
import glob
import os
import struct
import sys
import threading
import time
import serial

SOF1 = 0xAA
SOF2 = 0x55

MSG_HEARTBEAT = 0x01
MSG_WHEEL_CMD = 0x02
MSG_HEAD_CMD = 0x03
MSG_ENCODER_TICKS = 0x11
MSG_DIAGNOSTICS = 0x12
MSG_HEARTBEAT_ACK = 0x13

FLAG_WATCHDOG_TIMEOUT = 0x01
FLAG_HEAD_STALL = 0x04
FLAG_HEAD_LIMIT = 0x08


def crc8(data: bytes) -> int:
    poly = 0x07
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ poly
            else:
                crc = (crc << 1) & 0xFF
    return crc


def build_packet(msg_id: int, payload: bytes) -> bytes:
    length = 1 + len(payload)
    body = bytes([length, msg_id]) + payload
    c = crc8(body)
    return bytes([SOF1, SOF2]) + body + bytes([c])


def find_arduino_port() -> str:
    if os.path.exists("/dev/astro_arduino"):
        return "/dev/astro_arduino"
    for pattern in ("/dev/ttyCH341USB*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return "/dev/ttyUSB0"


class DirectHeadDiagnostic:
    def __init__(self, port: str = None, baud: int = 115200):
        self.port = port or find_arduino_port()
        self.baud = baud
        self.ser = None
        self.running = True
        self.arduino_alive = False
        self.last_hb_ack = 0.0
        self.hb_seq = 0
        self.tx_lock = threading.Lock()

        # Telemetry
        self.head_ticks = 0
        self.diag_flags = 0
        self.vbat_mv = 0
        self.mcu_temp = 0.0

    def connect(self) -> bool:
        print(f"🔌 Connecting to Arduino: {self.port} @ {self.baud} baud...")
        try:
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.05,
                write_timeout=0.05,
                rtscts=False,
                dsrdtr=False,
            )
            time.sleep(2.0)  # Arduino bootloader wait
            self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self.rx_thread.start()
            self.hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
            self.hb_thread.start()

            # Wait for handshake
            for _ in range(30):
                if self.arduino_alive:
                    print("✅ Arduino Connected & Heartbeat Healthy.")
                    return True
                time.sleep(0.1)
            print("❌ Heartbeat ACK timeout.")
            return False
        except Exception as exc:
            print(f"❌ Serial connect error: {exc}")
            return False

    def _hb_loop(self):
        while self.running and self.ser and self.ser.is_open:
            self.hb_seq += 1
            payload = struct.pack("<I", self.hb_seq)
            pkt = build_packet(MSG_HEARTBEAT, payload)
            with self.tx_lock:
                try:
                    self.ser.write(pkt)
                except Exception:
                    pass
            time.sleep(0.05)

    def _rx_loop(self):
        state = 0
        exp_len = 0
        buf = bytearray()

        while self.running and self.ser and self.ser.is_open:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if not chunk:
                    continue
                for b in chunk:
                    if state == 0:
                        if b == SOF1: state = 1
                    elif state == 1:
                        if b == SOF2: state = 2
                        elif b != SOF1: state = 0
                    elif state == 2:
                        exp_len = b
                        buf = bytearray([exp_len])
                        state = 3
                    elif state == 3:
                        buf.append(b)
                        if len(buf) == exp_len + 1:
                            c = buf[-1]
                            body = buf[:-1]
                            if crc8(body) == c:
                                msg_id = body[1]
                                payload = body[2:]
                                self._handle_packet(msg_id, payload)
                            state = 0
            except Exception:
                break

    def _handle_packet(self, msg_id: int, payload: bytes):
        if msg_id == MSG_HEARTBEAT_ACK:
            self.arduino_alive = True
            self.last_hb_ack = time.monotonic()
        elif msg_id == MSG_ENCODER_TICKS:
            if len(payload) >= 12:
                _, _, ht = struct.unpack("<iii", payload[:12])
                self.head_ticks = ht
        elif msg_id == MSG_DIAGNOSTICS:
            if len(payload) >= 8:
                vbat, temp, flags = struct.unpack("<HhI", payload[:8])
                self.vbat_mv = vbat
                self.mcu_temp = temp / 100.0
                self.diag_flags = flags

    def send_head_cmd(self, angle_deg: float):
        payload = struct.pack("<f", float(angle_deg))
        pkt = build_packet(MSG_HEAD_CMD, payload)
        with self.tx_lock:
            try:
                self.ser.write(pkt)
            except Exception as exc:
                print(f"Write error: {exc}")

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()


def run_test(angle_deg: float, duration_s: float = 4.0):
    diag = DirectHeadDiagnostic()
    if not diag.connect():
        sys.exit(1)

    try:
        time.sleep(0.5)
        start_ticks = diag.head_ticks
        print(f"\n▶ [DIRECT HEAD TEST] Commanding angle: {angle_deg:+.1f}° for {duration_s}s")
        print(f"  Initial Encoder Ticks = {start_ticks} ({start_ticks / 2.5882:.2f}°)")
        print(f"  Expected Target Ticks (2.5882 scale) = {round(angle_deg * 2.5882)} ticks")

        print(f"\n{'Time (s)':<10} | {'Cmd (°)':<10} | {'Ticks':<10} | {'Delta':<10} | {'Calc Angle':<12} | {'Flags':<12} | {'Stall':<8}")
        print("-" * 80)

        t0 = time.monotonic()
        last_t = 0.0
        while time.monotonic() - t0 < duration_s:
            t_now = time.monotonic() - t0
            diag.send_head_cmd(angle_deg)
            time.sleep(0.05)

            if t_now - last_t >= 0.10:
                last_t = t_now
                cur_ticks = diag.head_ticks
                delta = cur_ticks - start_ticks
                calc_angle = cur_ticks / 2.5882
                is_stall = bool(diag.diag_flags & FLAG_HEAD_STALL)
                flags_hex = hex(diag.diag_flags)
                print(f"{t_now:<10.2f} | {angle_deg:<10.1f} | {cur_ticks:<10} | {delta:<10} | {calc_angle:<12.2f} | {flags_hex:<12} | {str(is_stall):<8}")

        final_ticks = diag.head_ticks
        final_delta = final_ticks - start_ticks
        print("-" * 80)
        print(f"Summary: Initial Ticks = {start_ticks} -> Final Ticks = {final_ticks} (Delta = {final_delta} ticks)")
        print(f"Final Reported Angle = {final_ticks / 2.5882:.2f}° (Target = {angle_deg:+.1f}°)")
        print(f"Final Diag Flags = {hex(diag.diag_flags)} (Stall={bool(diag.diag_flags & FLAG_HEAD_STALL)})")

    finally:
        diag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Head Actuator Diagnostic")
    parser.add_argument("--angle", type=float, default=5.0, help="Target angle in degrees")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds")
    args = parser.parse_args()
    run_test(args.angle, args.duration)
