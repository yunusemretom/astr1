#!/usr/bin/env python3
"""ASTRO Robot - Isolated Head Actuator Calibration & Step Verification Tool.

Directly tests the serial bridge, Arduino Mega 2560 firmware PID, and optical encoder
without running audio, vision, or gaze FSM nodes.
"""

import math
import struct
import sys
import time
import serial
import glob
import os

import threading

SOF1 = 0xAA
SOF2 = 0x55
MSG_HEAD_CMD = 0x03
MSG_ENCODER_TICKS = 0x11
MSG_HEARTBEAT = 0x01
MSG_HEARTBEAT_ACK = 0x13



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


def find_serial_port():
    patterns = ["/dev/ttyCH341USB*", "/dev/ttyUSB*", "/dev/ttyACM*"]
    for pat in patterns:
        matches = glob.glob(pat)
        for m in matches:
            if "lidar" not in m:
                return m
    return "/dev/ttyCH341USB0"


class HeadCalibrator:
    def __init__(self, port: str = None, baud: int = 115200):
        self.port_name = port or find_serial_port()
        self.baud = baud
        self.ser = None
        self.ser_lock = threading.Lock()
        self.raw_head_ticks = 0
        self.received_packet_len = 0
        self.running = True
        self._heartbeat_active = False

    def connect(self):
        print(f"\n[1/3] Connecting to Arduino on {self.port_name} at {self.baud} baud...")
        try:
            self.ser = serial.Serial(self.port_name, self.baud, timeout=0.1)
            time.sleep(1.5)  # Clean Arduino boot
            print("  -> Connected successfully!")
            self._start_heartbeat_thread()
            # Check packet format
            time.sleep(0.5)
            self.read_telemetry(timeout_s=1.0)


            if self.received_packet_len == 12:
                print("\n" + "="*70)
                print("⚠️  UYARI: Arduino'da ESKİ firmware (12-byte paket) yüklü!")
                print("   Kafa enkoder telemetrisini okumak için Arduino'yu güncelleyin:")
                print("   cd ~/Desktop/astr1/arduino/astro_firmware && pio run -t upload")
                print("="*70 + "\n")
            elif self.received_packet_len == 16:
                print("  -> Firmware doğrulandı: 16-byte canlı kafa enkoder akışı AKTİF.")
        except Exception as e:
            print(f"  -> ERROR: Failed to open serial port {self.port_name}: {e}")
            sys.exit(1)

    def _start_heartbeat_thread(self):
        self._heartbeat_active = True
        def hb_loop():
            while self._heartbeat_active and self.ser and self.ser.is_open:
                try:
                    self.send_heartbeat()
                    time.sleep(0.10)
                except Exception:
                    break
        t = threading.Thread(target=hb_loop, daemon=True)
        t.start()

    def send_heartbeat(self):
        with self.ser_lock:
            if self.ser and self.ser.is_open:
                pkt = build_packet(MSG_HEARTBEAT, struct.pack("<I", 1))
                self.ser.write(pkt)
                self.ser.flush()

    def read_telemetry(self, timeout_s: float = 1.0) -> bool:
        """Reads latest encoder ticks from Arduino stream."""
        start = time.time()
        buf = bytearray()
        state = 0
        expected_len = 0

        while time.time() - start < timeout_s:
            if self.ser.in_waiting > 0:
                with self.ser_lock:
                    chunk = self.ser.read(self.ser.in_waiting)
                for b in chunk:
                    if state == 0:
                        if b == SOF1: state = 1
                    elif state == 1:
                        if b == SOF2: state = 2
                        elif b != SOF1: state = 0
                    elif state == 2:
                        expected_len = b or 1
                        buf = bytearray()
                        state = 3
                    elif state == 3:
                        buf.append(b)
                        if len(buf) >= expected_len:
                            state = 4
                    elif state == 4:
                        body = bytes([expected_len]) + bytes(buf)
                        c = crc8(body)
                        if c == b and len(buf) > 0:
                            msg_id = buf[0]
                            payload = bytes(buf[1:])
                            if msg_id == MSG_ENCODER_TICKS:
                                self.received_packet_len = len(payload)
                                if len(payload) == 16:
                                    dl, dr, head_ticks, dt_us = struct.unpack("<iiiI", payload)
                                    self.raw_head_ticks = head_ticks
                                    return True
                                elif len(payload) == 12:
                                    dl, dr, dt_us = struct.unpack("<iiI", payload)
                                    return True
                        state = 1 if b == SOF1 else 0
            time.sleep(0.01)
        return False

    def command_angle(self, angle_deg: float):
        """Sends binary head angle command."""
        payload = struct.pack("<f", float(angle_deg))
        pkt = build_packet(MSG_HEAD_CMD, payload)
        pkt_hex = " ".join(f"{b:02X}" for b in pkt)
        print(f"  -> Sending UART packet: {pkt_hex} (Target: {angle_deg:+.2f}°)")
        with self.ser_lock:
            if self.ser and self.ser.is_open:
                for _ in range(3):
                    self.ser.write(pkt)
                    self.ser.flush()
                    time.sleep(0.01)


    def run_live_monitor(self):
        """Displays real-time encoder stream."""

        print("\n========================================================")
        print("   CANLI ENKODER VE TELEMETRİ İZLEME")
        print("   (Çıkmak için Ctrl+C'ye basın)")
        print("========================================================")
        try:
            while True:
                self.read_telemetry(timeout_s=0.1)
                pkt_info = f"{self.received_packet_len} bytes"
                sys.stdout.write(f"\r[CANLI] Kafa Enkoder Ticki: {self.raw_head_ticks:8d} | Paket Boyutu: {pkt_info}   ")
                sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nCanlı izleme sonlandırıldı.")


    def run_multi_point_manual_calibration(self):
        """Measures raw encoder ticks across multiple physical reference angles with live interactive tracking."""
        print("\n========================================================")
        print("   MULTI-POINT LARGE-ANGLE ENCODER IDENTIFICATION")
        print("   (Motor is completely DISABLED. Move head by HAND)")
        print("========================================================")
        print("Instructions:")
        print("1. Set the physical head at TRUE MECHANICAL 0.0° (Center).")
        
        # Flush buffer and read live
        if self.ser: self.ser.reset_input_buffer()
        time.sleep(0.1)
        self.read_telemetry()
        
        input("Press [ENTER] when head is mechanically at 0.0° Center...")
        if self.ser: self.ser.reset_input_buffer()
        time.sleep(0.1)
        self.read_telemetry()
        zero_ticks = self.raw_head_ticks
        print(f"  -> Baseline 0.0° Reference Ticks = {zero_ticks}\n")

        test_points = [15.0, 30.0, 45.0, 90.0, -15.0, -30.0, -45.0, -90.0]
        recorded_data = [(0.0, 0)]  # (physical_angle, delta_ticks)

        print("For each target angle, manually rotate the head to the reference angle,")
        print("verify the ticks on screen, then press [ENTER] to record that point.")
        print("(Type 's' to skip any angle that is mechanically unreachable)\n")

        for angle in test_points:
            prompt = f"Align head to physical {angle:+.1f}° -> Press ENTER (or 's' to skip): "
            sys.stdout.write(prompt)
            sys.stdout.flush()
            
            user_in = input().strip().lower()
            if user_in == 's' or user_in == 'skip':
                continue

            # Flush OS serial buffer to get the absolute freshest packet
            if self.ser: self.ser.reset_input_buffer()
            time.sleep(0.08)
            self.read_telemetry(timeout_s=0.3)
            
            current_ticks = self.raw_head_ticks
            delta = current_ticks - zero_ticks
            recorded_data.append((angle, delta))
            print(f"  >>> RECORDED: Physical={angle:+.1f}° | Raw={current_ticks:d} | Delta={delta:+d} ticks <<<\n")


        if len(recorded_data) < 3:
            print("Not enough points recorded for regression.")
            return

        # Linear regression: ticks = a * angle + b
        n = len(recorded_data)
        sum_x = sum(pt[0] for pt in recorded_data)
        sum_y = sum(pt[1] for pt in recorded_data)
        sum_xx = sum(pt[0]**2 for pt in recorded_data)
        sum_xy = sum(pt[0]*pt[1] for pt in recorded_data)
        sum_yy = sum(pt[1]**2 for pt in recorded_data)

        denom = (n * sum_xx - sum_x**2)
        if abs(denom) < 1e-6:
            print("Regression calculation error: zero denominator.")
            return

        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y * sum_xx - sum_x * sum_xy) / denom

        # Calculate R^2 and residuals
        residuals = []
        for x, y in recorded_data:
            predicted = a * x + b
            residuals.append(y - predicted)

        ss_res = sum(r**2 for r in residuals)
        ss_tot = sum_yy - (sum_y**2 / n)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 1.0
        rmse = math.sqrt(ss_res / n)
        max_res = max(abs(r) for r in residuals)

        print("\n========================================================")
        print("   SYSTEM IDENTIFICATION RESULTS & STATISTICAL FIT")
        print("========================================================")
        print(f"Number of Points Recorded : {n}")
        print(f"Linear Fit Equation       : encoder_ticks = ({a:.4f}) * angle_deg + ({b:.2f})")
        print(f"Slope (ticks_per_degree)  : {abs(a):.4f} ticks / degree")
        print(f"Zero Offset Residual      : {b:+.2f} ticks")
        print(f"Correlation (R²)          : {r_squared:.6f}")
        print(f"RMSE                      : {rmse:.3f} ticks")
        print(f"Max Residual Error        : {max_res:.3f} ticks")

        # Sign verification
        sign_polarity = "POSITIVE (+)" if a > 0 else "NEGATIVE (-)"
        print(f"Actuator Direction Sign   : {sign_polarity} (Left/CCW generates positive ticks)")

        print("\n---------------- FINAL VERIFICATION TABLE ----------------")
        print("| Physical Head Angle | Raw Encoder Ticks | Model Predicted Ticks | Residual Error (Ticks) |")
        print("|--------------------:|------------------:|----------------------:|-----------------------:|")
        for x, y in recorded_data:
            pred = a * x + b
            err = y - pred
            print(f"| {x:+19.1f}° | {y:+17d} | {pred:+21.1f} | {err:+22.2f} |")

        print("\n---------------- UPDATED CONFIGURATION BLOCK ----------------")
        print("# Paste this directly into config/calibration_params.yaml:")
        print("head:")
        print(f"  ticks_per_deg: {abs(a):.4f}         # Measured via linear regression (R²={r_squared:.4f})")
        print(f"  zero_offset_deg: {(-b / a if abs(a) > 0 else 0.0):.2f}")
        print("  min_angle_deg: -90.0")
        print("  max_angle_deg: 90.0")
        print("\n// Paste this into AstroFirmware.ino & main.cpp:")
        print(f"static constexpr float HEAD_TICKS_PER_DEG = {abs(a):.4f}f;")

    def run_step_table_verification(self):
        """Runs safe micro-step commands (±2°, ±5°, ±10°) and builds verification table."""
        print("\n========================================================")
        print("   ISOLATED STEP COMMAND VERIFICATION TABLE")
        print("========================================================")
        test_angles = [0.0, 2.0, 0.0, -2.0, 0.0, 5.0, 0.0, -5.0, 0.0, 10.0, 0.0, -10.0, 0.0]

        print("\n| ROS cmd | Target Ticks | Encoder Before | Encoder After | Encoder Delta | Reported Angle | Status |")
        print("|--------:|-------------:|---------------:|--------------:|--------------:|---------------:|:-------|")

        for angle in test_angles:
            if self.ser: self.ser.reset_input_buffer()
            time.sleep(0.05)
            self.read_telemetry(timeout_s=0.2)
            enc_before = self.raw_head_ticks

            # Send command
            self.command_angle(angle)
            calc_ticks = round(angle * 1.5000)

            # Wait for physical settling (poll for 1.5 seconds)
            settled = False
            last_val = enc_before
            stable_count = 0
            for _ in range(30):
                time.sleep(0.05)
                self.read_telemetry(timeout_s=0.1)
                curr_val = self.raw_head_ticks
                if curr_val == last_val:
                    stable_count += 1
                    if stable_count >= 3:
                        settled = True
                        break
                else:
                    stable_count = 0
                    last_val = curr_val

            enc_after = self.raw_head_ticks
            delta = enc_after - enc_before
            reported_angle = enc_after / 1.5000




            status = "SETTLED" if settled else "MOVING/TIMEOUT"

            print(f"| {angle:+6.1f}° | {calc_ticks:12d} | {enc_before:14d} | {enc_after:13d} | {delta:+13d} | {reported_angle:+13.2f}° | {status} |")

        print("\nVerification complete.")



def main():
    print("========================================================")
    print("   ASTRO ROBOT - ISOLATED ACTUATOR IDENTIFICATION")
    print("========================================================")
    calib = HeadCalibrator()
    calib.connect()

    print("\nSelect Mode:")
    print("  [1] Multi-Point Large-Angle Identification (0°, ±15°, ±30°, ±45°, ±90° with linear fit)")
    print("  [2] Safe Step Response Table Test (±2°, ±5°, ±10° command sequence)")
    print("  [3] Single Direct Angle Test (Safe range: -10° to +10°)")
    print("  [4] Canlı Enkoder & Telemetri İzleme (Live Stream)")
    choice = input("\nEnter choice [1, 2, 3, or 4]: ").strip()

    if choice == "1":
        calib.run_multi_point_manual_calibration()
    elif choice == "2":
        calib.run_step_table_verification()
    elif choice == "3":
        ang_str = input("Enter test angle [-45.0 to +45.0 deg]: ").strip()
        try:
            ang = max(-45.0, min(45.0, float(ang_str)))
            if calib.ser: calib.ser.reset_input_buffer()
            time.sleep(0.05)
            calib.read_telemetry(timeout_s=0.2)
            enc_start = calib.raw_head_ticks

            print(f"\nSending command: {ang:+.2f}°...")
            calib.command_angle(ang)
            
            # Wait for settling
            for _ in range(30):
                time.sleep(0.08)
                calib.read_telemetry(timeout_s=0.1)

            enc_end = calib.raw_head_ticks
            delta = enc_end - enc_start
            print("\n---------------- STEP RESULT ----------------")
            print(f"Commanded Angle      : {ang:+.2f}°")
            print(f"Encoder Ticks Start  : {enc_start}")
            print(f"Encoder Ticks End    : {enc_end}")
            print(f"Encoder Delta Ticks  : {delta:+d} ticks")

            
            phys_str = input("\nPhysically measure: Approximately how many DEGREES did the head turn? (e.g. 5, 10, 15, 20): ").strip()
            try:
                phys_deg = float(phys_str)
                if abs(phys_deg) > 0.1:
                    real_scale = abs(delta) / abs(phys_deg)
                    print(f"\n🎯 >>> CALCULATED TICKS PER DEGREE: {real_scale:.4f} ticks/deg <<<")
                    print(f"   (360° full revolution = {real_scale * 360.0:.1f} ticks)")
                    print(f"\nRecommended action:")
                    print(f"1. In calibration_params.yaml : ticks_per_deg: {real_scale:.4f}")
                    print(f"2. In AstroFirmware.ino       : static constexpr float HEAD_TICKS_PER_DEG = {real_scale:.4f}f;")
            except ValueError:
                pass
        except ValueError:
            print("Invalid input.")
    elif choice == "4":
        calib.run_live_monitor()


if __name__ == "__main__":
    main()


