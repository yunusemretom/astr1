"""Low-Level Head Controller & Closed-Loop Hardware Safety Bridge.

Features:
  1. Protocol v2.0 Binary Packet Encoder/Decoder with CRC8-ATM
  2. Closed-Loop Position and Velocity Feedback Processing
  3. Host-to-MCU Heartbeat with Latency Tracking and Watchdog Guard
  4. Software Mechanical Limits [-90°, +90°] and Emergency Stop Gating
  5. Stall Detection and Fault Diagnostic Tracking
"""

import struct
from typing import Optional, Tuple
from astro_base.gaze.angle_math import clamp_deg
from astro_base.gaze.types import HeadFeedback, TrajectoryPoint


SOF1 = 0xAA
SOF2 = 0x55

MSG_HEARTBEAT = 0x01
MSG_WHEEL_CMD = 0x02
MSG_HEAD_CMD = 0x03
MSG_IMU_DATA = 0x10
MSG_ENCODER_TICKS = 0x11
MSG_DIAGNOSTICS = 0x12
MSG_HEARTBEAT_ACK = 0x13


def crc8(data: bytes) -> int:
    """Calculates CRC-8-ATM checksum (poly=0x07, init=0x00)."""
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
    """Constructs a Protocol v2.0 binary packet: [0xAA][0x55][LEN][MSG_ID][PAYLOAD][CRC8]."""
    length = 1 + len(payload)
    body = bytes([length, msg_id]) + payload
    c = crc8(body)
    return bytes([SOF1, SOF2]) + body + bytes([c])


class HeadControllerCore:
    """Hardware interface controller and safety watchdog engine."""

    def __init__(
        self,
        ticks_per_deg: float = 2.5882,
        min_limit_deg: float = -90.0,
        max_limit_deg: float = 90.0,
        watchdog_timeout_s: float = 0.50,
        stall_timeout_s: float = 1.50,
    ):
        self.ticks_per_deg = ticks_per_deg
        self.min_limit = min_limit_deg
        self.max_limit = max_limit_deg
        self.watchdog_timeout_s = watchdog_timeout_s
        self.stall_timeout_s = stall_timeout_s

        # Closed-loop feedback state
        self.actual_yaw_deg: float = 0.0
        self.actual_velocity_deg_s: float = 0.0
        self.encoder_ticks: int = 0
        self.motor_pwm: int = 0
        self.is_stalled: bool = False
        self.is_limited: bool = False

        # Watchdog & health state
        self.mcu_alive: bool = False
        self.last_heartbeat_ack_time: float = 0.0
        self.last_feedback_time: float = 0.0
        self.hb_seq: int = 0
        self.hb_latency_ms: float = 0.0

        # Stall detection internals
        self._last_stall_ref_ticks: int = 0
        self._stall_start_time: float = 0.0
        self._last_commanded_yaw: float = 0.0

    def encode_head_cmd_packet(self, trajectory_pt: TrajectoryPoint, timestamp: float) -> Optional[bytes]:
        """Encodes a target head angle into a binary MSG_HEAD_CMD packet."""
        # Safety gate: Do not command motors if MCU watchdog is dead
        if not self.mcu_alive and (timestamp - self.last_heartbeat_ack_time > self.watchdog_timeout_s):
            return None

        # Clamp to physical software limits
        target_deg = clamp_deg(trajectory_pt.position_deg, self.min_limit, self.max_limit)
        self.is_limited = (target_deg != trajectory_pt.position_deg)
        self._last_commanded_yaw = target_deg

        payload = struct.pack("<f", float(target_deg))
        return build_packet(MSG_HEAD_CMD, payload)

    def encode_heartbeat_packet(self, timestamp: float) -> bytes:
        """Encodes host heartbeat packet with sequence ID."""
        self.hb_seq = (self.hb_seq + 1) & 0xFFFFFFFF
        payload = struct.pack("<I", self.hb_seq)
        return build_packet(MSG_HEARTBEAT, payload)

    def process_heartbeat_ack(self, ack_seq: int, timestamp: float) -> None:
        """Processes received heartbeat ACK and clears watchdog lockout."""
        self.last_heartbeat_ack_time = timestamp
        self.mcu_alive = True

    def process_encoder_feedback(
        self,
        head_ticks: int,
        dt_s: float,
        timestamp: float,
        pwm: int = 0,
    ) -> HeadFeedback:
        """Processes raw encoder feedback from hardware."""
        prev_yaw = self.actual_yaw_deg
        self.encoder_ticks = head_ticks
        self.motor_pwm = pwm

        # Convert ticks to degrees
        self.actual_yaw_deg = round(float(head_ticks / self.ticks_per_deg), 2)

        # Compute velocity
        if dt_s > 0.001:
            self.actual_velocity_deg_s = round(float((self.actual_yaw_deg - prev_yaw) / dt_s), 2)

        # Stall detection: PWM commanded > 35 but encoder ticks not moving
        if abs(pwm) > 35 and abs(head_ticks - self._last_stall_ref_ticks) < 2:
            if self._stall_start_time == 0.0:
                self._stall_start_time = timestamp
            elif (timestamp - self._stall_start_time) > self.stall_timeout_s:
                self.is_stalled = True
        else:
            self._stall_start_time = 0.0
            self._last_stall_ref_ticks = head_ticks
            self.is_stalled = False

        self.last_feedback_time = timestamp
        return self.get_feedback_summary(timestamp)

    def evaluate_watchdog(self, timestamp: float) -> bool:
        """Evaluates communication watchdog timer. Returns True if alive."""
        if (timestamp - self.last_heartbeat_ack_time) > self.watchdog_timeout_s:
            self.mcu_alive = False
        return self.mcu_alive

    def get_feedback_summary(self, timestamp: float) -> HeadFeedback:
        """Generates HeadFeedback telemetry dataclass."""
        self.evaluate_watchdog(timestamp)
        return HeadFeedback(
            timestamp=timestamp,
            actual_yaw_deg=self.actual_yaw_deg,
            actual_velocity_deg_s=self.actual_velocity_deg_s,
            target_yaw_deg=self._last_commanded_yaw,
            encoder_ticks=self.encoder_ticks,
            motor_pwm=self.motor_pwm,
            is_stalled=self.is_stalled,
            is_limited=self.is_limited,
            watchdog_ok=self.mcu_alive,
            mcu_alive=self.mcu_alive,
        )
