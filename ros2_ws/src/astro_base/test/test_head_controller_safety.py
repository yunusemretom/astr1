"""Unit tests for Low-Level Head Controller and Safety Watchdog Bridge."""

import struct
import unittest
from astro_base.gaze.head_controller import (
    HeadControllerCore,
    build_packet,
    crc8,
    MSG_HEAD_CMD,
    MSG_HEARTBEAT,
    MSG_HEARTBEAT_ACK,
    SOF1,
    SOF2,
)
from astro_base.gaze.types import TrajectoryPoint


class TestHeadControllerSafety(unittest.TestCase):
    def setUp(self):
        self.ctrl = HeadControllerCore(
            ticks_per_deg=2.5882,
            min_limit_deg=-90.0,
            max_limit_deg=90.0,
            watchdog_timeout_s=0.50,
            stall_timeout_s=1.50,
        )

    def test_crc8_and_packet_framing(self):
        """Tests binary packet construction and CRC-8 calculation."""
        payload = struct.pack("<f", 45.0)
        pkt = build_packet(MSG_HEAD_CMD, payload)

        self.assertEqual(pkt[0], SOF1)
        self.assertEqual(pkt[1], SOF2)
        self.assertEqual(pkt[2], 1 + len(payload))  # length = 5
        self.assertEqual(pkt[3], MSG_HEAD_CMD)      # msg_id = 0x03

        # Verify CRC
        body = pkt[2:-1]
        calc_crc = crc8(body)
        self.assertEqual(pkt[-1], calc_crc)

    def test_watchdog_blocks_commands_when_mcu_dead(self):
        """Tests that commands are rejected if no heartbeat ACK received within watchdog window."""
        t = 1.0
        pt = TrajectoryPoint(timestamp=t, position_deg=30.0)

        # 1. Initially MCU is not alive -> Command rejected (None)
        pkt1 = self.ctrl.encode_head_cmd_packet(pt, timestamp=t)
        self.assertIsNone(pkt1)
        self.assertFalse(self.ctrl.evaluate_watchdog(t))

        # 2. Receive Heartbeat ACK -> MCU becomes alive -> Command accepted
        self.ctrl.process_heartbeat_ack(ack_seq=1, timestamp=t)
        self.assertTrue(self.ctrl.evaluate_watchdog(t))
        pkt2 = self.ctrl.encode_head_cmd_packet(pt, timestamp=t)
        self.assertIsNotNone(pkt2)

        # 3. Timeout expires (>0.5s without ACK) -> Watchdog trips -> Commands blocked
        t += 0.60
        self.assertFalse(self.ctrl.evaluate_watchdog(t))
        pkt3 = self.ctrl.encode_head_cmd_packet(pt, timestamp=t)
        self.assertIsNone(pkt3)

    def test_closed_loop_encoder_feedback(self):
        """Tests conversion from raw encoder ticks to degrees and velocity estimation."""
        t = 1.0

        # Step 1: 0 ticks at t=1.0 -> 0.0°
        fb1 = self.ctrl.process_encoder_feedback(head_ticks=0, dt_s=0.02, timestamp=t)
        self.assertEqual(fb1.actual_yaw_deg, 0.0)

        # Step 2: 259 ticks at t=1.1 (dt=0.1s) -> ~100.0°, velocity = 1000°/s
        t += 0.10
        fb2 = self.ctrl.process_encoder_feedback(head_ticks=259, dt_s=0.10, timestamp=t)
        self.assertAlmostEqual(fb2.actual_yaw_deg, 100.07, places=1)
        self.assertAlmostEqual(fb2.actual_velocity_deg_s, 1000.7, places=0)

    def test_stall_detection(self):
        """Tests stall detection when PWM is active but ticks do not move for >1.5s."""
        t = 1.0
        # Initialize
        self.ctrl.process_encoder_feedback(head_ticks=100, dt_s=0.02, timestamp=t, pwm=100)
        self.assertFalse(self.ctrl.is_stalled)

        # 1.0s later with same ticks -> Not stalled yet (<1.5s)
        t += 1.0
        self.ctrl.process_encoder_feedback(head_ticks=100, dt_s=0.02, timestamp=t, pwm=100)
        self.assertFalse(self.ctrl.is_stalled)

        # 1.6s after stall start -> Stalled!
        t += 1.60
        fb_stalled = self.ctrl.process_encoder_feedback(head_ticks=100, dt_s=0.02, timestamp=t, pwm=100)
        self.assertTrue(fb_stalled.is_stalled)


    def test_mechanical_limit_clamping(self):
        """Tests that commanded angles beyond [-90°, +90°] are clamped in the encoded packet."""
        t = 1.0
        self.ctrl.process_heartbeat_ack(ack_seq=1, timestamp=t)

        pt_over = TrajectoryPoint(timestamp=t, position_deg=110.0)
        pkt = self.ctrl.encode_head_cmd_packet(pt_over, timestamp=t)
        self.assertIsNotNone(pkt)
        self.assertTrue(self.ctrl.is_limited)

        # Decode angle from packet payload
        payload = pkt[4:-1]
        decoded_angle = struct.unpack("<f", payload)[0]
        self.assertAlmostEqual(decoded_angle, 90.0, places=2)


if __name__ == "__main__":
    unittest.main()
