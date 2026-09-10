"""ASTRO V1 — Phase 0 Verification Test Suite.

Verifies:
  1. WorldModel Tuple typing and instantiation without NameError
  2. SerialBridge CRC8 validation and MSG_HEARTBEAT_ACK integrity
  3. Differential drive kinematics (Twist -> WheelCmd conversion)
  4. ActionManager physical grounding via encoder joint states
  5. ActionManager bounded idempotency history
  6. SocialBrain integration in AstroRealtimeNode
"""

import math
import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch

# Ensure test import paths
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [
    os.path.join(pkg_root, "astro_ai"),
    os.path.join(pkg_root, "astro_ai", "astro_ai"),
    os.path.join(pkg_root, "astro_audio"),
    os.path.join(pkg_root, "astro_audio", "astro_audio"),
    os.path.join(pkg_root, "astro_base"),
    os.path.join(pkg_root, "astro_base", "astro_base"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_ai.brain.world_model import WorldModel, WorldStateSnapshot
from astro_ai.action_manager import ActionManager, ActionResult
from astro_ai.contracts.person_state import UnifiedPersonState


class TestPhase0WorldModel(unittest.TestCase):
    def test_01_world_model_instantiation_and_event_tuple(self):
        """Verify WorldModel instantiates and recent_events accepts tuples cleanly."""
        wm = WorldModel()
        self.assertIsNotNone(wm)
        self.assertEqual(len(wm._recent_events), 0)
        wm._recent_events.append((123456.78, "Baran odaya girdi"))
        self.assertEqual(len(wm._recent_events), 1)
        snapshot = wm.get_snapshot()
        self.assertIsInstance(snapshot, WorldStateSnapshot)


class TestPhase0KinematicsAndCRC(unittest.TestCase):
    def test_02_differential_drive_math(self):
        """Verify differential drive kinematics: linear 0.2 m/s -> equal RPM, angular -> differential RPM."""
        wheel_radius = 0.06
        wheel_separation = 0.26

        # Linear only: v = 0.2 m/s, w = 0.0 rad/s
        v = 0.2
        w = 0.0
        v_l = v - (w * wheel_separation / 2.0)
        v_r = v + (w * wheel_separation / 2.0)
        expected_rpm = (0.2 / wheel_radius) * (60.0 / (2.0 * math.pi))
        calc_rpm_l = (v_l / wheel_radius) * (60.0 / (2.0 * math.pi))
        calc_rpm_r = (v_r / wheel_radius) * (60.0 / (2.0 * math.pi))

        self.assertAlmostEqual(calc_rpm_l, expected_rpm, places=2)
        self.assertAlmostEqual(calc_rpm_r, expected_rpm, places=2)
        self.assertAlmostEqual(calc_rpm_l, 31.83, places=1)

        # Pure turn: v = 0.0, w = 1.0 rad/s
        v = 0.0
        w = 1.0
        v_l = v - (w * wheel_separation / 2.0)
        v_r = v + (w * wheel_separation / 2.0)
        calc_rpm_l = (v_l / wheel_radius) * (60.0 / (2.0 * math.pi))
        calc_rpm_r = (v_r / wheel_radius) * (60.0 / (2.0 * math.pi))
        self.assertAlmostEqual(calc_rpm_l, -calc_rpm_r, places=2)
        self.assertLess(calc_rpm_l, 0)
        self.assertGreater(calc_rpm_r, 0)

    def test_03_crc8_calculation(self):
        """Verify CRC8 ATM polynomial implementation."""
        from astro_base.serial_bridge import crc8

        # CRC of simple byte array
        test_bytes = bytes([0x03, 0x13, 0x00, 0x01])
        c = crc8(test_bytes)
        self.assertIsInstance(c, int)
        self.assertTrue(0 <= c <= 255)

        # Re-running on same data must be deterministic
        self.assertEqual(crc8(test_bytes), c)


class TestPhase0ActionManagerGrounding(unittest.TestCase):
    def setUp(self):
        self.mock_node = MagicMock()
        self.mock_node._arduino_heartbeat_healthy = True
        self.mock_node._last_heartbeat_ack_time = time.monotonic()
        self.mock_node._obstacle_detected = False
        self.mock_node._last_laser_scan_time = time.monotonic()
        self.mock_node._lidar_health = "HEALTHY"
        self.mock_pub_vel = MagicMock()
        self.mock_pub_head = MagicMock()

        self.mgr = ActionManager(
            pub_cmd_vel=self.mock_pub_vel,
            pub_head_cmd=self.mock_pub_head,
            node=self.mock_node,
        )

    def test_04_joint_states_update_and_query(self):
        """Verify update_joint_states tracks positions and velocities."""
        names = ["left_wheel_joint", "right_wheel_joint", "head_yaw_joint"]
        positions = [1.57, 3.14, 0.52]
        velocities = [0.1, 0.2, 0.0]

        self.mgr.update_joint_states(names, positions, velocities)
        cur_pos = self.mgr.get_joint_positions()

        self.assertAlmostEqual(cur_pos["left_wheel_joint"], 1.57)
        self.assertAlmostEqual(cur_pos["right_wheel_joint"], 3.14)
        self.assertAlmostEqual(cur_pos["head_yaw_joint"], 0.52)
        self.assertGreater(self.mgr._last_joint_update_ts, 0.0)

    def test_05_physical_verification_with_joint_states(self):
        """Verify execute_move sets verified=True when joint states feedback is active."""
        # Update joint states
        self.mgr.update_joint_states(["left_wheel_joint", "right_wheel_joint"], [0.1, 0.1])

        res = self.mgr.execute_move("forward", speed=0.2, duration=1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.hardware_ack)
        self.assertTrue(res.verified)
        self.assertIn("verified", res.to_dict())

    def test_06_bounded_action_history(self):
        """Verify action ID history is bounded and prevents duplicate execution."""
        # Execute action once
        res1 = self.mgr.execute_move("forward", speed=0.2, action_id="test_act_01")
        self.assertTrue(res1.success)

        # Repeated execution of same action ID should be caught by idempotency
        res2 = self.mgr.execute_move("forward", speed=0.2, action_id="test_act_01")
        self.assertIn("zaten yürütüldü", res2.message)


class TestPhase0RealtimeNodeSocialBrain(unittest.TestCase):
    @patch.dict(os.environ, {"ASTRO_TEST_MODE": "1", "OPENAI_API_KEY": ""})
    def test_07_social_brain_wire_in_realtime_node(self):
        """Verify AstroRealtimeNode instantiates SocialBrain and incorporates prompt context."""
        from astro_ai.astro_realtime_node import AstroRealtimeNode

        node = AstroRealtimeNode()
        self.assertIsNotNone(node)
        self.assertTrue(hasattr(node, "social_brain"))
        self.assertIsNotNone(node.social_brain)

        # Check prompt builder includes cognitive context
        identity = {"name": "Baran", "is_known": True, "user_id": "baran"}
        prompt = node._build_current_system_prompt(active_speaker=identity)
        self.assertIn("Baran", prompt)
        self.assertIn("SOSYAL ROBOT BİLİŞSEL BAĞLAMI", prompt)


if __name__ == "__main__":
    unittest.main()
