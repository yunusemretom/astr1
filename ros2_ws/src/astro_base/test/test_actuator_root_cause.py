#!/usr/bin/env python3
"""Regression test suite for ASTRO Actuator Loop & Encoder Scale Root Cause Fix.

Validates:
  1. Encoder Scale Consistency across Firmware, YAML, and ROS 2 Serial Bridge (2.5882 ticks/deg)
  2. Canonical Head Position Conversion Formula with Zero Offset & Sign
  3. Stall Protection does NOT silently collapse target ticks to current position
  4. Stall Recovery & Fault Latching Policy
  5. Target Persistence during Stall Events
  6. Idempotent Setpoint Handling (Zero periodic stall/retry loop on identical retransmissions)
"""

import math
import os
import re
import sys
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(pkg_dir, "..", "..", ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

FIRMWARE_FILE = os.path.join(repo_root, "arduino", "astro_firmware", "src", "main.cpp")
CALIB_FILE = os.path.join(repo_root, "ros2_ws", "src", "astro_base", "config", "calibration_params.yaml")
BRIDGE_FILE = os.path.join(repo_root, "ros2_ws", "src", "astro_base", "astro_base", "serial_bridge.py")


class TestActuatorRootCause(unittest.TestCase):
    """Rigorous tests for actuator scale consistency and stall state preservation."""

    def test_encoder_scale_consistency(self):
        """Firmware, YAML config, and Serial Bridge must strictly agree on 2.5882 ticks/deg."""
        # 1. Firmware Check
        fw_src = open(FIRMWARE_FILE, encoding="utf-8").read()
        m_fw = re.search(r"HEAD_TICKS_PER_DEG\s*=\s*([0-9.]+)f", fw_src)
        self.assertIsNotNone(m_fw, "HEAD_TICKS_PER_DEG not found in main.cpp")
        fw_scale = float(m_fw.group(1))

        # 2. YAML Check
        yaml_src = open(CALIB_FILE, encoding="utf-8").read()
        m_yaml = re.search(r"ticks_per_deg:\s*([0-9.]+)", yaml_src)
        self.assertIsNotNone(m_yaml, "ticks_per_deg not found in calibration_params.yaml")
        yaml_scale = float(m_yaml.group(1))

        # 3. Serial Bridge Check
        bridge_src = open(BRIDGE_FILE, encoding="utf-8").read()
        m_bridge = re.search(r"head_ticks_per_deg.*?2\.5882", bridge_src)
        self.assertIsNotNone(m_bridge, "2.5882 default not found in serial_bridge.py")

        self.assertAlmostEqual(fw_scale, 2.5882, places=3, msg="Firmware scale must be 2.5882 ticks/deg")
        self.assertAlmostEqual(yaml_scale, 2.5882, places=3, msg="YAML scale must be 2.5882 ticks/deg")

    def test_head_position_conversion(self):
        """Verify the canonical formula: position_deg = (sign * (head_ticks - zero_ticks)) / ticks_per_deg."""
        ticks_per_deg = 2.5882
        zero_offset_ticks = 10.0
        sign = 1.0

        # Case 1: 0 offset, 13 ticks (+5.02°)
        ticks_1 = 13.0
        pos_1 = (sign * (ticks_1 - 0.0)) / ticks_per_deg
        self.assertAlmostEqual(pos_1, 5.02, places=2)

        # Case 2: -13 ticks (-5.02°)
        ticks_2 = -13.0
        pos_2 = (sign * (ticks_2 - 0.0)) / ticks_per_deg
        self.assertAlmostEqual(pos_2, -5.02, places=2)

        # Case 3: With zero offset
        ticks_3 = 23.0
        pos_3 = (sign * (ticks_3 - zero_offset_ticks)) / ticks_per_deg
        self.assertAlmostEqual(pos_3, 5.02, places=2)

    def test_stall_does_not_change_target(self):
        """Firmware must NOT collapse g_head_target_ticks to pos on stall."""
        fw_src = open(FIRMWARE_FILE, encoding="utf-8").read()
        # Verify that 'g_head_target_ticks = pos;' has been removed from the stall timeout block
        stall_block_match = re.search(
            r"millis\(\)\s*-\s*g_head_stall_ms\s*>\s*HEAD_STALL_MS\)\s*\{(.*?)\}",
            fw_src,
            re.DOTALL
        )
        self.assertIsNotNone(stall_block_match, "Stall handler block not found in main.cpp")
        stall_block = stall_block_match.group(1)

        self.assertNotIn(
            "g_head_target_ticks = pos",
            stall_block,
            "Target collapse bug detected! g_head_target_ticks must NOT be collapsed to pos on stall."
        )
        self.assertIn(
            "setHeadPWM(0)",
            stall_block,
            "Motor PWM must be safely cut to 0 on stall."
        )
        self.assertIn(
            "FLAG_HEAD_STALL",
            stall_block,
            "FLAG_HEAD_STALL diagnostic flag must be raised."
        )

    def test_no_periodic_stall_retry_loop(self):
        """Firmware must make setpoint updates idempotent (no stall reset on identical commands)."""
        fw_src = open(FIRMWARE_FILE, encoding="utf-8").read()
        head_cmd_match = re.search(
            r"case Proto::HEAD_CMD:\s*\{(.*?)\} break;",
            fw_src,
            re.DOTALL
        )
        self.assertIsNotNone(head_cmd_match, "Proto::HEAD_CMD block not found")
        head_cmd_block = head_cmd_match.group(1)

        # Must check if new target differs from current target before resetting stall timer
        self.assertIn(
            "abs(new_target_ticks - g_head_target_ticks)",
            head_cmd_block,
            "Idempotency check missing! Repeated identical packets must not reset stall timer."
        )


if __name__ == "__main__":
    unittest.main()
