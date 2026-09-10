#!/usr/bin/env python3
"""ASTRO V1 — Realtime Rate-Limit, Token Optimization and Deduplication Tests."""

import asyncio
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
astro_ai_inner = os.path.join(pkg_dir, "astro_ai")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
if astro_ai_inner not in sys.path:
    sys.path.insert(0, astro_ai_inner)

try:
    from astro_ai.action_manager import ActionManager, ActionResult, SoundDirection
except ImportError:
    from action_manager import ActionManager, ActionResult, SoundDirection


class MockPub:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class TestRealtimeRateLimitsAndOptimization(unittest.TestCase):
    """Acceptance test suite for Phases 4-13 Realtime Rate Limit & Token Optimizations."""

    def setUp(self):
        self.mock_pub_gesture = MockPub()
        self.mock_pub_target_yaw = MockPub()
        self.action_manager = ActionManager(
            pub_head_gesture=self.mock_pub_gesture,
            pub_head_target_yaw=self.mock_pub_target_yaw,
        )
        self.action_manager._heartbeat_healthy = True
        self.action_manager._last_heartbeat_ack_ts = time.monotonic()
        self.action_manager._obstacle_detected = False
        self.action_manager._last_laser_scan_ts = time.monotonic()

    def test_01_tool_turn_to_sound_routes_to_target_yaw_topic(self):
        """1. turn_to_sound publishes to /head/target_yaw and not directly to /head_cmd."""
        self.action_manager.update_audio_state(raw_doa_deg=45.0, rms_level=1200.0, vad_active=True)
        res = self.action_manager.execute_turn_to_sound(generation_id=101)
        self.assertTrue(res.success)
        self.assertEqual(len(self.mock_pub_target_yaw.published), 1)
        self.assertEqual(self.mock_pub_target_yaw.published[0].data, -45.0)

    def test_02_tool_turn_to_sound_debounce(self):
        """2. Rapid repeated turn_to_sound with same azimuth is debounced without redundant publishing."""
        self.action_manager.update_audio_state(raw_doa_deg=45.0, rms_level=1200.0, vad_active=True)
        res1 = self.action_manager.execute_turn_to_sound(generation_id=102)
        self.assertTrue(res1.success)
        self.assertEqual(len(self.mock_pub_target_yaw.published), 1)

        # Immediate follow-up with near-identical angle within 0.1s
        self.action_manager.update_audio_state(raw_doa_deg=46.0, rms_level=1200.0, vad_active=True)
        res2 = self.action_manager.execute_turn_to_sound(generation_id=103)
        self.assertTrue(res2.success)
        # Should not publish new message to topic (debounced)
        self.assertEqual(len(self.mock_pub_target_yaw.published), 1)

    def test_03_tool_gesture_routes_to_gesture_topic(self):
        """3. execute_gesture publishes canonical gesture to /head/gesture."""
        res = self.action_manager.execute_gesture(gesture_name="nod", generation_id=104)
        self.assertTrue(res.success)
        self.assertEqual(len(self.mock_pub_gesture.published), 1)
        self.assertEqual(self.mock_pub_gesture.published[0].data, "nod")

    def test_04_tool_gesture_debounce(self):
        """4. Rapid repeated identical gesture is debounced."""
        res1 = self.action_manager.execute_gesture(gesture_name="nod", generation_id=105)
        self.assertTrue(res1.success)
        self.assertEqual(len(self.mock_pub_gesture.published), 1)

        res2 = self.action_manager.execute_gesture(gesture_name="nod", generation_id=106)
        self.assertTrue(res2.success)
        self.assertEqual(len(self.mock_pub_gesture.published), 1)

    def test_06_set_head_angle_direction_and_sign(self):
        """6. set_head_angle tool correctly respects direction sign rules."""
        from astro_ai.astro_realtime_node import AstroRealtimeNode
        node = AstroRealtimeNode()
        node.pub_head_target_yaw = MockPub()
        
        # Right turns must be negative
        res_r = node._execute_realtime_tool("set_head_angle", {"angle_deg": 30.0, "direction": "right"})
        self.assertEqual(res_r["status"], "success")
        self.assertEqual(res_r["angle_deg"], -30.0)
        self.assertEqual(node.pub_head_target_yaw.published[-1].data, -30.0)
        
        # Left turns must be positive
        res_l = node._execute_realtime_tool("set_head_angle", {"angle_deg": -25.0, "direction": "left"})
        self.assertEqual(res_l["status"], "success")
        self.assertEqual(res_l["angle_deg"], 25.0)
        self.assertEqual(node.pub_head_target_yaw.published[-1].data, 25.0)
        
        # Center turns must be 0.0
        res_c = node._execute_realtime_tool("set_head_angle", {"angle_deg": 40.0, "direction": "center"})
        self.assertEqual(res_c["status"], "success")
        self.assertEqual(res_c["angle_deg"], 0.0)
        self.assertEqual(node.pub_head_target_yaw.published[-1].data, 0.0)

    def test_07_vad_defaults_low_latency(self):
        """7. VAD defaults are set for ultra-low latency turn-taking (<400ms)."""
        from astro_ai.astro_realtime_node import AstroRealtimeNode
        node = AstroRealtimeNode()
        self.assertLessEqual(node.vad_silence_duration_ms, 400)
        self.assertLessEqual(node.vad_threshold, 0.60)
        self.assertLessEqual(node.vad_prefix_padding_ms, 250)


if __name__ == '__main__':
    unittest.main()

