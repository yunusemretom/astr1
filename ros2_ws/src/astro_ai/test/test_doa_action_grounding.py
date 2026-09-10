#!/usr/bin/env python3
"""Unit tests for ASTRO V1 — Physical Grounding, Sound Direction (DOA), Action ACK, and Persona Hotfix."""

import json
import logging
import math
import os
import sys
import time
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from astro_ai.action_manager import ActionManager, SoundDirection, ActionResult, circular_doa_to_yaw
from astro_ai.conversation_session import normalize_turkish_speech_input
from astro_ai.persona_engine import PersonaEngine, PERSONA_PROMPTS


class MockLogger(logging.Logger):
    def __init__(self, name="TestMockLogger"):
        super().__init__(name)
        self.log_messages = []

    def info(self, msg, *args, **kwargs):
        self.log_messages.append(str(msg))


class MockNode:
    """Mock ROS2 Node for ActionManager testing."""
    def __init__(self):
        self._arduino_heartbeat_healthy = True
        self._last_heartbeat_ack_time = time.monotonic()
        self._obstacle_detected = False
        self._last_laser_scan_time = time.monotonic()
        self._lidar_health = "HEALTHY"
        self.published_twists = []
        self.published_head_cmds = []

    class MockPublisher:
        def __init__(self, target_list):
            self.target_list = target_list
        def publish(self, msg):
            self.target_list.append(msg)

    @property
    def pub_cmd_vel(self):
        return self.MockPublisher(self.published_twists)

    @property
    def pub_head_cmd(self):
        return self.MockPublisher(self.published_head_cmds)


class TestSoundDirectionDOA(unittest.TestCase):
    """Tests for DOA geometric transformation and SoundDirection abstraction."""

    def test_circular_doa_to_yaw_conversion(self):
        # Body yaw is REP-103: positive = LEFT (URDF head_yaw_joint axis 0 0 1, and
        # GESTURE_PROFILES look_left=+35 / look_right=-35). ReSpeaker measures
        # clockwise, so a bearing on the right becomes a negative yaw.
        # 0° (front) -> 0.0°
        self.assertAlmostEqual(circular_doa_to_yaw(0.0), 0.0)
        # 35° (right) -> -35.0°
        self.assertAlmostEqual(circular_doa_to_yaw(35.0), -35.0)
        # 90° (right) -> -90.0°
        self.assertAlmostEqual(circular_doa_to_yaw(90.0), -90.0)
        # 180° (back) -> 180.0°
        self.assertAlmostEqual(abs(circular_doa_to_yaw(180.0)), 180.0)
        # 270° (left) -> +90.0°
        self.assertAlmostEqual(circular_doa_to_yaw(270.0), 90.0)
        # 325° (left) -> +35.0°
        self.assertAlmostEqual(circular_doa_to_yaw(325.0), 35.0)

    def test_test_a_turn_to_sound_no_direction_when_doa_unavailable(self):
        """Test A: User says 'Sesimin geldiği yöne dön'. DOA is unavailable -> NO_DIRECTION, no motor move."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        # No DOA provided
        res = action_mgr.execute_turn_to_sound(generation_id=1025)

        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "NO_DIRECTION")
        self.assertFalse(res.hardware_ack)
        self.assertEqual(len(mock_node.published_twists), 0)
        self.assertEqual(len(mock_node.published_head_cmds), 0)
        self.assertIn("tespit edilemedi", res.message)

    def test_test_b_turn_to_sound_success_with_valid_right_doa(self):
        """Test B: DOA bearing 35° (right) -> turn -35° (right in REP-103) executed with ACK."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        # Provide strong valid Right DOA (+35°)
        action_mgr.update_audio_state(
            raw_doa_deg=35.0,
            rms_level=2500.0,
            vad_active=True,
            is_speaking=False,
        )

        res = action_mgr.execute_turn_to_sound(generation_id=1026)

        self.assertTrue(res.success)
        self.assertEqual(res.action, "turn_to_sound")
        self.assertAlmostEqual(res.azimuth_deg, -35.0, places=1)
        self.assertGreaterEqual(res.confidence, 0.40)
        self.assertTrue(res.hardware_ack)
        self.assertGreater(len(mock_node.published_twists) + len(mock_node.published_head_cmds), 0)

    def test_controlled_left_sound_direction_execution(self):
        """Controlled Left (raw 315°) sound -> produces positive azimuth (+45°) and executes Left turn."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        action_mgr.update_audio_state(
            raw_doa_deg=315.0,  # bearing 315° = 45° to the left -> +45° body yaw
            rms_level=2200.0,
            vad_active=True,
            is_speaking=False,
        )

        res = action_mgr.execute_turn_to_sound(generation_id=1027)

        self.assertTrue(res.success)
        self.assertEqual(res.action, "turn_to_sound")
        self.assertAlmostEqual(res.azimuth_deg, 45.0, places=1)
        self.assertTrue(res.hardware_ack)

    def test_zero_degree_uncalibrated_idle_rejected(self):
        """0.0° default without active speech / low energy MUST produce valid=False and NO_DIRECTION."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        # Send 0.0° with no VAD and low energy (uncalibrated / idle state)
        action_mgr.update_audio_state(
            raw_doa_deg=0.0,
            rms_level=200.0,
            vad_active=False,
            is_speaking=False,
        )

        sd = action_mgr.get_sound_direction()
        self.assertIsNone(sd)

        res = action_mgr.execute_turn_to_sound(generation_id=1028)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "NO_DIRECTION")

    def test_zero_degree_confirmed_speech_accepted(self):
        """0.0° with strong active speech + high energy ratio MUST produce valid=True."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        # Send 0.0° with strong VAD and high energy (genuine front speaker)
        action_mgr.update_audio_state(
            raw_doa_deg=0.0,
            rms_level=3000.0,
            vad_active=True,
            is_speaking=False,
        )

        sd = action_mgr.get_sound_direction()
        self.assertIsNotNone(sd)
        self.assertTrue(sd.valid)
        self.assertAlmostEqual(sd.azimuth_deg, 0.0)

    def test_log_throttling_eliminates_spam(self):
        """10 consecutive identical 10Hz ticks should only log on initial change, eliminating spam."""
        mock_logger = MockLogger()
        action_mgr = ActionManager(logger=mock_logger)

        # 10 identical idle ticks
        for _ in range(10):
            action_mgr.update_audio_state(
                raw_doa_deg=0.0,
                rms_level=150.0,
                vad_active=False,
                is_speaking=False,
            )

        # Should only have logged once for the initial state
        doa_logs = [m for m in mock_logger.log_messages if "[DOA]" in m]
        self.assertLessEqual(len(doa_logs), 1)


class TestActionManagerPhysicalGrounding(unittest.TestCase):
    """Tests for physical grounding, hardware ACK, and safety interlocking."""

    def test_test_c_move_robot_success_hardware_ack(self):
        """Test C: Motor ACK: success = true -> structured action result verified."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        res = action_mgr.execute_move(direction="forward", speed=0.2, duration=1.0, generation_id=1027)

        self.assertTrue(res.success)
        self.assertTrue(res.hardware_ack)
        self.assertEqual(res.actual_direction, "forward")
        self.assertEqual(res.duration_ms, 1000)
        self.assertEqual(len(mock_node.published_twists), 1)

    def test_test_d_move_robot_failure_when_heartbeat_unhealthy(self):
        """Test D: Heartbeat unhealthy -> failure without motor execution, no hallucination."""
        mock_node = MockNode()
        mock_node._arduino_heartbeat_healthy = False
        action_mgr = ActionManager(node=mock_node)

        res = action_mgr.execute_move(direction="forward", speed=0.2, duration=1.0)

        self.assertFalse(res.success)
        self.assertFalse(res.hardware_ack)
        self.assertEqual(res.error_code, "MOTOR_CONTROLLER_UNAVAILABLE")
        self.assertEqual(res.reason, "heartbeat_unhealthy")
        self.assertEqual(len(mock_node.published_twists), 0)

    def test_action_idempotency(self):
        """Test that duplicate action IDs are rejected gracefully."""
        mock_node = MockNode()
        action_mgr = ActionManager(node=mock_node)

        res1 = action_mgr.execute_move(direction="left", speed=0.2, action_id="act_unique_1")
        self.assertTrue(res1.success)
        self.assertEqual(len(mock_node.published_twists), 1)

        # Duplicate call with identical action_id
        res2 = action_mgr.execute_move(direction="left", speed=0.2, action_id="act_unique_1")
        self.assertTrue(res2.success)
        # Should not publish additional twist
        self.assertEqual(len(mock_node.published_twists), 1)


class TestPersonaAndSTTHotfix(unittest.TestCase):
    """Tests for Profane/Roast persona behavior, anti-meta disclaimers, and STT normalization."""

    def test_test_e_profane_persona_prompt_no_meta_disclaimers(self):
        """Test E: PROFANE mode prompt contains zero meta-disclaimers and embodies roast character."""
        engine = PersonaEngine(current_persona="kufurbaz")
        prompt = engine.build_system_prompt()

        # Must mandate living the mode without robotic meta-disclaimers
        self.assertTrue("KÜFÜRBAZ / ROAST MODU" in prompt or "HAZIRCEVAP / ROAST MODU" in prompt)
        self.assertIn("SIFIR ROBOTİK DİSCLAIMER", prompt)
        self.assertIn("FİZİKSEL GERÇEKLİK VE EYLEM DÜRÜSTLÜĞÜ", prompt)
        self.assertIn("ANTI-NAME REPETITION", prompt)

        # Must NOT encourage repeating name in every turn
        self.assertNotIn("sık sık ve yerinde kullan", prompt.split("BİYOMETRİK DOĞRULAMA: BARAN")[-1] if "BARAN" in prompt else "")

    def test_turkish_stt_phonetic_normalization(self):
        """Test Turkish phonetic STT corrections (e.g. 'kürbat' -> 'küfürbaz')."""
        self.assertEqual(normalize_turkish_speech_input("kürbat moda geç"), "küfürbaz moda geç")
        self.assertEqual(normalize_turkish_speech_input("kurbat"), "küfürbaz")
        self.assertEqual(normalize_turkish_speech_input("küfür bat moduna geç"), "küfürbaz moduna geç")
        self.assertEqual(normalize_turkish_speech_input("ey aston naber"), "Hey Astro naber")


if __name__ == "__main__":
    unittest.main()
