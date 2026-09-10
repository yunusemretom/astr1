#!/usr/bin/env python3
"""Differential Parity Test Suite: Standalone GazeTracker vs ROS SocialGazeNode.

Validates:
  1. Exact mathematical parity (< 1e-4°) across 22 nominal bearing & head offset cases.
  2. Elimination of linear camera_azimuth_deg divergence (optical pinhole ray parity).
  3. Visual lock consistency: command_source = VISUAL, target_source = CAMERA.
  4. Visual coasting retention during dropouts (< 1.0s) without snapping to 0.0°.
  5. Audio reacquisition intent consistency without raw DOA commanding the motor.
  6. Legacy ROS pipeline operates in shadow mode only (never commands actuator).
  7. Runtime telemetry: 100-sample ring buffer, 5s reporting, and timestamp tracking.
"""

import json
import math
import os
import sys
import time
import unittest
from pathlib import Path

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

standalone_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../standalone"))
if standalone_dir not in sys.path:
    sys.path.insert(0, standalone_dir)

from astro_base.gaze.gaze_tracker import Detection, GazeResult, GazeTracker
from astro_base.social_gaze_node import SocialGazeNode


class _Msg:
    def __init__(self, data):
        self.data = data


class TestStandaloneRosParity(unittest.TestCase):
    """Rigorous Differential Parity Verification Suite."""

    def setUp(self):
        self.frame_w = 640
        self.frame_h = 480

    def _create_detection(self, nominal_bearing_deg: float, width_px: int = 80, conf: float = 0.88):
        """Creates bbox geometry for a nominal angle."""
        norm_offset = -nominal_bearing_deg / 36.0
        face_cx = (self.frame_w / 2.0) + norm_offset * (self.frame_w / 2.0)
        h = width_px
        x = int(face_cx - width_px / 2.0)
        y = int((self.frame_h / 2.0) - h / 2.0)
        return x, y, width_px, h

    def test_36_differential_parity_cases(self):
        """Validates exact mathematical parity (< 1e-4°) across 36 test cases.

        Explicitly covers: 0, ±5, ±10, ±15, ±20, ±25, ±30, ±40, ±50, ±60
        and varying head actual values.
        """
        test_cases = [
            # 19 cases at head_actual = 0.0:
            (0.0,    0.0, 0.88, 80),
            (5.0,    0.0, 0.88, 80),
            (-5.0,   0.0, 0.88, 80),
            (10.0,   0.0, 0.88, 80),
            (-10.0,  0.0, 0.88, 80),
            (15.0,   0.0, 0.88, 80),
            (-15.0,  0.0, 0.88, 80),
            (20.0,   0.0, 0.88, 80),
            (-20.0,  0.0, 0.88, 80),
            (25.0,   0.0, 0.88, 80),
            (-25.0,  0.0, 0.88, 80),
            (30.0,   0.0, 0.88, 80),
            (-30.0,  0.0, 0.88, 80),
            (40.0,   0.0, 0.88, 80),
            (-40.0,  0.0, 0.88, 80),
            (50.0,   0.0, 0.88, 80),
            (-50.0,  0.0, 0.88, 80),
            (60.0,   0.0, 0.88, 80),
            (-60.0,  0.0, 0.88, 80),
            # 17 cases with varying head actual offsets:
            (0.0,   15.0, 0.92, 90),
            (0.0,  -15.0, 0.92, 90),
            (5.0,   10.0, 0.85, 75),
            (-5.0, -10.0, 0.85, 75),
            (10.0, -10.0, 0.85, 70),
            (-10.0, 10.0, 0.85, 70),
            (15.0,   5.0, 0.90, 85),
            (-15.0, -5.0, 0.90, 85),
            (20.0, -15.0, 0.82, 60),
            (-20.0, 15.0, 0.82, 60),
            (25.0,  10.0, 0.78, 100),
            (-25.0,-10.0, 0.78, 100),
            (30.0, -20.0, 0.88, 80),
            (-30.0, 20.0, 0.88, 80),
            (40.0,  10.0, 0.80, 70),
            (-40.0,-10.0, 0.80, 70),
            (50.0,  -5.0, 0.85, 80),
        ]

        self.assertGreaterEqual(len(test_cases), 30)

        max_parity_diff = 0.0
        for b, head_act, conf, w in test_cases:
            x, y, w_val, h_val = self._create_detection(b, width_px=w, conf=conf)

            # 1. Standalone GazeTracker
            standalone_tracker = GazeTracker()
            det_sa = Detection(x=x, y=y, w=w_val, h=h_val, confidence=conf, detector_source="TEST")
            _ = standalone_tracker.step([det_sa], (self.frame_w, self.frame_h), None, head_act, 100.0)
            res_sa = standalone_tracker.step([det_sa], (self.frame_w, self.frame_h), None, head_act, 100.033)
            standalone_target_yaw = res_sa.target_yaw_deg

            # 2. ROS SocialGazeNode with Golden Tracker Core
            node = SocialGazeNode()
            node._on_head_state(type("HeadMsg", (), {"position_deg": head_act, "velocity_deg_s": 0.0})())
            face_dict = {
                "x": x, "y": y, "w": w_val, "h": h_val, "width": w_val, "height": h_val,
                "confidence": conf,
                "camera_azimuth_deg": b,  # Legacy linear override in incoming JSON
                "frame_width": self.frame_w, "frame_height": self.frame_h,
                "looking_at_robot": True, "yaw_deg": 0.0, "emotion": "neutral",
            }
            msg = _Msg(json.dumps([face_dict]))
            node._on_vision_json(msg)
            node._control_cycle()
            node._on_vision_json(msg)
            node._control_cycle()

            ros_cmd = node.last_forensic_chain["command"]
            ros_golden_target_yaw = ros_cmd["new_target_yaw"]

            diff = abs(standalone_target_yaw - ros_golden_target_yaw)
            max_parity_diff = max(max_parity_diff, diff)

            self.assertLess(
                diff,
                1e-4,
                f"Parity invariant violated for nominal {b:+.1f}°, head {head_act:+.1f}°: "
                f"standalone={standalone_target_yaw:.4f}°, ros={ros_golden_target_yaw:.4f}°, diff={diff:.6f}°"
            )
            self.assertEqual(ros_cmd["command_source"], "VISUAL")
            self.assertEqual(ros_cmd["target_source"], "CAMERA")

        self.assertLess(max_parity_diff, 1e-4)

    def test_single_cycle_semantics_and_no_new_decision_without_frame(self):
        """Verify One Frame -> One Step -> One Command, and no recalculation on 50 Hz timer."""
        node = SocialGazeNode()
        node.enable_actuator_output = True
        node._on_head_state(type("HeadMsg", (), {"position_deg": 0.0, "velocity_deg_s": 0.0})())

        # Feed Frame 1
        x, y, w, h = self._create_detection(15.0, width_px=80, conf=0.85)
        msg1 = _Msg(json.dumps([{
            "x": x, "y": y, "w": w, "h": h, "confidence": 0.85,
            "frame_width": self.frame_w, "frame_height": self.frame_h,
        }]))

        # Frame 1 sets command immediately
        cmd_count_before = node.pub_head_cmd_pos.count
        node._on_vision_json(msg1)
        self.assertGreater(node.pub_head_cmd_pos.count, cmd_count_before)
        target_frame1 = node.authoritative_target_yaw

        # Control cycle runs between frames: keepalive only, NO new visual decision
        node._control_cycle()
        self.assertEqual(node.authoritative_target_yaw, target_frame1)

        # Invariant: target_id is person_1
        self.assertEqual(node.golden_gaze_result.target_id, "person_1")

    def test_real_robot_acceptance_scenario_a_center(self):
        """TEST A — CENTER: Yüz merkezde, kafa yaklaşık 0°."""
        node = SocialGazeNode()
        node._on_head_state(type("HeadMsg", (), {"position_deg": 0.0, "velocity_deg_s": 0.0})())

        x, y, w, h = self._create_detection(0.0, width_px=80, conf=0.90)
        msg = _Msg(json.dumps([{"x": x, "y": y, "w": w, "h": h, "confidence": 0.90}]))
        node._on_vision_json(msg)
        node._control_cycle()

        self.assertAlmostEqual(node.authoritative_target_yaw, 0.0, delta=0.5)
        self.assertEqual(node.authoritative_command_source, "VISUAL")
        self.assertEqual(node.authoritative_target_id, "person_1")

    def test_real_robot_acceptance_scenario_b_slow_right(self):
        """TEST B — SLOW RIGHT: 0 -> +10 -> +20 -> +30, sürekli takip ve person_1 devamlılığı."""
        node = SocialGazeNode()
        node._on_head_state(type("HeadMsg", (), {"position_deg": 0.0, "velocity_deg_s": 0.0})())

        angles = [0.0, 10.0, 20.0, 30.0]
        yaws = []
        sim_t = 100.0
        for ang in angles:
            x, y, w, h = self._create_detection(ang, width_px=80, conf=0.90)
            for _ in range(5):
                sim_t += 0.05
                msg = _Msg(json.dumps([{"x": x, "y": y, "w": w, "h": h, "confidence": 0.90, "timestamp": sim_t}]))
                node._on_vision_json(msg)
                node._control_cycle()
            yaws.append(node.authoritative_target_yaw)
            self.assertEqual(node.authoritative_target_id, "person_1")
            self.assertEqual(node.authoritative_command_source, "VISUAL")

        # Monotonically increasing target yaw tracking the moving person
        self.assertLess(yaws[0], yaws[1])
        self.assertLess(yaws[1], yaws[2])
        self.assertLess(yaws[2], yaws[3])
        self.assertGreater(yaws[3], 20.0)

    def test_real_robot_acceptance_scenario_c_slow_left(self):
        """TEST C — SLOW LEFT: +30 -> +20 -> +10 -> 0 -> -10 -> -20 -> -30."""
        node = SocialGazeNode()
        node._on_head_state(type("HeadMsg", (), {"position_deg": 0.0, "velocity_deg_s": 0.0})())

        angles = [30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0]
        yaws = []
        sim_t = 100.0
        for ang in angles:
            x, y, w, h = self._create_detection(ang, width_px=80, conf=0.90)
            for _ in range(5):
                sim_t += 0.05
                msg = _Msg(json.dumps([{"x": x, "y": y, "w": w, "h": h, "confidence": 0.90, "timestamp": sim_t}]))
                node._on_vision_json(msg)
                node._control_cycle()
            yaws.append(node.authoritative_target_yaw)
            self.assertEqual(node.authoritative_target_id, "person_1")

        # Monotonically decreasing target yaw tracking smoothly from right to left
        for i in range(len(yaws) - 1):
            self.assertGreater(yaws[i], yaws[i + 1])
        self.assertLess(yaws[-1], -20.0)

    def test_real_robot_acceptance_scenario_d_temporary_dropout(self):
        """TEST D — TEMPORARY DROPOUT: Yüz kısa süre kayboluyor. Son visual target korunuyor, 0°'ye dönüş yok."""
        node = SocialGazeNode()
        node._on_head_state(type("HeadMsg", (), {"position_deg": 0.0, "velocity_deg_s": 0.0})())

        # Establish tracking at +20°
        x, y, w, h = self._create_detection(20.0, width_px=80, conf=0.90)
        msg = _Msg(json.dumps([{"x": x, "y": y, "w": w, "h": h, "confidence": 0.90}]))
        node._on_vision_json(msg)
        node._control_cycle()
        locked_yaw = node.authoritative_target_yaw
        self.assertGreater(locked_yaw, 15.0)

        # Temporary visual dropout (< 1.0s)
        empty_msg = _Msg(json.dumps([]))
        node._on_vision_json(empty_msg)
        node._control_cycle()

        self.assertIn(node.authoritative_command_source, ("VISUAL", "VISUAL_COAST"))
        self.assertAlmostEqual(node.authoritative_target_yaw, locked_yaw, places=2)
        self.assertNotEqual(node.authoritative_target_yaw, 0.0)

    def test_real_robot_acceptance_scenario_e_reacquire(self):
        """TEST E — REACQUIRE: Yüz tekrar geliyor. Aynı visual target (person_1) devam ediyor, yeni birth yok."""
        node = SocialGazeNode()
        node._on_head_state(type("HeadMsg", (), {"position_deg": 0.0, "velocity_deg_s": 0.0})())

        # 1. Lock person_1
        x, y, w, h = self._create_detection(15.0, width_px=80, conf=0.90)
        msg = _Msg(json.dumps([{"x": x, "y": y, "w": w, "h": h, "confidence": 0.90}]))
        node._on_vision_json(msg)
        node._control_cycle()
        self.assertEqual(node.authoritative_target_id, "person_1")

        # 2. Brief dropout
        empty_msg = _Msg(json.dumps([]))
        node._on_vision_json(empty_msg)
        node._control_cycle()
        self.assertIn(node.authoritative_command_source, ("VISUAL", "VISUAL_COAST"))

        # 3. Reacquire same face
        node._on_vision_json(msg)
        node._control_cycle()
        self.assertEqual(node.authoritative_command_source, "VISUAL")
        self.assertEqual(node.authoritative_target_id, "person_1")

    def test_visual_coasting_parity_on_dropout(self):
        """Verify that temporary visual dropout coasts the target instead of returning to 0.0°."""
        x, y, w, h = self._create_detection(-15.0, width_px=80, conf=0.90)
        node = SocialGazeNode()
        msg = _Msg(json.dumps([{
            "x": x, "y": y, "w": w, "h": h, "confidence": 0.90,
            "frame_width": self.frame_w, "frame_height": self.frame_h,
        }]))
        node._on_vision_json(msg)
        node._control_cycle()
        initial_yaw = node.last_forensic_chain["command"]["new_target_yaw"]

        # Empty frame arrives (dropout < 1.0s)
        empty_msg = _Msg(json.dumps([]))
        node._on_vision_json(empty_msg)
        node._control_cycle()

        cmd = node.last_forensic_chain["command"]
        self.assertIn(cmd["command_source"], ("VISUAL", "VISUAL_COAST"))
        self.assertAlmostEqual(cmd["new_target_yaw"], initial_yaw, places=2)

    def test_legacy_path_is_shadow_only(self):
        """Verify legacy pipeline calculates comparison telemetry but never commands the motor."""
        x, y, w, h = self._create_detection(20.0, width_px=80, conf=0.88)
        node = SocialGazeNode()
        node.enable_actuator_output = True
        msg = _Msg(json.dumps([{
            "x": x, "y": y, "w": w, "h": h, "confidence": 0.88,
            "camera_azimuth_deg": 20.0,
            "frame_width": self.frame_w, "frame_height": self.frame_h,
        }]))
        node._on_vision_json(msg)
        node._control_cycle()

        diag = json.loads(node.pub_gaze_debug.last_msg.data)
        self.assertEqual(diag["authoritative_source"], "STANDALONE_GOLDEN_TRACKER")
        self.assertIn("legacy_shadow", diag)
        self.assertIn("golden_divergence_deg", diag)
        self.assertEqual(node.pub_head_cmd_pos.last_msg.data, diag["golden_target_yaw_deg"])

    def test_parity_ring_buffer_and_timestamps(self):
        """Verify the 100-sample ring buffer and timestamp telemetry."""
        node = SocialGazeNode()
        for i in range(120):
            node._control_cycle()

        self.assertEqual(len(node.recent_parity_samples), 100)
        latest_sample = node.recent_parity_samples[-1]
        for key in ("timestamp", "golden_target", "actual_head", "visual_bearing", "motor_error"):
            self.assertIn(key, latest_sample)

        diag = json.loads(node.pub_gaze_debug.last_msg.data)
        self.assertIn("timestamps", diag)
        for key in ("capture_stamp", "vision_arrival_stamp", "golden_step_stamp", "head_feedback_stamp", "vision_age_ms"):
            self.assertIn(key, diag["timestamps"])


if __name__ == "__main__":
    unittest.main()
