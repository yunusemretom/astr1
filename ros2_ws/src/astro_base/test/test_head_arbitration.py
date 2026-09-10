#!/usr/bin/env python3
"""ASTRO V1 — Head Yaw Central Command Arbitration Acceptance Test Suite.

    1. Single Output Owner: Only HeadTrackerNode publishes /head_cmd
    2. Priority Hierarchy: SAFETY > GESTURE > TURN_TO_SOUND > CENTER > TRACKING
    3. Gesture Sequencer: Multi-step gestures execute smoothly without being clobbered by tracking
    4. Automatic Tracking Resume: Tracking resumes seamlessly when a gesture completes
    5. Slew-rate velocity limiting and software angle clamping [-70°, +70°]
    6. Rate optimization & deduplication: No redundant /head_cmd spam
"""

import collections
import json
import math
import os
import sys
import time
import unittest
import threading
from unittest.mock import MagicMock

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
astro_base_inner = os.path.join(pkg_dir, "astro_base")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
if astro_base_inner not in sys.path:
    sys.path.insert(0, astro_base_inner)

try:
    from astro_base.head_tracker_node import (
        CommandSource,
        HeadTrackerNode,
        SocialGazeStateMachine,
        angular_diff_deg,
        doa_to_robot_yaw,
    )
except ImportError:
    from head_tracker_node import (
        CommandSource,
        HeadTrackerNode,
        SocialGazeStateMachine,
        angular_diff_deg,
        doa_to_robot_yaw,
    )


bool_msg_class = type("MockMsg", (), {"data": None})

def MockMsg(data=None):
    m = bool_msg_class()
    m.data = data
    return m


class TestHeadYawCentralArbitration(unittest.TestCase):
    """Rigorous acceptance tests for Phase 1-3 Head Yaw Arbitration."""

    def setUp(self):
        self.node = HeadTrackerNode.__new__(HeadTrackerNode)
        self.node.enabled = True
        self.node.doa_offset_deg = 0.0
        self.node.doa_invert = False
        self.node.max_yaw_deg = 70.0
        self.node.min_yaw_deg = -70.0
        # This fixture pins a narrow neck range, so the seam-crossing shortcut is off.
        self.node._full_circle_travel = False
        self.node.deadband_deg = 12.0
        self.node.min_dwell_time_s = 3.0
        self.node.idle_return_timeout_s = 10.0
        self.node.max_speed_deg_s = 25.0
        self.node.update_rate_hz = 20.0
        self.node.min_rms_threshold = 1600.0
        self.node.noise_multiplier = 3.0
        self.node.head_motion_settle_s = 0.0  # motor-noise gate disabled for arbitration tests
        self.node.consensus_window_size = 7
        self.node.consensus_threshold = 5
        self.node.consensus_tolerance_deg = 22.0
        self.node.vision_fusion_enabled = True
        self.node.vision_gain = 0.6
        self.node.vision_timeout_s = 2.0
        self.node.lidar_fusion_enabled = True
        self.node.lidar_min_dist_m = 0.4
        self.node.lidar_max_dist_m = 2.8
        self.node.lidar_timeout_s = 2.5

        self.node._lock = threading.Lock()
        self.node._target_yaw = 0.0
        self.node._estimated_yaw = 0.0
        self.node._filtered_target_yaw = 0.0
        self.node._last_motion_cmd_time = 0.0
        self.node._vision_yaw_pending = False
        self.node._vision_ref_yaw = 0.0
        self.node._vision_anchor_yaw = 0.0
        self.node._vision_last_applied_yaw = None
        self.node.vision_max_correction_deg = 36.0
        self.node._state = SocialGazeStateMachine.IDLE
        self.node._command_source = CommandSource.IDLE

        self.node._active_gesture = None
        self.node._gesture_steps = []
        self.node._gesture_step_index = 0
        self.node._gesture_step_start_time = 0.0
        self.node._gesture_step_duration_s = 0.35

        self.node._turn_to_sound_active = False
        self.node._turn_to_sound_start_time = 0.0
        self.node._turn_to_sound_timeout_s = 3.0

        self.node._ambient_rms = 120.0
        self.node._latest_rms = 2000.0
        self.node._vad_active = True
        self.node._is_speaking = False
        self.node._is_playback_active = False
        self.node._is_sleeping = False
        self.node._last_speech_time = time.monotonic()
        self.node._last_gaze_switch_time = time.monotonic() - 10.0
        self.node._last_update_time = time.monotonic()

        self.node._vision_person_detected = False
        self.node._vision_head_yaw = 0.0
        self.node._vision_last_seen_time = 0.0
        self.node.vision_fusion_enabled = True

        self.node._lidar_person_detected = False
        self.node._lidar_target_yaw = 0.0
        self.node._lidar_distance_m = 0.0
        self.node._lidar_last_seen_time = 0.0

        self.node._doa_history = collections.deque(maxlen=7)
        self.published_head_cmds = []
        self.published_status = []

        self.node.pub_head_cmd = MagicMock()
        self.node.pub_head_cmd.publish = lambda msg: self.published_head_cmds.append(msg.angle_deg)
        self.node.pub_head_status = MagicMock()
        self.node.pub_head_status.publish = lambda msg: self.published_status.append(msg.data)
        self.node.get_logger = lambda: MagicMock()

    def test_01_tracking_only_behavior(self):
        """1. Tracking Only: DOA consensus sets target and slew limiter smoothly moves head."""
        for _ in range(5):
            self.node._on_doa(MockMsg(data=45.0))
        self.assertEqual(self.node._target_yaw, 45.0)

        for _ in range(10):
            self.node._last_update_time -= 0.05
            self.node._control_loop()

        self.assertGreater(self.node._estimated_yaw, 5.0)
        self.assertLessEqual(self.node._estimated_yaw, 45.0)
        self.assertTrue(len(self.published_head_cmds) > 0)

    def test_02_gesture_during_tracking_priority(self):
        """2. Gesture during Tracking: Gesture takes precedence and locks target until completed."""
        self.node._target_yaw = 45.0
        self.node._estimated_yaw = 45.0
        self.node._command_source = CommandSource.TRACKING

        self.node._on_gesture_cmd(MockMsg(data='nod'))
        self.assertEqual(self.node._command_source, CommandSource.GESTURE)
        self.assertEqual(self.node._target_yaw, 12.0)

        # new incoming DOA must NOT clobber active gesture
        self.node._on_doa(MockMsg(data=90.0))
        self.assertEqual(self.node._command_source, CommandSource.GESTURE)
        self.assertEqual(self.node._target_yaw, 12.0)

    def test_03_turn_to_sound_during_tracking(self):
        """3. Turn to Sound during Tracking: Explicit turn_to_sound updates target yaw cleanly."""
        self.node._target_yaw = 0.0
        self.node._estimated_yaw = 0.0
        self.node._command_source = CommandSource.TRACKING

        self.node._on_target_yaw_cmd(MockMsg(data=35.0))
        self.assertEqual(self.node._command_source, CommandSource.TURN_TO_SOUND)
        self.assertEqual(self.node._target_yaw, 35.0)

    def test_04_gesture_followed_by_tracking_resume(self):
        """4. Gesture followed by Tracking Resume: After gesture finishes, tracking resumes."""
        self.node._estimated_yaw = 0.0
        self.node._on_gesture_cmd(MockMsg(data='tilt'))
        self.assertEqual(self.node._command_source, CommandSource.GESTURE)

        # Advance step 1
        self.node._estimated_yaw = 16.0
        self.node._control_loop()

        # Advance step 2
        self.node._estimated_yaw = 0.0
        self.node._control_loop()

        # Gesture completion
        self.node._control_loop()
        self.assertEqual(self.node._command_source, CommandSource.TRACKING)
        self.assertIsNone(self.node._active_gesture)

    def test_05_center_during_gesture(self):
        """5. Center Command: 'center' gesture sets target immediately to 0.0°."""
        self.node._on_gesture_cmd(MockMsg(data='shake'))
        self.assertEqual(self.node._command_source, CommandSource.GESTURE)

        self.node._on_gesture_cmd(MockMsg(data='center'))
        self.assertEqual(self.node._command_source, CommandSource.CENTER)
        self.assertEqual(self.node._target_yaw, 0.0)

    def test_06_safety_command_during_gesture(self):
        """6. Safety Command: Emergency safety override instantly stops gesture and locks head."""
        self.node._on_gesture_cmd(MockMsg(data='shake'))
        self.assertEqual(self.node._command_source, CommandSource.GESTURE)

        # Trigger safety lock
        self.node._on_safety_cmd(MockMsg(data=True))
        self.assertEqual(self.node._command_source, CommandSource.SAFETY)
        self.assertTrue(self.node._is_sleeping)
        self.assertEqual(self.node._target_yaw, 0.0)

        # While safety locked, new gestures are rejected
        self.node._on_gesture_cmd(MockMsg(data='nod'))
        self.assertEqual(self.node._command_source, CommandSource.SAFETY)
        self.assertEqual(self.node._target_yaw, 0.0)

    def test_07_command_deduplication_and_rate_reduction(self):
        """7. Command Deduplication: Settled angles are not spammed over /head_cmd."""
        self.node._target_yaw = 0.0
        self.node._estimated_yaw = 0.0
        self.node._state = SocialGazeStateMachine.IDLE

        self.node._control_loop()
        first_count = len(self.published_head_cmds)

        for _ in range(50):
            self.node._control_loop()

        second_count = len(self.published_head_cmds)
        self.assertEqual(first_count, second_count)

    def test_08_angle_clamping(self):
        """8. Software Angle Clamp: Ensures targets never exceed [-70°, +70°]."""
        self.node._on_target_yaw_cmd(MockMsg(data=150.0))
        self.assertEqual(self.node._target_yaw, 70.0)

        self.node._on_target_yaw_cmd(MockMsg(data=-150.0))
        self.assertEqual(self.node._target_yaw, -70.0)

    def test_09_multi_speaker_spatial_mapping_and_turn_taking(self):
        """9. Multi-Speaker Spatial Mapping & Turn Taking: Maps visual faces and tracks active speaker."""
        # 1. Simulate vision seeing two people: Baran at -25° and Misafir at +35°
        faces_json = json.dumps([
            {"camera_azimuth_deg": -25.0, "recognized_name": "Baran", "is_known": True, "distance_m": 1.2},
            {"camera_azimuth_deg": 35.0, "recognized_name": "Misafir", "is_known": False, "distance_m": 1.5},
        ])
        self.node._estimated_yaw = 0.0
        self.node._on_vision_faces(MockMsg(data=faces_json))
        self.assertEqual(len(self.node._spatial_people_map), 2)
        self.assertEqual(self.node._spatial_people_map[0]["world_yaw"], -25.0)
        self.assertEqual(self.node._spatial_people_map[1]["world_yaw"], 35.0)

        # 2. Simulate acoustic speech from Baran (-25°)
        self.node.min_rms_threshold = 300.0
        self.node.consensus_threshold = 3
        self.node._vad_active = True
        self.node._latest_rms = 800.0
        self.node._ambient_rms = 100.0
        self.node._last_gaze_switch_time = 0.0
        for _ in range(3):
            self.node._on_doa(MockMsg(data=335.0)) # 335° in 360° frame = -25° in signed frame

        self.assertEqual(self.node._target_yaw, -25.0)

        # 3. Simulate speech turning to Misafir (+35°)
        self.node._doa_history.clear()
        self.node._last_gaze_switch_time = 0.0
        for _ in range(3):
            self.node._on_doa(MockMsg(data=35.0)) # 35° in 360° frame = +35° in signed frame

        self.assertEqual(self.node._target_yaw, 35.0)


if __name__ == '__main__':
    unittest.main()
