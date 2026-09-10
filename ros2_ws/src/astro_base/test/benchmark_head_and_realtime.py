#!/usr/bin/env python3
"""ASTRO V1 — Head Yaw & Realtime Rate Benchmark Suite."""

import collections
import os
import sys
import time
import unittest
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
    )
except ImportError:
    from head_tracker_node import (
        CommandSource,
        HeadTrackerNode,
        SocialGazeStateMachine,
    )


MockMsg = type('MockMsg', (), {'data': None})


class TestBenchmarkHeadAndRealtime(unittest.TestCase):
    """Measures before/after metrics for Head Yaw and Realtime Optimizations."""

    def setUp(self):
        self.node = HeadTrackerNode.__new__(HeadTrackerNode)
        self.node.enabled = True
        self.node.doa_offset_deg = 0.0
        self.node.doa_invert = False
        self.node.max_yaw_deg = 70.0
        self.node.min_yaw_deg = -70.0
        self.node.deadband_deg = 12.0
        self.node.min_dwell_time_s = 3.0
        self.node.idle_return_timeout_s = 10.0
        self.node.max_speed_deg_s = 25.0
        self.node.update_rate_hz = 20.0
        self.node.min_rms_threshold = 1600.0
        self.node.noise_multiplier = 3.0
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

        import threading
        self.node._lock = threading.Lock()
        self.node._target_yaw = 0.0
        self.node._estimated_yaw = 0.0
        self.node._filtered_target_yaw = 0.0
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

    def test_benchmark_idle_publish_rate_zero_spam(self):
        """Measures publish rate in settled idle state (must be 0 msgs/iteration)."""
        self.node._target_yaw = 0.0
        self.node._estimated_yaw = 0.0
        self.node._state = SocialGazeStateMachine.IDLE

        self.node._control_loop()
        self.published_head_cmds.clear()

        for _ in range(100):
            self.node._control_loop()

        # Assert zero redundant messages in settled idle
        self.assertEqual(len(self.published_head_cmds), 0)

    def test_benchmark_motion_publish_bounded(self):
        """Measures publish rate during a 45  motion ramp."""
        for _ in range(5):
            m = MockMsg()
            m.data = 45.0
            self.node._on_doa(m)

        self.published_head_cmds.clear()

        # Simulate 40 iterations (2 seconds of motion at 20 Hz)
        for _ in range(40):
            self.node._last_update_time -= 0.05
            self.node._control_loop()

        # Packets must be smoothly bounded, never exceeding control loop frequency
        published_count = len(self.published_head_cmds)
        self.assertLessEqual(published_count, 40)
        self.assertGreater(published_count, 0)


if __name__ == '__main__':
    unittest.main()
