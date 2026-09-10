#!/usr/bin/env python3
"""Unit tests for ASTRO social sound-localization head tracking node."""

import collections
import math
import os
import sys
import time
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
astro_base_inner = os.path.join(pkg_dir, "astro_base")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
if astro_base_inner not in sys.path:
    sys.path.insert(0, astro_base_inner)

try:
    from astro_base.head_tracker_node import (
        HeadTrackerNode,
        SocialGazeStateMachine,
        doa_to_robot_yaw,
    )
except ImportError:
    from head_tracker_node import (
        HeadTrackerNode,
        SocialGazeStateMachine,
        doa_to_robot_yaw,
    )


class TestDoaConversion(unittest.TestCase):
    def test_standard_angles(self):
        # 0 deg (straight ahead) -> 0.0
        self.assertAlmostEqual(doa_to_robot_yaw(0.0), 0.0)
        # 90 deg (right side) -> 90.0
        self.assertAlmostEqual(doa_to_robot_yaw(90.0), 90.0)
        # 180 deg (back) -> 180.0
        self.assertAlmostEqual(doa_to_robot_yaw(180.0), 180.0)
        # 270 deg (left side) -> -90.0
        self.assertAlmostEqual(doa_to_robot_yaw(270.0), -90.0)
        # 315 deg (front-left) -> -45.0
        self.assertAlmostEqual(doa_to_robot_yaw(315.0), -45.0)
        # 45 deg (front-right) -> 45.0
        self.assertAlmostEqual(doa_to_robot_yaw(45.0), 45.0)

    def test_offset_and_inversion(self):
        # 0 with 90 offset -> 90.0
        self.assertAlmostEqual(doa_to_robot_yaw(0.0, offset_deg=90.0), 90.0)
        # 90 with inversion -> -90.0
        self.assertAlmostEqual(doa_to_robot_yaw(90.0, invert=True), -90.0)


class MockMsg:
    def __init__(self, data):
        self.data = data


def setUpModule():
    try:
        import rclpy
        if not rclpy.ok():
            rclpy.init()
    except Exception:
        pass


def tearDownModule():
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


class TestSocialGazeLogic(unittest.TestCase):
    def setUp(self):
        try:
            import rclpy
            if not rclpy.ok():
                rclpy.init()
        except Exception:
            pass
        # Instantiate HeadTrackerNode with test defaults without requiring live ROS 2 daemon
        self.node = HeadTrackerNode()
        self.node.enabled = True
        self.node.deadband_deg = 12.0
        self.node.min_dwell_time_s = 2.0
        self.node.max_yaw_deg = 70.0
        self.node.min_yaw_deg = -70.0
        self.node.min_rms_threshold = 500.0
        self.node.noise_multiplier = 2.0
        self.node.consensus_threshold = 3
        self.node.consensus_tolerance_deg = 15.0
        self.node.doa_offset_deg = 0.0
        self.node.doa_invert = False
        self.node.lidar_fusion_enabled = True
        self.node._doa_history.clear()
        self.node._target_yaw = 0.0
        self.node._current_yaw = 0.0
        self.node._filtered_target_yaw = 0.0
        self.node._is_speaking = False
        self.node._is_playback_active = False
        self.node._is_sleeping = False
        self.node._vad_active = True
        self.node.vision_fusion_enabled = False
        self.node._vision_person_detected = False

    def test_sleep_mode_locks_head_and_rejects_doa(self):
        """Test that when robot is in sleep mode, DOA is ignored and head target stays 0.0°."""
        self.node._is_sleeping = True
        self.node._latest_rms = 5000.0
        self.node._vad_active = True
        for _ in range(10):
            self.node._on_doa(MockMsg(45.0))
        self.assertEqual(len(self.node._doa_history), 0)
        self.assertEqual(self.node._target_yaw, 0.0)

    def test_180_deg_speech_clamps_to_max_yaw(self):
        """Test that speech from behind (>70°) safely clamps to physical neck limit (+70° or -70°)."""
        self.node._doa_history.clear()
        self.node._current_yaw = 0.0
        self.node._target_yaw = 0.0
        self.node.doa_offset_deg = 0.0
        self.node.doa_invert = False
        self.node._latest_rms = 3000.0
        self.node._vad_active = True
        self.node._last_gaze_switch_time = 0.0
        for _ in range(8):
            self.node._on_doa(MockMsg(170.0))
        self.assertEqual(self.node._target_yaw, 70.0)

    def test_consensus_clustering(self):
        """Test that temporal consensus filters out isolated outlier spikes and averages true cluster."""
        # 3 samples around 45° and 2 outlier spikes at -60° and 150°
        self.node._doa_history = collections.deque([
            (1.0, 44.0),
            (1.1, -60.0),  # Outlier
            (1.2, 46.0),
            (1.3, 150.0),  # Outlier
            (1.4, 45.0),
        ], maxlen=5)

        consensus = self.node._evaluate_consensus()
        self.assertIsNotNone(consensus)
        # Average of (44.0, 46.0, 45.0) = 45.0
        self.assertAlmostEqual(consensus, 45.0, places=1)

    def test_consensus_rejects_chaotic_noise(self):
        """Test that when all samples are scattered, no consensus is reached."""
        self.node._doa_history = collections.deque([
            (1.0, 10.0),
            (1.1, 70.0),
            (1.2, -45.0),
            (1.3, 130.0),
            (1.4, -80.0),
        ], maxlen=5)

        consensus = self.node._evaluate_consensus()
        self.assertIsNone(consensus)

    def test_self_voice_suppression(self):
        """Test that DOA is ignored while the robot is speaking (no self-voice feedback loop)."""
        self.node._is_speaking = True
        self.node._current_yaw = 0.0
        self.node._target_yaw = 0.0

        # Feed loud DOA while speaking
        self.node._latest_rms = 5000.0
        self.node._on_doa(MockMsg(45.0))
        self.assertEqual(len(self.node._doa_history), 0)
        self.assertEqual(self.node._target_yaw, 0.0)

    def test_deadband_hysteresis(self):
        """Test that small angle variations (<12°) do not cause head movement."""
        self.node._doa_history.clear()
        self.node._current_yaw = 0.0
        self.node._target_yaw = 0.0
        self.node._latest_rms = 2500.0
        self.node._ambient_rms = 200.0
        self.node._last_gaze_switch_time = 0.0  # Dwell time satisfied

        # Feed consistent 8° DOA readings (below 12° deadband)
        for _ in range(8):
            self.node._on_doa(MockMsg(8.0))

        self.assertEqual(self.node._target_yaw, 0.0)

    def test_gaze_dwell_time(self):
        """Test that a new gaze direction is held for at least min_dwell_time_s."""
        self.node._doa_history.clear()
        self.node._current_yaw = 0.0
        self.node._target_yaw = 0.0
        self.node._latest_rms = 2500.0
        self.node._ambient_rms = 200.0
        self.node._last_gaze_switch_time = 0.0  # Dwell satisfied initially

        # 1. Turn to 45°
        for _ in range(12):
            self.node._on_doa(MockMsg(45.0))
        self.assertAlmostEqual(self.node._target_yaw, 45.0, places=1)

        # 2. Immediately try to switch to -45° within dwell window (0.1s later)
        import time
        now = time.monotonic()
        self.node._last_gaze_switch_time = now - 0.2  # only 0.2s elapsed (< 2.0s dwell)
        self.node._doa_history.clear()
        for _ in range(12):
            self.node._on_doa(MockMsg(315.0))  # 315° = -45°

        # Target should still be 45.0°, not -45.0°
        self.assertAlmostEqual(self.node._target_yaw, 45.0, places=1)

    def test_slew_rate_velocity_limiting(self):
        """Test that the control loop limits angular step size per cycle (smooth velocity profile)."""
        self.node._current_yaw = 0.0
        self.node._target_yaw = 60.0
        self.node.max_speed_deg_s = 40.0  # 40 deg/sec
        # Prevent idle timeout from resetting _target_yaw to 0° during this test
        self.node._last_speech_time = time.monotonic()
        # Ensure vision fusion doesn't interfere
        self.node._vision_person_detected = False

        # Simulate 1 step of 0.1s (dt = 0.1s -> max step = 4.0 deg)
        self.node._last_update_time = time.monotonic() - 0.1
        self.node._control_loop()
        self.assertAlmostEqual(self.node._current_yaw, 4.0, places=1)

    def test_lidar_radar_gaze_orientation(self):
        """Test that approaching person detected on LiDAR/Radar smoothly steers head target."""
        class MockLaserScan:
            def __init__(self, target_angle_deg=35.0, distance_m=1.5):
                self.angle_min = -math.pi
                self.angle_increment = math.radians(1.0)
                self.range_min = 0.15
                self.range_max = 12.0
                # 360 ranges (index 0 = -180°, index 180 = 0°, index 215 = +35°)
                self.ranges = [10.0] * 360
                idx = int((math.radians(target_angle_deg) - self.angle_min) / self.angle_increment)
                if 0 <= idx < 360:
                    self.ranges[idx] = distance_m

        self.node.enabled = True
        self.node.lidar_fusion_enabled = True
        self.node.lidar_min_dist_m = 0.4
        self.node.lidar_max_dist_m = 2.8
        self.node._is_sleeping = False
        self.node._current_yaw = 0.0
        self.node._target_yaw = 0.0
        self.node._vad_active = False
        self.node._vision_person_detected = False
        self.node._last_gaze_switch_time = time.monotonic() - 5.0  # Dwell satisfied

        scan_msg = MockLaserScan(target_angle_deg=35.0, distance_m=1.5)
        self.node._on_laser_scan(scan_msg)

        self.assertTrue(self.node._lidar_person_detected)
        self.assertAlmostEqual(self.node._lidar_target_yaw, 35.0, places=1)

        # Run control loop tick
        self.node._last_update_time = time.monotonic() - 0.05
        self.node._control_loop()

        self.assertAlmostEqual(self.node._target_yaw, 35.0, places=1)


if __name__ == "__main__":
    unittest.main()
