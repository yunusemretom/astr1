#!/usr/bin/env python3
"""Regression and Unit tests for Camera-on-Moving-Head Temporal Synchronization.

Verifies:
1. Interpolation accuracy in GazeRuntimeCore history buffer (linear, clamping, boundaries).
2. Elimination of artificial ghost error caused by camera transport latency while head is rotating.
3. Zero-skew exact parity with golden 2e0b70c standalone tracker.
4. Latency sweep stability across 0ms to 150ms transport delays.
"""

import math
import sys
from pathlib import Path
from typing import List

import pytest

# Resolve paths
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[3]
STANDALONE_DIR = REPO_ROOT / "standalone"
ASTRO_BASE_DIR = CUR_DIR.parents[1]

for path_str in (str(REPO_ROOT), str(STANDALONE_DIR), str(ASTRO_BASE_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from astro_base.gaze.gaze_runtime import GazeRuntimeCore
from astro_base.gaze.gaze_tracker import Detection
from astro_base.standalone_gaze_ros_node import StandaloneGazeRosNode
from tracker import GazeTracker


class TestTemporalSynchronization:
    """Tests temporal interpolation and lag compensation."""

    def test_history_buffer_interpolation(self):
        """Validates linear interpolation and boundary clamping in GazeRuntimeCore."""
        runtime = GazeRuntimeCore()
        assert not runtime.has_head_feedback

        # Empty fallback
        pos, vel = runtime.get_head_position_at(10.0)
        assert pos == 0.0
        assert vel == 0.0

        # Feed 5 samples at 50Hz (every 20ms)
        # Head moving at +50 deg/s
        t_base = 100.000
        for i in range(5):
            t = t_base + (i * 0.020)
            angle = float(i * 1.0)  # 0.0, 1.0, 2.0, 3.0, 4.0
            vel_val = 50.0
            runtime.update_head_feedback(angle, velocity_deg_s=vel_val, timestamp=t)

        # Exact sample points
        assert runtime.get_head_position_at(100.000)[0] == pytest.approx(0.0)
        assert runtime.get_head_position_at(100.020)[0] == pytest.approx(1.0)
        assert runtime.get_head_position_at(100.080)[0] == pytest.approx(4.0)

        # Midway interpolation (10ms into interval -> alpha=0.5)
        pos_mid, _ = runtime.get_head_position_at(100.010)
        assert pos_mid == pytest.approx(0.5, abs=1e-4)

        pos_3_4, _ = runtime.get_head_position_at(100.015)
        assert pos_3_4 == pytest.approx(0.75, abs=1e-4)

        # Boundary clamping
        # Prior to oldest sample -> clamp to oldest
        assert runtime.get_head_position_at(99.000)[0] == pytest.approx(0.0)
        # Newer than latest sample -> clamp to newest
        assert runtime.get_head_position_at(101.000)[0] == pytest.approx(4.0)

    def test_zero_skew_exact_parity(self):
        """When T_frame == T_encoder, runtime produces exact golden output."""
        runtime = GazeRuntimeCore()
        standalone = GazeTracker()

        detections = [Detection(x=280, y=200, w=80, h=80, confidence=0.9)]
        t = 10.0

        # Static head at 12.5 deg
        runtime.update_head_feedback(12.5, velocity_deg_s=0.0, timestamp=t)

        res_runtime = runtime.step(faces=detections, frame_size=(640, 480), timestamp=t)
        res_standalone = standalone.step(faces=detections, frame_size=(640, 480), doa_deg=None, measured_head_deg=12.5, timestamp=t)

        assert res_runtime.target_yaw_deg == pytest.approx(res_standalone.target_yaw_deg, abs=1e-3)
        assert res_runtime.gaze_state == res_standalone.gaze_state

    def test_moving_head_ghost_error_elimination(self):
        """Simulates head rotating across a centered person with transport latency.

        Proves that interpolating head pose at frame capture timestamp removes
        the artificial error that drives real-robot oscillation.
        """
        runtime = GazeRuntimeCore()

        # Person is stationary at body yaw = 0.0 deg.
        # Robot head is sweeping left-to-right from -10 deg to +10 deg at 50 deg/s.
        # At T_capture = 20.000s, head was at exactly 0.0 deg.
        # In the camera image, the person was centered (optical bearing = 0.0 deg).
        frame_w = 640
        person_centered_det = [Detection(x=280, y=200, w=80, h=80, confidence=0.9)]

        # Simulate 50Hz encoder feedback arriving from T=19.900s to T=20.100s
        for step_i in range(11):
            t_enc = 19.900 + (step_i * 0.020)
            # Head angle moves from -5.0 deg to +5.0 deg (50 deg/s)
            head_angle = -5.0 + (step_i * 1.0)
            runtime.update_head_feedback(head_angle, velocity_deg_s=50.0, timestamp=t_enc)

        # At T_proc = 20.100s (100ms later), the image captured at T_capture = 20.000s is processed.
        # Current encoder in runtime.actual_head_yaw_deg is now +5.0 deg.
        # But captured image photons were at T_capture = 20.000s when head was 0.0 deg.
        t_capture = 20.000

        # Step runtime with t_capture
        result = runtime.step(
            faces=person_centered_det,
            frame_size=(frame_w, 480),
            timestamp=t_capture,
        )

        # The head pose supplied to the tracker should be interpolated at T_capture = 0.0 deg!
        # If it used current encoder (+5.0 deg), target_yaw would shift to +5.0 deg.
        # With temporal alignment, target_yaw is centered around 0.0 deg.
        assert runtime.tracker.head_angle_deg == pytest.approx(0.0, abs=1e-2)
        assert abs(result.target_yaw_deg) < 1.0  # within center deadband / zero

    def test_latency_sweep_stability(self):
        """Verifies that across latencies from 0ms to 150ms, the aligned head pose matches capture pose."""
        latencies_ms = [0, 25, 50, 75, 100, 125, 150]

        for lat_ms in latencies_ms:
            runtime = GazeRuntimeCore()
            lat_s = lat_ms / 1000.0

            # Feed moving head history
            t_start = 50.0
            omega = 60.0  # deg/s
            for i in range(20):
                t_sample = t_start + (i * 0.020)
                pos = omega * (t_sample - t_start)  # moving at 60 deg/s
                runtime.update_head_feedback(pos, velocity_deg_s=omega, timestamp=t_sample)

            # Query at capture time = t_latest - latency
            t_latest = 50.0 + (19 * 0.020)
            t_capture = t_latest - lat_s
            expected_pos = omega * (t_capture - t_start)

            aligned_pos, _ = runtime.get_head_position_at(t_capture)
            assert aligned_pos == pytest.approx(expected_pos, abs=0.05), (
                f"Failed for latency {lat_ms}ms: got {aligned_pos}, expected {expected_pos}"
            )

    def test_ros_node_step_with_temporal_alignment(self):
        """Verifies StandaloneGazeRosNode integrates the history buffer and temporal alignment."""
        node = StandaloneGazeRosNode()
        assert hasattr(node, "camera_latency_s")
        assert node.camera_latency_s == 0.050

        # Simulate encoder stream
        t_base = 200.0
        for i in range(10):
            t = t_base + (i * 0.020)
            pos = -10.0 + (i * 2.0)
            node.runtime.update_head_feedback(pos, velocity_deg_s=100.0, timestamp=t, source="/head/state")

        # Step frame captured at t = 200.080 (pos should be -2.0 deg)
        t_capture = 200.080
        det = [Detection(x=320, y=240, w=100, h=100, confidence=0.9)]
        res = node.step_frame(det, frame_size=(640, 480), timestamp=t_capture)

        # Tracked head angle must be aligned to t_capture (-2.0 deg)
        assert node.runtime.tracker.head_angle_deg == pytest.approx(-2.0, abs=1e-2)
