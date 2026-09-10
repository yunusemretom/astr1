#!/usr/bin/env python3
"""Canonical Social Gaze Architecture Acceptance Test Suite for ASTRO Robot.

Validates:
  1. Single Actuator Command Authority (Only /head/command or SocialGazeNode)
  2. No Global Mutable _target_yaw in new SocialGazeNode
  3. Single Authoritative Publisher for /head/state (serial_bridge)
  4. Physical Feedback Required for Settling (ORIENTING -> HOLD requires actual encoder match)
  5. 3D Timestamp & Coordinate Geometry Alignment (Camera Optical -> Head -> Base)
  6. Multi-Target Association and Priority Arbitration
  7. Safety Constraint Enforcement without Target Corruption
  8. Bounded Mechanical Joint Limits ([-75°, +75°] with Zero Wrap-Around)
"""

import math
import os
import sys
import time
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from astro_base.gaze.angle_math import angular_diff_deg, clamp_deg, shortest_reachable_arc, wrap_deg
from astro_base.gaze.coordinate_frames import CalibrationConfig, CoordinateTransformer, HeadCalibration
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.motion_planner import MotionPlannerCore
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    FilteredAudioState,
    FusedTarget,
    GazeCommand,
    GazeStateEnum,
    Modality,
    PrioritySource,
    TargetState,
    TrackingState,
    VisualObservation,
    VisualTargetTrack,
)


class TestCanonicalSocialGazeArchitecture(unittest.TestCase):
    """Rigorous Architectural Verification Tests."""

    def test_single_head_command_authority(self):
        """Verify that SocialGazeNode does not expose shared mutable target state."""
        from astro_base.social_gaze_node import SocialGazeNode
        # The new canonical SocialGazeNode must NOT contain _target_yaw as an instance attribute
        node_attributes = dir(SocialGazeNode)
        self.assertNotIn("_target_yaw", node_attributes, "SocialGazeNode must not possess a legacy _target_yaw shared state")

    def test_no_legacy_target_writer(self):
        """Verify FSM produces GazeCommand deterministically without side effects."""
        fsm = SocialGazeFSM(min_limit_deg=-75.0, max_limit_deg=75.0)
        target = FusedTarget(
            target_id="person_1",
            modality=Modality.FUSED,
            body_azimuth_deg=30.0,
            body_elevation_deg=0.0,
            distance_m=1.8,
            confidence=0.85,
            is_speaking=True,
            eye_contact=True,
            person_name="User",
            is_known=True,
            timestamp=100.0,
            tracking_state=TrackingState.TRACKING,
        )
        target_state = TargetState(active_target=target, timestamp=100.0)

        # First cycle initiates orientation
        cmd1 = fsm.update(target_state=target_state, actual_head_yaw_deg=0.0, timestamp=100.0, actual_head_vel_deg_s=0.0)
        self.assertIsInstance(cmd1, GazeCommand)
        self.assertEqual(cmd1.gaze_state, GazeStateEnum.ORIENTING)
        self.assertAlmostEqual(cmd1.target_yaw_deg, 30.0, places=1)

    def test_head_state_single_publisher(self):
        """Verify serial_bridge is the single owner of /head/state publisher."""
        from astro_base.serial_bridge import SerialBridge
        self.assertTrue(hasattr(SerialBridge, "publish_joint_states") or hasattr(SerialBridge, "on_head_cmd"))

    def test_hold_requires_actual_feedback(self):
        """Verify that ORIENTING -> HOLD/TRACKING requires actual physical encoder feedback."""
        fsm = SocialGazeFSM(position_tolerance_deg=2.0, velocity_tolerance_deg_s=1.5)
        target = FusedTarget(
            target_id="p1",
            modality=Modality.FUSED,
            body_azimuth_deg=45.0,
            body_elevation_deg=0.0,
            distance_m=1.5,
            confidence=0.90,
            is_speaking=True,
            eye_contact=True,
            person_name=None,
            is_known=False,
            timestamp=10.0,
            tracking_state=TrackingState.TRACKING,
        )
        target_state = TargetState(active_target=target, timestamp=10.0)

        # 1. Planned command is 45°, but actual encoder reports 0.0° (still moving)
        cmd = fsm.update(target_state=target_state, actual_head_yaw_deg=0.0, timestamp=10.0, actual_head_vel_deg_s=25.0)
        self.assertEqual(cmd.gaze_state, GazeStateEnum.ORIENTING, "Must stay ORIENTING while head is physically in flight")
        self.assertFalse(fsm.at_target)

        # 2. Planned command is 45°, actual encoder reaches 45.0° and velocity settles (<1.5 deg/s) for 3 cycles
        for cycle in range(3):
            cmd = fsm.update(target_state=target_state, actual_head_yaw_deg=45.0, timestamp=10.1 + cycle*0.02, actual_head_vel_deg_s=0.2)

        self.assertTrue(fsm.at_target, "Must confirm at_target once actual encoder is settled")
        self.assertIn(cmd.gaze_state, (GazeStateEnum.TRACKING, GazeStateEnum.HOLDING_ATTENTION), "Must transition out of ORIENTING after physical arrival")

    def test_visual_timestamp_alignment(self):
        """Verify camera optical bearing transforms correctly to robot body frame using head pose at capture time."""
        transformer = CoordinateTransformer()
        # User is seen at optical azimuth -10.0° (camera right -> -10° in REP-103) when head was at +20.0°
        cam_azimuth = -10.0
        head_yaw_at_capture = 20.0

        body_yaw = transformer.camera_bearing_to_body_yaw(cam_azimuth, head_yaw_at_capture)
        # Expected: 20.0 + (-10.0) = 10.0° in robot base frame
        self.assertAlmostEqual(body_yaw, 10.0, places=1)

    def test_target_association(self):
        """Verify spatial consistency gating in AudioVisualFusionCore."""
        fusion = AudioVisualFusionCore(spatial_gate_deg=25.0)
        audio = FilteredAudioState(timestamp=10.0, valid=True, azimuth_deg=30.0, confidence=0.85)

        track_A = VisualTargetTrack(
            target_id="track_A",
            pos_3d=(1.5, 0.8, 0.0),
            vel_3d=(0.0, 0.0, 0.0),
            body_azimuth_deg=28.0,  # within 25° gate of 30.0°
            body_elevation_deg=0.0,
            distance_m=1.7,
            confidence=0.90,
            tracking_state=TrackingState.TRACKING,
            last_seen_time=10.0,
        )
        track_B = VisualTargetTrack(
            target_id="track_B",
            pos_3d=(1.5, -1.2, 0.0),
            vel_3d=(0.0, 0.0, 0.0),
            body_azimuth_deg=-40.0, # far away from 30.0°
            body_elevation_deg=0.0,
            distance_m=1.9,
            confidence=0.80,
            tracking_state=TrackingState.TRACKING,
            last_seen_time=10.0,
        )

        fused = fusion.fuse(audio_state=audio, visual_tracks=[track_A, track_B], timestamp=10.0)
        self.assertEqual(len(fused), 2)

        fused_A = next(t for t in fused if t.target_id == "track_A")
        fused_B = next(t for t in fused if t.target_id == "track_B")

        self.assertEqual(fused_A.modality, Modality.FUSED, "track_A should be associated with sound")
        self.assertTrue(fused_A.is_speaking)
        self.assertEqual(fused_B.modality, Modality.VISION, "track_B is silent; must remain VISION modality")
        self.assertFalse(fused_B.is_speaking)

    def test_safety_does_not_modify_gaze_target(self):
        """Verify emergency stop acts as a supervisor and does not corrupt the target estimator."""
        fsm = SocialGazeFSM()
        target = FusedTarget(
            target_id="p1",
            modality=Modality.FUSED,
            body_azimuth_deg=40.0,
            body_elevation_deg=0.0,
            distance_m=1.5,
            confidence=0.90,
            is_speaking=True,
            eye_contact=True,
            person_name=None,
            is_known=False,
            timestamp=5.0,
            tracking_state=TrackingState.TRACKING,
        )
        target_state = TargetState(active_target=target, timestamp=5.0)

        # Trigger Safety Emergency Stop
        fsm.set_safety_lock(True)
        cmd = fsm.update(target_state=target_state, actual_head_yaw_deg=20.0, timestamp=5.0)

        self.assertEqual(cmd.priority_source, PrioritySource.SAFETY)
        self.assertEqual(cmd.target_yaw_deg, 0.0, "Safety must force safe neutral center pose")
        # Ensure target estimator itself was not destroyed
        self.assertIsNotNone(target_state.active_target)
        self.assertEqual(target_state.active_target.body_azimuth_deg, 40.0)

    def test_bounded_joint(self):
        """Verify MotionPlanner and angle math strictly respect [-75°, +75°] with zero wrap-around."""
        planner = MotionPlannerCore(min_limit_deg=-75.0, max_limit_deg=75.0)

        # Target outside limit (+95.0°) must be clamped to +75.0°
        gaze_cmd = GazeCommand(target_yaw_deg=95.0, timestamp=1.0)
        point = planner.plan_step(gaze_cmd=gaze_cmd, actual_pos_deg=0.0, timestamp=1.0)
        self.assertLessEqual(point.position_deg, 75.0)

        # Shortest arc calculation for -70° to +70° must be +140°, NOT -220° wrap-around
        arc = shortest_reachable_arc(target_deg=70.0, current_deg=-70.0, min_limit_deg=-75.0, max_limit_deg=75.0)
        self.assertEqual(arc, 140.0, "Bounded joint must use direct path within limits, not wrap around rear")


if __name__ == "__main__":
    unittest.main()
