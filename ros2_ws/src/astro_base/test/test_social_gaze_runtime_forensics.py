"""ASTRO — Runtime Social Gaze Forensic Integration Test Suite.

Verifies:
  1. test_runtime_invalid_doa_cannot_create_target
  2. test_runtime_invalid_doa_cannot_move_head
  3. test_runtime_empty_room_is_stationary
  4. test_runtime_idle_cannot_enter_holding_attention
  5. test_runtime_same_target_motion_is_continuous
  6. test_runtime_holding_attention_has_persistence
  7. test_runtime_explicit_gaze_preempts_visual_target
  8. test_runtime_explicit_gaze_bypasses_turntaking
  9. test_runtime_current_speaker_has_spatial_target
  10. test_runtime_short_visual_dropout_does_not_return_center
  11. test_runtime_every_head_command_has_attention_reason
"""

import json
import math
import time
import unittest
from typing import List, Optional

from astro_base.gaze.angle_math import angular_diff_deg, circular_distance_deg, clamp_deg
from astro_base.gaze.attention_arbiter import AttentionArbiterCore
from astro_base.gaze.audio_filter import AudioFilterCore
from astro_base.gaze.audio_perception import AudioPerceptionCore
from astro_base.gaze.coordinate_frames import CalibrationConfig, CoordinateTransformer
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.motion_planner import MotionPlannerCore
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    ActuatorStateEnum,
    AudioMeasurement,
    AudioObservation,
    ExplicitGazeIntent,
    FilteredAudioState,
    FusedTarget,
    GazeCommand,
    GazeStateEnum,
    Modality,
    PrioritySource,
    TargetSelectorType,
    TargetState,
    TrackingState,
    VisualObservation,
    VisualTargetTrack,
)
from astro_base.gaze.visual_perception import VisualPerceptionCore
from astro_base.gaze.visual_tracker import VisualTrackerCore


class SimulatedSocialGazeRuntime:
    """End-to-end simulated node runtime matching SocialGazeNode pipeline."""

    def __init__(self):
        calib = CalibrationConfig()
        self.transformer = CoordinateTransformer(calib=calib)
        self.audio_perception = AudioPerceptionCore(transformer=self.transformer)
        self.audio_filter = AudioFilterCore()
        self.visual_perception = VisualPerceptionCore(transformer=self.transformer)
        self.visual_tracker = VisualTrackerCore()
        self.fusion = AudioVisualFusionCore(spatial_gate_deg=25.0, audio_freshness_half_life_s=0.40)
        self.target_manager = TargetManagerCore()
        self.arbiter = AttentionArbiterCore()
        self.fsm = SocialGazeFSM(
            arbiter=self.arbiter,
            min_attention_dwell_s=2.5,
            target_lost_timeout_s=5.0,
        )
        self.planner = MotionPlannerCore(max_velocity_deg_s=75.0, max_acceleration_deg_s2=180.0)

        # Node State
        self.actual_head_yaw_deg: float = 0.0
        self.actual_head_vel_deg_s: float = 0.0
        self.latest_audio_state: Optional[FilteredAudioState] = None
        self.latest_visual_tracks: List[VisualTargetTrack] = []
        self.is_robot_speaking: bool = False

        # Commanded outputs & Telemetry history
        self.commanded_head_angles: List[float] = []
        self.commanded_gaze_states: List[GazeStateEnum] = []
        self.commanded_priorities: List[PrioritySource] = []
        self.telemetry_history: List[dict] = []

    def feed_doa_deg(self, raw_doa_deg: float, timestamp: float, confidence: float = 0.85):
        """Simulates incoming /audio/doa message."""
        obs = self.audio_perception.process_raw_doa(
            raw_doa_deg=raw_doa_deg,
            timestamp=timestamp,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
            confidence=confidence,
            is_robot_speaking=self.is_robot_speaking,
        )
        self.latest_audio_state = self.audio_filter.filter_observation(
            obs=obs,
            head_velocity_deg_s=self.actual_head_vel_deg_s,
        )

    def feed_vision_world_yaw(self, world_yaw_deg: float, depth_m: float, timestamp: float, conf: float = 0.85):
        """Simulates a face at a fixed world bearing relative to base frame as head rotates."""
        cam_azimuth = angular_diff_deg(world_yaw_deg, self.actual_head_yaw_deg)
        half_hfov = self.transformer.calib.camera.hfov_deg / 2.0
        norm_u = -cam_azimuth / half_hfov
        u_px = 320.0 + norm_u * 320.0
        x_px = int(clamp_deg(u_px - 30, 0, 580))
        self.feed_vision_detection(x=x_px, y=210, w=60, h=60, depth_m=depth_m, timestamp=timestamp, conf=conf)

    def feed_vision_detection(self, x: int, y: int, w: int, h: int, depth_m: float, timestamp: float, conf: float = 0.85):
        """Simulates incoming /vision/json face detection."""
        obs = self.visual_perception.process_detection(
            x=x, y=y, w=w, h=h, depth_m=depth_m,
            timestamp=timestamp,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
            confidence=conf,
        )
        self.latest_visual_tracks = self.visual_tracker.update(
            observations=[obs],
            timestamp=timestamp,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
        )

    def feed_vision_dropout(self, timestamp: float):
        """Simulates video frame with no faces."""
        self.latest_visual_tracks = self.visual_tracker.update(
            observations=[],
            timestamp=timestamp,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
        )

    def feed_explicit_gaze(self, selector: str = "CURRENT_SPEAKER", target_yaw_deg: Optional[float] = None, timestamp: float = 0.0):
        """Simulates incoming /behavior/explicit_gaze command."""
        intent = ExplicitGazeIntent(
            selector=TargetSelectorType(selector),
            target_yaw_deg=target_yaw_deg,
            confidence=1.0,
            timestamp=timestamp,
            expiry_time=timestamp + 4.0,
            valid=True,
            reason=f"EXPLICIT_TEST_{selector}",
        )
        self.fsm.set_explicit_gaze_intent(intent)

    def step(self, timestamp: float):
        """Executes 50 Hz control loop cycle."""
        # 1. Multimodal Sensor Fusion
        fused_targets = self.fusion.fuse(
            audio_state=self.latest_audio_state,
            visual_tracks=self.latest_visual_tracks,
            timestamp=timestamp,
        )

        # 2. Target Manager
        target_state = self.target_manager.update(
            fused_targets=fused_targets,
            timestamp=timestamp,
        )

        # 3. Social Gaze FSM
        gaze_cmd = self.fsm.update(
            target_state=target_state,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
            timestamp=timestamp,
            actual_head_vel_deg_s=self.actual_head_vel_deg_s,
        )

        # 4. Motion Planner
        traj_point = self.planner.plan_step(
            gaze_cmd=gaze_cmd,
            actual_pos_deg=self.actual_head_yaw_deg,
            timestamp=timestamp,
        )

        # 5. Actuator Plant (simulated closed-loop physical follow)
        dt = 0.02
        lag = 0.25
        self.actual_head_yaw_deg += lag * (traj_point.position_deg - self.actual_head_yaw_deg)
        self.actual_head_vel_deg_s = traj_point.velocity_deg_s

        # 6. Record telemetry
        face_bearing = self.latest_visual_tracks[0].body_azimuth_deg if self.latest_visual_tracks else None
        face_to_desired = round(float(face_bearing - gaze_cmd.target_yaw_deg), 2) if face_bearing is not None else 0.0
        desired_to_actual = round(float(gaze_cmd.target_yaw_deg - self.actual_head_yaw_deg), 2)

        telemetry = {
            "timestamp": timestamp,
            "gaze_state": gaze_cmd.gaze_state,
            "priority": gaze_cmd.priority_source,
            "attention_reason": self.fsm.last_decision.reason if self.fsm.last_decision else "NONE",
            "active_target_id": gaze_cmd.active_target_id,
            "target_yaw_deg": gaze_cmd.target_yaw_deg,
            "actual_yaw_deg": self.actual_head_yaw_deg,
            "face_to_desired_error_deg": face_to_desired,
            "desired_to_actual_error_deg": desired_to_actual,
            "hold_enter_reason": self.fsm.hold_enter_reason,
            "hold_exit_reason": self.fsm.hold_exit_reason,
        }
        self.telemetry_history.append(telemetry)
        self.commanded_head_angles.append(gaze_cmd.target_yaw_deg)
        self.commanded_gaze_states.append(gaze_cmd.gaze_state)
        self.commanded_priorities.append(gaze_cmd.priority_source)

        return gaze_cmd, traj_point


class TestSocialGazeRuntimeForensics(unittest.TestCase):
    """Forensic verification test suite for runtime integration requirements."""

    def setUp(self):
        self.runtime = SimulatedSocialGazeRuntime()

    def test_runtime_invalid_doa_cannot_create_target(self):
        """Invalid audio DOA angles (+180°, +178.9°, +135°, -115.9°, +119.4°, -180°) must be marked invalid and produce 0 candidate targets."""
        invalid_angles = [180.0, 178.9, 135.0, -115.9, 119.4, -180.0]
        t = 1.0

        for raw in invalid_angles:
            self.runtime.feed_doa_deg(raw, timestamp=t)
            self.assertFalse(self.runtime.latest_audio_state.valid, f"DOA {raw}° should be marked invalid!")
            self.assertEqual(self.runtime.latest_audio_state.confidence, 0.0)

            # Step runtime
            cmd, _ = self.runtime.step(t)
            self.assertIsNone(cmd.active_target_id)
            self.assertEqual(cmd.target_yaw_deg, 0.0, f"Head commanded to non-zero for invalid angle {raw}°!")
            self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
            t += 0.02

    def test_runtime_invalid_doa_cannot_move_head(self):
        """Feeding 100 invalid DOA samples over 2 seconds must result in exactly 0.0° head movement."""
        t = 1.0
        for i in range(100):
            noisy_rear_angle = 180.0 if i % 2 == 0 else 135.0
            self.runtime.feed_doa_deg(noisy_rear_angle, timestamp=t)
            cmd, pt = self.runtime.step(t)
            self.assertEqual(cmd.target_yaw_deg, 0.0)
            self.assertEqual(pt.position_deg, 0.0)
            self.assertAlmostEqual(self.runtime.actual_head_yaw_deg, 0.0, delta=0.01)
            t += 0.02

    def test_runtime_empty_room_is_stationary(self):
        """Empty room (120 seconds / 6000 steps @ 50 Hz): Spontaneous head motion = 0, attention changes = 0."""
        t = 1.0
        for _ in range(6000):
            cmd, pt = self.runtime.step(t)
            self.assertEqual(cmd.target_yaw_deg, 0.0)
            self.assertEqual(cmd.priority_source, PrioritySource.IDLE)
            self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
            t += 0.02

        self.assertAlmostEqual(self.runtime.actual_head_yaw_deg, 0.0, delta=0.01)

    def test_runtime_idle_cannot_enter_holding_attention(self):
        """IDLE state with no valid target must never jump to HOLDING_ATTENTION."""
        t = 1.0
        for _ in range(50):
            cmd, _ = self.runtime.step(t)
            self.assertNotEqual(cmd.gaze_state, GazeStateEnum.HOLDING_ATTENTION)
            self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
            t += 0.02

    def test_runtime_same_target_motion_is_continuous(self):
        """A person moving across FOV maintains continuous target identity without orienting resets."""
        t = 1.0
        # Person moves from optical center (320px) to right (440px -> 560px)
        pixels = [320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560]

        target_ids = set()
        for px in pixels:
            for _ in range(10):
                self.runtime.feed_vision_detection(x=px-30, y=210, w=60, h=60, depth_m=1.5, timestamp=t)
                cmd, _ = self.runtime.step(t)
                if cmd.active_target_id:
                    target_ids.add(cmd.active_target_id)
                t += 0.02

        # Exactly 1 persistent target identity throughout the trajectory
        self.assertEqual(len(target_ids), 1, f"Expected 1 persistent target ID, got: {target_ids}")

    def test_runtime_holding_attention_has_persistence(self):
        """Stationary target in view stays in HOLDING_ATTENTION without frame-by-frame TRACKING chatter."""
        t = 1.0
        # Initialize target and settle head
        for _ in range(40):
            self.runtime.feed_vision_detection(x=290, y=210, w=60, h=60, depth_m=1.5, timestamp=t)
            self.runtime.step(t)
            t += 0.02

        # Verify settled in HOLDING_ATTENTION
        cmd, _ = self.runtime.step(t)
        self.assertEqual(cmd.gaze_state, GazeStateEnum.HOLDING_ATTENTION)

        # Feed 50 frames with slight micro-jitter (±2 pixels)
        transitions = []
        for i in range(50):
            jitter_x = 290 + (1 if i % 2 == 0 else -1)
            self.runtime.feed_vision_detection(x=jitter_x, y=210, w=60, h=60, depth_m=1.5, timestamp=t)
            cmd, _ = self.runtime.step(t)
            transitions.append(cmd.gaze_state)
            t += 0.02

        # Must maintain 100% HOLDING_ATTENTION without chattering to TRACKING
        tracking_chatter_count = sum(1 for s in transitions if s == GazeStateEnum.TRACKING)
        self.assertEqual(tracking_chatter_count, 0, f"Detected {tracking_chatter_count} unwanted chatter transitions to TRACKING!")

    def test_runtime_explicit_gaze_preempts_visual_target(self):
        """Explicit command ('Astro bana dön') immediately preempts active visual track of Person B."""
        t = 1.0
        # Step 1: Lock on Person B at +30°
        for _ in range(40):
            self.runtime.feed_vision_detection(x=150, y=210, w=60, h=60, depth_m=1.5, timestamp=t)
            self.runtime.step(t)
            t += 0.02

        # Verify settled on Person B
        cmd_before, _ = self.runtime.step(t)
        self.assertEqual(cmd_before.priority_source, PrioritySource.VISUAL_TRACKING)

        # Step 2: Person A issues explicit gaze command from -40°
        self.runtime.feed_doa_deg(raw_doa_deg=320.0, timestamp=t)  # -40° sound
        self.runtime.feed_explicit_gaze(selector="CURRENT_SPEAKER", target_yaw_deg=-40.0, timestamp=t)

        cmd_after, _ = self.runtime.step(t)
        self.assertEqual(cmd_after.priority_source, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertAlmostEqual(cmd_after.target_yaw_deg, -40.0, delta=1.0)

    def test_runtime_explicit_gaze_bypasses_turntaking(self):
        """Explicit command switches attention on step 0 without waiting for 0.8s turn-taking dwell."""
        t = 1.0
        # Person B tracking
        for _ in range(30):
            self.runtime.feed_vision_detection(x=200, y=210, w=60, h=60, depth_m=1.5, timestamp=t)
            self.runtime.step(t)
            t += 0.02

        # Voice command arrives on step 0
        self.runtime.feed_explicit_gaze(selector="CURRENT_SPEAKER", target_yaw_deg=45.0, timestamp=t)
        cmd, _ = self.runtime.step(t)

        self.assertEqual(cmd.priority_source, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertAlmostEqual(cmd.target_yaw_deg, 45.0, delta=1.0)

    def test_runtime_current_speaker_has_spatial_target(self):
        """CURRENT_SPEAKER with no spatial direction reports UNRESOLVED_CURRENT_SPEAKER_POSITION."""
        t = 1.0
        # No DOA and no visual candidates
        self.runtime.feed_explicit_gaze(selector="CURRENT_SPEAKER", target_yaw_deg=None, timestamp=t)
        cmd, _ = self.runtime.step(t)

        self.assertEqual(cmd.priority_source, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertEqual(self.runtime.fsm.last_decision.reason, "UNRESOLVED_CURRENT_SPEAKER_POSITION")
        self.assertEqual(cmd.target_yaw_deg, 0.0)

    def test_runtime_short_visual_dropout_does_not_return_center(self):
        """Brief visual dropout (2.0s) coasts and does not immediately return to 0.0° center."""
        t = 1.0
        # Track person stationary in the room at +30.0° body yaw
        for _ in range(60):
            self.runtime.feed_vision_world_yaw(world_yaw_deg=30.0, depth_m=1.5, timestamp=t)
            self.runtime.step(t)
            t += 0.02

        last_target_yaw = self.runtime.commanded_head_angles[-1]
        self.assertAlmostEqual(last_target_yaw, 30.0, delta=2.0)

        # Drop vision for 2.0s (100 frames)
        for _ in range(100):
            self.runtime.feed_vision_dropout(timestamp=t)
            cmd, _ = self.runtime.step(t)
            # Head target yaw must be preserved at last seen position during dropout (+30.0°, not 0.0°)
            self.assertAlmostEqual(cmd.target_yaw_deg, 30.0, delta=2.0)
            self.assertGreater(cmd.target_yaw_deg, 20.0, "Head must NOT return to 0° center during short dropout!")
            self.assertIn(cmd.gaze_state, (GazeStateEnum.HOLDING_ATTENTION, GazeStateEnum.TARGET_LOST))
            t += 0.02

    def test_runtime_every_head_command_has_attention_reason(self):
        """Every head command produced has an upstream explainable attention reason and separate errors."""
        t = 1.0
        for i in range(50):
            if i < 25:
                self.runtime.feed_vision_detection(x=290, y=210, w=60, h=60, depth_m=1.5, timestamp=t)
            cmd, _ = self.runtime.step(t)
            telemetry = self.runtime.telemetry_history[-1]

            self.assertTrue(len(telemetry["attention_reason"]) > 0)
            self.assertIn("face_to_desired_error_deg", telemetry)
            self.assertIn("desired_to_actual_error_deg", telemetry)
            t += 0.02


if __name__ == "__main__":
    unittest.main()
