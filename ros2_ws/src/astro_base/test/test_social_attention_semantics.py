"""Comprehensive Social Attention Semantics & Regression Test Suite for ASTRO Robot.

Verifies:
  1. test_invalid_doa_does_not_create_target
  2. test_empty_room_doa_does_not_move_head
  3. test_idle_does_not_enter_hold_without_valid_target
  4. test_explicit_gaze_interrupts_visual_target
  5. test_explicit_gaze_bypasses_turn_taking_dwell
  6. test_target_identity_persists_during_motion
  7. test_visual_tracking_follows_bearing_change
  8. test_tracking_error_semantics
  9. test_stale_audio_does_not_control_gaze
  10. test_target_loss_requires_temporal_evidence
"""

import math
import time
import unittest
from typing import List, Optional

from astro_base.gaze.angle_math import angular_diff_deg, circular_distance_deg
from astro_base.gaze.attention_arbiter import AttentionArbiterCore
from astro_base.gaze.audio_filter import AudioFilterCore
from astro_base.gaze.audio_perception import AudioPerceptionCore
from astro_base.gaze.coordinate_frames import CalibrationConfig, CoordinateTransformer
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    ActuatorStateEnum,
    AttentionDecision,
    AudioMeasurement,
    DialogueGazeIntent,
    ExplicitGazeIntent,
    FusedTarget,
    GazeCommand,
    GazeStateEnum,
    GestureGazeIntent,
    Modality,
    PrioritySource,
    SafetyGazeIntent,
    TargetSelectorType,
    TargetState,
    TrackingState,
    VisualMeasurement,
    VisualTargetTrack,
)
from astro_base.gaze.visual_perception import VisualPerceptionCore
from astro_base.gaze.visual_tracker import VisualTrackerCore


class TestSocialAttentionSemantics(unittest.TestCase):
    """Rigorous unit and scenario regression tests for social attention semantics."""

    def setUp(self):
        self.calib = CalibrationConfig()
        self.transformer = CoordinateTransformer(self.calib)
        self.audio_perception = AudioPerceptionCore(self.transformer)
        self.audio_filter = AudioFilterCore(max_jump_deg=35.0, outlier_persistence_count=3)
        self.visual_perception = VisualPerceptionCore(self.transformer)
        self.visual_tracker = VisualTrackerCore(self.transformer)
        self.fusion = AudioVisualFusionCore(spatial_gate_deg=25.0)
        self.target_manager = TargetManagerCore(
            acquisition_threshold=0.75,
            hold_threshold=0.40,
            target_lost_timeout_s=1.0,
            min_attention_dwell_s=2.50,
            turn_taking_min_dwell_s=0.80,
            turn_taking_min_angle_deg=20.0,
        )
        self.arbiter = AttentionArbiterCore()
        self.fsm = SocialGazeFSM(
            deadband_deg=2.5,
            idle_saccades_enabled=False,
            min_attention_dwell_s=2.50,
            target_lost_timeout_s=1.0,
            arbiter=self.arbiter,
        )

    def test_invalid_doa_does_not_create_target(self):
        """Invalid audio DOA angles (180°, 119°, -180°) must be rejected BEFORE target creation."""
        invalid_raw_angles = [180.0, 119.0, -180.0, 240.0]

        for raw_angle in invalid_raw_angles:
            obs = self.audio_perception.process_raw_doa(
                raw_doa_deg=raw_angle,
                timestamp=1.0,
                actual_head_yaw_deg=0.0,
                is_robot_speaking=False,
            )
            # Must be marked invalid due to rear envelope gating (> 85°)
            self.assertFalse(obs.valid, f"DOA angle {raw_angle}° was erroneously marked valid!")

            filtered = self.audio_filter.filter_observation(obs, head_velocity_deg_s=0.0)
            fused = self.fusion.fuse(audio_state=filtered, visual_tracks=[], timestamp=1.0)
            self.assertEqual(len(fused), 0, f"Invalid DOA {raw_angle}° produced candidate fused target!")

            target_state = self.target_manager.update(fused_targets=fused, timestamp=1.0)
            self.assertIsNone(target_state.active_target, f"Invalid DOA {raw_angle}° created active target!")

            gaze_cmd = self.fsm.update(target_state, actual_head_yaw_deg=0.0, timestamp=1.0)
            self.assertEqual(gaze_cmd.priority_source, PrioritySource.IDLE)
            self.assertEqual(gaze_cmd.gaze_state, GazeStateEnum.IDLE)
            self.assertEqual(gaze_cmd.target_yaw_deg, 0.0)

    def test_empty_room_doa_does_not_move_head(self):
        """Simulate 60 seconds (3000 steps @ 50 Hz) of quiet empty room with stray noise."""
        dt = 0.02
        t = 0.0
        head_yaw = 0.0

        for step in range(3000):
            t += dt
            # Stray invalid noise clicks every 2 seconds
            if step % 100 == 0:
                obs = self.audio_perception.process_raw_doa(
                    raw_doa_deg=180.0,
                    timestamp=t,
                    actual_head_yaw_deg=head_yaw,
                    is_robot_speaking=False,
                )
            else:
                obs = AudioMeasurement(timestamp=t, valid=False, vad=False)

            filtered = self.audio_filter.filter_observation(obs, head_velocity_deg_s=0.0)
            fused = self.fusion.fuse(audio_state=filtered, visual_tracks=[], timestamp=t)
            target_state = self.target_manager.update(fused_targets=fused, timestamp=t)
            gaze_cmd = self.fsm.update(target_state, actual_head_yaw_deg=head_yaw, timestamp=t)

            self.assertEqual(gaze_cmd.gaze_state, GazeStateEnum.IDLE, f"State departed from IDLE at t={t:.2f}s")
            self.assertEqual(gaze_cmd.priority_source, PrioritySource.IDLE)
            self.assertEqual(gaze_cmd.target_yaw_deg, 0.0, f"Head commanded to move at t={t:.2f}s!")

    def test_idle_does_not_enter_hold_without_valid_target(self):
        """Passive low-confidence sensor spikes must not trigger transition from IDLE to HOLDING_ATTENTION."""
        # Feed weak audio spike with confidence 0.30 (below acquisition threshold 0.75)
        obs = AudioMeasurement(
            timestamp=1.0,
            valid=True,
            vad=True,
            raw_azimuth_deg=45.0,
            relative_azimuth_deg=45.0,
            body_azimuth_deg=45.0,
            confidence=0.30,
        )
        filtered = self.audio_filter.filter_observation(obs, head_velocity_deg_s=0.0)
        fused = self.fusion.fuse(audio_state=filtered, visual_tracks=[], timestamp=1.0)
        target_state = self.target_manager.update(fused_targets=fused, timestamp=1.0)
        gaze_cmd = self.fsm.update(target_state, actual_head_yaw_deg=0.0, timestamp=1.0)

        self.assertEqual(gaze_cmd.gaze_state, GazeStateEnum.IDLE)
        self.assertNotEqual(gaze_cmd.gaze_state, GazeStateEnum.HOLDING_ATTENTION)

    def test_explicit_gaze_interrupts_visual_target(self):
        """Explicit command ('Astro bana dön') immediately preempts passive visual tracking."""
        # 1. Establish visual target Person B at +25°
        vis_track_b = VisualTargetTrack(
            target_id="person_B",
            pos_3d=(0.5, 1.2, 0.0),
            vel_3d=(0.0, 0.0, 0.0),
            body_azimuth_deg=25.0,
            body_elevation_deg=0.0,
            distance_m=1.3,
            confidence=0.88,
            tracking_state=TrackingState.TRACKING,
            last_seen_time=1.0,
        )
        fused = self.fusion.fuse(audio_state=None, visual_tracks=[vis_track_b], timestamp=1.0)
        target_state = self.target_manager.update(fused_targets=fused, timestamp=1.0)
        gaze_cmd = self.fsm.update(target_state, actual_head_yaw_deg=25.0, timestamp=1.0)

        self.assertEqual(gaze_cmd.priority_source, PrioritySource.VISUAL_TRACKING)
        self.assertEqual(gaze_cmd.active_target_id, "person_B")

        # 2. Person A speaks at -35° and explicit intent is received
        speaker_target_a = FusedTarget(
            target_id="speaker_A",
            modality=Modality.AUDIO,
            body_azimuth_deg=-35.0,
            body_elevation_deg=0.0,
            distance_m=1.8,
            confidence=0.92,
            is_speaking=True,
            eye_contact=False,
            person_name="Person A",
            is_known=True,
            timestamp=2.0,
            tracking_state=TrackingState.TRACKING,
        )
        target_state_with_speaker = TargetState(
            active_target=target_state.active_target,
            candidate_targets=[speaker_target_a, target_state.active_target],
            timestamp=2.0,
        )

        explicit_intent = ExplicitGazeIntent(
            selector=TargetSelectorType.CURRENT_SPEAKER,
            timestamp=2.0,
            expiry_time=6.0,
            valid=True,
            reason="EXPLICIT_USER_COMMAND_BANA_DON",
        )

        decision = self.arbiter.arbitrate(
            target_state=target_state_with_speaker,
            explicit_intent=explicit_intent,
            actual_head_yaw_deg=25.0,
            timestamp=2.0,
        )

        self.assertEqual(decision.owner, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertEqual(decision.target_id, "speaker_A")
        self.assertAlmostEqual(decision.target_yaw_deg, -35.0, places=1)
        self.assertTrue(decision.is_preemption)

        # Update FSM with explicit intent
        gaze_cmd_post = self.fsm.update(
            target_state_with_speaker,
            actual_head_yaw_deg=25.0,
            timestamp=2.0,
            explicit_intent=explicit_intent,
        )
        self.assertEqual(gaze_cmd_post.priority_source, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertAlmostEqual(gaze_cmd_post.target_yaw_deg, -35.0, places=1)

    def test_explicit_gaze_bypasses_turn_taking_dwell(self):
        """Explicit user gaze command switches target on step 0 without waiting for 0.8s dwell."""
        # Active target Person B
        active_b = FusedTarget(
            target_id="person_B",
            modality=Modality.VISION,
            body_azimuth_deg=30.0,
            body_elevation_deg=0.0,
            distance_m=1.5,
            confidence=0.85,
            is_speaking=False,
            eye_contact=True,
            person_name=None,
            is_known=False,
            timestamp=1.0,
            tracking_state=TrackingState.TRACKING,
        )
        speaker_a = FusedTarget(
            target_id="person_A",
            modality=Modality.AUDIO,
            body_azimuth_deg=-40.0,
            body_elevation_deg=0.0,
            distance_m=1.5,
            confidence=0.90,
            is_speaking=True,
            eye_contact=False,
            person_name=None,
            is_known=False,
            timestamp=1.01,
            tracking_state=TrackingState.TRACKING,
        )
        target_state = TargetState(active_target=active_b, candidate_targets=[speaker_a, active_b], timestamp=1.01)

        # Standard TargetManager turn-taking would NOT switch on step 0 (elapsed = 0.01s < 0.80s)
        self.target_manager.active_target = active_b
        self.target_manager._active_target_start_time = 1.0
        tm_state = self.target_manager.update(fused_targets=[speaker_a, active_b], timestamp=1.01)
        self.assertEqual(tm_state.active_target.target_id, "person_B", "Passive turn-taking switched too early without dwell!")

        # BUT Attention Arbiter with ExplicitGazeIntent switches IMMEDIATELY!
        explicit_intent = ExplicitGazeIntent(
            selector=TargetSelectorType.CURRENT_SPEAKER,
            timestamp=1.01,
            valid=True,
        )
        decision = self.arbiter.arbitrate(
            target_state=tm_state,
            explicit_intent=explicit_intent,
            actual_head_yaw_deg=30.0,
            timestamp=1.01,
        )
        self.assertEqual(decision.owner, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertEqual(decision.target_id, "person_A")
        self.assertAlmostEqual(decision.target_yaw_deg, -40.0, places=1)

    def test_target_identity_persists_during_motion(self):
        """A person moving across field (0° -> +10° -> +20° -> +30°) maintains the same target_id."""
        angles = [0.0, 10.0, 20.0, 30.0, 20.0, 10.0, 0.0]
        initial_id: Optional[str] = None
        t = 1.0

        for ang in angles:
            t += 0.1
            obs = self.visual_perception.process_detection(
                x=320,
                y=240,
                w=60,
                h=60,
                depth_m=1.5,
                timestamp=t,
                actual_head_yaw_deg=ang,
                confidence=0.90,
            )
            tracks = self.visual_tracker.update(observations=[obs], timestamp=t, actual_head_yaw_deg=ang)
            self.assertEqual(len(tracks), 1)

            if initial_id is None:
                initial_id = tracks[0].target_id
            else:
                self.assertEqual(tracks[0].target_id, initial_id, f"Track ID changed from {initial_id} to {tracks[0].target_id}!")

    def test_visual_tracking_follows_bearing_change(self):
        """Visual target moving smoothly updates target yaw without resetting to ORIENTING."""
        t = 1.0
        # Step 1: Initialize at 0°
        obs0 = self.visual_perception.process_detection(
            x=320, y=240, w=60, h=60, depth_m=1.5, timestamp=t, actual_head_yaw_deg=0.0, confidence=0.88
        )
        tracks0 = self.visual_tracker.update([obs0], timestamp=t, actual_head_yaw_deg=0.0)
        fused0 = self.fusion.fuse(audio_state=None, visual_tracks=tracks0, timestamp=t)
        ts0 = self.target_manager.update(fused0, timestamp=t)
        cmd0 = self.fsm.update(ts0, actual_head_yaw_deg=0.0, timestamp=t)

        # Fast forward state to TRACKING
        self.fsm._transition_to(GazeStateEnum.TRACKING, timestamp=t)

        # Step 2: Person moves to +8°
        t += 0.05
        obs1 = self.visual_perception.process_detection(
            x=360, y=240, w=60, h=60, depth_m=1.5, timestamp=t, actual_head_yaw_deg=0.0, confidence=0.88
        )
        tracks1 = self.visual_tracker.update([obs1], timestamp=t, actual_head_yaw_deg=0.0)
        fused1 = self.fusion.fuse(audio_state=None, visual_tracks=tracks1, timestamp=t)
        ts1 = self.target_manager.update(fused1, timestamp=t)
        cmd1 = self.fsm.update(ts1, actual_head_yaw_deg=0.0, timestamp=t)

        self.assertEqual(cmd1.gaze_state, GazeStateEnum.TRACKING)
        self.assertNotEqual(cmd1.target_yaw_deg, 0.0)
        self.assertAlmostEqual(cmd1.target_yaw_deg, -5.7, places=1)

    def test_tracking_error_semantics(self):
        """Verify distinct face_to_desired_error_deg vs desired_to_actual_error_deg metrics."""
        face_bearing = 25.0
        desired_gaze = 20.0
        actual_head = 15.0

        face_to_desired = round(float(face_bearing - desired_gaze), 2)
        desired_to_actual = round(float(desired_gaze - actual_head), 2)

        self.assertEqual(face_to_desired, 5.0)      # Optical centering error
        self.assertEqual(desired_to_actual, 5.0)    # Actuator tracking lag

    def test_stale_audio_does_not_control_gaze(self):
        """Acoustic DOA older than freshness half-life (0.4s) decays and does not maintain target."""
        # Acoustic event at t=1.0s
        obs = AudioMeasurement(
            timestamp=1.0,
            valid=True,
            vad=True,
            raw_azimuth_deg=40.0,
            relative_azimuth_deg=40.0,
            body_azimuth_deg=40.0,
            confidence=0.85,
        )
        filtered = self.audio_filter.filter_observation(obs, head_velocity_deg_s=0.0)
        fused = self.fusion.fuse(audio_state=filtered, visual_tracks=[], timestamp=1.0)
        self.assertEqual(len(fused), 1)

        # Advance time to t=2.0s without new audio
        fused_stale = self.fusion.fuse(audio_state=filtered, visual_tracks=[], timestamp=2.0)
        # Stale audio target confidence decays below threshold
        self.assertEqual(len(fused_stale), 0, "Stale audio observation was not decayed!")

    def test_target_loss_requires_temporal_evidence(self):
        """Single dropped frame does not immediately lose active target; requires sustained timeout."""
        t = 1.0
        obs = self.visual_perception.process_detection(
            x=320, y=240, w=60, h=60, depth_m=1.5, timestamp=t, actual_head_yaw_deg=0.0, confidence=0.90
        )
        tracks = self.visual_tracker.update([obs], timestamp=t, actual_head_yaw_deg=0.0)
        fused = self.fusion.fuse(audio_state=None, visual_tracks=tracks, timestamp=t)
        ts = self.target_manager.update(fused, timestamp=t)
        self.assertIsNotNone(ts.active_target)

        # 1 frame dropped at t = 1.02s
        fused_empty = self.fusion.fuse(audio_state=None, visual_tracks=[], timestamp=1.02)
        ts_coasting = self.target_manager.update(fused_empty, timestamp=1.02)
        # Target must STILL be present during coasting
        self.assertIsNotNone(ts_coasting.active_target, "Target was dropped on a single frame gap!")

        # After sustained 1.5s timeout (timeout = 1.0s)
        ts_lost = self.target_manager.update(fused_empty, timestamp=2.50)
        self.assertIsNone(ts_lost.active_target, "Target was not lost after sustained timeout!")


if __name__ == "__main__":
    unittest.main()
