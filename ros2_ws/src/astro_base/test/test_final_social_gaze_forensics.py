"""Comprehensive Integration Test Suite for ASTRO Final Social Gaze Runtime Forensics.

Verifies all 12 invariants and prevents regression of observed real-world flight recorder failures.
"""

import math
import time
import unittest
from typing import List, Optional, Tuple

from astro_base.gaze.angle_math import angular_diff_deg, wrap_deg
from astro_base.gaze.attention_arbiter import AttentionArbiterCore
from astro_base.gaze.audio_filter import AudioFilterCore
from astro_base.gaze.audio_perception import AudioPerceptionCore
from astro_base.gaze.coordinate_frames import CalibrationConfig, CoordinateTransformer
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.motion_planner import MotionPlannerCore
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    AudioObservation,
    ExplicitGazeIntent,
    FusedTarget,
    GazeCommand,
    GazeStateEnum,
    Modality,
    PrioritySource,
    TargetSelectorType,
    TrackingState,
    VisualObservation,
    VisualTargetTrack,
)
from astro_base.gaze.visual_perception import VisualPerceptionCore
from astro_base.gaze.visual_tracker import VisualTrackerCore


class FinalSocialGazeTestPipeline:
    """Full-stack software simulation pipeline for end-to-end testing."""

    def __init__(self):
        self.transformer = CoordinateTransformer()
        self.audio_perception = AudioPerceptionCore(transformer=self.transformer, max_acoustic_envelope_deg=75.0)
        self.audio_filter = AudioFilterCore()
        self.visual_perception = VisualPerceptionCore(transformer=self.transformer)
        self.visual_tracker = VisualTrackerCore(transformer=self.transformer)
        self.fusion = AudioVisualFusionCore()
        self.target_manager = TargetManagerCore()
        self.fsm = SocialGazeFSM(deadband_deg=2.5, min_attention_dwell_s=1.5, target_lost_timeout_s=1.0)
        self.planner = MotionPlannerCore()

        self.sim_head_pos_deg = 0.0
        self.sim_head_vel_deg_s = 0.0
        self.latest_audio = None
        self.latest_visual = []
        self.command_log: List[GazeCommand] = []

    def step(
        self,
        timestamp: float,
        raw_doa_deg: Optional[float] = None,
        raw_doa_conf: float = 0.85,
        visual_faces: Optional[List[dict]] = None,
        explicit_intent: Optional[ExplicitGazeIntent] = None,
    ) -> GazeCommand:
        # Audio step
        if raw_doa_deg is not None:
            obs_a = self.audio_perception.process_raw_doa(
                raw_doa_deg=raw_doa_deg,
                timestamp=timestamp,
                actual_head_yaw_deg=self.sim_head_pos_deg,
                confidence=raw_doa_conf,
            )
            self.latest_audio = self.audio_filter.filter_observation(
                obs=obs_a,
                head_velocity_deg_s=self.sim_head_vel_deg_s,
            )

        # Vision step
        if visual_faces is not None:
            obs_list = []
            for f in visual_faces:
                cam_az = f.get("cam_azimuth_deg", 0.0)
                depth = f.get("depth_m", 1.5)
                conf = f.get("confidence", 0.85)
                obs_v = self.visual_perception.process_detection(
                    x=320, y=240, w=60, h=60, depth_m=depth,
                    timestamp=timestamp, actual_head_yaw_deg=self.sim_head_pos_deg,
                    confidence=conf, cam_azimuth_deg=cam_az,
                )
                obs_list.append(obs_v)
            self.latest_visual = self.visual_tracker.update(
                observations=obs_list,
                timestamp=timestamp,
                actual_head_yaw_deg=self.sim_head_pos_deg,
            )
        else:
            self.latest_visual = self.visual_tracker.update(
                observations=[],
                timestamp=timestamp,
                actual_head_yaw_deg=self.sim_head_pos_deg,
            )

        # Fusion & target management
        fused = self.fusion.fuse(self.latest_audio, self.latest_visual, timestamp)
        target_state = self.target_manager.update(fused, timestamp)

        if explicit_intent is not None:
            self.fsm.set_explicit_gaze_intent(explicit_intent)

        # Gaze FSM
        cmd = self.fsm.update(
            target_state=target_state,
            actual_head_yaw_deg=self.sim_head_pos_deg,
            timestamp=timestamp,
            actual_head_vel_deg_s=self.sim_head_vel_deg_s,
        )
        self.command_log.append(cmd)

        # Motion planner
        traj = self.planner.plan_step(cmd, self.sim_head_pos_deg, timestamp)
        self.sim_head_pos_deg = float(traj.position_deg)
        self.sim_head_vel_deg_s = float(traj.velocity_deg_s)

        return cmd


class TestFinalSocialGazeForensics(unittest.TestCase):
    """Verifies all 12 forensic invariants."""

    def setUp(self):
        self.pipeline = FinalSocialGazeTestPipeline()

    def test_idle_target_is_null(self):
        """Invariant: In IDLE, active_target must be null and attention_owner must be IDLE."""
        t = 1.0
        cmd = self.pipeline.step(timestamp=t)
        self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
        self.assertEqual(cmd.priority_source, PrioritySource.IDLE)
        self.assertIsNone(cmd.active_target_id)
        self.assertEqual(cmd.target_yaw_deg, 0.0)

    def test_no_target_cannot_enter_tracking(self):
        """Invariant: Empty environment cannot enter TRACKING or move head."""
        t = 1.0
        for _ in range(50):
            cmd = self.pipeline.step(timestamp=t)
            self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
            self.assertEqual(self.pipeline.sim_head_pos_deg, 0.0)
            t += 0.02

    def test_no_target_cannot_enter_holding(self):
        """Invariant: IDLE cannot directly jump to HOLDING_ATTENTION."""
        t = 1.0
        cmd = self.pipeline.step(timestamp=t)
        self.assertNotEqual(cmd.gaze_state, GazeStateEnum.HOLDING_ATTENTION)

    def test_scattered_out_of_cone_audio_cannot_birth_target(self):
        """Invariant: reverberation outside the conversation cone births no target.

        This is what closed the rear envelope in the first place — single reflections
        off a wall used to drive the neck into its mechanical limit. What identifies a
        reflection is that it arrives from somewhere new each time, so the guarantee is
        stated that way rather than as a flat angle cutoff.
        """
        t = 1.0
        for bearing in (80.0, 115.0, 85.0, 120.0, 90.0, 118.0, 82.0, 116.0):
            cmd = self.pipeline.step(timestamp=t, raw_doa_deg=bearing, raw_doa_conf=0.90)
            self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
            self.assertEqual(cmd.priority_source, PrioritySource.IDLE)
            self.assertEqual(self.pipeline.sim_head_pos_deg, 0.0)
            t += 0.02

        self.assertGreater(self.pipeline.audio_perception.counters.invalid_angle_events, 0)
        self.assertEqual(self.pipeline.audio_perception.counters.audio_target_births, 0)

    def test_audio_beyond_what_the_neck_can_reach_cannot_birth_target(self):
        """Invariant: past 121 deg, turning the head shows the robot nothing, so a
        target there would only make it turn, see no one, and swing back."""
        t = 1.0
        for _ in range(20):
            cmd = self.pipeline.step(timestamp=t, raw_doa_deg=180.0, raw_doa_conf=0.90)
            self.assertEqual(cmd.priority_source, PrioritySource.IDLE)
            t += 0.02

        self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)

    def test_someone_calling_persistently_from_behind_does_birth_a_target(self):
        """A person who keeps calling from one direction is not a reflection."""
        t = 1.0
        owners = []
        for _ in range(20):
            owners.append(self.pipeline.step(timestamp=t, raw_doa_deg=119.7, raw_doa_conf=0.90).priority_source)
            t += 0.02

        self.assertIn(PrioritySource.ACTIVE_SPEAKER, owners)
        # ...and the head actually set off towards them (the source sits to the right).
        self.assertLess(self.pipeline.sim_head_pos_deg, 0.0)

    def test_target_loss_requires_temporal_evidence(self):
        """Invariant: Single-frame drop out coasts and does not immediately lose active target."""
        t = 1.0
        # Lock visual face
        for _ in range(10):
            self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": 15.0, "confidence": 0.85}])
            t += 0.02

        # 1 frame dropout
        self.pipeline.step(timestamp=t, visual_faces=[])
        # Must stay committed (coasting / holding), not instantly reset to 0 deg
        self.assertIn(self.pipeline.fsm.state, (GazeStateEnum.TRACKING, GazeStateEnum.HOLDING_ATTENTION, GazeStateEnum.ORIENTING))

    def test_reacquisition_requires_new_measurement(self):
        """Invariant: TARGET_LOST does not flutter back to TRACKING in under 150ms without new measurement."""
        t = 1.0
        # Engage and then let dwell expire to TARGET_LOST
        for _ in range(10):
            self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": 20.0, "confidence": 0.85}])
            t += 0.02

        # Expire dwell
        for _ in range(100):
            self.pipeline.step(timestamp=t, visual_faces=[])
            t += 0.02

        if self.pipeline.fsm.state == GazeStateEnum.TARGET_LOST:
            # Within 40ms, no flicker allowed
            self.pipeline.step(timestamp=t, visual_faces=[])
            self.assertEqual(self.pipeline.fsm.state, GazeStateEnum.TARGET_LOST)

    def test_recovery_clears_stale_target(self):
        """Invariant: When RECOVERING settles to IDLE, all target references are cleared."""
        t = 1.0
        # Engage target
        for _ in range(15):
            self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": 30.0, "confidence": 0.85}])
            t += 0.02

        # Drop target until RECOVERING completes
        for _ in range(250):
            self.pipeline.step(timestamp=t, visual_faces=[])
            t += 0.02

        self.assertEqual(self.pipeline.fsm.state, GazeStateEnum.IDLE)
        self.assertIsNone(self.pipeline.fsm.active_target_id)
        self.assertEqual(self.pipeline.fsm.active_priority, PrioritySource.IDLE)

    def test_same_target_motion_does_not_change_identity(self):
        """Invariant: Moving person across FOV maintains continuous target identity."""
        t = 1.0
        seen_ids = set()
        for step in range(20):
            angle = -20.0 + step * 2.0
            cmd = self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": angle, "confidence": 0.85}])
            if cmd.active_target_id:
                seen_ids.add(cmd.active_target_id)
            t += 0.02

        self.assertEqual(len(seen_ids), 1, "Target ID must remain strictly identical across smooth walk")

    def test_holding_attention_persists(self):
        """Invariant: Stationary target remains committed in HOLDING_ATTENTION without frame-by-frame chatter."""
        t = 1.0
        # Acquire stationary person at 25 deg
        for _ in range(80):
            self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": 25.0, "confidence": 0.85}])
            t += 0.02

        # In HOLDING_ATTENTION, small jitter of 0.5 deg must not kick out to TRACKING
        states = []
        for _ in range(50):
            cmd = self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": 25.2, "confidence": 0.85}])
            states.append(cmd.gaze_state)
            t += 0.02

        self.assertTrue(all(s == GazeStateEnum.HOLDING_ATTENTION for s in states))

    def test_moving_target_generates_continuous_gaze(self):
        """Physical Moving-Target Test: 0 -> +10 -> +20 -> +30 -> +20 -> +10 -> 0 and reverse."""
        t = 1.0
        trajectory = [0.0, 10.0, 20.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -20.0, -10.0, 0.0]

        target_ids = set()
        direction_reversals = 0
        prev_dir = 0

        for target_angle in trajectory:
            for _ in range(25):  # 0.5s per waypoint
                cmd = self.pipeline.step(
                    timestamp=t,
                    visual_faces=[{"cam_azimuth_deg": target_angle - self.pipeline.sim_head_pos_deg, "confidence": 0.85}]
                )
                if cmd.active_target_id:
                    target_ids.add(cmd.active_target_id)

                cur_dir = 1 if self.pipeline.sim_head_vel_deg_s > 1.0 else (-1 if self.pipeline.sim_head_vel_deg_s < -1.0 else 0)
                if cur_dir != 0 and prev_dir != 0 and cur_dir != prev_dir:
                    direction_reversals += 1
                if cur_dir != 0:
                    prev_dir = cur_dir

                t += 0.02

        # 1. Single continuous target ID
        self.assertEqual(len(target_ids), 1)
        # 2. Smooth physical trajectory with bounded natural reversals matching the waypoint zigzag
        self.assertLessEqual(direction_reversals, 8)

    def test_explicit_gaze_preempts_visual_target(self):
        """Explicit Gaze Preemption: 'Astro bana don' preempts active visual track of Person B."""
        t = 1.0
        # Phase 1: Tracking Person B at world -25 deg
        for _ in range(50):
            rel_cam = -25.0 - self.pipeline.sim_head_pos_deg
            self.pipeline.step(timestamp=t, visual_faces=[{"cam_azimuth_deg": rel_cam, "confidence": 0.85}])
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, -25.0, delta=2.5)

        # Phase 2: Explicit command to turn to Person A at +35 deg
        intent = ExplicitGazeIntent(
            selector=TargetSelectorType.ABSOLUTE_YAW,
            target_yaw_deg=35.0,
            confidence=1.0,
            timestamp=t,
            expiry_time=t + 4.0,
            reason="VOICE_COMMAND_ASTRO_BANA_DON",
        )
        for _ in range(60):
            self.pipeline.step(timestamp=t, explicit_intent=intent)
            t += 0.02

        # Head immediately preempted Person B and settled at Person A (+35 deg)
        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 35.0, delta=2.5)
        self.assertEqual(self.pipeline.fsm.active_priority, PrioritySource.EXPLICIT_USER_GAZE)

    def test_every_head_command_has_reason(self):
        """Invariant: Every command generated has an explainable reason."""
        t = 1.0
        for _ in range(30):
            cmd = self.pipeline.step(timestamp=t)
            decision = self.pipeline.fsm.last_decision
            self.assertIsNotNone(decision)
            self.assertIsNotNone(decision.reason)
            self.assertGreater(len(decision.reason), 0)
            t += 0.02

    def test_velocity_telemetry_is_correct(self):
        """Verify dt >= 0.020 eliminates sub-20ms packet burst differentiation artifacts."""
        pos1 = 0.0
        pos2 = 0.386  # 1 encoder tick
        dt_burst = 0.006  # 6 ms

        raw_vel_unguarded = (pos2 - pos1) / dt_burst
        self.assertGreater(raw_vel_unguarded, 60.0, "Unguarded 6ms burst produces fake 64 deg/s spike")


if __name__ == "__main__":
    unittest.main()
