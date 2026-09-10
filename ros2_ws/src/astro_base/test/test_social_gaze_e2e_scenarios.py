"""Comprehensive End-to-End Replay Test Suite for ASTRO Social Gaze System.

Executes all 11 mandatory behavioral scenarios through the full integrated pipeline:
  Scenario 1: Stationary Speaker
  Scenario 2: Two Alternating Speakers (Turn-Taking)
  Scenario 3: Continuous Moving Person
  Scenario 4: Fast Sound Spike / Acoustic Outlier
  Scenario 5: Head Motion Self-Noise Suppression
  Scenario 6: Blind Spot Audio Cue -> Visual Re-Acquisition
  Scenario 7: Camera Blinking / Visual Coasting
  Scenario 8: Robot Self-Speech Suppression
  Scenario 9: Dialogue Gaze & Social Gestures
  Scenario 10: Watchdog & Emergency Safety Stop
  Scenario 11: Mechanical Joint Limit Boundary Clamping
"""

import math
import unittest
from typing import List, Optional

from astro_base.gaze.audio_filter import AudioFilterCore
from astro_base.gaze.audio_perception import AudioPerceptionCore
from astro_base.gaze.coordinate_frames import CalibrationConfig, CoordinateTransformer
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.head_controller import HeadControllerCore
from astro_base.gaze.motion_planner import MotionPlannerCore
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    AudioObservation,
    GazeStateEnum,
    Modality,
    PrioritySource,
    TrajectoryPoint,
    VisualObservation,
)
from astro_base.gaze.visual_perception import VisualPerceptionCore
from astro_base.gaze.visual_tracker import VisualTrackerCore


class IntegratedGazePipeline:
    """Full integrated headless instance of the social gaze pipeline for deterministic replay testing."""

    def __init__(self):
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
            min_attention_dwell_s=2.50,
            turn_taking_min_dwell_s=0.80,
        )
        self.fsm = SocialGazeFSM(deadband_deg=3.0, idle_saccades_enabled=False)
        self.planner = MotionPlannerCore(
            max_velocity_deg_s=75.0,
            max_acceleration_deg_s2=180.0,
            profile_type="trapezoidal",
        )
        self.head_ctrl = HeadControllerCore()

        # Simulated physical state
        self.sim_head_pos_deg = 0.0
        self.sim_head_vel_deg_s = 0.0

    def step(
        self,
        timestamp: float,
        dt: float = 0.02,
        raw_doa_deg: Optional[float] = None,
        raw_doa_conf: float = 0.85,
        visual_obs: Optional[List[VisualObservation]] = None,
        is_robot_speaking: bool = False,
    ) -> TrajectoryPoint:
        """Executes one 50 Hz control loop cycle through the entire pipeline."""
        # 1. Audio Pipeline
        audio_state = None
        if raw_doa_deg is not None:
            raw_obs = self.audio_perception.process_raw_doa(
                raw_doa_deg=raw_doa_deg,
                timestamp=timestamp,
                actual_head_yaw_deg=self.sim_head_pos_deg,
                confidence=raw_doa_conf,
                is_robot_speaking=is_robot_speaking,
            )
            audio_state = self.audio_filter.filter_observation(
                obs=raw_obs,
                head_velocity_deg_s=self.sim_head_vel_deg_s,
            )

        # 2. Visual Pipeline
        visual_tracks = []
        if visual_obs is not None:
            visual_tracks = self.visual_tracker.update(
                observations=visual_obs,
                timestamp=timestamp,
                actual_head_yaw_deg=self.sim_head_pos_deg,
            )

        # 3. Sensor Fusion
        fused = self.fusion.fuse(
            audio_state=audio_state,
            visual_tracks=visual_tracks,
            timestamp=timestamp,
        )

        # 4. Target Manager
        targets = self.target_manager.update(fused_targets=fused, timestamp=timestamp)

        # 5. Social Gaze FSM
        gaze_cmd = self.fsm.update(
            target_state=targets,
            actual_head_yaw_deg=self.sim_head_pos_deg,
            timestamp=timestamp,
        )

        # 6. Motion Planner
        traj_point = self.planner.plan_step(
            gaze_cmd=gaze_cmd,
            actual_pos_deg=self.sim_head_pos_deg,
            timestamp=timestamp,
        )

        # Simulate ideal physical motor following
        self.sim_head_pos_deg = traj_point.position_deg
        self.sim_head_vel_deg_s = traj_point.velocity_deg_s

        return traj_point


    def body_bearing_to_respeaker_doa(self, target_body_yaw_deg: float) -> float:
        """Converts desired body frame acoustic bearing into ReSpeaker raw CW 0..360° format."""
        head_yaw = self.sim_head_pos_deg
        head_rel = target_body_yaw_deg - head_yaw
        return (-head_rel) % 360.0


class TestAll11GazeScenarios(unittest.TestCase):
    def setUp(self):
        self.pipeline = IntegratedGazePipeline()

    def test_scenario_1_stationary_speaker(self):
        """Scenario 1: Speaker at +40° talks -> head turns and smoothly settles at +40°."""
        t = 1.0
        for _ in range(100):  # 2.0 seconds
            raw_doa = self.pipeline.body_bearing_to_respeaker_doa(40.0)
            pt = self.pipeline.step(timestamp=t, raw_doa_deg=raw_doa, raw_doa_conf=0.85)
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 40.0, delta=1.5)
        self.assertEqual(self.pipeline.fsm.state, GazeStateEnum.VISUAL_ACQUIRE)

    def test_scenario_2_two_alternating_speakers(self):
        """Scenario 2: Speaker A at +20°, then Speaker B at -35° speaks for >0.8s -> turn-taking switch."""
        t = 1.0
        # Phase 1: Speaker A engages for 1.5s
        for _ in range(75):
            raw_doa_a = self.pipeline.body_bearing_to_respeaker_doa(20.0)
            self.pipeline.step(timestamp=t, raw_doa_deg=raw_doa_a, raw_doa_conf=0.85)
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 20.0, delta=2.0)

        # Phase 2: Speaker B starts speaking at -35° for 1.2s
        for _ in range(60):
            raw_doa_b = self.pipeline.body_bearing_to_respeaker_doa(-35.0)
            self.pipeline.step(timestamp=t, raw_doa_deg=raw_doa_b, raw_doa_conf=0.85)
            t += 0.02

        # Head smoothly switched and settled at Speaker B (-35°)
        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, -35.0, delta=2.0)

    def test_scenario_3_continuous_moving_person(self):
        """Scenario 3: Person walks smoothly from -30° to +30° -> head tracks smoothly without jitter."""
        t = 1.0
        for i in range(150):  # 3.0s
            person_angle = -30.0 + (60.0 * (i / 150.0))
            head_yaw = self.pipeline.sim_head_pos_deg
            rel_angle = person_angle - head_yaw
            rad = math.radians(rel_angle)
            # Optical frame: Forward = +Z, Right = +X (so Left is -X)
            pos_3d = (-1.5 * math.sin(rad), 0.0, 1.5 * math.cos(rad))

            obs = VisualObservation(
                timestamp=t, valid=True, pos_3d_camera=pos_3d, depth_m=1.5,
                confidence=0.90, body_azimuth_deg=person_angle
            )
            self.pipeline.step(timestamp=t, visual_obs=[obs])
            t += 0.02

        # Settle at final position (+30°) for 0.3s
        for _ in range(15):
            head_yaw = self.pipeline.sim_head_pos_deg
            rel_angle = 30.0 - head_yaw
            rad = math.radians(rel_angle)
            obs_final = VisualObservation(
                timestamp=t, valid=True, pos_3d_camera=(-1.5 * math.sin(rad), 0.0, 1.5 * math.cos(rad)),
                depth_m=1.5, confidence=0.90, body_azimuth_deg=30.0
            )
            self.pipeline.step(timestamp=t, visual_obs=[obs_final])
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 30.0, delta=1.5)
        self.assertIn(self.pipeline.fsm.state, (GazeStateEnum.TRACKING, GazeStateEnum.HOLDING_ATTENTION))

    def test_scenario_4_fast_sound_spike_outlier(self):
        """Scenario 4: Single frame acoustic spike (+80°) while tracking person at 0° -> rejected."""
        t = 1.0
        # Establish stable target at 0°
        for _ in range(30):
            raw_doa = self.pipeline.body_bearing_to_respeaker_doa(0.0)
            self.pipeline.step(timestamp=t, raw_doa_deg=raw_doa, raw_doa_conf=0.85)
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 0.0, delta=1.0)

        # Single frame 80° spike
        raw_spike = self.pipeline.body_bearing_to_respeaker_doa(80.0)
        pt_spike = self.pipeline.step(timestamp=t, raw_doa_deg=raw_spike, raw_doa_conf=0.85)
        t += 0.02

        # Head position must NOT jump towards 80°
        self.assertLess(abs(self.pipeline.sim_head_pos_deg), 3.0)

    def test_scenario_5_head_motion_self_noise_attenuation(self):
        """Scenario 5: High head velocity attenuates confidence, preventing false positive lock."""
        t = 1.0
        # Artificially set high head velocity
        self.pipeline.sim_head_vel_deg_s = 65.0
        obs = AudioObservation(
            timestamp=t, valid=True, vad=True, body_azimuth_deg=45.0, confidence=0.80
        )
        filt = self.pipeline.audio_filter.filter_observation(obs, head_velocity_deg_s=65.0)
        # Confidence must be attenuated (<0.40) and motion_attenuated = True
        self.assertTrue(filt.motion_attenuated)
        self.assertLess(filt.confidence, 0.40)

    def test_scenario_6_blind_spot_audio_to_visual(self):
        """Scenario 6: Off-camera sound at +60° -> Head orients -> Visual face locks."""
        t = 1.0
        # Phase 1: Only sound at +60°
        for _ in range(50):
            raw_doa = self.pipeline.body_bearing_to_respeaker_doa(60.0)
            self.pipeline.step(timestamp=t, raw_doa_deg=raw_doa, raw_doa_conf=0.85)
            t += 0.02

        self.assertGreater(self.pipeline.sim_head_pos_deg, 45.0)


        # Phase 2: Camera now sees face at +60°
        for _ in range(50):
            head_yaw = self.pipeline.sim_head_pos_deg
            rel_angle = 60.0 - head_yaw
            rad = math.radians(rel_angle)
            pos_3d = (-2.0 * math.sin(rad), 0.0, 2.0 * math.cos(rad))
            obs = VisualObservation(
                timestamp=t, valid=True, pos_3d_camera=pos_3d, depth_m=2.0,
                confidence=0.90, body_azimuth_deg=60.0
            )
            self.pipeline.step(timestamp=t, visual_obs=[obs])
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 60.0, delta=1.5)
        self.assertIn(self.pipeline.fsm.state, (GazeStateEnum.TRACKING, GazeStateEnum.HOLDING_ATTENTION))

    def test_scenario_7_camera_blinking_coasting(self):
        """Scenario 7: Target locked, drops out for 3 frames -> coasting maintains gaze smoothly."""
        t = 1.0
        # Lock target at 25°
        for _ in range(50):
            head_yaw = self.pipeline.sim_head_pos_deg
            rel_angle = 25.0 - head_yaw
            rad = math.radians(rel_angle)
            pos_3d = (-1.65 * math.sin(rad), 0.0, 1.65 * math.cos(rad))
            obs = VisualObservation(
                timestamp=t, valid=True, pos_3d_camera=pos_3d, depth_m=1.65,
                confidence=0.90, body_azimuth_deg=25.0
            )
            self.pipeline.step(timestamp=t, visual_obs=[obs])
            t += 0.02

        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 25.0, delta=1.5)

        # 3 missing frames (blink / dropout)
        for _ in range(3):
            self.pipeline.step(timestamp=t, visual_obs=[])
            t += 0.02

        # Head must stay solidly held at ~25° without jerking to center
        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 25.0, delta=1.5)


    def test_scenario_8_robot_speech_suppression(self):
        """Scenario 8: Robot speaks -> raw sound is suppressed and not tracked."""
        t = 1.0
        # Robot is speaking
        for _ in range(50):
            self.pipeline.step(
                timestamp=t, raw_doa_deg=50.0, raw_doa_conf=0.85, is_robot_speaking=True
            )
            t += 0.02

        # Head remains at 0° (does not chase own echo)
        self.assertAlmostEqual(self.pipeline.sim_head_pos_deg, 0.0, delta=1.0)
        self.assertEqual(self.pipeline.fsm.state, GazeStateEnum.IDLE)

    def test_scenario_9_dialogue_gaze_and_gestures(self):
        """Scenario 9: Scripted nod gesture executes with priority over background tracking."""
        t = 1.0
        # Trigger nod gesture
        self.assertTrue(self.pipeline.fsm.trigger_gesture("nod", timestamp=t))
        pt = self.pipeline.step(timestamp=t)
        self.assertEqual(self.pipeline.fsm.active_priority, PrioritySource.GESTURE)

    def test_scenario_10_emergency_safety_stop(self):
        """Scenario 10: Emergency safety lock instantly commands 0° with SAFETY priority."""
        t = 1.0
        # Tracking at 40°
        obs = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(1.5, 1.25, 0.0), depth_m=1.95,
            confidence=0.90, body_azimuth_deg=40.0
        )
        for _ in range(40):
            self.pipeline.step(timestamp=t, visual_obs=[obs])
            t += 0.02

        # Trigger Safety Lock
        self.pipeline.fsm.set_safety_lock(True)
        pt_safe = self.pipeline.step(timestamp=t)
        self.assertEqual(self.pipeline.fsm.active_priority, PrioritySource.SAFETY)

    def test_scenario_11_mechanical_limit_wrapping(self):
        """Scenario 11: Target from +120° is clamped strictly to mechanical software limit."""
        t = 1.0
        obs = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(2.0, 1.5, 0.0), depth_m=2.5,
            confidence=0.90, body_azimuth_deg=120.0
        )
        for _ in range(80):
            self.pipeline.step(timestamp=t, visual_obs=[obs])
            t += 0.02

        self.assertLessEqual(abs(self.pipeline.sim_head_pos_deg), 90.0)
        self.assertAlmostEqual(abs(self.pipeline.sim_head_pos_deg), 75.0, delta=1.5)



if __name__ == "__main__":
    unittest.main()
