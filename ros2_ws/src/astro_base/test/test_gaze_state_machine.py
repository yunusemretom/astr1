"""Unit tests for Social Gaze State Machine (FSM) and Priority Arbitration."""

import unittest
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.types import (
    FusedTarget,
    GazeStateEnum,
    Modality,
    PrioritySource,
    TargetState,
    TrackingState,
)


class TestGazeStateMachine(unittest.TestCase):
    def setUp(self):
        self.fsm = SocialGazeFSM(
            deadband_deg=3.0,
            idle_return_timeout_s=20.0,
            min_attention_dwell_s=2.50,
            target_lost_timeout_s=1.0,
            idle_saccades_enabled=False,
        )

    def test_initial_state_idle(self):
        """Initial state without targets is IDLE at 0.0°."""
        cmd = self.fsm.update(TargetState(active_target=None), actual_head_yaw_deg=0.0, timestamp=1.0)
        self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
        self.assertEqual(cmd.target_yaw_deg, 0.0)
        self.assertEqual(cmd.priority_source, PrioritySource.IDLE)

    def test_audio_target_initiates_orienting_then_visual_acquire(self):
        """Acoustic target at 45° triggers ORIENTING when error is large (>15°), then VISUAL_ACQUIRE."""
        t = 1.0
        aud_target = FusedTarget(
            target_id="spk_1", modality=Modality.AUDIO, body_azimuth_deg=45.0,
            body_elevation_deg=0.0, distance_m=1.8, confidence=0.80,
            is_speaking=True, eye_contact=False, person_name=None, is_known=False,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

        # 1. Head is at 0°, target is at 45° (error = 45° > 15°) -> ORIENTING
        cmd1 = self.fsm.update(TargetState(active_target=aud_target), actual_head_yaw_deg=0.0, timestamp=t)
        self.assertEqual(cmd1.gaze_state, GazeStateEnum.ORIENTING)
        self.assertEqual(cmd1.target_yaw_deg, 45.0)
        self.assertEqual(cmd1.priority_source, PrioritySource.ACTIVE_SPEAKER)

        # 2. Head has rotated to 44.5° and settled -> VISUAL_ACQUIRE
        for _ in range(3):
            t += 0.02
            cmd2 = self.fsm.update(
                TargetState(active_target=aud_target),
                actual_head_yaw_deg=44.5,
                timestamp=t,
                actual_head_vel_deg_s=0.1,
            )
        self.assertEqual(cmd2.gaze_state, GazeStateEnum.VISUAL_ACQUIRE)
        self.assertEqual(cmd2.target_yaw_deg, 45.0)


    def test_visual_target_triggers_tracking(self):
        """Visual target when aligned enters TRACKING state."""
        t = 1.0
        vis_target = FusedTarget(
            target_id="person_1", modality=Modality.FUSED, body_azimuth_deg=20.0,
            body_elevation_deg=0.0, distance_m=1.5, confidence=0.85,
            is_speaking=True, eye_contact=True, person_name="Alice", is_known=True,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

        cmd = self.fsm.update(TargetState(active_target=vis_target), actual_head_yaw_deg=18.0, timestamp=t)
        self.assertEqual(cmd.gaze_state, GazeStateEnum.TRACKING)
        self.assertEqual(cmd.target_yaw_deg, 20.0)

    def test_deadband_rejects_micro_jitter(self):
        """Small target changes (<3.0° deadband) do NOT update commanded angle."""
        t = 1.0
        target1 = FusedTarget(
            target_id="person_1", modality=Modality.FUSED, body_azimuth_deg=20.0,
            body_elevation_deg=0.0, distance_m=1.5, confidence=0.85,
            is_speaking=True, eye_contact=True, person_name="Alice", is_known=True,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )
        cmd1 = self.fsm.update(TargetState(active_target=target1), actual_head_yaw_deg=20.0, timestamp=t)
        self.assertEqual(cmd1.target_yaw_deg, 20.0)

        # Micro-jitter: Target reports 21.2° (delta = 1.2° < 3.0° deadband)
        t += 0.1
        target2 = FusedTarget(
            target_id="person_1", modality=Modality.FUSED, body_azimuth_deg=21.2,
            body_elevation_deg=0.0, distance_m=1.5, confidence=0.85,
            is_speaking=True, eye_contact=True, person_name="Alice", is_known=True,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )
        cmd2 = self.fsm.update(TargetState(active_target=target2), actual_head_yaw_deg=20.0, timestamp=t)
        # Commanded target must stay locked at 20.0°!
        self.assertEqual(cmd2.target_yaw_deg, 20.0)

    def test_safety_override_preempts_all_behaviors(self):
        """Safety lock immediately forces head to 0.0° and sets SAFETY priority."""
        t = 1.0
        target = FusedTarget(
            target_id="person_1", modality=Modality.FUSED, body_azimuth_deg=40.0,
            body_elevation_deg=0.0, distance_m=1.5, confidence=0.85,
            is_speaking=True, eye_contact=True, person_name="Alice", is_known=True,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

        self.fsm.set_safety_lock(True)
        cmd = self.fsm.update(TargetState(active_target=target), actual_head_yaw_deg=40.0, timestamp=t)
        self.assertEqual(cmd.priority_source, PrioritySource.SAFETY)
        self.assertEqual(cmd.target_yaw_deg, 0.0)

    def test_gesture_execution_sequence(self):
        """Tests sequential execution of a scripted nod gesture."""
        t = 1.0
        self.assertTrue(self.fsm.trigger_gesture("nod", timestamp=t))

        # First step of nod is +12.0°
        cmd1 = self.fsm.update(TargetState(active_target=None), actual_head_yaw_deg=0.0, timestamp=t)
        self.assertEqual(cmd1.priority_source, PrioritySource.GESTURE)
        self.assertEqual(cmd1.target_yaw_deg, 12.0)

    def test_fsm_does_not_hold_before_actual_head_reaches_target(self):
        """CRITICAL REGRESSION: FSM must NEVER transition to HOLD or VISUAL_ACQUIRE while head is en route."""
        t = 1.0
        target = FusedTarget(
            target_id="spk_neg35", modality=Modality.AUDIO, body_azimuth_deg=-35.0,
            body_elevation_deg=0.0, distance_m=2.0, confidence=0.85,
            is_speaking=True, eye_contact=False, person_name=None, is_known=False,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

        # 1. Cue at -35.0° triggers ORIENTING from initial 0.0°
        cmd1 = self.fsm.update(TargetState(active_target=target), actual_head_yaw_deg=0.0, timestamp=t)
        self.assertEqual(cmd1.gaze_state, GazeStateEnum.ORIENTING)
        self.assertEqual(cmd1.target_yaw_deg, -35.0)

        # 2. Audio cue vanishes (like a single ros2 topic pub --once message)
        # Head is moving and currently at actual_head_yaw_deg = -9.97°, vel = -45°/s
        # FSM MUST REMAIN IN ORIENTING! It must NOT enter HOLD!
        t += 0.02
        cmd2 = self.fsm.update(
            TargetState(active_target=None),
            actual_head_yaw_deg=-9.97,
            timestamp=t,
            actual_head_vel_deg_s=-45.0,
        )
        self.assertEqual(cmd2.gaze_state, GazeStateEnum.ORIENTING,
                         "BUG DETECTED: FSM prematurely transitioned out of ORIENTING before head arrived!")
        self.assertEqual(cmd2.target_yaw_deg, -35.0)

        # 3. Head is at -25.0° (still en route) -> MUST STAY in ORIENTING
        t += 0.10
        cmd3 = self.fsm.update(
            TargetState(active_target=None),
            actual_head_yaw_deg=-25.0,
            timestamp=t,
            actual_head_vel_deg_s=-30.0,
        )
        self.assertEqual(cmd3.gaze_state, GazeStateEnum.ORIENTING)

        # 4. Head arrives at -34.8° with low velocity, but first cycle of settling -> Still ORIENTING
        t += 0.10
        cmd4 = self.fsm.update(
            TargetState(active_target=None),
            actual_head_yaw_deg=-34.8,
            timestamp=t,
            actual_head_vel_deg_s=-1.2,
        )
        # 1st cycle of settling
        self.assertEqual(cmd4.gaze_state, GazeStateEnum.ORIENTING)

        # 5. Head stays settled at -34.9° for 2 more cycles -> NOW transitions to VISUAL_ACQUIRE
        t += 0.02
        self.fsm.update(TargetState(active_target=None), actual_head_yaw_deg=-34.9, timestamp=t, actual_head_vel_deg_s=0.1)
        t += 0.02
        cmd5 = self.fsm.update(TargetState(active_target=None), actual_head_yaw_deg=-35.0, timestamp=t, actual_head_vel_deg_s=0.0)
        self.assertEqual(cmd5.gaze_state, GazeStateEnum.VISUAL_ACQUIRE)
        self.assertEqual(cmd5.target_yaw_deg, -35.0)

    def test_all_motion_phases_with_closed_loop_feedback(self):
        """Tests closed-loop trajectory settling across all canonical motion phases."""
        test_cases = [
            (0.0, -35.0),
            (0.0, 35.0),
            (0.0, 60.0),
            (60.0, -60.0),
            (-60.0, 60.0),
        ]

        for start_pos, target_pos in test_cases:
            fsm = SocialGazeFSM()
            t = 10.0

            target = FusedTarget(
                target_id="tgt", modality=Modality.AUDIO, body_azimuth_deg=target_pos,
                body_elevation_deg=0.0, distance_m=2.0, confidence=0.85,
                is_speaking=True, eye_contact=False, person_name=None, is_known=False,
                timestamp=t, tracking_state=TrackingState.TRACKING
            )

            # Start saccade
            cmd = fsm.update(TargetState(active_target=target), actual_head_yaw_deg=start_pos, timestamp=t)
            self.assertEqual(cmd.gaze_state, GazeStateEnum.ORIENTING)

            # Midway check (50% progress) -> MUST stay in ORIENTING
            mid_pos = (start_pos + target_pos) / 2.0
            t += 0.1
            cmd_mid = fsm.update(TargetState(active_target=None), actual_head_yaw_deg=mid_pos, timestamp=t, actual_head_vel_deg_s=25.0)
            self.assertEqual(cmd_mid.gaze_state, GazeStateEnum.ORIENTING)

            # Complete motion and settle
            for _ in range(4):
                t += 0.02
                cmd_settled = fsm.update(TargetState(active_target=None), actual_head_yaw_deg=target_pos, timestamp=t, actual_head_vel_deg_s=0.0)

            self.assertEqual(cmd_settled.gaze_state, GazeStateEnum.VISUAL_ACQUIRE)
            self.assertEqual(cmd_settled.target_yaw_deg, target_pos)

    def test_recovering_settles_with_deadband_offset(self):
        """Head settling within deadband/tolerance (e.g. -2.3°) successfully transitions from RECOVERING to IDLE."""
        fsm = SocialGazeFSM(recovery_timeout_s=2.0)
        t = 10.0
        fsm.state = GazeStateEnum.RECOVERING
        fsm._state_entry_time = t

        # Head is at -2.3° (stopped, vel=0.0) -> Within max(2.0, 2.5) = 2.5°
        cmd = fsm.update(TargetState(active_target=None), actual_head_yaw_deg=-2.3, timestamp=t + 0.1, actual_head_vel_deg_s=0.0)
        self.assertEqual(cmd.gaze_state, GazeStateEnum.IDLE)
        self.assertEqual(cmd.priority_source, PrioritySource.IDLE)
        self.assertEqual(fsm.last_transition_reason, "RECOVERY_SETTLED_IDLE")

    def test_recovering_timeout_fallback_if_stuck(self):
        """If head is stuck outside deadband (e.g. -5.0°), recovery_timeout_s guarantees return to IDLE."""
        fsm = SocialGazeFSM(recovery_timeout_s=2.0)
        t = 10.0
        fsm.state = GazeStateEnum.RECOVERING
        fsm._state_entry_time = t

        # Head stuck at -5.0° before timeout -> stays RECOVERING
        cmd1 = fsm.update(TargetState(active_target=None), actual_head_yaw_deg=-5.0, timestamp=t + 1.0, actual_head_vel_deg_s=0.0)
        self.assertEqual(cmd1.gaze_state, GazeStateEnum.RECOVERING)

        # After timeout (t + 2.1s >= 2.0s) -> forces IDLE
        cmd2 = fsm.update(TargetState(active_target=None), actual_head_yaw_deg=-5.0, timestamp=t + 2.1, actual_head_vel_deg_s=0.0)
        self.assertEqual(cmd2.gaze_state, GazeStateEnum.IDLE)
        self.assertEqual(cmd2.priority_source, PrioritySource.IDLE)
        self.assertEqual(fsm.last_transition_reason, "RECOVERY_TIMEOUT_IDLE")


if __name__ == "__main__":
    unittest.main()


