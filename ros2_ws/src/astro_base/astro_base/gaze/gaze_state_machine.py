"""Social Gaze Finite State Machine (FSM) and Gaze Policy Engine.

Semantic Social Gaze States:
  IDLE | SEARCHING | ACQUIRING | ORIENTING | TRACKING | HOLDING_ATTENTION | TARGET_LOST | RECOVERING

Attention Arbitration:
  Delegated to AttentionArbiterCore (EMERGENCY_STOP > EXPLICIT_USER_GAZE > DIALOGUE > GESTURE > ACTIVE_SPEAKER > VISUAL_TRACKING > IDLE).
"""

import math
from typing import Dict, List, Optional, Tuple

from astro_base.gaze.angle_math import (
    angular_diff_deg,
    circular_distance_deg,
    clamp_deg,
    wrap_deg,
)
from astro_base.gaze.attention_arbiter import AttentionArbiterCore
from astro_base.gaze.spatial_memory import EpistemicSpatialMemory
from astro_base.gaze.types import (
    AttentionDecision,
    DialogueGazeIntent,
    ExplicitGazeIntent,
    FusedTarget,
    GazeCommand,
    GazeStateEnum,
    GestureGazeIntent,
    Modality,
    PrioritySource,
    SafetyGazeIntent,
    TargetState,
    TrackingState,
)


class SocialGazeFSM:
    """Central authoritative behavioral Gaze Policy Manager."""

    GESTURE_PROFILES: Dict[str, List[float]] = {
        "nod": [12.0, -8.0, 0.0],
        "shake": [20.0, -20.0, 10.0, 0.0],
        "tilt": [14.0, 0.0],
        "scan": [-35.0, 35.0, 0.0],
        "center": [0.0],
        "look_left": [35.0],
        "look_right": [-35.0],
        "glance_left": [18.0, 0.0],
        "glance_right": [-18.0, 0.0],
    }

    GESTURE_ALIASES: Dict[str, str] = {
        "head_nod": "nod",
        "head_shake": "shake",
        "head_tilt": "tilt",
        "head_center": "center",
        "reset": "center",
    }

    def __init__(
        self,
        deadband_deg: float = 2.50,
        idle_return_timeout_s: float = 8.0,
        min_attention_dwell_s: float = 2.50,
        target_lost_timeout_s: float = 1.0,
        idle_saccades_enabled: bool = False,
        idle_saccade_interval_s: float = 4.0,
        min_limit_deg: float = -75.0,
        max_limit_deg: float = 75.0,
        position_tolerance_deg: float = 2.0,
        velocity_tolerance_deg_s: float = 3.0,
        settling_persistence_required: int = 3,
        acquisition_threshold: float = 0.75,
        audio_acquisition_threshold: float = 0.45,
        recovery_timeout_s: float = 2.0,
        arbiter: Optional[AttentionArbiterCore] = None,
        spatial_memory: Optional[EpistemicSpatialMemory] = None,
    ):
        self.deadband_deg = deadband_deg
        self.idle_return_timeout_s = idle_return_timeout_s
        self.min_attention_dwell_s = min_attention_dwell_s
        self.target_lost_timeout_s = target_lost_timeout_s
        self.idle_saccades_enabled = idle_saccades_enabled
        self.idle_saccade_interval_s = idle_saccade_interval_s
        self.min_limit_deg = min_limit_deg
        self.max_limit_deg = max_limit_deg
        self.position_tolerance_deg = position_tolerance_deg
        self.velocity_tolerance_deg_s = velocity_tolerance_deg_s
        self.settling_persistence_required = settling_persistence_required
        self.acquisition_threshold = acquisition_threshold
        self.audio_acquisition_threshold = audio_acquisition_threshold
        self.recovery_timeout_s = recovery_timeout_s

        # Spatial Memory for situational awareness and negative evidence
        self.spatial_memory = spatial_memory or EpistemicSpatialMemory()

        # Attention Arbiter authority
        self.arbiter = arbiter or AttentionArbiterCore(
            min_limit_deg=min_limit_deg,
            max_limit_deg=max_limit_deg,
            spatial_memory=self.spatial_memory,
        )
        if self.arbiter.spatial_memory is None:
            self.arbiter.spatial_memory = self.spatial_memory

        # Internal FSM state
        self.state = GazeStateEnum.IDLE
        self.target_yaw_deg: float = 0.0
        self.target_pitch_deg: float = 0.0
        self.active_priority: PrioritySource = PrioritySource.IDLE
        self.active_target_id: Optional[str] = None
        self.at_target: bool = True
        self._settling_persistence_count: int = 0
        self.last_decision: Optional[AttentionDecision] = None
        self.hold_enter_reason: str = "NONE"
        self.hold_exit_reason: str = "NONE"
        self.last_transition_reason: str = "INIT"

        # Intents state storage
        self._safety_intent: SafetyGazeIntent = SafetyGazeIntent()
        self._explicit_intent: Optional[ExplicitGazeIntent] = None
        self._dialogue_intent: Optional[DialogueGazeIntent] = None
        self._gesture_intent: Optional[GestureGazeIntent] = None

        # Gesture execution state
        self._active_gesture: Optional[str] = None
        self._gesture_steps: List[float] = []
        self._gesture_step_idx: int = 0
        self._gesture_step_start_time: float = 0.0
        self._gesture_step_duration_s: float = 0.35

        # Timing tracking
        self._state_entry_time: float = 0.0
        self._last_speech_time: float = 0.0
        self._last_target_observed_time: float = 0.0
        self._last_idle_saccade_time: float = 0.0
        self._saccade_direction: int = 1

    @property
    def safety_lock(self) -> bool:
        return self._safety_intent.is_locked

    @property
    def is_sleeping(self) -> bool:
        return self._safety_intent.is_sleeping

    def set_safety_lock(self, locked: bool) -> None:
        """Emergency stop or hardware safety lock."""
        self._safety_intent.is_locked = locked
        if locked:
            self._active_gesture = None
            self._gesture_steps = []
            self.state = GazeStateEnum.IDLE
            self.target_yaw_deg = 0.0
            self.active_priority = PrioritySource.EMERGENCY_STOP

    def set_sleep_mode(self, sleeping: bool) -> None:
        """Sets robot sleep state; locks head at center during deep sleep."""
        self._safety_intent.is_sleeping = sleeping
        if sleeping:
            self._active_gesture = None
            self._gesture_steps = []
            self.state = GazeStateEnum.IDLE
            self.target_yaw_deg = 0.0
            self.active_priority = PrioritySource.EMERGENCY_STOP

    def set_explicit_gaze_intent(self, intent: Optional[ExplicitGazeIntent]) -> None:
        """Sets an explicit user gaze command (e.g. 'Astro bana dön')."""
        self._explicit_intent = intent

    def set_dialogue_target(self, yaw_deg: float, duration_s: float, timestamp: float) -> None:
        """Sets explicit cognitive dialogue gaze intent (e.g. looking at interlocutor)."""
        if self.safety_lock or self.is_sleeping:
            return
        self._dialogue_intent = DialogueGazeIntent(
            target_yaw_deg=clamp_deg(yaw_deg, self.min_limit_deg, self.max_limit_deg),
            confidence=0.90,
            timestamp=timestamp,
            expiry_time=timestamp + max(0.5, duration_s),
            valid=True,
            reason="AI_DIALOGUE_INTERACTION",
        )

    def trigger_gesture(self, gesture_name: str, timestamp: float) -> bool:
        """Queues a social head gesture sequence."""
        if self.safety_lock or self.is_sleeping:
            return False

        canonical = self.GESTURE_ALIASES.get(gesture_name.lower().strip(), gesture_name.lower().strip())
        if canonical not in self.GESTURE_PROFILES:
            return False

        self._active_gesture = canonical
        self._gesture_steps = list(self.GESTURE_PROFILES[canonical])
        self._gesture_step_idx = 0
        self._gesture_step_start_time = timestamp
        self._gesture_intent = GestureGazeIntent(
            gesture_name=canonical,
            target_yaw_deg=clamp_deg(self._gesture_steps[0], self.min_limit_deg, self.max_limit_deg),
            confidence=1.0,
            timestamp=timestamp,
            valid=True,
        )
        return True

    def _transition_to(self, new_state: GazeStateEnum, timestamp: float, reason: str = "") -> None:
        """Executes FSM state transition."""
        if self.state != new_state:
            prev_state = self.state
            if prev_state == GazeStateEnum.HOLDING_ATTENTION:
                self.hold_exit_reason = reason or "STATE_EXIT"
            if new_state == GazeStateEnum.HOLDING_ATTENTION:
                self.hold_enter_reason = reason or "STATE_ENTER"
            self.last_transition_reason = reason
            self.state = new_state
            self._state_entry_time = timestamp
            self._settling_persistence_count = 0

    def update(
        self,
        target_state: TargetState,
        actual_head_yaw_deg: float,
        timestamp: float,
        actual_head_vel_deg_s: float = 0.0,
        explicit_intent: Optional[ExplicitGazeIntent] = None,
        dialogue_intent: Optional[DialogueGazeIntent] = None,
        gesture_intent: Optional[GestureGazeIntent] = None,
        safety_intent: Optional[SafetyGazeIntent] = None,
    ) -> GazeCommand:
        """Evaluates sensory inputs, invokes AttentionArbiter, and executes social gaze policy."""
        # 1. Update active intent stores if passed explicitly
        if explicit_intent is not None:
            self._explicit_intent = explicit_intent
        if dialogue_intent is not None:
            self._dialogue_intent = dialogue_intent
        if gesture_intent is not None:
            self._gesture_intent = gesture_intent
        if safety_intent is not None:
            self._safety_intent = safety_intent

        # 2. Advance active gesture sequence if in progress
        if self._active_gesture and self._gesture_steps:
            elapsed_step = timestamp - self._gesture_step_start_time
            step_target = self._gesture_steps[self._gesture_step_idx]
            head_settled = (
                abs(angular_diff_deg(actual_head_yaw_deg, step_target)) <= self.position_tolerance_deg
                and abs(actual_head_vel_deg_s) <= self.velocity_tolerance_deg_s
            )

            if elapsed_step >= self._gesture_step_duration_s or head_settled:
                self._gesture_step_idx += 1
                if self._gesture_step_idx < len(self._gesture_steps):
                    next_step = self._gesture_steps[self._gesture_step_idx]
                    self._gesture_step_start_time = timestamp
                    self._gesture_intent = GestureGazeIntent(
                        gesture_name=self._active_gesture,
                        target_yaw_deg=clamp_deg(next_step, self.min_limit_deg, self.max_limit_deg),
                        confidence=1.0,
                        timestamp=timestamp,
                        valid=True,
                    )
                else:
                    self._active_gesture = None
                    self._gesture_steps = []
                    self._gesture_intent = None

        # 3. Arbitrate Attention Ownership
        decision = self.arbiter.arbitrate(
            target_state=target_state,
            explicit_intent=self._explicit_intent,
            dialogue_intent=self._dialogue_intent,
            gesture_intent=self._gesture_intent,
            safety_intent=self._safety_intent,
            actual_head_yaw_deg=actual_head_yaw_deg,
            timestamp=timestamp,
        )
        self.last_decision = decision
        self.active_priority = decision.owner
        self.active_target_id = decision.target_id

        # 4. Check physical settling detection
        pos_err = abs(angular_diff_deg(actual_head_yaw_deg, self.target_yaw_deg))
        pos_ok = (pos_err <= self.position_tolerance_deg)
        vel_ok = (abs(actual_head_vel_deg_s) <= self.velocity_tolerance_deg_s)

        if pos_ok and vel_ok:
            self._settling_persistence_count += 1
        else:
            self._settling_persistence_count = 0

        self.at_target = (self._settling_persistence_count >= self.settling_persistence_required)

        # 5. Gaze Policy Lifecycle & Invariant Transitions
        if decision.owner == PrioritySource.EMERGENCY_STOP:
            self.target_yaw_deg = 0.0
            if abs(actual_head_yaw_deg) > 1.5:
                self._transition_to(GazeStateEnum.RECOVERING, timestamp, reason="EMERGENCY_STOP_RECOVERY")
            else:
                self.active_target_id = None
                self._transition_to(GazeStateEnum.IDLE, timestamp, reason="EMERGENCY_STOP_IDLE")

        elif decision.is_preemption:
            # Explicit user command immediately preempts active attention without turn-taking dwell
            self.target_yaw_deg = decision.target_yaw_deg
            err_to_target = abs(angular_diff_deg(self.target_yaw_deg, actual_head_yaw_deg))
            if err_to_target > 8.0 and not self.at_target:
                self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason=f"PREEMPTION_SACCADE_{decision.reason}")
            else:
                self._transition_to(GazeStateEnum.HOLDING_ATTENTION, timestamp, reason=f"PREEMPTION_HOLD_{decision.reason}")

        elif decision.owner in (PrioritySource.ACTIVE_SPEAKER, PrioritySource.VISUAL_TRACKING):
            target_yaw = decision.target_yaw_deg
            err_deg = abs(angular_diff_deg(target_yaw, actual_head_yaw_deg))
            has_vision = (
                target_state.active_target is not None
                and target_state.active_target.modality in (Modality.FUSED, Modality.VISION)
            )

            if self.state == GazeStateEnum.ORIENTING:
                # Saccade in progress: update target if shifted by more than deadband
                if abs(angular_diff_deg(target_yaw, self.target_yaw_deg)) >= self.deadband_deg:
                    self.target_yaw_deg = target_yaw

                # Complete orientation when arrived or timeout
                if self.at_target or (timestamp - self._state_entry_time) >= 2.5:
                    if has_vision or decision.owner == PrioritySource.VISUAL_TRACKING:
                        if self.at_target:
                            self._transition_to(GazeStateEnum.HOLDING_ATTENTION, timestamp, reason="ORIENTING_ARRIVED_HOLD")
                        else:
                            self._transition_to(GazeStateEnum.TRACKING, timestamp, reason="ORIENTING_ARRIVED_TRACKING")
                    else:
                        self._transition_to(GazeStateEnum.ACQUIRING, timestamp, reason="ORIENTING_COMPLETE_ACQUIRING")

            elif self.state == GazeStateEnum.ACQUIRING:
                # Active target updated while acquiring (e.g. vision lock or speaker switch)
                if abs(angular_diff_deg(target_yaw, self.target_yaw_deg)) >= self.deadband_deg:
                    self.target_yaw_deg = target_yaw

                if has_vision or decision.owner == PrioritySource.VISUAL_TRACKING:
                    if self.at_target or err_deg <= self.position_tolerance_deg:
                        self._transition_to(GazeStateEnum.HOLDING_ATTENTION, timestamp, reason="ACQUIRE_VISION_HOLD")
                    else:
                        self._transition_to(GazeStateEnum.TRACKING, timestamp, reason="ACQUIRE_VISION_TRACK")
                elif err_deg > 8.0:
                    self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason="ACQUIRE_AUDIO_STEP_SACCADE")

            elif self.state == GazeStateEnum.HOLDING_ATTENTION:
                # Stable Social Attention Commitment (Failure 9 & 10):
                # Continuously follow target setpoint smoothly if moved by more than deadband
                if abs(angular_diff_deg(target_yaw, self.target_yaw_deg)) >= self.deadband_deg:
                    self.target_yaw_deg = target_yaw

                # Only leave HOLDING_ATTENTION if target ID switched with significant spatial jump (> 12.0°) or large step jump (> 15.0°)
                target_id_changed = (
                    target_state.active_target is not None
                    and self.active_target_id is not None
                    and target_state.active_target.target_id != self.active_target_id
                )
                if target_id_changed:
                    if err_deg > 12.0:
                        self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason=f"TARGET_SWITCH_{target_state.active_target.target_id}")
                    else:
                        # Nearby target re-identification: seamlessly maintain attention
                        self.active_target_id = target_state.active_target.target_id
                elif err_deg > 15.0:
                    self._transition_to(GazeStateEnum.TRACKING, timestamp, reason="LARGE_TARGET_STEP_PURSUIT")

            elif self.state == GazeStateEnum.TRACKING:
                # Smooth Visual Pursuit
                if abs(angular_diff_deg(target_yaw, self.target_yaw_deg)) >= self.deadband_deg:
                    self.target_yaw_deg = target_yaw

                # Settle into committed HOLDING_ATTENTION once arrived and velocity settled
                if self.at_target or (err_deg <= self.position_tolerance_deg and abs(actual_head_vel_deg_s) <= self.velocity_tolerance_deg_s):
                    self._transition_to(GazeStateEnum.HOLDING_ATTENTION, timestamp, reason="PURSUIT_ARRIVED_HOLD")
                elif err_deg > 25.0:
                    self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason="LARGE_TARGET_STEP")

            elif self.state == GazeStateEnum.TARGET_LOST:
                # Explicit Reacquisition Semantics (Failure 3):
                # Debounce window (150ms) to reject single-frame glitch oscillations
                time_in_lost = timestamp - self._state_entry_time
                if time_in_lost >= 0.15:
                    req_conf = (
                        self.audio_acquisition_threshold
                        if (target_state.active_target is not None and target_state.active_target.modality == Modality.AUDIO)
                        else self.acquisition_threshold
                    )
                    if (
                        target_state.active_target is not None
                        and target_state.active_target.confidence >= req_conf
                        and (timestamp - target_state.active_target.timestamp) <= 0.25
                    ):
                        reacq_reason = (
                            f"REACQUIRED_SAME_{target_state.active_target.target_id}"
                            if target_state.active_target.target_id == self.active_target_id
                            else f"REACQUIRED_NEW_{target_state.active_target.target_id}"
                        )
                        self.target_yaw_deg = target_state.active_target.body_azimuth_deg
                        if has_vision:
                            self._transition_to(GazeStateEnum.TRACKING, timestamp, reason=reacq_reason)
                        else:
                            self._transition_to(GazeStateEnum.ACQUIRING, timestamp, reason=reacq_reason)
                    elif time_in_lost >= self.target_lost_timeout_s:
                        self._transition_to(GazeStateEnum.RECOVERING, timestamp, reason="TARGET_LOST_TIMEOUT_RECOVER")

            elif self.state == GazeStateEnum.IDLE:
                # Strict IDLE Entry Guards (Failure 1):
                # IDLE can NEVER jump directly to HOLDING_ATTENTION!
                req_conf = (
                    self.audio_acquisition_threshold
                    if (target_state.active_target is not None and target_state.active_target.modality == Modality.AUDIO)
                    else self.acquisition_threshold
                )
                if target_state.active_target is not None and target_state.active_target.confidence >= req_conf:
                    self.target_yaw_deg = target_yaw
                    if err_deg > 6.0:
                        self._settling_persistence_count = 0
                        self.at_target = False
                        self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason=f"IDLE_TARGET_ACQUIRED_SACCADE_{target_state.active_target.target_id}")
                    elif has_vision:
                        self._transition_to(GazeStateEnum.TRACKING, timestamp, reason=f"IDLE_TARGET_ACQUIRED_TRACK_{target_state.active_target.target_id}")
                    else:
                        self._transition_to(GazeStateEnum.ACQUIRING, timestamp, reason=f"IDLE_AUDIO_ACQUIRE_{target_state.active_target.target_id}")

            elif self.state == GazeStateEnum.RECOVERING:
                # Target detected during return to center
                req_conf = (
                    self.audio_acquisition_threshold
                    if (target_state.active_target is not None and target_state.active_target.modality == Modality.AUDIO)
                    else self.acquisition_threshold
                )
                if target_state.active_target is not None and target_state.active_target.confidence >= req_conf:
                    self.target_yaw_deg = target_yaw
                    if err_deg > 8.0:
                        self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason="RECOVERY_PREEMPTED_SACCADE")
                    elif has_vision:
                        self._transition_to(GazeStateEnum.TRACKING, timestamp, reason="RECOVERY_PREEMPTED_TRACK")
                    else:
                        self._transition_to(GazeStateEnum.ACQUIRING, timestamp, reason="RECOVERY_PREEMPTED_AUDIO_ACQUIRE")

        elif decision.owner in (PrioritySource.DIRECT_DIALOGUE_INTENT, PrioritySource.GESTURE_INTENT):
            self.target_yaw_deg = decision.target_yaw_deg
            err_deg = abs(angular_diff_deg(self.target_yaw_deg, actual_head_yaw_deg))
            if err_deg > 12.0 and not self.at_target:
                self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason="INTENT_SACCADE")
            else:
                self._transition_to(GazeStateEnum.HOLDING_ATTENTION, timestamp, reason="INTENT_HOLD")

        else:
            # PrioritySource.IDLE: No active targets or intents
            if self.state == GazeStateEnum.ORIENTING:
                if self.at_target or (timestamp - self._state_entry_time) >= 2.5:
                    self._transition_to(GazeStateEnum.ACQUIRING, timestamp, reason="IDLE_ORIENT_TIMEOUT")

            elif self.state == GazeStateEnum.ACQUIRING:
                scan_elapsed = timestamp - self._state_entry_time
                if scan_elapsed >= 0.80:
                    # Register Negative Evidence: This head yaw has no face -> mark as Reverb Zone!
                    if self.spatial_memory is not None:
                        self.spatial_memory.register_negative_acoustic_evidence(
                            bearing_deg=actual_head_yaw_deg,
                            timestamp=timestamp,
                            reason="NO_FACE_IN_ACQUIRE"
                        )
                    # Conscious check: Do we know where a real human is in the room?
                    known_person_yaw = self.spatial_memory.get_most_likely_person_location(timestamp, max_age_s=15.0) if self.spatial_memory else None
                    if known_person_yaw is not None and abs(angular_diff_deg(known_person_yaw, actual_head_yaw_deg)) > 15.0:
                        self.target_yaw_deg = known_person_yaw
                        self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason="REORIENT_TO_KNOWN_HUMAN")
                    else:
                        self._transition_to(GazeStateEnum.TARGET_LOST, timestamp, reason="ACQUISITION_TIMEOUT_NO_TARGET")

            elif self.state == GazeStateEnum.TRACKING:
                self._transition_to(GazeStateEnum.HOLDING_ATTENTION, timestamp, reason="TRACKING_DROPOUT_COAST")

            elif self.state == GazeStateEnum.HOLDING_ATTENTION:
                dwell_elapsed = timestamp - self._state_entry_time
                if dwell_elapsed >= self.min_attention_dwell_s:
                    self._transition_to(GazeStateEnum.TARGET_LOST, timestamp, reason="ATTENTION_DWELL_EXPIRED")

            elif self.state == GazeStateEnum.TARGET_LOST:
                time_lost = timestamp - self._state_entry_time
                if time_lost >= self.target_lost_timeout_s:
                    known_person_yaw = self.spatial_memory.get_most_likely_person_location(timestamp, max_age_s=15.0) if self.spatial_memory else None
                    if known_person_yaw is not None and abs(angular_diff_deg(known_person_yaw, actual_head_yaw_deg)) > 15.0:
                        self.target_yaw_deg = known_person_yaw
                        self._transition_to(GazeStateEnum.ORIENTING, timestamp, reason="LOST_REORIENT_TO_KNOWN_HUMAN")
                    else:
                        self._transition_to(GazeStateEnum.RECOVERING, timestamp, reason="TARGET_LOST_TIMEOUT_RECOVER")

            elif self.state == GazeStateEnum.RECOVERING:
                self.target_yaw_deg = 0.0
                time_in_recovering = timestamp - self._state_entry_time
                settled_at_center = (
                    abs(actual_head_yaw_deg) <= max(self.position_tolerance_deg, 2.5)
                    and abs(actual_head_vel_deg_s) <= self.velocity_tolerance_deg_s
                )
                if settled_at_center or time_in_recovering >= self.recovery_timeout_s:
                    self.active_target_id = None
                    self.active_priority = PrioritySource.IDLE
                    reason = "RECOVERY_SETTLED_IDLE" if settled_at_center else "RECOVERY_TIMEOUT_IDLE"
                    self._transition_to(GazeStateEnum.IDLE, timestamp, reason=reason)

            elif self.state == GazeStateEnum.IDLE:
                self.target_yaw_deg = 0.0
                self.active_target_id = None
                self.active_priority = PrioritySource.IDLE
                if self.idle_saccades_enabled:
                    time_since_saccade = timestamp - self._last_idle_saccade_time
                    if time_since_saccade >= self.idle_saccade_interval_s:
                        self._saccade_direction *= -1
                        self.target_yaw_deg = float(self._saccade_direction * 3.5)
                        self._last_idle_saccade_time = timestamp

        return GazeCommand(
            target_yaw_deg=self.target_yaw_deg,
            target_pitch_deg=self.target_pitch_deg,
            priority_source=self.active_priority,
            gaze_state=self.state,
            active_target_id=self.active_target_id,
            confidence=round(decision.confidence, 2),
            timestamp=timestamp,
        )
