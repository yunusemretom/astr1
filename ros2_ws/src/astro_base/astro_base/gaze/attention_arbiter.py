"""Attention Arbiter for ASTRO Social Robot Gaze Architecture.

Decides WHICH target, speaker, intent, or safety lock currently owns the robot's attention.

Priority Hierarchy (Strict & Configurable):
  1. EMERGENCY_STOP       (Hardware safety lock or deep sleep)
  2. EXPLICIT_USER_GAZE   (Direct user gaze command: 'Astro bana dön' - Immediate Preemption)
  3. DIRECT_DIALOGUE_INTENT (AI Cognitive brain dialogue turn intent)
  4. GESTURE_INTENT       (Scripted social gesture sequence)
  5. ACTIVE_SPEAKER       (Multimodal active speaking person)
  6. VISUAL_TRACKING      (Passive visual human presence / Visual Primacy)
  7. IDLE                 (Ambient baseline / No active target)
"""

import math
from typing import List, Optional

from astro_base.gaze.angle_math import clamp_deg, wrap_deg
from astro_base.gaze.spatial_memory import EpistemicSpatialMemory
from astro_base.gaze.types import (
    AttentionDecision,
    DialogueGazeIntent,
    ExplicitGazeIntent,
    FusedTarget,
    GestureGazeIntent,
    Modality,
    PrioritySource,
    SafetyGazeIntent,
    TargetSelectorType,
    TargetState,
)


class AttentionArbiterCore:
    """Arbitrates social attention ownership across measurements, candidate targets, and explicit cognitive intents."""

    def __init__(
        self,
        min_limit_deg: float = -75.0,
        max_limit_deg: float = 75.0,
        min_speaker_confidence: float = 0.50,
        min_visual_confidence: float = 0.40,
        visual_primacy_enabled: bool = True,
        spatial_memory: Optional[EpistemicSpatialMemory] = None,
    ):
        self.min_limit_deg = min_limit_deg
        self.max_limit_deg = max_limit_deg
        self.min_speaker_confidence = min_speaker_confidence
        self.min_visual_confidence = min_visual_confidence
        self.visual_primacy_enabled = visual_primacy_enabled
        self.spatial_memory = spatial_memory

        self.last_decision: Optional[AttentionDecision] = None

    def arbitrate(
        self,
        target_state: TargetState,
        explicit_intent: Optional[ExplicitGazeIntent] = None,
        dialogue_intent: Optional[DialogueGazeIntent] = None,
        gesture_intent: Optional[GestureGazeIntent] = None,
        safety_intent: Optional[SafetyGazeIntent] = None,
        actual_head_yaw_deg: float = 0.0,
        timestamp: float = 0.0,
    ) -> AttentionDecision:
        """Evaluates all candidate sources and returns the authoritative AttentionDecision."""
        # =========================================================================
        # Priority 1: EMERGENCY_STOP / SAFETY LOCK
        # =========================================================================
        if safety_intent is not None and safety_intent.valid and (safety_intent.is_locked or safety_intent.is_sleeping):
            decision = AttentionDecision(
                owner=PrioritySource.EMERGENCY_STOP,
                target_id=None,
                target_yaw_deg=clamp_deg(safety_intent.target_yaw_deg, self.min_limit_deg, self.max_limit_deg),
                confidence=1.0,
                reason=safety_intent.reason,
                timestamp=timestamp,
                is_preemption=True,
            )
            self.last_decision = decision
            return decision

        # =========================================================================
        # Priority 2: EXPLICIT_USER_GAZE (Direct Command Preemption: "Astro bana dön")
        # =========================================================================
        if explicit_intent is not None and explicit_intent.valid:
            if explicit_intent.expiry_time == 0.0 or timestamp <= explicit_intent.expiry_time:
                chosen_target_id: Optional[str] = None
                chosen_yaw: float = actual_head_yaw_deg
                preempted_tid: Optional[str] = target_state.active_target.target_id if target_state.active_target else None
                decision_reason: str = explicit_intent.reason

                if explicit_intent.selector == TargetSelectorType.CURRENT_SPEAKER:
                    # Look for active speaker candidate in target_state
                    speaker = next(
                        (t for t in target_state.candidate_targets if t.is_speaking and t.confidence >= self.min_speaker_confidence),
                        None
                    )
                    if speaker is not None:
                        chosen_target_id = speaker.target_id
                        chosen_yaw = speaker.body_azimuth_deg
                        decision_reason = f"EXPLICIT_SPEAKER_ACQUIRED_{speaker.target_id}"
                    elif explicit_intent.target_yaw_deg is not None:
                        chosen_yaw = explicit_intent.target_yaw_deg
                        decision_reason = "EXPLICIT_EXACT_ANGLE"
                    elif target_state.candidate_targets:
                        # Fallback: Best visual candidate
                        best_vis = target_state.candidate_targets[0]
                        chosen_target_id = best_vis.target_id
                        chosen_yaw = best_vis.body_azimuth_deg
                        decision_reason = f"EXPLICIT_FALLBACK_VISUAL_{best_vis.target_id}"
                    elif self.spatial_memory is not None and self.spatial_memory.get_most_likely_person_location(timestamp) is not None:
                        chosen_yaw = self.spatial_memory.get_most_likely_person_location(timestamp)
                        decision_reason = "EXPLICIT_SPATIAL_MEMORY_PERSON"
                    else:
                        chosen_yaw = actual_head_yaw_deg
                        decision_reason = "UNRESOLVED_CURRENT_SPEAKER_POSITION"

                elif explicit_intent.selector == TargetSelectorType.TARGET_ID:
                    matched = next((t for t in target_state.candidate_targets if t.target_id == explicit_intent.target_id), None)
                    if matched is not None:
                        chosen_target_id = matched.target_id
                        chosen_yaw = matched.body_azimuth_deg
                    elif explicit_intent.target_yaw_deg is not None:
                        chosen_yaw = explicit_intent.target_yaw_deg

                elif explicit_intent.selector == TargetSelectorType.ABSOLUTE_YAW:
                    if explicit_intent.target_yaw_deg is not None:
                        chosen_yaw = explicit_intent.target_yaw_deg

                decision = AttentionDecision(
                    owner=PrioritySource.EXPLICIT_USER_GAZE,
                    target_id=chosen_target_id,
                    target_yaw_deg=clamp_deg(chosen_yaw, self.min_limit_deg, self.max_limit_deg),
                    confidence=explicit_intent.confidence,
                    reason=decision_reason,
                    timestamp=timestamp,
                    is_preemption=True,
                    preempted_target_id=preempted_tid,
                )
                self.last_decision = decision
                return decision

        # =========================================================================
        # Priority 3: DIRECT_DIALOGUE_INTENT (Cognitive brain conversational turn)
        # =========================================================================
        if dialogue_intent is not None and dialogue_intent.valid:
            if dialogue_intent.expiry_time == 0.0 or timestamp <= dialogue_intent.expiry_time:
                decision = AttentionDecision(
                    owner=PrioritySource.DIRECT_DIALOGUE_INTENT,
                    target_id=None,
                    target_yaw_deg=clamp_deg(dialogue_intent.target_yaw_deg, self.min_limit_deg, self.max_limit_deg),
                    confidence=dialogue_intent.confidence,
                    reason=dialogue_intent.reason,
                    timestamp=timestamp,
                    is_preemption=False,
                )
                self.last_decision = decision
                return decision

        # =========================================================================
        # Priority 4: GESTURE_INTENT (Social gesture sequence)
        # =========================================================================
        if gesture_intent is not None and gesture_intent.valid:
            decision = AttentionDecision(
                owner=PrioritySource.GESTURE_INTENT,
                target_id=None,
                target_yaw_deg=clamp_deg(gesture_intent.target_yaw_deg, self.min_limit_deg, self.max_limit_deg),
                confidence=gesture_intent.confidence,
                reason=f"GESTURE_{gesture_intent.gesture_name}",
                timestamp=timestamp,
                is_preemption=False,
            )
            self.last_decision = decision
            return decision

        # =========================================================================
        # Priority 5 & 6: ACTIVE_SPEAKER & VISUAL_TRACKING (Target Manager state)
        # =========================================================================
        active_target = target_state.active_target
        if active_target is not None and active_target.confidence >= self.min_visual_confidence:
            if active_target.is_speaking:
                owner = PrioritySource.ACTIVE_SPEAKER
                reason = f"ACTIVE_SPEAKER_{active_target.target_id}"
            else:
                owner = PrioritySource.VISUAL_TRACKING
                reason = f"VISUAL_TRACKING_{active_target.target_id}"

            decision = AttentionDecision(
                owner=owner,
                target_id=active_target.target_id,
                target_yaw_deg=clamp_deg(active_target.body_azimuth_deg, self.min_limit_deg, self.max_limit_deg),
                confidence=active_target.confidence,
                reason=reason,
                timestamp=timestamp,
                is_preemption=False,
            )
            self.last_decision = decision
            return decision

        # =========================================================================
        # Priority 7: IDLE (No valid active target or intent)
        # =========================================================================
        decision = AttentionDecision(
            owner=PrioritySource.IDLE,
            target_id=None,
            target_yaw_deg=0.0,
            confidence=0.0,
            reason="NO_ACTIVE_TARGET",
            timestamp=timestamp,
            is_preemption=False,
        )
        self.last_decision = decision
        return decision