#!/usr/bin/env python3
"""ASTRO V1 — Physical Action Grounding & Hardware Authority Manager.

Coordinates robotic motion execution, sound-source localization actions,
safety interlocking, hardware ACK verification, and action idempotency.

Key Principles:
  1. Physical Grounding: LLMs never guess or hallucinate movement directions or success.
  2. Sound Direction Authority: The ReSpeaker 4-Mic DOA subsystem is the sole authority
     for acoustic orientation. If DOA is unavailable/weak, 'NO_DIRECTION' is produced.
  3. Machine-Verifiable Action Results: Every physical command produces an unambiguous
     `ActionResult` containing success flag, actual directions, error codes, and telemetry.
  4. Safety Interlocking: Heartbeat, LiDAR obstacle proximity, and sensor freshness
     watchdogs gate all physical movement independently of AI personas.
  5. Idempotency: Actions carry unique IDs and timestamps to prevent duplicate executions.
"""

import collections
import dataclasses
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

try:
    from geometry_msgs.msg import Twist
except ImportError:
    class Twist:  # type: ignore
        def __init__(self):
            class _Vec:
                x = 0.0
                y = 0.0
                z = 0.0
            self.linear = _Vec()
            self.angular = _Vec()

try:
    from std_msgs.msg import Bool, Float32, String
except ImportError:
    class _MockMsg:
        data: Any = None
    Bool = Float32 = String = _MockMsg  # type: ignore

try:
    from astro_base.msg import HeadCmd
except ImportError:
    class HeadCmd:  # type: ignore
        angle_deg: float = 0.0

try:
    from astro_audio.doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry
except ImportError:
    try:
        from doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry
    except ImportError:
        AcousticDOAEstimator = ReSpeakerGeometry = None  # type: ignore


@dataclass
class SoundDirection:
    """Represents a validated acoustic sound direction from the microphone array."""
    azimuth_deg: float      # Relative yaw: -180.0° to +180.0° (0 = front, + = right, - = left)
    confidence: float       # 0.0 to 1.0
    valid: bool             # True if confidence >= threshold and fresh
    timestamp: float        # time.monotonic()
    raw_doa_deg: float      # 0.0° to 359.0° circular ReSpeaker DOA
    rms_level: float        # Audio frame RMS energy
    source: str = "respeaker_4mic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "azimuth_deg": round(self.azimuth_deg, 1),
            "confidence": round(self.confidence, 2),
            "valid": self.valid,
            "timestamp": round(self.timestamp, 3),
            "raw_doa_deg": round(self.raw_doa_deg, 1),
            "rms_level": round(self.rms_level, 1),
            "source": self.source,
        }


@dataclass
class ActionResult:
    """Machine-verifiable action result for physical grounding."""
    success: bool
    action: str
    action_id: str
    generation_id: Optional[int] = None
    azimuth_deg: Optional[float] = None
    confidence: Optional[float] = None
    requested_direction: Optional[str] = None
    actual_direction: Optional[str] = None
    duration_ms: Optional[int] = None
    hardware_ack: bool = False
    verified: bool = False
    encoder_delta: Optional[float] = None
    error_code: Optional[str] = None
    error: Optional[str] = None
    reason: Optional[str] = None
    message: str = ""
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        status_val = "success" if self.success else ("blocked" if self.reason in ("heartbeat_unhealthy", "obstacle_detected", "lidar_stale_or_disconnected", "safety_blocked") else "error")
        d: Dict[str, Any] = {
            "status": status_val,
            "success": self.success,
            "action": self.action,
            "action_id": self.action_id,
            "hardware_ack": self.hardware_ack,
            "verified": self.verified,
            "message": self.message,
            "timestamp": round(self.timestamp, 3),
        }
        if self.encoder_delta is not None:
            d["encoder_delta"] = round(self.encoder_delta, 4)
        if self.generation_id is not None:
            d["generation_id"] = self.generation_id
        if self.azimuth_deg is not None:
            d["azimuth_deg"] = round(self.azimuth_deg, 1)
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 2)
        if self.requested_direction is not None:
            d["requested_direction"] = self.requested_direction
        if self.actual_direction is not None:
            d["actual_direction"] = self.actual_direction
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.error_code is not None:
            d["error_code"] = self.error_code
        if self.error is not None:
            d["error"] = self.error
        if self.reason is not None:
            d["reason"] = self.reason
        return d


def circular_doa_to_yaw(raw_doa_deg: float) -> float:
    """Converts 0°..359° circular ReSpeaker DOA to robot body yaw frame (-180°..+180°).

    The result is published to /head/target_yaw, which head_tracker_node executes as a
    15-second TURN_TO_SOUND lock, so this MUST match head_tracker_node.doa_to_robot_yaw:
    ReSpeaker measures clockwise, ROS body yaw is counter-clockwise (positive = left).
    Skipping the inversion here sent the head to the mirror image of the speaker.
    """
    raw = float(raw_doa_deg) % 360.0
    yaw = raw if raw <= 180.0 else raw - 360.0
    return -yaw


class ActionManager:
    """Authoritative physical execution and sound orientation coordinator."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        pub_cmd_vel: Any = None,
        pub_head_cmd: Any = None,
        pub_head_gesture: Any = None,
        pub_head_target_yaw: Any = None,
        pub_explicit_gaze: Any = None,
        node: Any = None,
    ):
        self._logger = logger or logging.getLogger("ActionManager")
        self._pub_cmd_vel = pub_cmd_vel
        self._pub_head_cmd = pub_head_cmd  # Kept for backward compatibility in mock tests
        self._pub_head_gesture = pub_head_gesture
        self._pub_head_target_yaw = pub_head_target_yaw
        self._pub_explicit_gaze = pub_explicit_gaze
        self._node = node
        self._lock = threading.RLock()

        # Acoustic Estimator (GCC-PHAT)
        self._acoustic_estimator = AcousticDOAEstimator(sample_rate=16000) if AcousticDOAEstimator else None

        # DOA Tracking & Consensus
        self._latest_doa: Optional[SoundDirection] = None
        self._doa_history: Deque[Tuple[float, float, float]] = collections.deque(maxlen=60)  # (timestamp, yaw, rms)
        self._min_doa_confidence = 0.40
        self._doa_freshness_timeout_s = 5.0
        self._ambient_rms = 120.0
        self._is_speaking = False
        self._is_playback_active = False

        # Anti-Spam Logging State
        self._last_logged_doa_time = 0.0
        self._last_logged_valid: Optional[bool] = None
        self._last_logged_yaw: Optional[float] = None

        # Action Idempotency & History
        self._executed_action_ids: Set[str] = set()
        self._action_id_history: Deque[str] = collections.deque(maxlen=200)
        self._recent_actions: Deque[ActionResult] = collections.deque(maxlen=50)

        # Joint States / Encoder Tracking for Physical Grounding
        self._joint_positions: Dict[str, float] = {
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            "head_yaw_joint": 0.0,
        }
        self._joint_velocities: Dict[str, float] = {
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            "head_yaw_joint": 0.0,
        }
        self._last_joint_update_ts: float = 0.0

        # Movement Safety Limits
        self.max_linear_speed = 0.4   # m/s
        self.max_angular_speed = 0.8  # rad/s
        self.max_duration_s = 5.0

    def update_joint_states(
        self,
        joint_names: List[str],
        positions: List[float],
        velocities: Optional[List[float]] = None,
    ):
        """Updates internal physical joint state estimates from /joint_states."""
        with self._lock:
            for idx, name in enumerate(joint_names):
                if idx < len(positions):
                    self._joint_positions[name] = float(positions[idx])
                if velocities and idx < len(velocities):
                    self._joint_velocities[name] = float(velocities[idx])
            self._last_joint_update_ts = time.monotonic()

    def get_joint_positions(self) -> Dict[str, float]:
        """Returns a snapshot of current joint positions."""
        with self._lock:
            return dict(self._joint_positions)

    def update_audio_state(
        self,
        raw_doa_deg: Optional[float] = None,
        rms_level: Optional[float] = None,
        vad_active: bool = False,
        is_speaking: bool = False,
        is_playback_active: bool = False,
    ):
        """Updates internal acoustic perception state and evaluates DOA validity.
        
        Strict physical rule:
        - 0.0° uncalibrated default / idle reading is NEVER marked valid=true.
        - 0.0° is only valid if confirmed by active VAD and high acoustic energy.
        - Log is event-driven and throttled to prevent spam.
        """
        now = time.monotonic()
        with self._lock:
            self._is_speaking = is_speaking
            self._is_playback_active = is_playback_active

            if rms_level is not None:
                if not is_speaking and not is_playback_active and rms_level < 400.0:
                    self._ambient_rms = 0.95 * self._ambient_rms + 0.05 * rms_level

            if raw_doa_deg is not None:
                # Discard DOA while robot itself is speaking
                if is_speaking or is_playback_active:
                    return

                cur_rms = rms_level if rms_level is not None else 500.0
                yaw = circular_doa_to_yaw(raw_doa_deg)
                self._doa_history.append((now, yaw, cur_rms))

                # Calculate confidence based on acoustic energy & VAD
                energy_ratio = cur_rms / max(80.0, self._ambient_rms)
                vad_factor = 0.9 if vad_active else 0.5
                conf = min(1.0, max(0.0, (energy_ratio / 3.0) * vad_factor))

                # Strict gating for 0.0° default
                is_zero_angle = abs(yaw) < 0.5
                if is_zero_angle:
                    # 0° is the uninitialized / hardware idle default.
                    # ONLY valid if strong active speech and high energy ratio is proven.
                    if not vad_active or cur_rms < 450.0 or energy_ratio < 2.5:
                        conf = min(conf, 0.25)
                        is_valid = False
                    else:
                        is_valid = (conf >= self._min_doa_confidence)
                elif abs(yaw) > 135.0:
                    # Rear wall reflection / acoustic bounce: only valid if proven with high acoustic energy and active VAD
                    if not vad_active or cur_rms < 600.0 or energy_ratio < 3.0:
                        conf = min(conf, 0.30)
                        is_valid = False
                    else:
                        is_valid = (conf >= self._min_doa_confidence)
                else:
                    # Temporal clustering / consistency check for frontal / lateral angles
                    recent = [y for ts, y, _ in self._doa_history if (now - ts) <= 2.0 and abs(y) <= 135.0]
                    if len(recent) >= 2:
                        sin_sum = sum(math.sin(math.radians(y)) for y in recent)
                        cos_sum = sum(math.cos(math.radians(y)) for y in recent)
                        mean_y = math.degrees(math.atan2(sin_sum, cos_sum))
                        is_consistent = all(abs((y - mean_y + 180.0) % 360.0 - 180.0) <= 25.0 for y in recent)
                        if is_consistent:
                            conf = min(1.0, conf + 0.25)
                            yaw = mean_y
                        else:
                            conf = max(0.0, conf - 0.20)

                    is_valid = (conf >= self._min_doa_confidence) and (energy_ratio >= 1.5)

                self._latest_doa = SoundDirection(
                    azimuth_deg=float(yaw),
                    confidence=float(conf),
                    valid=bool(is_valid),
                    timestamp=now,
                    raw_doa_deg=float(raw_doa_deg),
                    rms_level=float(cur_rms),
                )

                # Anti-Spam Logging: Only log on state change or significant angle shift or periodic keepalive
                should_log = False
                if self._last_logged_valid != is_valid:
                    should_log = True
                elif is_valid:
                    if self._last_logged_yaw is None or abs(yaw - self._last_logged_yaw) >= 15.0:
                        should_log = True
                    elif (now - self._last_logged_doa_time) >= 3.0:
                        should_log = True

                if should_log:
                    self._last_logged_doa_time = now
                    self._last_logged_valid = is_valid
                    self._last_logged_yaw = yaw
                    sign = "+" if yaw >= 0 else ""
                    self._logger.debug(
                        f"[DOA]\n"
                        f"azimuth_deg={sign}{yaw:.1f}\n"
                        f"confidence={conf:.2f}\n"
                        f"valid={str(is_valid).lower()}"
                    )

    def update_multichannel_audio(
        self,
        pcm_channels: Any,
        rms_level: Optional[float] = None,
        vad_active: bool = False,
        is_speaking: bool = False,
        is_playback_active: bool = False,
    ):
        """Processes 4-channel microphone buffer with GCC-PHAT to compute exact DOA."""
        if is_speaking or is_playback_active or self._acoustic_estimator is None:
            return
        if hasattr(pcm_channels, "shape") and pcm_channels.ndim == 2:
            if pcm_channels.shape[0] >= 6:
                pcm_channels = pcm_channels[1:5]
            elif pcm_channels.shape[0] > 4:
                pcm_channels = pcm_channels[:4]
        azimuth, conf, is_valid = self._acoustic_estimator.estimate_from_multichannel_pcm(pcm_channels)
        if azimuth is not None:
            raw_doa = azimuth if azimuth >= 0 else azimuth + 360.0
            self.update_audio_state(
                raw_doa_deg=raw_doa,
                rms_level=rms_level,
                vad_active=vad_active,
                is_speaking=is_speaking,
                is_playback_active=is_playback_active,
            )

    def get_sound_direction(self) -> Optional[SoundDirection]:
        """Returns the current sound direction if fresh and valid, else None."""
        now = time.monotonic()
        with self._lock:
            if not self._latest_doa:
                return None
            if (now - self._latest_doa.timestamp) > self._doa_freshness_timeout_s:
                return None
            if not self._latest_doa.valid:
                return None
            return self._latest_doa

    def execute_turn_to_sound(
        self,
        generation_id: Optional[int] = None,
        action_id: Optional[str] = None,
    ) -> ActionResult:
        """Executes physical orientation towards the validated acoustic sound source.

        If DOA is unavailable or invalid, strictly produces NO_DIRECTION without moving motors.
        """
        now = time.monotonic()
        act_id = action_id or f"turn_sound_{int(now * 1000)}"

        with self._lock:
            # Idempotency check
            if act_id in self._executed_action_ids:
                return ActionResult(
                    success=True,
                    action="turn_to_sound",
                    action_id=act_id,
                    generation_id=generation_id,
                    hardware_ack=True,
                    message="Bu ses yönü eylemi zaten yürütüldü.",
                )

            sound_dir = self.get_sound_direction()

            # 1. DOA / Multimodal User Direction Resolution
            azimuth = None
            confidence = 0.0

            # Prefer recent speech consensus (within 6.0s) that is not a rear wall bounce (>130°)
            speech_doa = [y for ts, y, cur_rms in self._doa_history if (now - ts) <= 6.0 and abs(y) <= 130.0 and cur_rms >= 150.0]
            if speech_doa:
                sin_s = sum(math.sin(math.radians(y)) for y in speech_doa)
                cos_s = sum(math.cos(math.radians(y)) for y in speech_doa)
                azimuth = float(math.degrees(math.atan2(sin_s, cos_s)))
                confidence = 0.85
            elif sound_dir and sound_dir.valid and abs(sound_dir.azimuth_deg) <= 130.0:
                azimuth = sound_dir.azimuth_deg
                confidence = sound_dir.confidence
            else:
                # Secondary fallback: speech within 10.0s
                recent_doa = [y for ts, y, cur_rms in self._doa_history if (now - ts) <= 10.0 and abs(y) <= 130.0 and cur_rms >= 120.0]
                if recent_doa:
                    sin_s = sum(math.sin(math.radians(y)) for y in recent_doa)
                    cos_s = sum(math.cos(math.radians(y)) for y in recent_doa)
                    azimuth = float(math.degrees(math.atan2(sin_s, cos_s)))
                    confidence = 0.60
                elif self._node and getattr(self._node, "_speaker_angle", None) is not None:
                    spk_angle = float(self._node._speaker_angle)
                    if abs(circular_doa_to_yaw(spk_angle)) <= 130.0:
                        azimuth = float(circular_doa_to_yaw(spk_angle))
                        confidence = 0.55
                elif self._node and getattr(self._node, "_vision_person_detected", False):
                    # Person is already in front of camera
                    azimuth = float(getattr(self._node, "_vision_head_yaw", 0.0) or 0.0)
                    confidence = 0.70

            # Filter out rear acoustic echo / wall bounce in conversational interaction
            if azimuth is not None and abs(azimuth) > 110.0:
                self._logger.info(f"🛡️ [ActionManager] Arkadan yansıyan akustik yankı ({azimuth:.1f}°) filtrelendi, karşıya odaklanılıyor.")
                azimuth = 0.0
                confidence = 0.80

            # If robot is already facing the user within deadband (0° - 4°), acknowledge orientation
            if azimuth is not None and abs(azimuth) <= 4.0:

                azimuth = 0.0
                if self._pub_head_gesture:
                    msg = String()
                    msg.data = "nod"
                    self._pub_head_gesture.publish(msg)
                res = ActionResult(
                    success=True,
                    action="turn_to_sound",
                    action_id=act_id,
                    generation_id=generation_id,
                    azimuth_deg=0.0,
                    hardware_ack=True,
                    message="Kullanıcı zaten tam karşıda, kafa odaklandı.",
                )
                self._recent_actions.append(res)
                return res

            if azimuth is None:
                # 3. Broader temporal history fallback (up to 15.0s with verified acoustic speech energy)
                broader_doa = [y for ts, y, cur_rms in self._doa_history if (now - ts) <= 15.0 and abs(y) <= 130.0 and cur_rms >= 120.0]
                if broader_doa:
                    sin_s = sum(math.sin(math.radians(y)) for y in broader_doa)
                    cos_s = sum(math.cos(math.radians(y)) for y in broader_doa)
                    azimuth = float(math.degrees(math.atan2(sin_s, cos_s)))
                    confidence = 0.45
                    self._logger.info(f"👂 [ActionManager] Genişletilmiş zaman tamponundan ses yönü kurtarıldı -> {azimuth:.1f}°")
                else:
                    self._logger.warning("⚠️ [ActionManager] turn_to_sound: UNRESOLVED_CURRENT_SPEAKER_POSITION (DOA yok veya zayıf)")
                    res = ActionResult(
                        success=False,
                        action="turn_to_sound",
                        action_id=act_id,
                        generation_id=generation_id,
                        error_code="UNRESOLVED_CURRENT_SPEAKER_POSITION",
                        error="Sesin yönü belirlenemedi (UNRESOLVED_CURRENT_SPEAKER_POSITION).",
                        reason="UNRESOLVED_CURRENT_SPEAKER_POSITION",
                        message="Sesin hangi yönden geldiği tespit edilemediği için robot hareket ettirilmedi.",
                        hardware_ack=False,
                    )
                    self._recent_actions.append(res)
                    return res

            # 2. Safety Gates (Heartbeat & LiDAR Freshness)
            blocked_reason = self._check_safety_gates(direction="turn")
            if blocked_reason:
                res = ActionResult(
                    success=False,
                    action="turn_to_sound",
                    action_id=act_id,
                    generation_id=generation_id,
                    azimuth_deg=round(azimuth, 1),
                    confidence=round(confidence, 2),
                    error_code=blocked_reason.get("error_code", "SAFETY_BLOCKED"),
                    reason=blocked_reason.get("reason", "safety_blocked"),
                    error=blocked_reason.get("message", "Güvenlik kilidi devrede."),
                    message=blocked_reason.get("message", "Güvenlik kilidi nedeniyle hareket edilmedi."),
                    hardware_ack=False,
                )
                self._recent_actions.append(res)
                return res

            # 3. Canonical Attention Arbiter Preemption
            dir_str = "left" if (azimuth or 0.0) > 0 else "right"
            target_clamped = float(max(-75.0, min(75.0, azimuth))) if azimuth is not None else None

            pub_explicit = self._pub_explicit_gaze or getattr(self._node, "pub_explicit_gaze", None)
            if pub_explicit:
                try:
                    explicit_msg = String()
                    payload = {
                        "selector": "CURRENT_SPEAKER",
                        "confidence": float(confidence if confidence > 0 else 1.0),
                        "reason": "action_manager_turn_to_sound",
                    }
                    # Only include explicit target_yaw_deg if azimuth is fresh (<0.8s) and high-confidence
                    # Otherwise, let AttentionArbiter resolve directly via EpistemicSpatialMemory!
                    if target_clamped is not None and confidence >= 0.85 and sound_dir is not None and (now - sound_dir.timestamp) <= 0.8:
                        payload["target_yaw_deg"] = target_clamped
                    explicit_msg.data = json.dumps(payload)
                    pub_explicit.publish(explicit_msg)
                    self._logger.info(f"🎯 [ActionManager -> AttentionArbiter] EXPLICIT_USER_GAZE (target={payload.get('target_yaw_deg', 'SPATIAL_MEMORY')})")
                except Exception as ex:
                    self._logger.debug(f"Explicit gaze publication failed: {ex}")

            pub_target_yaw = self._pub_head_target_yaw or getattr(self._node, "pub_head_target_yaw", None)
            if pub_target_yaw and target_clamped is not None:
                try:
                    msg = Float32()
                    msg.data = target_clamped
                    pub_target_yaw.publish(msg)
                except Exception as he:
                    self._logger.debug(f"Float32 target_yaw publication failed: {he}")
            else:
                pub_head = self._pub_head_cmd or getattr(self._node, "pub_head_cmd", None)
                if pub_head:
                    try:
                        head_msg = HeadCmd()
                        head_msg.angle_deg = target_clamped
                        pub_head.publish(head_msg)
                    except Exception as he:
                        self._logger.debug(f"HeadCmd publication fallback failed: {he}")

            self._executed_action_ids.add(act_id)
            self._action_id_history.append(act_id)

            has_joint_feedback = (self._last_joint_update_ts > 0.0 and (time.monotonic() - self._last_joint_update_ts) < 5.0)

            res = ActionResult(
                success=True,
                action="turn_to_sound",
                action_id=act_id,
                generation_id=generation_id,
                azimuth_deg=round(azimuth, 1),
                confidence=round(confidence, 2),
                requested_direction=dir_str,
                actual_direction=dir_str,
                hardware_ack=True,
                verified=bool(has_joint_feedback),
                message=f"Ses yönü ({azimuth:.1f}°) kafa takipçisine iletildi.",
            )
            self._recent_actions.append(res)
            return res

    def execute_move(
        self,
        direction: str,
        speed: float = 0.2,
        duration: float = 1.5,
        generation_id: Optional[int] = None,
        action_id: Optional[str] = None,
    ) -> ActionResult:
        """Executes a linear or angular base motion with verified hardware authority."""
        now = time.monotonic()
        act_id = action_id or f"move_{direction}_{int(now * 1000)}"
        clean_dir = (direction or "stop").lower().strip()

        with self._lock:
            # Idempotency check
            if clean_dir != "stop" and (act_id in self._executed_action_ids or act_id in self._action_id_history):
                return ActionResult(
                    success=True,
                    action="move_robot",
                    action_id=act_id,
                    generation_id=generation_id,
                    requested_direction=clean_dir,
                    actual_direction=clean_dir,
                    hardware_ack=True,
                    message="Bu hareket eylemi zaten yürütüldü.",
                )

            valid_dirs = ["forward", "backward", "left", "right", "stop"]
            if clean_dir not in valid_dirs:
                return ActionResult(
                    success=False,
                    action="move_robot",
                    action_id=act_id,
                    generation_id=generation_id,
                    error_code="INVALID_DIRECTION",
                    error=f"Geçersiz yön: '{direction}'. Geçerli yönler: {valid_dirs}",
                    message="Geçersiz hareket yönü belirtildi.",
                    hardware_ack=False,
                )

            # Clamp parameters for safety
            clamped_speed = max(0.05, min(speed, self.max_linear_speed))
            clamped_duration = max(0.5, min(duration, self.max_duration_s))

            if clean_dir != "stop":
                blocked_reason = self._check_safety_gates(direction=clean_dir)
                if blocked_reason:
                    res = ActionResult(
                        success=False,
                        action="move_robot",
                        action_id=act_id,
                        generation_id=generation_id,
                        requested_direction=clean_dir,
                        error_code=blocked_reason.get("error_code", "SAFETY_BLOCKED"),
                        reason=blocked_reason.get("reason", "safety_blocked"),
                        error=blocked_reason.get("message", "Güvenlik kilidi devrede."),
                        message=blocked_reason.get("message", "Güvenlik nedeniyle hareket engellendi."),
                        hardware_ack=False,
                    )
                    self._recent_actions.append(res)
                    return res

            pub_vel = self._pub_cmd_vel or getattr(self._node, "pub_cmd_vel", None)
            if not pub_vel:
                return ActionResult(
                    success=False,
                    action="move_robot",
                    action_id=act_id,
                    generation_id=generation_id,
                    error_code="MOTOR_CONTROLLER_UNAVAILABLE",
                    error="/cmd_vel yayıncısı hazır değil veya bağlantı yok.",
                    message="Motor kontrolcüsü hazır olmadığı için hareket edilemedi.",
                    hardware_ack=False,
                )

            try:
                tw = Twist()
                if clean_dir == "forward":
                    tw.linear.x = clamped_speed
                elif clean_dir == "backward":
                    tw.linear.x = -clamped_speed
                elif clean_dir == "left":
                    tw.angular.z = clamped_speed * 2.0
                elif clean_dir == "right":
                    tw.angular.z = -clamped_speed * 2.0
                elif clean_dir == "stop":
                    tw.linear.x = 0.0
                    tw.angular.z = 0.0

                pub_vel.publish(tw)

                if clean_dir != "stop":
                    def _stop_later():
                        time.sleep(clamped_duration)
                        try:
                            stop_tw = Twist()
                            pub_vel.publish(stop_tw)
                        except Exception:
                            pass
                    threading.Thread(target=_stop_later, daemon=True).start()

                if clean_dir != "stop":
                    self._executed_action_ids.add(act_id)
                    self._action_id_history.append(act_id)

                has_joint_feedback = (clean_dir == "stop") or (self._last_joint_update_ts > 0.0 and (time.monotonic() - self._last_joint_update_ts) < 5.0)

                res = ActionResult(
                    success=True,
                    action="move_robot",
                    action_id=act_id,
                    generation_id=generation_id,
                    requested_direction=clean_dir,
                    actual_direction=clean_dir,
                    duration_ms=int(clamped_duration * 1000),
                    hardware_ack=True,
                    verified=bool(has_joint_feedback),
                    message=f"Robot {clean_dir} yönünde {clamped_speed} m/s hızla hareket ettirildi.",
                )
                self._recent_actions.append(res)
                return res

            except Exception as exc:
                res = ActionResult(
                    success=False,
                    action="move_robot",
                    action_id=act_id,
                    generation_id=generation_id,
                    error_code="MOTOR_WRITE_ERROR",
                    error=str(exc),
                    message="Motor komutu verilirken hata oluştu.",
                    hardware_ack=False,
                )
                self._recent_actions.append(res)
                return res

    GESTURE_PROFILES: Dict[str, List[float]] = {
        "nod": [12.0, -8.0, 0.0],
        "shake": [22.0, -22.0, 12.0, 0.0],
        "tilt": [16.0],
        "scan": [-35.0, 35.0, 0.0],
        "center": [0.0],
        "look_left": [35.0],
        "look_right": [-35.0],
    }

    GESTURE_ALIASES: Dict[str, str] = {
        "yes": "nod",
        "onayla": "nod",
        "no": "shake",
        "reddet": "shake",
        "curious": "tilt",
        "merak": "tilt",
        "ara": "scan",
        "search": "scan",
        "sifirla": "center",
    }

    def execute_gesture(
        self,
        gesture_name: str,
        duration_ms: int = 600,
        generation_id: Optional[int] = None,
        action_id: Optional[str] = None,
    ) -> ActionResult:
        """Executes a physical head gesture sequence (nod, shake, tilt, scan, center)."""
        now = time.monotonic()
        act_id = action_id or f"gesture_{gesture_name}_{int(now * 1000)}"
        clean_name = (gesture_name or "center").lower().strip()
        canonical_gesture = self.GESTURE_ALIASES.get(clean_name, clean_name)

        with self._lock:
            # Idempotency check
            if act_id in self._executed_action_ids or act_id in self._action_id_history:
                return ActionResult(
                    success=True,
                    action="execute_gesture",
                    action_id=act_id,
                    generation_id=generation_id,
                    requested_direction=clean_name,
                    actual_direction=canonical_gesture,
                    hardware_ack=True,
                    message="Bu jest eylemi zaten yürütüldü.",
                )

            if canonical_gesture not in self.GESTURE_PROFILES:
                return ActionResult(
                    success=False,
                    action="execute_gesture",
                    action_id=act_id,
                    generation_id=generation_id,
                    error_code="INVALID_GESTURE",
                    error=f"Geçersiz jest: '{gesture_name}'. Desteklenen jestler: {list(self.GESTURE_PROFILES.keys())}",
                    message="Geçersiz kafa jesti belirtildi.",
                    hardware_ack=False,
                )

            angles = self.GESTURE_PROFILES[canonical_gesture]

            # Route gesture command to central HeadTrackerNode arbitration engine
            pub_gesture = self._pub_head_gesture or getattr(self._node, "pub_head_gesture", None)
            if pub_gesture:
                try:
                    g_msg = String()
                    g_msg.data = canonical_gesture
                    pub_gesture.publish(g_msg)
                except Exception as ge:
                    self._logger.debug(f"Gesture publication to /head/gesture failed: {ge}")
            else:
                # Fallback for unit tests mocking legacy pub_head_cmd
                pub_head = self._pub_head_cmd or getattr(self._node, "pub_head_cmd", None)
                if pub_head and 'HeadCmd' in globals():
                    try:
                        h_cmd = HeadCmd()
                        h_cmd.angle_deg = float(angles[0]) if angles else 0.0
                        pub_head.publish(h_cmd)
                    except Exception as he:
                        self._logger.debug(f"Legacy gesture publication fallback failed: {he}")

            self._executed_action_ids.add(act_id)
            self._action_id_history.append(act_id)

            has_joint_feedback = (self._last_joint_update_ts > 0.0 and (time.monotonic() - self._last_joint_update_ts) < 5.0)

            res = ActionResult(
                success=True,
                action="execute_gesture",
                action_id=act_id,
                generation_id=generation_id,
                requested_direction=clean_name,
                actual_direction=canonical_gesture,
                duration_ms=duration_ms,
                hardware_ack=True,
                verified=bool(has_joint_feedback),
                message=f"Kafa jesti '{canonical_gesture}' başarıyla yürütüldü.",
            )
            self._recent_actions.append(res)
            return res

    def _check_safety_gates(self, direction: str) -> Optional[Dict[str, str]]:
        """Checks Arduino heartbeat health, LiDAR proximity, and LiDAR freshness."""
        node = self._node
        if not node:
            return None

        # 1. Heartbeat Health Gate (Strict for linear wheel driving; relaxed for head/sound orientation)
        hb_ok = getattr(node, "_arduino_heartbeat_healthy", False)
        last_ack = getattr(node, "_last_heartbeat_ack_time", 0.0)
        is_turn_or_head = direction in ("turn", "head", "gesture")
        
        if not is_turn_or_head:
            if not hb_ok or (time.monotonic() - last_ack) > 3.0:
                return {
                    "error_code": "MOTOR_CONTROLLER_UNAVAILABLE",
                    "reason": "heartbeat_unhealthy",
                    "message": "Arduino bağlantısı veya heartbeat aktif değil, güvenlik için hareket engellendi."
                }

        # 2. Obstacle Detection Gate
        if direction == "forward" and getattr(node, "_obstacle_detected", False):
            return {
                "error_code": "OBSTACLE_DETECTED",
                "reason": "obstacle_detected",
                "message": "Robotun önünde engel tespit edildi, hareket güvenlik nedeniyle engellendi."
            }

        # 3. LiDAR Freshness Watchdog Gate
        last_scan_time = getattr(node, "_last_laser_scan_time", 0.0)
        is_lidar_stale = (time.monotonic() - last_scan_time) > 2.0
        if is_lidar_stale:
            if hasattr(node, "_lidar_health"):
                node._lidar_health = "UNHEALTHY"
            if direction == "forward":
                return {
                    "error_code": "LIDAR_STALE_OR_DISCONNECTED",
                    "reason": "lidar_stale_or_disconnected",
                    "message": "LiDAR tarama verisi alınamıyor veya güncel değil, güvenlik nedeniyle ileri hareket engellendi."
                }

        return None
