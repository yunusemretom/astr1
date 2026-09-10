#!/usr/bin/env python3
"""ASTRO V1 — Social Sound-Localization & Head Tracking Node.

Controls robot head yaw orientation using ReSpeaker 4-Mic DOA (Direction of Arrival)
and multimodal perception with human-like social gaze behavior:
  - Acoustic energy & VAD gating (resists ambient background noise)
  - Self-voice suppression (zero tracking while robot is speaking)
  - Temporal consensus filtering (rejects momentary noise spikes & reverberation)
  - Deadband hysteresis (prevents micro-jitter and neck twitching)
  - Gaze dwell time (holds attention for >= 2.5s before switching target)
  - Smooth S-curve / slew-rate velocity limiting (max 45°/s, organic motion)
  - Idle return-to-center after sustained silence
  - OAK-D Lite vision fusion: camera-based face angle overrides DOA when a face is detected
"""

import collections
import json
import math
import os
import threading
import time
from typing import Deque, Optional, Tuple, Any

# Single source of truth for every declared parameter.
#
# doa_invert: ReSpeaker reports azimuth clockwise (90 deg = right), while ROS body
# yaw is counter-clockwise — positive = LEFT, which is the convention this node's own
# GESTURE_PROFILES already use ("look_left": +35, "look_right": -35). Inverting is
# therefore the correct default, not an opt-in: without it the head turns toward the
# mirror image of the speaker and the DOA loop diverges instead of converging.
HEAD_TRACKER_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "doa_offset_deg": 0.0,
    "doa_invert": True,
    # Full circle: the neck turns freely, and clipping the range is what forced a rear
    # source to be approached by a 140 deg sweep across the front (the "long way round").
    # Must stay <= the firmware's HEAD_MIN_DEG/HEAD_MAX_DEG in arduino/astro_firmware.
    "max_yaw_deg": 180.0,
    "min_yaw_deg": -180.0,
    # Small enough that the head actually ends up facing the talker rather than parked
    # up to 16 deg off them; large enough to sit still under ReSpeaker's coarse DOA.
    "deadband_deg": 8.0,
    "min_dwell_time_s": 3.5,
    "idle_return_timeout_s": 30.0,
    "max_speed_deg_s": 45.0,
    "update_rate_hz": 20.0,
    "min_rms_threshold": 1600.0,
    "noise_multiplier": 3.0,
    "consensus_window_size": 7,
    "consensus_threshold": 5,
    "consensus_tolerance_deg": 22.0,
    # OAK-D Lite vision fusion
    "vision_fusion_enabled": True,
    "vision_gain": 0.85,
    "vision_timeout_s": 2.0,
    # Camera half field of view: the furthest a face can honestly be from where the lens
    # was pointing, and therefore the cap on what one visual acquisition may command.
    "vision_max_correction_deg": 36.0,
    # 2D LiDAR radar fusion
    "lidar_fusion_enabled": True,
    "lidar_min_dist_m": 0.4,
    "lidar_max_dist_m": 2.8,
    "lidar_timeout_s": 2.5,
    # Quiet window after the head stops moving, before acoustic tracking resumes.
    "head_motion_settle_s": 0.5,
}


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool, Float32, String
except ImportError:
    rclpy = None
    qos_profile_sensor_data = 10  # type: ignore
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def declare_parameter(self, name, default):
            pass
        def get_parameter(self, name):
            class _Param:
                def __init__(self, val): self.value = val
            return _Param(HEAD_TRACKER_DEFAULTS.get(name, 0.0))
        def get_logger(self):
            import logging
            return logging.getLogger("HeadTrackerNode")
        def create_subscription(self, *args, **kwargs):
            return None
        def create_publisher(self, *args, **kwargs):
            class _Pub:
                def publish(self, msg): pass
            return _Pub()
        def create_timer(self, *args, **kwargs):
            return None
        def destroy_node(self):
            pass
    class _MockMsg:
        data: Any = None
    Bool = Float32 = String = LaserScan = _MockMsg  # type: ignore

class DummyHeadCmd:
    angle_deg: float = 0.0

try:
    from astro_base.msg import HeadCmd
except ImportError:
    try:
        from astro_base.msg._head_cmd import HeadCmd
    except ImportError:
        HeadCmd = DummyHeadCmd


def doa_to_robot_yaw(doa_deg: float, offset_deg: float = 0.0, invert: bool = False) -> float:
    """Converts ReSpeaker circular DOA (0°..359° clockwise) to robot body yaw frame (-180°..+180°).

    Robot body yaw follows REP-103: rotation about +Z, so POSITIVE yaw is LEFT. That is
    the convention of the URDF head_yaw_joint (axis 0 0 1) and of GESTURE_PROFILES
    ("look_left": +35, "look_right": -35).

    ReSpeaker measures clockwise, which is the opposite sense, so invert=True is the
    correct setting for the standard mounting:
      - DOA   0° = straight ahead    ->    0° yaw
      - DOA  90° = right            ->  -90° yaw (invert=True)
      - DOA 270° = left             ->  +90° yaw (invert=True)
      - DOA 180° = directly behind  -> ±180° yaw
    """
    raw = (doa_deg + offset_deg) % 360.0
    if raw <= 180.0:
        yaw = raw
    else:
        yaw = raw - 360.0

    if invert:
        yaw = -yaw
    return yaw


def angular_diff_deg(a: float, b: float) -> float:
    """Calculates minimal signed angular difference between two angles in degrees (-180..+180)."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


def wrap_deg(angle: float) -> float:
    """Folds any angle into (-180, +180], the range every head target lives in."""
    return (angle + 180.0) % 360.0 - 180.0


class SocialGazeStateMachine:
    """Manages social gaze states: IDLE, ATTENDING, ENGAGED, RETURNING."""
    IDLE = "IDLE"
    ATTENDING = "ATTENDING"
    ENGAGED = "ENGAGED"
    RETURNING = "RETURNING"


class CommandSource:
    """Arbitration priority sources for head movement."""
    SAFETY = "SAFETY"
    GESTURE = "GESTURE"
    TURN_TO_SOUND = "TURN_TO_SOUND"
    CENTER = "CENTER"
    TRACKING = "TRACKING"
    IDLE = "IDLE"


class HeadTrackerNode(Node):
    """ROS 2 Node for central head yaw arbitration and multimodal sound-localization."""

    GESTURE_PROFILES: dict[str, list[float]] = {
        "nod": [12.0, -8.0, 0.0],
        "shake": [22.0, -22.0, 12.0, 0.0],
        "tilt": [16.0, 0.0],
        "scan": [-35.0, 35.0, 0.0],
        "center": [0.0],
        "look_left": [35.0],
        "look_right": [-35.0],
    }

    GESTURE_ALIASES: dict[str, str] = {
        "yes": "nod",
        "onayla": "nod",
        "no": "shake",
        "reddet": "shake",
        "merak": "tilt",
        "curious": "tilt",
        "ara": "scan",
        "search": "scan",
        "sifirla": "center",
        "reset": "center",
    }

    def __init__(self):
        super().__init__("head_tracker_node")

        # Declare parameters from the single defaults table
        for _name, _default in HEAD_TRACKER_DEFAULTS.items():
            self.declare_parameter(_name, _default)

        # Load parameters
        def _get_val(name, default=None):
            default = HEAD_TRACKER_DEFAULTS[name] if default is None else default
            try:
                p = self.get_parameter(name)
                val = getattr(p, "value", default)
                return val if val is not None else default
            except Exception:
                return default

        self.enabled = bool(_get_val("enabled"))
        self.doa_offset_deg = float(_get_val("doa_offset_deg"))
        self.doa_invert = bool(_get_val("doa_invert"))
        self.max_yaw_deg = float(_get_val("max_yaw_deg"))
        self.min_yaw_deg = float(_get_val("min_yaw_deg"))
        # A band covering the whole circle means there is no mechanical stop to respect.
        self._full_circle_travel = (self.max_yaw_deg - self.min_yaw_deg) >= 359.9
        self.deadband_deg = float(_get_val("deadband_deg"))
        self.min_dwell_time_s = float(_get_val("min_dwell_time_s"))
        self.idle_return_timeout_s = float(_get_val("idle_return_timeout_s"))
        self.max_speed_deg_s = float(_get_val("max_speed_deg_s"))
        self.update_rate_hz = float(_get_val("update_rate_hz"))
        self.min_rms_threshold = float(_get_val("min_rms_threshold"))
        self.noise_multiplier = float(_get_val("noise_multiplier"))
        self.head_motion_settle_s = float(_get_val("head_motion_settle_s"))
        self.consensus_window_size = max(1, int(_get_val("consensus_window_size")))
        self.consensus_threshold = max(1, int(_get_val("consensus_threshold")))
        self.consensus_tolerance_deg = float(_get_val("consensus_tolerance_deg"))
        # OAK-D Lite vision fusion
        self.vision_fusion_enabled = bool(_get_val("vision_fusion_enabled"))
        self.vision_gain = float(_get_val("vision_gain"))
        self.vision_max_correction_deg = float(_get_val("vision_max_correction_deg"))
        self.vision_timeout_s = float(_get_val("vision_timeout_s"))
        # 2D LiDAR Radar tracking fusion
        self.lidar_fusion_enabled = bool(_get_val("lidar_fusion_enabled"))
        self.lidar_min_dist_m = float(_get_val("lidar_min_dist_m"))
        self.lidar_max_dist_m = float(_get_val("lidar_max_dist_m"))
        self.lidar_timeout_s = float(_get_val("lidar_timeout_s"))

        # Publishers — SINGLE AUTHORITATIVE OUTPUT OWNER FOR /head_cmd
        if HeadCmd is not None and hasattr(HeadCmd, "_TYPE_SUPPORT"):
            self.pub_head_cmd = self.create_publisher(HeadCmd, "/head_cmd", 10)
        else:
            class _DummyPub:
                def publish(inner_self, msg): pass
            self.pub_head_cmd = _DummyPub()
        self.pub_head_status = self.create_publisher(String, "/head/status", 10)

        # External Intent / Action Subscriptions (Central Arbitration)
        self.create_subscription(String, "/head/gesture", self._on_gesture_cmd, 10)
        self.create_subscription(Float32, "/head/target_yaw", self._on_target_yaw_cmd, 10)
        self.create_subscription(Bool, "/head/safety", self._on_safety_cmd, 10)

        # Subscribers — acoustic
        self.create_subscription(Float32, "/audio/doa", self._on_doa, 10)
        self.create_subscription(Float32, "/audio/mic_level", self._on_mic_level, 10)
        self.create_subscription(Bool, "/audio/vad", self._on_vad, 10)
        self.create_subscription(Bool, "/tts/speaking", self._on_tts_speaking, 10)
        self.create_subscription(Bool, "/audio/playback_active", self._on_playback_active, 10)
        self.create_subscription(String, "/robot/emotion", self._on_emotion, 10)
        # Subscribers — vision (OAK-D Lite face_detector_node)
        self.create_subscription(Bool, "/vision/person_detected", self._on_person_detected, 10)
        self.create_subscription(Float32, "/vision/head_yaw", self._on_vision_head_yaw, 10)
        self.create_subscription(String, "/vision/faces", self._on_vision_faces, 10)
        # Subscribers — radar/LiDAR (RPLIDAR scan_filter_node with SensorData QoS)
        self.create_subscription(LaserScan, "/scan", self._on_laser_scan, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan_filtered", self._on_laser_scan, qos_profile_sensor_data)

        # State & Thread Safety
        self._lock = threading.Lock()
        # Semantics:
        # - _target_yaw: arbitrated goal angle [-70°, +70°]
        # - _estimated_yaw: software trajectory angle simulated with slew limiter
        # - commanded_yaw: the actual value sent over /head_cmd
        self._target_yaw = 0.0
        self._estimated_yaw = 0.0
        self._filtered_target_yaw = 0.0
        self._state = SocialGazeStateMachine.IDLE
        self._command_source = CommandSource.IDLE
        self._spatial_people_map: list[dict] = []
        self._last_speaker_switch_time: float = 0.0

        # Gesture sequencing state
        self._active_gesture: Optional[str] = None
        self._gesture_steps: list[float] = []
        self._gesture_step_index: int = 0
        self._gesture_step_start_time: float = 0.0
        self._gesture_step_duration_s: float = 0.35

        # Turn to sound state
        self._turn_to_sound_active: bool = False
        self._turn_to_sound_start_time: float = 0.0
        self._turn_to_sound_timeout_s: float = 3.0

        # Acoustic & Noise Tracking
        self._ambient_rms = 120.0
        self._latest_rms = 0.0
        self._vad_active = False
        self._is_speaking = False
        self._is_playback_active = False
        self._is_sleeping = True
        self._last_speech_time = 0.0
        self._last_gaze_switch_time = time.monotonic()
        self._last_update_time = time.monotonic()
        self._last_motion_cmd_time = 0.0

        # OAK-D Lite vision state
        self._vision_person_detected = False
        self._vision_head_yaw = 0.0          # Relative camera yaw: negative=left of frame, positive=right
        self._vision_ref_yaw = 0.0           # Head angle the pending measurement was taken from
        self._vision_anchor_yaw = 0.0        # Head angle this servo "budget" started from
        self._vision_last_applied_yaw = None # Bearing that produced the last correction
        self._vision_yaw_pending = False     # True only while a fresh measurement awaits servoing
        self._vision_last_seen_time = 0.0    # monotonic time of last person_detected=True

        # 2D LiDAR radar tracking state
        self._lidar_person_detected = False
        self._lidar_target_yaw = 0.0
        self._lidar_distance_m = 0.0
        self._lidar_last_seen_time = 0.0

        # Circular buffer for DOA temporal consensus
        self._doa_history: Deque[Tuple[float, float]] = collections.deque(maxlen=self.consensus_window_size)

        # Periodic control loop timer (20 Hz)
        timer_period = 1.0 / max(1.0, self.update_rate_hz)
        self._control_timer = self.create_timer(timer_period, self._control_loop)

        self.get_logger().info(
            f"🤖 [Head Tracker] Central Arbiter Başlatıldı | Limitler: [{self.min_yaw_deg}°, {self.max_yaw_deg}°] | "
            f"Maks Hız: {self.max_speed_deg_s}°/s | Deadband: {self.deadband_deg}° | Dwell: {self.min_dwell_time_s}s"
        )

    def _acoustic_bearing_to_body_yaw(self, relative_deg: float) -> float:
        """Turns a head-relative acoustic bearing into an absolute body-frame yaw target.

        The mic array is rigidly attached to head_link (URDF mic_joint), so it turns with
        the head and reports where the talker is relative to the FACE, not to the body.
        Assigning that number straight to _target_yaw makes the loop chase its own tail:
        the head swings out to the bearing, the array then reads ~0 because it is pointing
        at the talker, 0 is taken as "face forward", the head returns to centre, and the
        cycle repeats. Adding the bearing to where the head already points gives a fixed
        point at the talker's true body angle, which is what actually converges.

        The base is the angle last written to /head_cmd, because that is the setpoint the
        Arduino's own position PID is driving to. _estimated_yaw is only the node's slew
        trajectory towards it and lags the real neck by seconds on a large move, so
        composing onto it produces a wrong target for the whole flight.
        """
        return wrap_deg(self._commanded_head_yaw() + relative_deg)

    def _commanded_head_yaw(self) -> float:
        """Best available statement of where the neck is: the last setpoint sent out.

        There is no encoder feedback -- the firmware prints its head ticks only to its
        Serial2 debug port -- so the commanded angle is as close to ground truth as this
        node can get. Streaming those ticks back would make it exact.
        """
        last = getattr(self, "_last_published_cmd_yaw", None)
        return float(last) if last is not None else float(self._estimated_yaw)

    def _shortest_reachable_arc(self, target_yaw: float, current_yaw: float) -> float:
        """Signed rotation to get from current_yaw to target_yaw the short way.

        A plain subtraction sends the neck 340 deg around the houses whenever the two
        angles sit either side of the +-180 seam, which is the "goes the long way" the
        operator sees. The wrapped difference is the short arc; it is only unusable when
        it would run past a mechanical limit, and then the long arc is the one that fits.
        On a range-limited neck the result therefore always stays inside
        [min_yaw_deg, max_yaw_deg], so the angle handed to the firmware never jumps and
        its dead-reckoned zero keeps its meaning.
        """
        short = angular_diff_deg(target_yaw, current_yaw)

        if self._full_circle_travel:
            # +-180 is a seam in how the angle is written down, not a mechanical stop.
            # With the whole circle reachable the short arc is always the right one, and
            # _estimated_yaw is re-wrapped after the step so it stays in (-180, +180].
            # This only holds because the firmware wraps its position error too
            # (HEAD_CONTINUOUS_ROTATION in arduino/astro_firmware/src/main.cpp); without
            # that, a +179 -> -179 setpoint would be executed as 358 deg the wrong way.
            return short

        if self.min_yaw_deg <= current_yaw + short <= self.max_yaw_deg:
            return short

        long_way = short - math.copysign(360.0, short) if short != 0.0 else 0.0
        if self.min_yaw_deg <= current_yaw + long_way <= self.max_yaw_deg:
            return long_way

        # Neither arc lands inside the band: aim at the closer limit.
        clamped = max(self.min_yaw_deg, min(self.max_yaw_deg, current_yaw + short))
        return clamped - current_yaw

    def _head_in_motion(self, now: float) -> bool:
        """True for a bounded window after the neck was last commanded to move.

        ROS never learns the real head angle — the firmware keeps its encoder ticks to
        itself — so the exact moment the motor stops is unknowable here. Gating on the
        software trajectory instead would over-block for seconds, because _estimated_yaw
        crawls at max_speed_deg_s while the Arduino PID has long since arrived. A fixed
        quiet window after each command is the honest approximation; streaming head ticks
        back over the serial link would let this become exact.
        """
        return (now - self._last_motion_cmd_time) < self.head_motion_settle_s

    @property
    def _current_yaw(self) -> float:
        """Backward-compatible property returning software estimated trajectory yaw."""
        return self._estimated_yaw

    @_current_yaw.setter
    def _current_yaw(self, val: float):
        self._estimated_yaw = float(val)

    def _on_safety_cmd(self, msg: Bool):
        """Emergency safety lock: immediately overrides all commands and locks head at center."""
        with self._lock:
            if msg.data:
                self._is_sleeping = True
                self._command_source = CommandSource.SAFETY
                self._active_gesture = None
                self._gesture_steps = []
                self._target_yaw = 0.0
                self.get_logger().info("🛑 [Safety Arbiter]: Acil güvenlik kilidi devrede -> Hedef 0°")
            else:
                self._is_sleeping = False
                self._command_source = CommandSource.IDLE
                self.get_logger().info("🟢 [Safety Arbiter]: Güvenlik kilidi kaldırıldı.")

    def _on_gesture_cmd(self, msg: String):
        """Queues a gesture sequence in the central arbitration engine."""
        if not self.enabled:
            return
        g_name = (msg.data or "").lower().strip()
        if not g_name:
            return
        canonical = self.GESTURE_ALIASES.get(g_name, g_name)
        if canonical not in self.GESTURE_PROFILES:
            self.get_logger().warning(f"Unknown gesture: '{g_name}'")
            return

        now = time.monotonic()
        with self._lock:
            if self._command_source == CommandSource.SAFETY:
                return
            self._is_sleeping = False
            if canonical == "center":
                self._active_gesture = None
                self._gesture_steps = []
                self._command_source = CommandSource.CENTER
                self._target_yaw = 0.0
                self._last_speech_time = now
                self.get_logger().info("🎯 [Gesture Arbiter]: Merkeze yönelme (CENTER) komutu alındı.")
            else:
                self._active_gesture = canonical
                self._gesture_steps = list(self.GESTURE_PROFILES[canonical])
                self._gesture_step_index = 0
                self._gesture_step_start_time = now
                self._gesture_step_duration_s = 0.35
                self._command_source = CommandSource.GESTURE
                self._target_yaw = max(self.min_yaw_deg, min(self.max_yaw_deg, self._gesture_steps[0]))
                self._last_speech_time = now
                self._state = SocialGazeStateMachine.ATTENDING
                self.get_logger().info(
                    f"🎭 [Gesture Arbiter]: Jest başlatıldı -> '{canonical}' "
                    f"(Adım 1/{len(self._gesture_steps)}: {self._target_yaw:.1f}°)"
                )

    def _on_target_yaw_cmd(self, msg: Float32):
        """Handles /head/target_yaw, the explicit "turn to that sound" command.

        Its only publisher is action_manager, which derives the angle from the same
        ReSpeaker DOA, so the number is a bearing RELATIVE TO THE HEAD and has to be
        composed onto where the head already points — see _acoustic_bearing_to_body_yaw.
        """
        now = time.monotonic()
        req_yaw = float(msg.data)
        with self._lock:
            body_yaw = self._acoustic_bearing_to_body_yaw(req_yaw)
            clamped_yaw = max(self.min_yaw_deg, min(self.max_yaw_deg, body_yaw))
            if self._command_source in (CommandSource.SAFETY, CommandSource.GESTURE):


                self.get_logger().info(f"⏳ [Arbitration]: turn_to_sound ({clamped_yaw:.1f}°) ertelendi / öncelik: {self._command_source}")
                return
            self._is_sleeping = False
            self._command_source = CommandSource.TURN_TO_SOUND
            self._target_yaw = clamped_yaw
            self._filtered_target_yaw = clamped_yaw
            self._turn_to_sound_active = True
            self._turn_to_sound_start_time = now
            self._turn_to_sound_timeout_s = 2.5  # Initial acoustic orientation lock (2.5s), then hands over to vision
            self._last_speech_time = now
            self._last_gaze_switch_time = now
            self._doa_history.clear()
            self._state = SocialGazeStateMachine.ATTENDING
            self.get_logger().info(f"🎯 [Arbitration]: turn_to_sound hedefi ayarlandı -> {self._target_yaw:.1f}° (2.5s kilitlendi)")

    def _on_doa(self, msg: Float32):
        """Processes raw acoustic Direction of Arrival from ReSpeaker."""
        if not self.enabled:
            return

        now = time.monotonic()
        raw_doa = float(msg.data)

        with self._lock:
            # 1. Sleep & Self-Voice Gate: Ignore DOA if robot is sleeping, speaking, or playing audio
            if self._is_sleeping or self._is_speaking or self._is_playback_active:
                return

            # 2. Priority check: If gesture, safety, or active turn_to_sound is running, do NOT override
            if self._command_source in (CommandSource.SAFETY, CommandSource.GESTURE):
                return
            if self._command_source == CommandSource.TURN_TO_SOUND and self._turn_to_sound_active:
                return

            # 3. Vision Priority: If camera actively sees a person/face, do NOT override with acoustic DOA
            if self.vision_fusion_enabled and self._vision_person_detected and (now - self._vision_last_seen_time) <= self.vision_timeout_s:
                return

            # 3b. Head-Motion Gate: the ReSpeaker array is mounted on the head, so the
            #     yaw motor's structure-borne noise reaches the capsules directly and
            #     the array reports an unusable bearing while the neck is turning.
            if self._head_in_motion(now):
                self._doa_history.clear()
                return

            # 4. Acoustic Energy & VAD Gate: Require BOTH active VAD AND energy significantly above ambient
            #    (Prevents random room noises, chair movements, or TV spikes from moving the head)
            adaptive_threshold = max(self.min_rms_threshold, self._ambient_rms * self.noise_multiplier)
            if not self._vad_active or self._latest_rms < adaptive_threshold:
                return

            # Bearing of the talker as seen FROM THE HEAD, in REP-103 sense.
            relative_yaw = doa_to_robot_yaw(raw_doa, offset_deg=self.doa_offset_deg, invert=self.doa_invert)

            # ...then expressed in the robot body frame. Storing the body-frame angle is
            # also what makes the consensus window meaningful: samples taken while the
            # neck was at different angles are only comparable once they share a frame.
            body_yaw = self._acoustic_bearing_to_body_yaw(relative_yaw)





            # Record the UNCLAMPED bearing: clamping before the consensus step splits a
            # single source behind the robot into alternating +max/-max samples.
            # Mechanical clamping happens once, after consensus (step 7).
            self._doa_history.append((now, body_yaw))
            self._last_speech_time = now

            # 6. Temporal Consensus Clustering with Angular Wrap-Around Handling
            candidate_yaw = self._evaluate_consensus()
            if candidate_yaw is None:
                return

            # Multi-Speaker Spatial Face Association (Snap acoustic DOA to visual face if in vicinity)
            matched_person_name = None
            for person in getattr(self, "_spatial_people_map", []):
                if (now - person.get("timestamp", 0.0)) <= 4.0:
                    if abs(angular_diff_deg(candidate_yaw, person["world_yaw"])) <= 25.0:
                        candidate_yaw = person["world_yaw"]
                        matched_person_name = person["name"]
                        break

            # 2D LiDAR (Radar) Fusion Association: Snap coarse acoustic angle to precise physical human/obstacle detected in zone
            if not matched_person_name and self.lidar_fusion_enabled and getattr(self, "_lidar_person_detected", False):
                if (now - getattr(self, "_lidar_last_seen_time", 0.0)) <= self.lidar_timeout_s:
                    lidar_target = getattr(self, "_lidar_target_yaw", 0.0)
                    lidar_dist = getattr(self, "_lidar_distance_m", 0.0)
                    if abs(angular_diff_deg(candidate_yaw, lidar_target)) <= 35.0:
                        self.get_logger().info(
                            f"📡 [Radar Fusion] Akustik DOA ({candidate_yaw:.1f}°) LiDAR radar hedefiyle ({lidar_target:.1f}°, {lidar_dist:.2f}m) kilitlendi."
                        )
                        candidate_yaw = lidar_target

            # 7. Mechanical & Safety Clamping
            clamped_target = max(self.min_yaw_deg, min(self.max_yaw_deg, candidate_yaw))

            # Confidence calculation
            cluster_size = getattr(self, "_last_cluster_size", self.consensus_threshold)
            total_samples = getattr(self, "_last_total_samples", len(self._doa_history))
            energy_ratio = self._latest_rms / max(80.0, self._ambient_rms)
            conf = min(1.0, max(0.1, (cluster_size / float(total_samples)) * min(1.0, energy_ratio / 2.0)))

            last_logged_filtered = getattr(self, "_last_logged_filtered_yaw", None)
            last_logged_time = getattr(self, "_last_logged_doa_time", 0.0)
            if last_logged_filtered is None or abs(clamped_target - last_logged_filtered) >= 10.0 or (now - last_logged_time) >= 3.0:
                self._last_logged_filtered_yaw = clamped_target
                self._last_logged_doa_time = now
                self.get_logger().info(
                    f"[DOA FILTER]\n"
                    f"raw={raw_doa:.1f}\n"
                    f"confidence={conf:.2f}\n"
                    f"filtered={clamped_target:.1f}\n"
                    f"target_yaw={clamped_target:.1f}"
                )

            # 8. Gaze Dwell Time & Deadband Hysteresis (with Multi-Speaker Turn-Taking Responsiveness)
            dwell_elapsed = now - self._last_gaze_switch_time
            angle_diff = abs(angular_diff_deg(clamped_target, self._current_yaw))

            # Allow speaker switch if dwell time elapsed OR if a distinct multi-speaker turn is detected (>20° after min 0.8s)
            min_dwell = 0.8 if (matched_person_name and angle_diff >= 20.0) else self.min_dwell_time_s
            if angle_diff >= self.deadband_deg and dwell_elapsed >= min_dwell:
                self._target_yaw = clamped_target
                self._filtered_target_yaw = clamped_target
                self._last_gaze_switch_time = now
                self._state = SocialGazeStateMachine.ATTENDING
                spk_tag = f" ({matched_person_name})" if matched_person_name else ""
                self.get_logger().info(
                    f"🎯 [Head Tracker Gaze]: DOA={raw_doa:.1f}° -> Hedef Yaw={clamped_target:.1f}°{spk_tag} "
                    f"(Fark={angle_diff:.1f}°, RMS={self._latest_rms:.0f})"
                )

    def _evaluate_consensus(self) -> Optional[float]:
        """Finds cluster consensus in recent DOA history to filter reverberation & outlier spikes."""
        if len(self._doa_history) < self.consensus_threshold:
            return None

        recent_samples = [yaw for ts, yaw in self._doa_history]
        best_cluster = []

        for candidate in recent_samples:
            cluster = [y for y in recent_samples if abs(angular_diff_deg(y, candidate)) <= self.consensus_tolerance_deg]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if len(best_cluster) >= self.consensus_threshold:
            sin_sum = sum(math.sin(math.radians(y)) for y in best_cluster)
            cos_sum = sum(math.cos(math.radians(y)) for y in best_cluster)
            avg_yaw = math.degrees(math.atan2(sin_sum, cos_sum))
            self._last_cluster_size = len(best_cluster)
            self._last_total_samples = len(recent_samples)
            return float(avg_yaw)
        return None

    def _on_mic_level(self, msg: Float32):
        """Tracks RMS audio input energy and background ambient noise floor."""
        rms = float(msg.data)
        with self._lock:
            self._latest_rms = rms
            if not self._is_speaking and not self._is_playback_active and rms < 400.0:
                # Slowly track ambient background level
                self._ambient_rms = 0.95 * self._ambient_rms + 0.05 * rms

    def _on_vad(self, msg: Bool):
        with self._lock:
            self._vad_active = bool(msg.data)
            if self._vad_active:
                self._last_speech_time = time.monotonic()

    def _on_tts_speaking(self, msg: Bool):
        with self._lock:
            self._is_speaking = bool(msg.data)

    def _on_playback_active(self, msg: Bool):
        with self._lock:
            self._is_playback_active = bool(msg.data)
            self._doa_history.clear()
            if not self._is_playback_active:
                self._last_speech_time = time.monotonic()

    def _on_emotion(self, msg: String):
        """Monitors robot emotional and sleep state to lock head during sleep/deep-idle."""
        with self._lock:
            emo = str(msg.data).strip().lower()
            if emo in ("sleeping", "sleep", "deep_idle"):
                if not self._is_sleeping:
                    self.get_logger().info(f"💤 [Head Tracker]: Uyku modu aktif ({emo}) — Kafa 0.0° merkezde kilitlendi.")
                self._is_sleeping = True
                self._doa_history.clear()
                self._target_yaw = 0.0
                self._state = SocialGazeStateMachine.IDLE
            else:
                if self._is_sleeping:
                    self.get_logger().info(f"⏰ [Head Tracker]: Uyanma algılandı ({emo}) — Ses ve bakış takibi devrede.")
                self._is_sleeping = False
                self._last_speech_time = time.monotonic()

    def _on_person_detected(self, msg: Bool):
        """OAK-D Lite: called when face detector publishes person presence."""
        with self._lock:
            was_detected = self._vision_person_detected
            self._vision_person_detected = bool(msg.data)
            if self._vision_person_detected:
                self._vision_last_seen_time = time.monotonic()
                if not was_detected:
                    self._vision_anchor_yaw = self._commanded_head_yaw()
                    self._vision_last_applied_yaw = None
                    self.get_logger().info("👁️ [Vision Fusion] Yüz tespit edildi — Kamera yönelimi aktif")

    def _on_vision_head_yaw(self, msg: Float32):
        """OAK-D Lite: face_detector_node publishes relative head yaw inside camera frame."""
        if not self.vision_fusion_enabled:
            return
        with self._lock:
            self._vision_head_yaw = float(msg.data)
            # Where the neck was pointing when this bearing was measured. The correction
            # is anchored here rather than on the running target, so replaying a stale
            # measurement keeps describing the same spot in the room instead of walking
            # the head another step every time -- see _control_loop's vision branch.
            self._vision_ref_yaw = self._commanded_head_yaw()
            self._vision_yaw_pending = True

    def _on_vision_faces(self, msg: String):
        """Maintains local spatial map of detected people in robot coordinate frame."""
        if not self.vision_fusion_enabled:
            return
        try:
            faces = json.loads(msg.data)
            if not isinstance(faces, list):
                return
            now = time.monotonic()
            with self._lock:
                updated_map = []
                for f in faces:
                    cam_azimuth = float(f.get("camera_azimuth_deg", 0.0))
                    world_yaw = max(self.min_yaw_deg, min(self.max_yaw_deg, self._estimated_yaw + cam_azimuth))
                    updated_map.append({
                        "name": f.get("recognized_name") or "Misafir",
                        "is_known": bool(f.get("is_known", False)),
                        "cam_azimuth": cam_azimuth,
                        "world_yaw": world_yaw,
                        "distance_m": float(f.get("distance_m", 1.0)),
                        "timestamp": now,
                    })
                if updated_map:
                    self._spatial_people_map = updated_map
        except Exception as _exc:
            self.get_logger().debug(f"_on_vision_faces json error: {_exc}")

    def _on_laser_scan(self, msg: Any):
        """Processes 2D LiDAR planar scan to detect humans/objects in the social zone around the robot."""
        if not self.enabled or not self.lidar_fusion_enabled or self._is_sleeping:
            return

        now = time.monotonic()
        ranges = getattr(msg, "ranges", None)
        if not ranges:
            return

        angle_min = getattr(msg, "angle_min", -math.pi)
        angle_increment = getattr(msg, "angle_increment", 0.01745)

        points: list[tuple[float, float]] = []
        for i, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < self.lidar_min_dist_m or r > self.lidar_max_dist_m:
                continue
            ang_rad = angle_min + (i * angle_increment)
            ang_deg = math.degrees(ang_rad)
            # Normalize to [-180, 180]
            ang_deg = (ang_deg + 180.0) % 360.0 - 180.0
            points.append((float(r), float(ang_deg)))

        if not points:
            return

        # Sort by distance (closest obstacle / human)
        points.sort(key=lambda p: p[0])
        closest_r, raw_closest_angle = points[0]
        clamped_closest_angle = max(self.min_yaw_deg, min(self.max_yaw_deg, raw_closest_angle))

        with self._lock:
            self._lidar_person_detected = True
            self._lidar_target_yaw = clamped_closest_angle
            self._lidar_distance_m = closest_r
            self._lidar_last_seen_time = now

    def _control_loop(self):
        """Periodic 20 Hz trajectory generator with central arbitration and slew-rate limiting."""
        if not self.enabled:
            return

        now = time.monotonic()
        with self._lock:
            dt = max(0.001, now - self._last_update_time)
            self._last_update_time = now

            # =========================================================
            # CENTRAL ARBITRATION PRIORITY
            # Priority: SAFETY > GESTURE > TURN_TO_SOUND > CENTER > TRACKING
            # =========================================================

            # 1. SAFETY / SLEEP (Highest Priority)
            if self._is_sleeping or self._command_source == CommandSource.SAFETY:
                self._target_yaw = 0.0
                if abs(self._estimated_yaw) > 1.0:
                    self._state = SocialGazeStateMachine.RETURNING
                else:
                    self._state = SocialGazeStateMachine.IDLE

            # 2. GESTURE SEQUENCE (High Priority — cannot be clobbered by tracking)
            elif self._command_source == CommandSource.GESTURE and self._gesture_steps:
                elapsed_step = now - self._gesture_step_start_time
                step_settled = abs(self._estimated_yaw - self._target_yaw) <= 1.5
                if elapsed_step >= self._gesture_step_duration_s or step_settled:
                    self._gesture_step_index += 1
                    if self._gesture_step_index < len(self._gesture_steps):
                        self._target_yaw = max(self.min_yaw_deg, min(self.max_yaw_deg, self._gesture_steps[self._gesture_step_index]))
                        self._gesture_step_start_time = now
                    else:
                        # Gesture sequence fully completed! Transition smoothly back to tracking
                        self.get_logger().info(f"✅ [Gesture Arbiter]: Tamamlandı -> '{self._active_gesture}', Tracking'e dönülüyor.")
                        self._active_gesture = None
                        self._gesture_steps = []
                        self._command_source = CommandSource.TRACKING
                        self._last_speech_time = now
                        self._last_gaze_switch_time = now
                self._state = SocialGazeStateMachine.ATTENDING

            # 3. TURN_TO_SOUND (Explicit sound orientation & manual angle commands)
            elif self._command_source == CommandSource.TURN_TO_SOUND and self._turn_to_sound_active:
                elapsed_tts = now - self._turn_to_sound_start_time
                vision_acquired = (
                    self.vision_fusion_enabled
                    and self._vision_person_detected
                    and (now - self._vision_last_seen_time) <= self.vision_timeout_s
                    and elapsed_tts >= 0.6  # Give at least 0.6s for head to slew toward initial acoustic heading
                )
                if elapsed_tts >= self._turn_to_sound_timeout_s or vision_acquired:
                    if vision_acquired:
                        self.get_logger().info("👁️ [Vision Fusion] turn_to_sound sırasında yüz yakalandı -> Kamera kilitlendi.")
                    self._turn_to_sound_active = False
                    self._command_source = CommandSource.TRACKING
                    self._state = SocialGazeStateMachine.ATTENDING
                    self._last_speech_time = now
                    self._last_gaze_switch_time = now
                else:
                    self._state = SocialGazeStateMachine.ATTENDING

            # 4. BEHAVIORAL CENTER (Commanded neutral return)
            elif self._command_source == CommandSource.CENTER:
                self._target_yaw = 0.0
                if abs(self._estimated_yaw) <= 1.0:
                    self._command_source = CommandSource.IDLE
                    self._state = SocialGazeStateMachine.IDLE
                else:
                    self._state = SocialGazeStateMachine.RETURNING

            # 5. MULTIMODAL TRACKING (Vision, Radar/LiDAR, Acoustic DOA)
            else:
                # --- Vision Fusion (OAK-D Lite) ---
                vision_active = (
                    self.vision_fusion_enabled
                    and self._vision_person_detected
                    and (now - self._vision_last_seen_time) <= self.vision_timeout_s
                )
                if vision_active:
                    self._last_speech_time = now  # prevent idle return while face is visible
                    self._state = SocialGazeStateMachine.ATTENDING
                    # Confident visual gaze centering (proportional visual servoing)
                    if self._vision_yaw_pending and abs(self._vision_head_yaw) >= 1.5:
                        self._vision_yaw_pending = False
                        
                        # Akıcı ve organik yüz takibi: Görsel servo gecikmesini 80ms'ye indirerek pürüzsüz takip sağla
                        if (now - self._last_motion_cmd_time) > 0.08:

                            # Anchored on the pose the bearing was measured from, not on
                            # the running target. Adding to the target integrates instead
                            # of correcting: the face detector replays its last bounding
                            # box for several frames after a dropout, and each replay used
                            # to push the neck another gain*bearing further, walking it to
                            # the travel limit on a single acquisition. Anchored, the same
                            # measurement always resolves to the same place in the room.
                            bearing = self._vision_head_yaw
                            last = self._vision_last_applied_yaw

                            # The error falling means the corrections are landing, so the
                            # servo has earned a fresh budget from where it now points.
                            if last is None or abs(bearing) < abs(last) - 2.0:
                                self._vision_anchor_yaw = self._vision_ref_yaw

                            desired_yaw = self._vision_ref_yaw + (self.vision_gain * bearing)

                            # A face is only ever seen inside the lens, so it cannot lie
                            # further than half the field of view from where the camera was
                            # looking. Corrections that never shrink the error are not
                            # landing -- a replayed bounding box, a frozen stream, a false
                            # positive -- and without this bound they compound into the
                            # steady 120 deg sweep to the travel limit seen in the field.
                            budget = self.vision_max_correction_deg
                            overshoot = angular_diff_deg(desired_yaw, self._vision_anchor_yaw)
                            if abs(overshoot) > budget:
                                desired_yaw = self._vision_anchor_yaw + math.copysign(budget, overshoot)

                            self._vision_last_applied_yaw = bearing
                            self._target_yaw = max(max(self.min_yaw_deg, -85.0), min(min(self.max_yaw_deg, 85.0), wrap_deg(desired_yaw)))
                elif self._vision_person_detected and (now - self._vision_last_seen_time) > self.vision_timeout_s:

                    self._vision_person_detected = False
                    self.get_logger().info("👁️ [Vision Fusion] Yüz kayboldu — DOA / LiDAR takibine dönülüyor")

                # --- LiDAR / Radar Tracking (RPLIDAR) ---
                lidar_active = (
                    self.lidar_fusion_enabled
                    and getattr(self, "_lidar_person_detected", False)
                    and (now - getattr(self, "_lidar_last_seen_time", 0.0)) <= self.lidar_timeout_s
                    and not self._vad_active
                    and not vision_active
                )
                if lidar_active and (now - self._last_gaze_switch_time) >= self.min_dwell_time_s:
                    target_diff = abs(angular_diff_deg(self._lidar_target_yaw, self._estimated_yaw))
                    if target_diff >= self.deadband_deg:
                        self._target_yaw = max(self.min_yaw_deg, min(self.max_yaw_deg, self._lidar_target_yaw))
                        self._last_speech_time = now
                        self._last_gaze_switch_time = now
                        self._state = SocialGazeStateMachine.ATTENDING
                        self.get_logger().info(
                            f"📡 [LiDAR Radar Gaze]: İnsan/Nesne algılandı -> Hedef Yaw={self._target_yaw:.1f}° "
                            f"(Mesafe={self._lidar_distance_m:.2f}m)"
                        )
                elif getattr(self, "_lidar_person_detected", False) and (now - getattr(self, "_lidar_last_seen_time", 0.0)) > self.lidar_timeout_s:
                    self._lidar_person_detected = False

                # --- Idle Timeout ---
                if (now - self._last_speech_time) > self.idle_return_timeout_s:
                    if abs(self._estimated_yaw) > 2.0:
                        self._target_yaw = 0.0
                        self._state = SocialGazeStateMachine.RETURNING
                    else:
                        self._target_yaw = 0.0
                        self._state = SocialGazeStateMachine.IDLE

            vision_active_flag = bool(
                self.vision_fusion_enabled
                and self._vision_person_detected
                and (now - self._vision_last_seen_time) <= self.vision_timeout_s
                and not self._is_sleeping
            )

            # --- Smooth Trajectory Generation: Central Slew-Rate Limiting & Soft-Landing ---
            err = self._shortest_reachable_arc(self._target_yaw, self._estimated_yaw)
            abs_err = abs(err)

            # Soft landing: gradually decelerate when within 15° of target for organic, non-abrupt stopping
            if abs_err < 15.0:
                cur_speed = max(4.0, self.max_speed_deg_s * (abs_err / 15.0))
            else:
                cur_speed = self.max_speed_deg_s

            max_step = cur_speed * dt

            if abs_err <= max_step:
                self._estimated_yaw = self._target_yaw
            else:
                self._estimated_yaw += math.copysign(max_step, err)

            if self._full_circle_travel:
                self._estimated_yaw = wrap_deg(self._estimated_yaw)

            commanded_yaw_to_send = self._estimated_yaw
            target_yaw_snapshot = self._target_yaw
            state_snapshot = self._state

        # --- Publish /head_cmd (SINGLE OUTPUT OWNER) ---
        # Microcontroller (Arduino) executes its own 50 Hz PID position controller.
        # Publish target yaw setpoint directly on target change (>= 1.0°) or state change.
        last_pub = getattr(self, "_last_published_cmd_yaw", None)
        is_idle_settled = (state_snapshot == SocialGazeStateMachine.IDLE and abs(target_yaw_snapshot) < 0.1)

        should_publish = False
        if is_idle_settled:
            # Send settling 0° command once when returning to idle
            if last_pub is None or abs(last_pub) >= 0.5:
                should_publish = True
        else:
            # Publish on target change >= 1.0°
            if last_pub is None or abs(target_yaw_snapshot - last_pub) >= 1.0:
                should_publish = True

        if should_publish:
            cmd = HeadCmd()
            cmd.angle_deg = float(target_yaw_snapshot)
            if self.pub_head_cmd is not None:
                self.pub_head_cmd.publish(cmd)
            self._last_published_cmd_yaw = target_yaw_snapshot
            self._last_motion_cmd_time = now

        # Periodic status telemetry
        status = {
            "current_yaw_deg": round(commanded_yaw_to_send, 1),
            "target_yaw_deg": round(target_yaw_snapshot, 1),
            "state": state_snapshot,
            "source": getattr(self, "_command_source", CommandSource.IDLE),
            "ambient_rms": round(self._ambient_rms, 1),
            "is_speaking": self._is_speaking,
            "vision_active": vision_active_flag,
        }
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.pub_head_status.publish(status_msg)



def main(args=None):
    rclpy.init(args=args)
    node = HeadTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
