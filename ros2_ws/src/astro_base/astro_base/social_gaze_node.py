#!/usr/bin/env python3
"""Social Gaze ROS 2 Node for ASTRO Robot Head.


Integrates:
  1. Audio Perception & GCC-PHAT Filtering
  2. 3D Visual Face Tracking & Kalman Estimation
  3. Multimodal Audio-Visual Association & Fusion
  4. Dual-Threshold Target Management & Turn-Taking Arbitration
  5. 9-State Social Gaze FSM & Priority Arbiter
  6. Jerk-Limited S-Curve Motion Planner (50 Hz Control Loop)
  7. Closed-Loop Hardware Safety & Watchdog Monitor
"""

from collections import deque, namedtuple
import json
import math
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from astro_base.gaze.gaze_runtime import GazeRuntimeCore
from astro_base.gaze.gaze_tracker import Detection, GazeResult, GazeTracker

SpeechEstimate = namedtuple("SpeechEstimate", ["is_speech", "confidence"])

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float32, Header, Int32, String
    from sensor_msgs.msg import JointState
    try:
        from astro_base.msg import GazeStatus, HeadCmd, HeadState
    except ImportError:
        GazeStatus = HeadCmd = HeadState = None
except ImportError:
    class _MockRclpy:
        @staticmethod
        def ok(): return True
        @staticmethod
        def shutdown(): pass
        @staticmethod
        def init(*args, **kwargs): pass
    rclpy = _MockRclpy()

    class _MockParam:
        def __init__(self, val): self.value = val
        def get_parameter_value(self):
            class _Val:
                def __init__(self, v):
                    self.string_value = str(v) if v is not None else ""
                    self.double_value = float(v) if isinstance(v, (int, float)) else 0.0
                    self.integer_value = int(v) if isinstance(v, (int, float)) else 0
                    self.bool_value = bool(v)
            return _Val(self.value)

    class _MockPublisher:
        """Captures published messages so headless tests can assert on node output."""
        def __init__(self, topic): self.topic = topic; self.last_msg = None; self.count = 0
        def publish(self, msg): self.last_msg = msg; self.count += 1

    class _MockClock:
        def now(self):
            class _Time:
                def to_msg(self): return None
            return _Time()

    class Node:
        def __init__(self, *args, **kwargs):
            self._params = {}
        def create_publisher(self, msg_type, topic, *args, **kwargs):
            return _MockPublisher(topic)
        def create_subscription(self, *args, **kwargs): return None
        def create_timer(self, *args, **kwargs): return None
        def get_clock(self): return _MockClock()
        def get_logger(self):
            import logging
            return logging.getLogger("SocialGazeNode")
        def declare_parameter(self, name, value=None, *args, **kwargs):
            # Remembering the declared default is what lets the headless harness
            # exercise the node's real thresholds instead of empty-string stubs.
            self._params[name] = value
            return _MockParam(value)
        def get_parameter(self, name):
            return _MockParam(self._params.get(name))
    class QoSProfile:
        def __init__(self, *args, **kwargs): pass

    class ReliabilityPolicy:
        BEST_EFFORT = 0
        RELIABLE = 1

    class _MockMsg:
        """Assignable stand-in for std_msgs/sensor_msgs types."""
        def __init__(self, data=None, **kwargs):
            self.data = data
            for key, val in kwargs.items():
                setattr(self, key, val)

    Bool = Float32 = Header = Int32 = String = JointState = _MockMsg
    GazeStatus = HeadCmd = HeadState = None

from astro_base.gaze.angle_math import angular_diff_deg
from astro_base.gaze.audio_filter import AudioFilterCore
from astro_base.gaze.audio_perception import AudioPerceptionCore
from astro_base.gaze.coordinate_frames import CalibrationConfig, CoordinateTransformer
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.head_controller import HeadControllerCore
from astro_base.gaze.motion_planner import MotionPlannerCore
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.spatial_memory import EpistemicSpatialMemory
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
    HeadFeedback,
    Modality,
    PrioritySource,
    TargetSelectorType,
    TargetState,
    TrackingState,
    TrajectoryPoint,
    VisualMeasurement,
    VisualObservation,
    VisualTargetTrack,
)
from astro_base.gaze.visual_perception import VisualPerceptionCore
from astro_base.gaze.visual_tracker import VisualTrackerCore


# Credited to a detection whose publisher reports no confidence of its own.
# It has to clear the target manager's 0.40 hold threshold — an unscored person is
# still a person, and dropping them would be worse than the bug — while staying
# under the 0.75 acquisition threshold, so a lone unscored frame cannot seize the
# head. Corroboration (a detector score, or speech from the same bearing) is what
# lifts such a candidate to active target.
UNSCORED_DETECTION_CONFIDENCE = 0.65


class SocialGazeNode(Node):
    """Authoritative Social Gaze Controller Node for ASTRO Robot."""

    def __init__(self):
        super().__init__("social_gaze_node")

        # -------------------------------------------------------------------------
        # 1. Parameter Declaration & Loading
        # -------------------------------------------------------------------------
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("config_file", "")
        self.declare_parameter("calibration_file", "")

        self.declare_parameter("max_velocity_deg_s", 75.0)
        self.declare_parameter("max_acceleration_deg_s2", 180.0)
        self.declare_parameter("max_jerk_deg_s3", 360.0)
        self.declare_parameter("gaze_deadband_deg", 3.0)
        self.declare_parameter("min_attention_dwell_s", 2.50)
        self.declare_parameter("turn_taking_min_dwell_s", 0.80)
        self.declare_parameter("spatial_gate_deg", 25.0)
        self.declare_parameter("idle_saccades_enabled", False)
        self.declare_parameter("self_speech_suppression", True)
        self.declare_parameter("enable_actuator_output", False)
        self.enable_actuator_output = bool(self.get_parameter("enable_actuator_output").value)

        # Load Calibration
        calib_file = self.get_parameter("calibration_file").get_parameter_value().string_value
        if calib_file and os.path.exists(calib_file):
            self.calib = CalibrationConfig.from_yaml_file(calib_file)
            self.get_logger().info(f"Loaded calibration from {calib_file}")
        else:
            self.calib = CalibrationConfig()

        self.transformer = CoordinateTransformer(self.calib)

        # Epistemic Situational Spatial Memory
        self.spatial_memory = EpistemicSpatialMemory()

        # -------------------------------------------------------------------------
        # 2. Pipeline Core Modules Initialization
        # -------------------------------------------------------------------------
        self.audio_perception = AudioPerceptionCore(
            transformer=self.transformer,
            self_speech_suppression_factor=0.15,
        )
        self.audio_filter = AudioFilterCore(
            max_jump_deg=35.0,
            outlier_persistence_count=3,
            kalman_q=0.08,
            kalman_r=0.45,
        )
        self.visual_perception = VisualPerceptionCore(
            transformer=self.transformer,
            min_confidence=0.50,
            direct_gaze_max_yaw_deg=22.0,
        )
        self.visual_tracker = VisualTrackerCore(
            transformer=self.transformer,
            gating_distance_m=0.85,
            coasting_timeout_s=0.70,
        )
        self.fusion = AudioVisualFusionCore(
            spatial_gate_deg=float(self.get_parameter("spatial_gate_deg").value),
            audio_freshness_half_life_s=0.80,
            vision_freshness_half_life_s=1.20,
            spatial_memory=self.spatial_memory,
        )
        self.target_manager = TargetManagerCore(
            acquisition_threshold=0.75,
            hold_threshold=0.40,
            target_lost_timeout_s=2.5,
            min_attention_dwell_s=float(self.get_parameter("min_attention_dwell_s").value),
            turn_taking_min_dwell_s=float(self.get_parameter("turn_taking_min_dwell_s").value),
        )
        self.fsm = SocialGazeFSM(
            deadband_deg=float(self.get_parameter("gaze_deadband_deg").value),
            idle_return_timeout_s=20.0,
            min_attention_dwell_s=float(self.get_parameter("min_attention_dwell_s").value),
            idle_saccades_enabled=bool(self.get_parameter("idle_saccades_enabled").value),
            min_limit_deg=self.calib.head.min_angle_deg,
            max_limit_deg=self.calib.head.max_angle_deg,
            spatial_memory=self.spatial_memory,
        )
        self.planner = MotionPlannerCore(
            max_velocity_deg_s=float(self.get_parameter("max_velocity_deg_s").value),
            max_acceleration_deg_s2=float(self.get_parameter("max_acceleration_deg_s2").value),
            max_jerk_deg_s3=float(self.get_parameter("max_jerk_deg_s3").value),
            min_limit_deg=self.calib.head.min_angle_deg,
            max_limit_deg=self.calib.head.max_angle_deg,
        )
        self.head_ctrl = HeadControllerCore(
            ticks_per_deg=self.calib.head.ticks_per_deg,
            min_limit_deg=self.calib.head.min_angle_deg,
            max_limit_deg=self.calib.head.max_angle_deg,
        )

        # -------------------------------------------------------------------------
        # 3. Inter-Module State Variables
        # -------------------------------------------------------------------------
        self.latest_audio_state: Optional[FilteredAudioState] = None
        self.latest_visual_tracks: List[VisualTargetTrack] = []
        self.actual_head_yaw_deg: float = 0.0
        self.actual_head_vel_deg_s: float = 0.0
        self.raw_encoder_deg: float = 0.0
        # Her kerteriz `body_azimuth = actual_head_yaw + kamera_acisi` ile hesaplanir.
        # Encoder hic konusmazsa bu deger 0'da kalir, kafa fiziksel olarak donse bile:
        # 20 derece donup kisiyi tam ortaya alan kafa, kisiyi 0 derecede sanip komutu
        # merkeze geri cekiyor ve orada bekliyor. Sessizce olmasin diye izliyoruz.
        self._head_feedback_seen: bool = False
        self._head_state_received: bool = False
        self.diagnostic_joint_yaw_deg: float = 0.0
        self.diagnostic_joint_vel_deg_s: float = 0.0
        self._warned_no_head_feedback: bool = False
        self.is_robot_speaking: bool = False
        self.is_playback_active: bool = False
        # The person's own head-pose yaw (from _estimate_head_yaw), carried separately
        # from the visual tracker.  Used only as an eye-contact cue for detections
        # that arrive without a per-face yaw_deg of their own.
        self.latest_person_head_yaw_deg: float = 0.0
        # Bug #4 fix: GCC-PHAT PSR confidence delivered via /audio/doa_confidence
        # companion topic.  Updated in _on_doa_confidence, consumed in _on_doa_deg.
        # Default 0.70 = reasonable prior while no confidence measurement has arrived.
        self._latest_doa_confidence: float = 0.70
        self.commands_from_audio: int = 0
        self.commands_from_visual: int = 0

        self.coast_timeout_s: float = 1.0
        self._last_visual_target_yaw: Optional[float] = None
        self._last_visual_seen_time: float = 0.0
        self._last_visual_target_id: Optional[str] = None
        self._was_visually_tracking: bool = False
        self._audio_reacq_active: bool = False
        self._audio_reacq_target_yaw: Optional[float] = None
        self._audio_reacq_start_time: float = 0.0
        self._speech_in_progress: bool = False
        self.audio_reacquisition_count: int = 0
        self.visual_handover_count: int = 0
        self.is_speech_verified: bool = False

        # Forensic telemetry state
        self._cycle_id: int = 0
        self._frame_index: int = 0
        self._new_frame_arrived: bool = False
        self.authoritative_target_yaw: float = 0.0
        self.authoritative_command_source: str = "SAFETY_ZERO"
        self.authoritative_target_source: str = "NONE"
        self.authoritative_target_id: Optional[str] = None
        self.authoritative_frame_id: int = 0
        self.authoritative_cycle_id: int = 0
        self._latest_doa_time: float = 0.0
        self._latest_vad_time: float = 0.0
        self._last_target_yaw_telemetry: float = 0.0
        self._latest_detections_telemetry: List[dict] = []
        self.last_forensic_chain: Optional[dict] = None

        # Parity tracking & ring buffer (last 100 samples)
        self.recent_parity_samples: deque = deque(maxlen=100)
        self._last_parity_report_time: float = 0.0

        # Timestamp telemetry
        self.capture_stamp: float = 0.0
        self.vision_arrival_stamp: float = 0.0
        self.golden_step_stamp: float = 0.0
        self.head_feedback_stamp: float = 0.0

        # Authoritative Golden Decision Reference (GazeRuntimeCore wrapping GazeTracker)
        self.runtime = GazeRuntimeCore(
            calibration=self.calib,
            coast_timeout_s=self.coast_timeout_s,
        )
        self.golden_tracker = self.runtime.tracker
        self.latest_face_detections: List[Detection] = []
        self.latest_frame_size: Tuple[int, int] = (640, 480)
        self.latest_detection_time: float = 0.0
        self.golden_gaze_result: Optional[GazeResult] = None
        self.golden_divergence_deg: float = 0.0
        self._latest_doa_deg: Optional[float] = None

        # -------------------------------------------------------------------------
        # 4. ROS 2 Publishers & Subscriptions
        # -------------------------------------------------------------------------
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Canonical Actuator Command Publisher (/head/command -> HeadCmd)
        if HeadCmd is not None:
            self.pub_head_command = self.create_publisher(HeadCmd, "/head/command", 10)
        else:
            self.pub_head_command = None
        # Compatibility topic (/head/cmd_pos -> Float32)
        self.pub_head_cmd_pos = self.create_publisher(Float32, "/head/cmd_pos", 10)

        # Typed Gaze Status publisher
        if GazeStatus is not None:
            self.pub_gaze_state = self.create_publisher(GazeStatus, "/gaze/state", 10)
        else:
            self.pub_gaze_state = None

        # Diagnostics & Visualization topics
        self.pub_gaze_debug = self.create_publisher(String, "/gaze/debug", 10)
        self.pub_active_target = self.create_publisher(String, "/gaze/active_target", 10)


        # Subscriptions
        if HeadState is not None:
            self.create_subscription(HeadState, "/head/state", self._on_head_state, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, qos_best_effort)
        self.create_subscription(Float32, "/audio/doa", self._on_doa_deg, 10)
        self.create_subscription(Float32, "/audio/doa_deg", self._on_doa_deg, 10)
        self.create_subscription(Int32, "/audio/doa_raw", self._on_doa_raw, 10)
        # Bug #4 fix: GCC-PHAT PSR confidence companion — latched into
        # self._latest_doa_confidence so _on_doa_deg can pass it through.
        self.create_subscription(Float32, "/audio/doa_confidence", self._on_doa_confidence, 10)
        self.create_subscription(Bool, "/audio/vad", self._on_vad, 10)
        self.create_subscription(String, "/vision/detections_json", self._on_vision_json, 10)
        self.create_subscription(String, "/vision/faces", self._on_vision_json, 10)
        self.create_subscription(Float32, "/vision/head_yaw", self._on_vision_head_yaw, 10)
        self.create_subscription(String, "/behavior/gesture", self._on_gesture, 10)
        self.create_subscription(Float32, "/behavior/gaze_intent", self._on_gaze_intent, 10)
        self.create_subscription(Float32, "/head/target_yaw", self._on_gaze_intent, 10)
        self.create_subscription(String, "/behavior/explicit_gaze", self._on_explicit_gaze, 10)
        self.create_subscription(Bool, "/robot/is_speaking", self._on_speaking_status, 10)
        self.create_subscription(Bool, "/audio/playback_active", self._on_playback_active, 10)
        self.create_subscription(String, "/robot/emotion", self._on_emotion, 10)
        self.create_subscription(Bool, "/safety/emergency_stop", self._on_emergency_stop, 10)
        self.create_subscription(Bool, "/system/sleep", self._on_sleep_mode, 10)

        # 50 Hz Control Loop Timer
        rate_hz = float(self.get_parameter("control_rate_hz").value)
        timer_period_s = 1.0 / max(1.0, rate_hz)
        self.timer = self.create_timer(timer_period_s, self._control_cycle)

        self.get_logger().info(f"SocialGazeNode initialized at {rate_hz:.1f} Hz (AttentionArbiter enabled)")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_head_state(self, msg) -> None:
        """Authoritative reader for real encoder position and velocity from HeadState message."""
        t = time.monotonic()
        self.head_feedback_stamp = t
        if hasattr(msg, "position_deg") and not math.isnan(msg.position_deg):
            pos = float(msg.position_deg)
            self.raw_encoder_deg = pos
            self.actual_head_yaw_deg = pos
            self._head_feedback_seen = True
            self._head_state_received = True
            vel_val = float(msg.velocity_deg_s) if hasattr(msg, "velocity_deg_s") and not math.isnan(msg.velocity_deg_s) else 0.0
            self.actual_head_vel_deg_s = vel_val
            self.runtime.update_head_feedback(pos, vel_val, timestamp=t, source="/head/state")

    def _on_joint_states(self, msg: JointState) -> None:
        """Diagnostic reader for head_yaw_joint actual position and velocity.

        /head/state is the sole authoritative source. When /head/state is active,
        /joint_states is strictly diagnostic and will not overwrite authoritative feedback.
        """
        if "head_yaw_joint" in msg.name:
            idx = msg.name.index("head_yaw_joint")
            pos_val = msg.position[idx]
            if not math.isnan(pos_val):
                deg_pos = math.degrees(pos_val)
                vel_val = math.degrees(msg.velocity[idx]) if (len(msg.velocity) > idx and not math.isnan(msg.velocity[idx])) else 0.0
                self.diagnostic_joint_yaw_deg = float(deg_pos)
                self.diagnostic_joint_vel_deg_s = float(vel_val)
                if not getattr(self, "_head_state_received", False):
                    t = time.monotonic()
                    self.head_feedback_stamp = t
                    self.raw_encoder_deg = float(deg_pos)
                    self.actual_head_yaw_deg = float(deg_pos)
                    self._head_feedback_seen = True
                    self.actual_head_vel_deg_s = vel_val
                    self.runtime.update_head_feedback(deg_pos, vel_val, timestamp=t, source="/joint_states")

    def _on_doa_raw(self, msg: Int32) -> None:
        """Processes raw integer DOA from ReSpeaker firmware."""
        t = time.monotonic()
        raw_val = float(msg.data)
        self._latest_doa_deg = raw_val
        self._latest_doa_time = t
        obs = self.audio_perception.process_raw_doa(
            raw_doa_deg=raw_val,
            timestamp=t,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
            is_robot_speaking=self.is_robot_speaking,
        )
        self.latest_audio_state = self.audio_filter.filter_observation(
            obs=obs,
            head_velocity_deg_s=self.actual_head_vel_deg_s,
        )

    def _on_doa_confidence(self, msg: Float32) -> None:
        """Latches the GCC-PHAT PSR confidence from /audio/doa_confidence.

        This companion topic is published by audio_stream_node immediately before
        (or after) /audio/doa so that the confidence value is not lost at the
        Float32 topic boundary (Bug #4).
        """
        raw = float(msg.data)
        # Clamp to [0, 1] — guard against any sensor noise in the PSR ratio
        self._latest_doa_confidence = max(0.0, min(1.0, raw))

    def _on_vad(self, msg: Bool) -> None:
        """Sets verified speech activity flag (/audio/vad)."""
        self.is_speech_verified = bool(msg.data)
        self._latest_vad_time = time.monotonic() if bool(msg.data) else 0.0

    def _on_doa_deg(self, msg: Float32) -> None:
        """Processes float DOA angle in degrees."""
        t = time.monotonic()
        raw_val = float(msg.data)
        self._latest_doa_deg = raw_val
        self._latest_doa_time = t
        obs = self.audio_perception.process_raw_doa(
            raw_doa_deg=raw_val,
            timestamp=t,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
            # Bug #4 fix: use the latched GCC-PHAT confidence instead of 0.85
            confidence=self._latest_doa_confidence,
            is_robot_speaking=self.is_robot_speaking,
        )
        if not obs.valid:
            self.get_logger().info(
                f"[AUDIO REJECT] raw_angle={raw_val:+.1f}° rel_bearing={obs.relative_azimuth_deg:+.1f}° "
                f"confidence={obs.confidence:.2f} reason=OUT_OF_CONVERSATIONAL_FOV target_created=False "
                f"attention_owner={self.fsm.active_priority.value}"
            )
        else:
            self.get_logger().debug(
                f"[AUDIO ACCEPT] raw_angle={raw_val:+.1f}° rel_bearing={obs.relative_azimuth_deg:+.1f}° "
                f"confidence={obs.confidence:.2f} body_yaw={obs.body_azimuth_deg:+.1f}°"
            )

        self.latest_audio_state = self.audio_filter.filter_observation(
            obs=obs,
            head_velocity_deg_s=self.actual_head_vel_deg_s,
        )

    def _on_vision_head_yaw(self, msg: Float32) -> None:
        """Records the observed person's head POSE yaw — not a bearing to them.

        Every astro_vision publisher fills this topic from `_estimate_head_yaw()`,
        which measures how far the eye midpoint sits from the centre of the face
        ROI: it answers "which way is this person facing", never "where are they".
        Feeding it to the tracker as a camera azimuth fabricated a target out of
        thin air — and since the detectors emit a literal 45.0 whenever they fail
        to find eyes, a lost blink threw the head 45 degrees off to the side.

        Bearings come from /vision/faces alone. This value is kept only as the
        eye-contact cue for detections that carry no per-face yaw of their own.
        """
        self.latest_person_head_yaw_deg = float(msg.data)

    def _on_vision_json(self, msg: String) -> None:
        """Processes JSON array of detected faces from OAK-D Lite vision pipeline.

        Implements strict Single Cycle Semantics:
        ONE FRAME -> ONE TRACKER STEP -> ONE GazeResult -> ONE HEAD COMMAND
        """
        t = time.monotonic()
        self.vision_arrival_stamp = t
        self._new_frame_arrived = True
        self._cycle_id += 1
        self._frame_index += 1
        try:
            raw_data = json.loads(msg.data)
            if isinstance(raw_data, dict):
                detections = raw_data.get("faces", [])
                self.capture_stamp = float(raw_data.get("timestamp", raw_data.get("capture_stamp", t)))
            elif isinstance(raw_data, list):
                detections = raw_data
                self.capture_stamp = float(detections[0].get("timestamp", t)) if detections else t
            else:
                detections = []
                self.capture_stamp = t

            # 1. Adapt to Canonical Golden Detection Data Model (No camera_azimuth_deg override!)
            frame_w_val = 640
            frame_h_val = 480
            det_objs: List[Detection] = []
            for d in detections:
                w_val = int(d.get("w", d.get("width", 50)))
                h_val = int(d.get("h", d.get("height", 50)))
                frame_w_val = int(d.get("frame_width", d.get("frame_w", frame_w_val)))
                frame_h_val = int(d.get("frame_height", d.get("frame_h", frame_h_val)))
                conf = float(d.get("confidence", UNSCORED_DETECTION_CONFIDENCE))
                det_src = str(d.get("detector_source", d.get("source", "vision_json")))
                det_objs.append(
                    Detection(
                        x=int(d.get("x", 0)),
                        y=int(d.get("y", 0)),
                        w=w_val,
                        h=h_val,
                        confidence=conf,
                        detector_source=det_src,
                    )
                )

            self.latest_face_detections = det_objs
            self.latest_frame_size = (frame_w_val, frame_h_val)
            self.latest_detection_time = t

            # 2. Step Golden Reference Tracker (Authoritative Core: Single Step per Frame)
            is_speech_fresh = self.is_speech_verified and ((t - self._latest_vad_time) <= 0.5)
            speech = SpeechEstimate(is_speech=True, confidence=self._latest_doa_confidence) if is_speech_fresh else None
            doa_fresh = self._latest_doa_deg is not None and ((t - self._latest_doa_time) <= 0.5)
            doa_val = self._latest_doa_deg if (speech is not None and speech.is_speech and doa_fresh) else None
            measured_head = None if self.head_feedback_missing() else self.actual_head_yaw_deg

            self.golden_gaze_result = self.runtime.step(
                faces=det_objs,
                frame_size=self.latest_frame_size,
                doa_deg=doa_val,
                speech=speech,
                measured_head_deg=measured_head,
                timestamp=self.capture_stamp,
                is_robot_speaking=self.is_robot_speaking,
            )
            self.golden_step_stamp = time.monotonic()
            res = self.golden_gaze_result
            authoritative_target_yaw = float(res.target_yaw_deg)
            self.authoritative_target_yaw = authoritative_target_yaw
            self.authoritative_command_source = res.command_source
            self.authoritative_target_source = res.target_source
            self.authoritative_target_id = res.target_id
            self.authoritative_frame_id = self._frame_index
            self.authoritative_cycle_id = self._cycle_id

            # Invariant 1: Log warning if DETECTION target != COMMAND target during VISUAL tracking
            det_target_id = res.active_target_at_command
            cmd_target_id = res.target_id
            if res.command_source == "VISUAL":
                if det_target_id != "NONE" and cmd_target_id is not None and det_target_id != cmd_target_id:
                    self.get_logger().warn(
                        f"⚠️ [INVARIANT DIVERGENCE] DETECTION target != COMMAND target! "
                        f"det_target={det_target_id} cmd_target={cmd_target_id}"
                    )

            # 3. Actuator Command Publishing (Disabled by default: standalone_gaze_ros_node is sole actuator authority)
            target_goal_deg = float(authoritative_target_yaw)
            if self.enable_actuator_output:
                if self.pub_head_command is not None:
                    hcmd = HeadCmd()
                    hcmd.angle_deg = target_goal_deg
                    self.pub_head_command.publish(hcmd)

                cmd_msg = Float32()
                cmd_msg.data = target_goal_deg
                self.pub_head_cmd_pos.publish(cmd_msg)

            prev_target_yaw = float(self._last_target_yaw_telemetry)
            delta_yaw = abs(angular_diff_deg(prev_target_yaw, authoritative_target_yaw))
            self._last_target_yaw_telemetry = authoritative_target_yaw

            # 4. Critical Forensic Telemetry (FRAME & COMMAND blocks, 8 Required Fields)
            face_bearing = res.face_bearings_deg[0] if res.face_bearings_deg else None
            face_bearing_str = f"{face_bearing:+.1f}°" if face_bearing is not None else "NONE"
            primary_track_id = res.active_track_at_command
            primary_target_id = res.target_id or "NONE"
            vision_age_ms = round(max(0.0, (t - self.capture_stamp) * 1000.0), 1)
            bbox_str = f"[{det_objs[0].x},{det_objs[0].y},{det_objs[0].w},{det_objs[0].h}]" if det_objs else "NONE"
            conf_str = f"{det_objs[0].confidence:.2f}" if det_objs else "0.00"

            frame_log = (
                f"FRAME\n"
                f"cycle_id={self._cycle_id}\n"
                f"frame_id={self._frame_index}\n"
                f"capture_ts={self.capture_stamp:.3f}\n"
                f"arrival_ts={self.vision_arrival_stamp:.3f}\n"
                f"step_ts={self.golden_step_stamp:.3f}\n"
                f"vision_age_ms={vision_age_ms:.1f}\n"
                f"bbox={bbox_str}\n"
                f"confidence={conf_str}\n"
                f"visual_bearing={face_bearing_str}\n"
                f"track_id={primary_track_id}\n"
                f"target_id={primary_target_id}"
            )
            tracker_head = getattr(self.golden_tracker, "head_angle_deg", self.actual_head_yaw_deg)
            raw_enc = getattr(self, "raw_encoder_deg", self.actual_head_yaw_deg)
            fb_deg, fb_age, fb_src = self.runtime.get_feedback_telemetry(now=t)

            sync_line = (
                f"visual_bearing={face_bearing_str} "
                f"command_yaw={authoritative_target_yaw:+.1f}° "
                f"actual_head={self.actual_head_yaw_deg:+.1f}° "
                f"head_feedback_deg={fb_deg:+.1f}° "
                f"head_feedback_age_ms={fb_age:.1f}ms "
                f"head_feedback_source={fb_src}"
            )

            cmd_log = (
                f"COMMAND\n"
                f"cycle_id={self._cycle_id}\n"
                f"frame_id={self._frame_index}\n"
                f"target_id={primary_target_id}\n"
                f"golden_target_yaw={authoritative_target_yaw:+.1f}°\n"
                f"command_yaw={authoritative_target_yaw:+.1f}°\n"
                f"actual_head={self.actual_head_yaw_deg:+.1f}°\n"
                f"raw_encoder={raw_enc:+.1f}°\n"
                f"golden_tracker.head_angle_deg={tracker_head:+.1f}°\n"
                f"head_feedback_deg={fb_deg:+.1f}°\n"
                f"head_feedback_age_ms={fb_age:.1f}ms\n"
                f"head_feedback_source={fb_src}\n"
                f"FEEDBACK_SYNC: {sync_line}\n"
                f"source={res.command_source}"
            )
            forensic_msg = f"\n{frame_log}\n{cmd_log}"
            try:
                print(forensic_msg)
            except UnicodeEncodeError:
                print(forensic_msg.encode("ascii", errors="replace").decode("ascii"))
            self.get_logger().info(forensic_msg)

            # 5. Legacy ROS Pipeline (Shadow Mode for Diagnostic Telemetry Comparison Only)
            obs_list: List[VisualObservation] = []
            for d in detections:
                w_val = int(d.get("w", d.get("width", 50)))
                h_val = int(d.get("h", d.get("height", 50)))
                depth_val = float(d.get("depth_m", d.get("distance_m", 1.5)))
                recog_name = d.get("name", d.get("recognized_name"))
                is_known_val = bool(d.get("is_known", recog_name is not None))
                head_yaw_val = float(d.get("yaw_deg", d.get("head_yaw_deg", self.latest_person_head_yaw_deg)))
                eyes_vis = bool(d.get("eyes_visible", d.get("looking_at_robot", True)))
                cam_az_val = d.get("camera_azimuth_deg", d.get("cam_azimuth_deg"))
                if cam_az_val is not None:
                    cam_az_val = float(cam_az_val)
                frame_w_val = int(d.get("frame_width", d.get("frame_w", 640)))
                frame_h_val = int(d.get("frame_height", d.get("frame_h", 480)))

                obs = self.visual_perception.process_detection(
                    x=int(d.get("x", 0)),
                    y=int(d.get("y", 0)),
                    w=w_val,
                    h=h_val,
                    depth_m=depth_val,
                    timestamp=t,
                    actual_head_yaw_deg=self.actual_head_yaw_deg,
                    frame_width=frame_w_val,
                    frame_height=frame_h_val,
                    confidence=float(d.get("confidence", UNSCORED_DETECTION_CONFIDENCE)),
                    eyes_visible=eyes_vis,
                    head_yaw_deg=head_yaw_val,
                    emotion=str(d.get("emotion", "neutral")),
                    person_name=recog_name,
                    is_known=is_known_val,
                    cam_azimuth_deg=cam_az_val,
                )
                obs_list.append(obs)

            self.latest_visual_tracks = self.visual_tracker.update(
                observations=obs_list,
                timestamp=t,
                actual_head_yaw_deg=self.actual_head_yaw_deg,
            )

            last_assocs = getattr(self.visual_tracker, "last_associations", {})
            tracks_dict = getattr(self.visual_tracker, "tracks", {})
            for face_idx, (d, obs) in enumerate(zip(detections, obs_list)):
                tid = last_assocs.get(face_idx, "NONE") if isinstance(last_assocs, dict) else "NONE"
                t_state = "NONE"
                if tid != "NONE" and tid in tracks_dict:
                    tr_obj = tracks_dict[tid]
                    state_attr = getattr(tr_obj, "state", getattr(tr_obj, "tracking_state", None))
                    t_state = getattr(state_attr, "value", str(state_attr)) if state_attr is not None else "NONE"
                det_src = str(d.get("detector_source", d.get("source", "vision_json")))
                det_telem = {
                    "frame_id": self._frame_index,
                    "timestamp": round(float(t), 3),
                    "bbox": [
                        int(d.get("x", 0)),
                        int(d.get("y", 0)),
                        int(d.get("w", d.get("width", 50))),
                        int(d.get("h", d.get("height", 50))),
                    ],
                    "bearing": round(float(obs.body_azimuth_deg), 1),
                    "confidence": round(float(d.get("confidence", UNSCORED_DETECTION_CONFIDENCE)), 2),
                    "detector_source": det_src,
                    "track_id": str(tid),
                    "track_state": str(t_state),
                }
                self._latest_detections_telemetry.append(det_telem)

            if delta_yaw > 3.0:
                cmd_telem = {
                    "previous_target_yaw": round(prev_target_yaw, 2),
                    "new_target_yaw": round(authoritative_target_yaw, 2),
                    "command_source": res.command_source,
                    "target_source": res.target_source,
                    "reason": res.command_generation_reason,
                    "active_target_at_command": primary_target_id,
                    "active_track_at_command": primary_track_id,
                }
                tm_telem = {
                    "previous_active_target": "NONE",
                    "new_active_target": primary_target_id,
                    "reason": res.command_generation_reason,
                }
                att_telem = {
                    "old_owner": "NONE",
                    "new_owner": res.owner.value if hasattr(res.owner, "value") else str(res.owner),
                    "reason": res.command_generation_reason,
                    "preempted_target": "NONE",
                }
                self._print_causal_chain(
                    delta_yaw=delta_yaw,
                    detections=self._latest_detections_telemetry,
                    tm=tm_telem,
                    att=att_telem,
                    cmd=cmd_telem,
                    tracks=self.latest_visual_tracks,
                )
            self._new_frame_arrived = False
        except Exception as exc:
            self._new_frame_arrived = False
            self.get_logger().error(f"Error parsing vision JSON: {exc}")

    def _on_gesture(self, msg: String) -> None:
        t = time.monotonic()
        ok = self.fsm.trigger_gesture(msg.data, timestamp=t)
        self.golden_tracker.fsm.trigger_gesture(msg.data, timestamp=t)
        if ok:
            self.get_logger().info(f"Triggered gesture: {msg.data}")

    def _on_gaze_intent(self, msg: Float32) -> None:
        t = time.monotonic()
        self.fsm.set_dialogue_target(yaw_deg=msg.data, duration_s=3.0, timestamp=t)
        self.golden_tracker.fsm.set_dialogue_target(yaw_deg=msg.data, duration_s=3.0, timestamp=t)
        self.authoritative_target_yaw = float(msg.data)
        self.authoritative_command_source = "DIRECT_DIALOGUE_INTENT"

    def _on_explicit_gaze(self, msg: String) -> None:
        """Handles explicit user gaze command (e.g. 'Astro bana dön')."""
        t = time.monotonic()
        text = str(msg.data).strip().lower()
        selector = TargetSelectorType.CURRENT_SPEAKER
        target_yaw: Optional[float] = None

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                sel_str = data.get("selector", "CURRENT_SPEAKER")
                selector = TargetSelectorType(sel_str)
                target_yaw = data.get("target_yaw_deg")
        except Exception:
            pass

        intent = ExplicitGazeIntent(
            selector=selector,
            target_yaw_deg=target_yaw,
            confidence=1.0,
            timestamp=t,
            expiry_time=t + 4.0,
            valid=True,
            reason=f"EXPLICIT_COMMAND_{text}",
        )
        self.fsm.set_explicit_gaze_intent(intent)
        self.golden_tracker.fsm.set_explicit_gaze_intent(intent)
        if target_yaw is not None:
            self.authoritative_target_yaw = float(target_yaw)
            self.authoritative_command_source = "EXPLICIT_USER_GAZE"
        self.get_logger().info(f"Explicit gaze intent received: selector={selector.value}, reason={intent.reason}")

    def _on_speaking_status(self, msg: Bool) -> None:
        """Sets TTS/NLG-level speaking flag (/robot/is_speaking)."""
        self.is_robot_speaking = bool(msg.data) or self.is_playback_active

    def _on_playback_active(self, msg: Bool) -> None:
        """Sets audio-player playback flag (/audio/playback_active, 10 Hz).

        This is separate from _on_speaking_status so that a 10 Hz playback
        heartbeat cannot overwrite a is_speaking=True that just arrived from
        the NLG side.  The combined robot-inhibit flag is the logical OR of
        both sources.
        """
        self.is_playback_active = bool(msg.data)
        self.is_robot_speaking = bool(msg.data) or self.is_robot_speaking

    def _on_emergency_stop(self, msg: Bool) -> None:
        self.fsm.set_safety_lock(msg.data)
        self.golden_tracker.fsm.set_safety_lock(msg.data)
        if msg.data:
            self.authoritative_target_yaw = float(self.actual_head_yaw_deg)
            self.authoritative_command_source = "EMERGENCY_STOP"

    def _on_sleep_mode(self, msg: Bool) -> None:
        self.fsm.set_sleep_mode(msg.data)
        self.golden_tracker.fsm.set_sleep_mode(msg.data)
        if msg.data:
            self.authoritative_target_yaw = 0.0
            self.authoritative_command_source = "SAFETY_ZERO"

    def _on_emotion(self, msg: String) -> None:
        emo = str(msg.data).strip().lower()
        if emo in ("sleeping", "sleep", "deep_idle"):
            self.fsm.set_sleep_mode(True)
            self.golden_tracker.fsm.set_sleep_mode(True)
            self.authoritative_target_yaw = 0.0
            self.authoritative_command_source = "SAFETY_ZERO"
        else:
            self.fsm.set_sleep_mode(False)
            self.golden_tracker.fsm.set_sleep_mode(False)

    # =========================================================================
    # 50 Hz Synchronous Control Cycle
    # =========================================================================

    def head_feedback_missing(self) -> bool:
        """Encoderdan hic konum gelmediyse True.

        Bu durumda tum kerterizler kafa 0 derecedeymis gibi hesaplanir ve hedef
        merkeze cokerek takip sessizce olur.
        """
        return not self._head_feedback_seen

    def _control_cycle(self) -> None:
        t = time.monotonic()

        if self.head_feedback_missing() and not self._warned_no_head_feedback:
            self._warned_no_head_feedback = True
            self.get_logger().warning(
                "Kafa geri beslemesi yok: /head/state veya /joint_states'ten hic konum "
                "gelmedi. Butun kerterizler kafa 0 derecedeymis gibi hesaplanacak, yani "
                "kafa donunce hedef merkeze cokup takip duracak. serial_bridge calisiyor "
                "ve MCU head encoder tick gonderiyor mu kontrol edin."
            )

        # Invariant: FRAME SYNCHRONOUS VISUAL LOGIC
        # _control_cycle NEVER makes visual decisions, never calls golden_tracker.step(),
        # and never overwrites golden_tracker state. It only runs legacy shadow fusion/telemetry
        # and streams 50Hz motor keepalive setpoints from self.authoritative_target_yaw.

        # 1. Multimodal Sensor Fusion (Passive Measurements -> Fused Targets)
        fused_targets = self.fusion.fuse(
            audio_state=self.latest_audio_state,
            visual_tracks=self.latest_visual_tracks,
            timestamp=t,
        )

        # 1. Target-Manager telemetry snapshot before update
        prev_target = self.target_manager.active_target
        prev_target_id = prev_target.target_id if prev_target else "NONE"

        # 2. Target Management (Candidate Targets, Track Continuity, Hysteresis)
        target_state = self.target_manager.update(
            fused_targets=fused_targets,
            timestamp=t,
        )

        # 1. Target-Manager telemetry snapshot after update
        new_target = target_state.active_target
        new_target_id = new_target.target_id if new_target else "NONE"

        if self.target_manager.last_target_birth is not None:
            tm_reason = self.target_manager.last_target_birth.get("reason", f"TARGET_BIRTH_{new_target_id}")
        elif prev_target is None and new_target is not None:
            tm_reason = f"TARGET_ACQUIRED_{new_target.target_id}_{new_target.modality.value}"
        elif prev_target is not None and new_target is None:
            tm_reason = f"TARGET_LOST_{prev_target.target_id}"
        elif prev_target is not None and new_target is not None and prev_target.target_id != new_target.target_id:
            tm_reason = f"TURN_TAKING_SWITCH_{prev_target.target_id}_TO_{new_target.target_id}"
        elif prev_target is not None and new_target is not None and prev_target.target_id == new_target.target_id:
            tm_reason = f"TARGET_MAINTAINED_{new_target.target_id}"
        else:
            tm_reason = "NO_ACTIVE_TARGET"

        tm_telemetry = {
            "previous_active_target": prev_target_id,
            "new_active_target": new_target_id,
            "reason": tm_reason,
        }

        # Auto-wake social gaze from sleep mode when a person is detected or speaking
        if target_state.active_target is not None and self.fsm.is_sleeping and not self.fsm.safety_lock:
            self.fsm.set_sleep_mode(False)

        # 2. Attention decision telemetry snapshot before update
        old_owner = self.fsm.active_priority.value

        # 3. Social Gaze FSM & Attention Arbitration
        gaze_cmd = self.fsm.update(
            target_state=target_state,
            actual_head_yaw_deg=self.actual_head_yaw_deg,
            timestamp=t,
            actual_head_vel_deg_s=self.actual_head_vel_deg_s,
        )

        # 2. Attention decision telemetry snapshot after update
        new_owner = self.fsm.active_priority.value
        decision = self.fsm.last_decision
        att_reason = decision.reason if decision else self.fsm.last_transition_reason
        preempted_target = (
            decision.preempted_target_id
            if (decision and decision.is_preemption and decision.preempted_target_id)
            else "NONE"
        )
        att_telemetry = {
            "old_owner": old_owner,
            "new_owner": new_owner,
            "reason": att_reason,
            "preempted_target": preempted_target,
        }

        # Determine visual target grounding and direct camera lock
        active_target = target_state.active_target
        has_visual_target = bool(
            active_target is not None
            and active_target.modality in (Modality.FUSED, Modality.VISION)
        )

        # Active track resolution: only the track corresponding to active_target
        active_track = None
        if active_target is not None and self.latest_visual_tracks:
            for tr in self.latest_visual_tracks:
                tr_id = getattr(tr, "target_id", getattr(tr, "track_id", None))
                if tr_id == active_target.target_id:
                    active_track = tr
                    break

        active_track_seen = bool(
            active_track is not None
            and getattr(active_track, "missed_frames", 0) == 0
            and getattr(active_track, "tracking_state", getattr(active_track, "state", None))
            in (TrackingState.TRACKING, TrackingState.DETECTED)
        )
        has_visual_lock = bool(has_visual_target and active_track_seen)

        # Invalidate old coast cache if target dropped or target switched
        if not has_visual_target:
            self._last_visual_target_yaw = None
            self._last_visual_target_id = None
            self._last_visual_seen_time = 0.0
            self._was_visually_tracking = False
        elif self._last_visual_target_id is not None and self._last_visual_target_id != active_target.target_id:
            self._last_visual_target_yaw = None
            self._last_visual_target_id = None
            self._last_visual_seen_time = 0.0
            self._was_visually_tracking = False

        if has_visual_lock:
            self._last_visual_target_yaw = float(active_target.body_azimuth_deg)
            self._last_visual_seen_time = t
            self._last_visual_target_id = active_target.target_id
            self._was_visually_tracking = True

        # Calculate visual target elapsed time
        time_since_visual = (t - self._last_visual_seen_time) if self._last_visual_seen_time > 0.0 else 999.0
        visual_target_age_ms = max(0.0, time_since_visual * 1000.0) if self._last_visual_seen_time > 0.0 else 99999.0

        # Verified human speech and stabilized audio bearing evaluation
        has_verified_speech = bool(
            self.is_speech_verified
            or (
                self.latest_audio_state is not None
                and self.latest_audio_state.valid
                and self.latest_audio_state.confidence >= 0.50
            )
        )
        audio_evidence = bool(self.latest_audio_state is not None and self.latest_audio_state.valid)
        audio_bearing = float(self.latest_audio_state.azimuth_deg) if audio_evidence else 0.0
        audio_bearing_age_ms = max(0.0, (t - self.latest_audio_state.timestamp) * 1000.0) if audio_evidence else 99999.0
        audio_bearing_valid = bool(
            audio_evidence
            and audio_bearing_age_ms < 600.0
            and not self.latest_audio_state.is_outlier
            and getattr(self.latest_audio_state, "variance", 0.0) <= 1.0
        )

        # Single mechanical motor envelope: strictly [-75.0, +75.0]
        # Angles outside this envelope (e.g. 119°, 153°, 180°, -135°) are rejected as INVALID_AUDIO_REACQUISITION
        # and MUST NOT be clamped to ±75°
        is_in_audio_envelope = bool(-75.0 <= audio_bearing <= 75.0)

        # ---------------------------------------------------------------------
        # 5-STATE ATTENTION OWNERSHIP & REACQUISITION MACHINE
        # ---------------------------------------------------------------------
        coast_active = False
        audio_reacq_active = False

        cmd_reason = "NONE"

        if gaze_cmd.priority_source in (
            PrioritySource.EXPLICIT_USER_GAZE,
            PrioritySource.DIRECT_DIALOGUE_INTENT,
            PrioritySource.GESTURE_INTENT,
            PrioritySource.EMERGENCY_STOP,
        ):
            command_source = gaze_cmd.priority_source.value
            target_source = "CAMERA" if has_visual_lock else "NONE"
            cmd_reason = f"PRIORITY_COMMAND_{command_source}"
        elif has_visual_lock:
            # STATE 1: VISUAL_LOCK & STATE 4: VISUAL_HANDOVER
            is_handover = self._audio_reacq_active
            if self._audio_reacq_active:
                # STATE 4: VISUAL_HANDOVER on first valid visual detection
                self._audio_reacq_active = False
                self._audio_reacq_target_yaw = None
                self.visual_handover_count += 1
            command_source = "VISUAL"
            target_source = "CAMERA"
            self.commands_from_visual += 1
            cmd_reason = f"VISUAL_HANDOVER_TARGET_{active_target.target_id}" if is_handover else f"VISUAL_LOCK_TARGET_{active_target.target_id}"
        elif (
            has_visual_target
            and active_target is not None
            and self._was_visually_tracking
            and self._last_visual_target_id == active_target.target_id
            and self._last_visual_target_yaw is not None
            and time_since_visual <= self.coast_timeout_s
        ):
            # STATE 2: VISUAL_COAST (0.0 - 1.0s: hold last visual bearing, do not snap to 0.0°, block audio reacq)
            coast_active = True
            command_source = "VISUAL_COAST"
            target_source = "COAST"
            gaze_cmd = replace(
                gaze_cmd,
                target_yaw_deg=float(self._last_visual_target_yaw),
                priority_source=PrioritySource.VISUAL_TRACKING,
            )
            cmd_reason = f"COASTING_LAST_VISUAL_{self._last_visual_target_id}_AGE_{visual_target_age_ms:.0f}MS"
        elif (
            time_since_visual > self.coast_timeout_s
            and has_verified_speech
            and audio_bearing_valid
            and not self.is_robot_speaking
            and is_in_audio_envelope
        ):
            # STATE 3: AUDIO_REACQUISITION (strictly within [-75°, +75°], single-episode limited)
            if self._audio_reacq_active:
                # Ongoing episode: target yaw is fixed, do NOT wander with repeated audio samples!
                audio_reacq_active = True
                command_source = "AUDIO_REACQUISITION"
                target_source = "AUDIO_REACQUISITION"
                gaze_cmd = replace(
                    gaze_cmd,
                    target_yaw_deg=float(self._audio_reacq_target_yaw),
                    priority_source=PrioritySource.ACTIVE_SPEAKER,
                    gaze_state=GazeStateEnum.ORIENTING,
                )
                err = abs(angular_diff_deg(self.actual_head_yaw_deg, self._audio_reacq_target_yaw))
                elapsed = t - self._audio_reacq_start_time
                if elapsed >= 2.5 or (err <= 2.5 and abs(self.actual_head_vel_deg_s) <= 3.0 and elapsed >= 0.3):
                    # Arrived at orienting target without visual acquisition
                    pass
                cmd_reason = f"AUDIO_REACQ_ONGOING_ORIENTING_TO_{self._audio_reacq_target_yaw:+.1f}DEG"
            elif not self._speech_in_progress:
                # New verified speech episode starts single-shot reacquisition action
                self._audio_reacq_active = True
                self._audio_reacq_target_yaw = float(audio_bearing)
                self._audio_reacq_start_time = t
                self._speech_in_progress = True
                self.audio_reacquisition_count += 1
                audio_reacq_active = True
                command_source = "AUDIO_REACQUISITION"
                target_source = "AUDIO_REACQUISITION"
                gaze_cmd = replace(
                    gaze_cmd,
                    target_yaw_deg=float(self._audio_reacq_target_yaw),
                    priority_source=PrioritySource.ACTIVE_SPEAKER,
                    gaze_state=GazeStateEnum.ORIENTING,
                )
                cmd_reason = f"AUDIO_REACQ_NEW_ORIENTING_TO_{self._audio_reacq_target_yaw:+.1f}DEG"
            else:
                # Speech episode already generated its one action and is still continuing: stay stationary
                command_source = "SAFETY_ZERO"
                target_source = "NONE"
                gaze_cmd = replace(
                    gaze_cmd,
                    target_yaw_deg=float(self.actual_head_yaw_deg),
                    priority_source=PrioritySource.IDLE,
                )
                cmd_reason = "AUDIO_REACQ_EPISODE_EXHAUSTED_HOLD_STATIONARY"
        else:
            # STATE 5: IDLE / STATIONARY (no visual, no coast, no verified speech, or outside envelope)
            if not has_verified_speech:
                self._speech_in_progress = False
                self._audio_reacq_active = False
                self._audio_reacq_target_yaw = None
            command_source = "SAFETY_ZERO"
            target_source = "NONE"
            gaze_cmd = replace(
                gaze_cmd,
                target_yaw_deg=float(self.actual_head_yaw_deg),
                priority_source=PrioritySource.IDLE,
            )
            cmd_reason = f"STATIONARY_HOLD_HEAD_AT_{self.actual_head_yaw_deg:+.1f}DEG"

        # CRITICAL ACCEPTANCE INVARIANT GUARDS
        if command_source in ("VISUAL_COAST", "VISUAL"):
            if not has_visual_target or target_state.active_target is None or self._last_visual_target_id != active_target.target_id:
                # Under NO_ACTIVE_TARGET / IDLE, VISUAL_COAST or VISUAL commands are strictly forbidden!
                command_source = "SAFETY_ZERO"
                target_source = "NONE"
                gaze_cmd = replace(
                    gaze_cmd,
                    target_yaw_deg=float(self.actual_head_yaw_deg),
                    priority_source=PrioritySource.IDLE,
                )
                if not cmd_reason.startswith("STATIONARY") and not cmd_reason.startswith("IDLE"):
                    cmd_reason = f"IDLE_STATIONARY_HOLD_HEAD_AT_{self.actual_head_yaw_deg:+.1f}DEG"

        if command_source == "VISUAL":
            target_source = "CAMERA"
        elif command_source == "VISUAL_COAST":
            target_source = "COAST"
        elif command_source in ("IDLE", "SAFETY_ZERO"):
            target_source = "NONE"

        # Architectural Invariant: raw audio DOA -> head command MUST BE ZERO
        self.commands_from_audio = 0

        # Head command telemetry computation
        prev_target_yaw = float(self._last_target_yaw_telemetry)
        legacy_target_yaw = float(gaze_cmd.target_yaw_deg)
        legacy_command_source = str(command_source)
        legacy_target_source = str(target_source)
        legacy_active_target = str(target_state.active_target.target_id) if target_state.active_target else "NONE"

        # ---------------------------------------------------------------------
        # AUTHORITATIVE DECISION SELECTION (GOLDEN TRACKER AUTHORITY)
        # ---------------------------------------------------------------------
        golden_target_yaw = float(self.authoritative_target_yaw)
        self.golden_divergence_deg = abs(angular_diff_deg(golden_target_yaw, legacy_target_yaw))

        if gaze_cmd.priority_source in (
            PrioritySource.EXPLICIT_USER_GAZE,
            PrioritySource.DIRECT_DIALOGUE_INTENT,
            PrioritySource.GESTURE_INTENT,
            PrioritySource.EMERGENCY_STOP,
        ):
            authoritative_target_yaw = legacy_target_yaw
            final_command_source = gaze_cmd.priority_source.value
            final_target_source = "CAMERA" if has_visual_lock else "NONE"
            final_reason = f"PRIORITY_COMMAND_{final_command_source}"
        else:
            authoritative_target_yaw = golden_target_yaw
            final_command_source = self.authoritative_command_source
            final_target_source = self.authoritative_target_source
            final_reason = self.golden_gaze_result.command_generation_reason if self.golden_gaze_result else cmd_reason

        new_target_yaw = float(authoritative_target_yaw)
        self._last_target_yaw_telemetry = new_target_yaw

        active_target_at_command = str(self.golden_gaze_result.active_target_at_command if self.golden_gaze_result else legacy_active_target)
        active_track_at_command = str(self.golden_gaze_result.active_track_at_command if self.golden_gaze_result else (active_track.target_id if active_track else "NONE"))
        command_generation_reason = str(final_reason)

        cmd_telemetry = {
            "previous_target_yaw": round(prev_target_yaw, 2),
            "new_target_yaw": round(new_target_yaw, 2),
            "command_source": final_command_source,
            "target_source": final_target_source,
            "reason": final_reason,
            "active_target_at_command": active_target_at_command,
            "active_track_at_command": active_track_at_command,
            "command_generation_reason": command_generation_reason,
            "golden_target_yaw": round(golden_target_yaw, 2),
            "legacy_target_yaw": round(legacy_target_yaw, 2),
            "golden_divergence_deg": round(self.golden_divergence_deg, 2),
        }

        delta_yaw = abs(angular_diff_deg(prev_target_yaw, new_target_yaw))
        if delta_yaw > 3.0 and gaze_cmd.priority_source != PrioritySource.VISUAL_TRACKING:
            self._print_causal_chain(
                delta_yaw=delta_yaw,
                detections=self._latest_detections_telemetry,
                tm=tm_telemetry,
                att=att_telemetry,
                cmd=cmd_telemetry,
                tracks=self.latest_visual_tracks,
            )

        forensic_payload = {
            "detections": list(self._latest_detections_telemetry),
            "target_manager": tm_telemetry,
            "attention": att_telemetry,
            "command": cmd_telemetry,
            "delta_yaw": round(delta_yaw, 2),
        }
        self.last_forensic_chain = forensic_payload

        # Record ring buffer sample (last 100 samples)
        vis_bearing = (self.golden_gaze_result.face_bearings_deg[0]
                       if self.golden_gaze_result and self.golden_gaze_result.face_bearings_deg
                       else None)
        sample = {
            "timestamp": round(t, 3),
            "golden_target": round(golden_target_yaw, 2),
            "actual_head": round(self.actual_head_yaw_deg, 2),
            "visual_bearing": round(vis_bearing, 2) if vis_bearing is not None else None,
            "motor_error": round(abs(authoritative_target_yaw - self.actual_head_yaw_deg), 2),
        }
        self.recent_parity_samples.append(sample)

        # 5-Second Periodic Parity Reporting
        vision_age_ms = round(max(0.0, (t - self.latest_detection_time) * 1000.0), 1) if self.latest_detection_time > 0.0 else 99999.0
        if (t - self._last_parity_report_time) >= 5.0:
            self._last_parity_report_time = t
            self.get_logger().info(
                f"📊 [RUNTIME PARITY 5s] golden_yaw={golden_target_yaw:+.1f}° "
                f"legacy_yaw={legacy_target_yaw:+.1f}° divergence={self.golden_divergence_deg:.2f}° "
                f"actual_head={self.actual_head_yaw_deg:+.1f}° vision_age={vision_age_ms:.1f}ms "
                f"samples_recorded={len(self.recent_parity_samples)}"
            )

        # 4. Kinematic Motion Planning & Trajectory Generation
        gaze_cmd = replace(gaze_cmd, target_yaw_deg=authoritative_target_yaw)
        measured_pos = None if self.head_feedback_missing() else self.actual_head_yaw_deg
        traj_point = self.planner.plan_step(
            gaze_cmd=gaze_cmd,
            actual_pos_deg=measured_pos,
            timestamp=t,
        )

        if self.head_feedback_missing():
            self.actual_head_yaw_deg = float(traj_point.position_deg)
            self.actual_head_vel_deg_s = float(traj_point.velocity_deg_s)

        # 5. Actuator Command Publishing (Disabled by default: standalone_gaze_ros_node is sole actuator authority)
        target_goal_deg = float(authoritative_target_yaw)
        if self.enable_actuator_output:
            if self.pub_head_command is not None:
                hcmd = HeadCmd()
                hcmd.angle_deg = target_goal_deg
                self.pub_head_command.publish(hcmd)

            cmd_msg = Float32()
            cmd_msg.data = target_goal_deg
            self.pub_head_cmd_pos.publish(cmd_msg)

        # 6. Lifecycle Purging on IDLE (Failure 4)
        if self.fsm.state == GazeStateEnum.IDLE and not self.latest_visual_tracks and not (self.latest_audio_state and self.latest_audio_state.valid):
            self.target_manager.reset_lifecycle()

        # 7. TARGET_BIRTH Telemetry Dispatch (Failure 2 & 10)
        if self.target_manager.last_target_birth is not None:
            tb = self.target_manager.last_target_birth
            if tb.get("source") == "AUDIO":
                self.audio_perception.counters.audio_target_births += 1
            tb_msg = String()
            tb_msg.data = json.dumps(tb)
            self.pub_active_target.publish(tb_msg)
            self.get_logger().info(
                f"🎯 [TARGET_BIRTH] id={tb['target_id']} source={tb['source']} "
                f"bearing={tb['bearing']:+.1f}° conf={tb['confidence']:.2f} reason={tb['reason']}"
            )
            self.target_manager.last_target_birth = None

        # 8. Separate Error Metrics Calculation (Failure 7)
        face_bearing_deg = self.latest_visual_tracks[0].body_azimuth_deg if self.latest_visual_tracks else None
        face_to_desired_error_deg = round(float(face_bearing_deg - gaze_cmd.target_yaw_deg), 2) if face_bearing_deg is not None else 0.0
        desired_to_actual_error_deg = round(float(gaze_cmd.target_yaw_deg - self.actual_head_yaw_deg), 2)
        actuator_state_val = "MOVING" if (abs(self.actual_head_vel_deg_s) > 2.0 or abs(desired_to_actual_error_deg) > 2.0) else "SETTLED"

        audio_age_s = round(float(t - self.latest_audio_state.timestamp), 2) if self.latest_audio_state else 999.0
        visual_age_s = round(float(t - self.latest_visual_tracks[0].last_seen_time), 2) if self.latest_visual_tracks else 999.0

        target_identity_correctness = bool(gaze_cmd.active_target_id is not None and target_state.active_target is not None and gaze_cmd.active_target_id == target_state.active_target.target_id)
        attention_owner_correctness = bool(gaze_cmd.priority_source != PrioritySource.IDLE or target_state.active_target is None)

        # 9. Publish Typed GazeStatus Message
        if self.pub_gaze_state is not None:
            status_msg = GazeStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = "base_link"

            state_enum_map = {
                GazeStateEnum.IDLE: 0,
                GazeStateEnum.SEARCHING: 1,
                GazeStateEnum.ACQUIRING: 2,
                GazeStateEnum.ORIENTING: 3,
                GazeStateEnum.TRACKING: 4,
                GazeStateEnum.HOLDING_ATTENTION: 5,
                GazeStateEnum.TARGET_LOST: 6,
                GazeStateEnum.RECOVERING: 7,
            }
            priority_enum_map = {
                PrioritySource.IDLE: 0,
                PrioritySource.VISUAL_TRACKING: 1,
                PrioritySource.ACTIVE_SPEAKER: 2,
                PrioritySource.GESTURE_INTENT: 3,
                PrioritySource.DIRECT_DIALOGUE_INTENT: 4,
                PrioritySource.EXPLICIT_USER_GAZE: 5,
                PrioritySource.EMERGENCY_STOP: 6,
            }
            status_msg.state = state_enum_map.get(gaze_cmd.gaze_state, 0)
            status_msg.priority = priority_enum_map.get(gaze_cmd.priority_source, 0)
            status_msg.desired_yaw_deg = float(gaze_cmd.target_yaw_deg)
            status_msg.planned_yaw_deg = float(traj_point.position_deg)
            status_msg.actual_yaw_deg = float(self.actual_head_yaw_deg)
            status_msg.target_confidence = float(gaze_cmd.confidence)
            status_msg.target_valid = bool(gaze_cmd.confidence > 0.10)
            status_msg.motion_active = bool(actuator_state_val == "MOVING")
            status_msg.at_target = bool(self.fsm.at_target and traj_point.is_settled)
            status_msg.active_target_id = str(gaze_cmd.active_target_id or "")
            self.pub_gaze_state.publish(status_msg)

        # 10. Publish JSON Debug Telemetry
        state_diag = {
            "timestamp": round(t, 3),
            "authoritative_source": "STANDALONE_GOLDEN_TRACKER",
            "golden_target_yaw_deg": round(golden_target_yaw, 2),
            "legacy_target_yaw_deg": round(legacy_target_yaw, 2),
            "golden_divergence_deg": round(self.golden_divergence_deg, 2),
            "timestamps": {
                "capture_stamp": round(self.capture_stamp, 3),
                "vision_arrival_stamp": round(self.vision_arrival_stamp, 3),
                "golden_step_stamp": round(self.golden_step_stamp, 3),
                "head_feedback_stamp": round(self.head_feedback_stamp, 3),
                "vision_age_ms": vision_age_ms,
            },
            "legacy_shadow": {
                "legacy_target_yaw": round(legacy_target_yaw, 2),
                "legacy_command_source": legacy_command_source,
                "legacy_target_source": legacy_target_source,
                "legacy_active_target": legacy_active_target,
            },
            "authoritative_motor_command": {
                "target_yaw_deg": round(authoritative_target_yaw, 2),
                "command_source": final_command_source,
                "target_source": final_target_source,
                "reason": final_reason,
            },
            "gaze_state": gaze_cmd.gaze_state.value,
            "actuator_state": actuator_state_val,
            "attention_owner": gaze_cmd.priority_source.value,
            "attention_priority": gaze_cmd.priority_source.value,
            "attention_reason": self.fsm.last_decision.reason if self.fsm.last_decision else "NONE",
            "active_target_id": gaze_cmd.active_target_id,
            "target_source": final_target_source,
            "visual_target": has_visual_target,
            "audio_evidence": audio_evidence,
            "command_source": final_command_source,
            "commands_from_audio": self.commands_from_audio,
            "commands_from_visual": self.commands_from_visual,
            "target_confidence": round(gaze_cmd.confidence, 2),
            "desired_yaw_deg": round(gaze_cmd.target_yaw_deg, 2),
            "planned_yaw_deg": round(traj_point.position_deg, 2),
            "actual_yaw_deg": round(self.actual_head_yaw_deg, 2),
            "face_to_desired_error_deg": face_to_desired_error_deg,
            "desired_to_actual_error_deg": desired_to_actual_error_deg,
            "target_identity_correctness": target_identity_correctness,
            "attention_owner_correctness": attention_owner_correctness,
            "audio_valid": bool(self.latest_audio_state.valid) if self.latest_audio_state else False,
            "audio_bearing_valid": audio_bearing_valid,
            "audio_bearing": round(audio_bearing, 2),
            "audio_bearing_age_ms": round(audio_bearing_age_ms, 1),
            "visual_target_age_ms": round(visual_target_age_ms, 1),
            "coast_active": coast_active,
            "audio_reacquisition_active": audio_reacq_active,
            "audio_reacquisition_count": self.audio_reacquisition_count,
            "visual_handover_count": self.visual_handover_count,
            "audio_age_s": audio_age_s,
            "visual_valid": bool(self.latest_visual_tracks[0].confidence > 0.4) if self.latest_visual_tracks else False,
            "visual_age_s": visual_age_s,
            "audio_counters": {
                "raw": self.audio_perception.counters.raw_audio_events,
                "accepted": self.audio_perception.counters.accepted_audio_events,
                "rejected": self.audio_perception.counters.rejected_audio_events,
                "invalid_angle": self.audio_perception.counters.invalid_angle_events,
                "stale": self.audio_perception.counters.stale_audio_events,
                "births": self.audio_perception.counters.audio_target_births,
            },
            "at_target": self.fsm.at_target,
            "is_speaking": self.is_robot_speaking,
            "hold_enter_reason": getattr(self.fsm, "hold_enter_reason", "NONE"),
            "hold_exit_reason": getattr(self.fsm, "hold_exit_reason", "NONE"),
            "last_transition_reason": getattr(self.fsm, "last_transition_reason", "NONE"),
            "forensic": self.last_forensic_chain,
        }
        msg_str = String()
        msg_str.data = json.dumps(state_diag)
        self.pub_gaze_debug.publish(msg_str)

    def _print_causal_chain(
        self,
        delta_yaw: float,
        detections: List[dict],
        tm: dict,
        att: dict,
        cmd: dict,
        tracks: Optional[List] = None,
    ) -> str:
        lines = [
            f"[FORENSIC CAUSAL CHAIN] Δyaw={delta_yaw:+.1f}° (>3.0°)",
            "DETECTION → TRACK → TARGET → ATTENTION → COMMAND",
        ]
        # 1. DETECTION
        if detections:
            det_strs = []
            for d in detections:
                det_strs.append(
                    f"frame_id={d['frame_id']} ts={d['timestamp']:.3f} bbox={d['bbox']} "
                    f"bearing={d['bearing']:+.1f}° conf={d['confidence']:.2f} "
                    f"src={d['detector_source']} track_id={d['track_id']} track_state={d['track_state']}"
                )
            lines.append("  DETECTION: " + " | ".join(det_strs))
        else:
            lines.append("  DETECTION: NONE")

        # 2. TRACK
        target_id = tm.get("new_active_target")
        track_info = None
        if tracks:
            for tr in tracks:
                tr_id = getattr(tr, "target_id", getattr(tr, "track_id", None))
                if tr_id == target_id:
                    state_val = getattr(getattr(tr, "tracking_state", getattr(tr, "state", None)), "value", "NONE")
                    bearing_val = getattr(tr, "body_azimuth_deg", 0.0)
                    conf_val = getattr(tr, "confidence", 0.0)
                    track_info = f"track_id={tr_id} state={state_val} bearing={bearing_val:+.1f}° conf={conf_val:.2f}"
                    break
            if not track_info and tracks:
                tr = tracks[0]
                tr_id = getattr(tr, "target_id", getattr(tr, "track_id", "NONE"))
                state_val = getattr(getattr(tr, "tracking_state", getattr(tr, "state", None)), "value", "NONE")
                bearing_val = getattr(tr, "body_azimuth_deg", 0.0)
                conf_val = getattr(tr, "confidence", 0.0)
                track_info = f"track_id={tr_id} state={state_val} bearing={bearing_val:+.1f}° conf={conf_val:.2f}"
        if not track_info:
            track_info = f"track_id={target_id} state=NONE"
        lines.append(f"  TRACK    : {track_info}")

        # 3. TARGET
        lines.append(
            f"  TARGET   : prev={tm['previous_active_target']} -> new={tm['new_active_target']} "
            f"reason={tm['reason']}"
        )

        # 4. ATTENTION
        lines.append(
            f"  ATTENTION: old_owner={att['old_owner']} -> new_owner={att['new_owner']} "
            f"reason={att['reason']} preempted={att['preempted_target']}"
        )

        # 5. COMMAND
        lines.append(
            f"  COMMAND  : prev_yaw={cmd['previous_target_yaw']:+.1f}° -> new_yaw={cmd['new_target_yaw']:+.1f}° "
            f"cmd_src={cmd['command_source']} target_src={cmd['target_source']} reason={cmd['reason']} "
            f"act_target={cmd.get('active_target_at_command', 'NONE')} "
            f"act_track={cmd.get('active_track_at_command', 'NONE')}"
        )
        msg = "\n".join(lines)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        self.get_logger().info(msg)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SocialGazeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

