"""Data contracts, type definitions, and enums for the ASTRO Gaze & Tracking system.

All structures use pure Python dataclasses and enums with standard SI units:
  - Angles in degrees (°), internally wrapped to (-180.0, +180.0]
  - Distances in meters (m)
  - Velocities in degrees/second (°/s) or meters/second (m/s)
  - Accelerations in degrees/second² (°/s²)
  - Jerk in degrees/second³ (°/s³)
  - Timestamps in monotonic seconds (float)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class Modality(str, Enum):
    """Sensing modality originating target observations."""
    NONE = "NONE"
    AUDIO = "AUDIO"
    VISION = "VISION"
    FUSED = "FUSED"
    RADAR = "RADAR"


class TrackingState(str, Enum):
    """Lifecycle state of a tracked target."""
    DETECTED = "DETECTED"    # Tentative first observation
    TRACKING = "TRACKING"    # Confirmed active track with consistent updates
    COASTING = "COASTING"    # Temporarily unobserved; predictive state extrapolation
    LOST = "LOST"            # Expired track exceeding coast timeout


class GazeStateEnum(str, Enum):
    """Social Gaze Finite State Machine (FSM) semantic states."""
    IDLE = "IDLE"                        # Resting neutral pose with gentle social breathing
    SEARCHING = "SEARCHING"              # Deliberate, bounded search behavior with reason
    ACQUIRING = "ACQUIRING"              # Candidate target confirmation before saccade
    ORIENTING = "ORIENTING"              # Fast orienting saccade towards target bearing
    TRACKING = "TRACKING"                # Face locked in camera FOV; smooth visual pursuit
    HOLDING_ATTENTION = "HOLDING_ATTENTION"  # Intentionally maintaining social attention on target
    TARGET_LOST = "TARGET_LOST"          # Target absent; coasting last known state before timeout
    RECOVERING = "RECOVERING"            # Smoothly returning to neutral 0° center

    # Backward compatibility aliases
    HOLD = "HOLDING_ATTENTION"
    AUDIO_ACQUIRE = "ACQUIRING"
    VISUAL_ACQUIRE = "ACQUIRING"
    RETURNING = "RECOVERING"


class ActuatorStateEnum(str, Enum):
    """Physical actuator execution state (separate from cognitive social gaze state)."""
    MOVING = "MOVING"      # Joint actively executing trajectory
    SETTLED = "SETTLED"    # Joint settled within deadband and low velocity
    FAULT = "FAULT"        # Actuator stall, mechanical limit, or watchdog fault


# Alias for backward compatibility
ActuatorState = ActuatorStateEnum


class PrioritySource(str, Enum):
    """Priority hierarchy for social attention arbitration."""
    EMERGENCY_STOP = "EMERGENCY_STOP"            # Priority 1: Hardware E-Stop / Safety lock
    EXPLICIT_USER_GAZE = "EXPLICIT_USER_GAZE"    # Priority 2: Direct user command ("Astro bana dön")
    DIRECT_DIALOGUE_INTENT = "DIRECT_DIALOGUE_INTENT"  # Priority 3: AI Cognitive dialogue gaze intent
    GESTURE_INTENT = "GESTURE_INTENT"            # Priority 4: Scripted social gesture sequence
    ACTIVE_SPEAKER = "ACTIVE_SPEAKER"            # Priority 5: Multimodal active speaker tracking
    VISUAL_TRACKING = "VISUAL_TRACKING"          # Priority 6: Visual human face tracking (Visual Primacy)
    IDLE = "IDLE"                                # Priority 7: Neutral ambient baseline

    # Backward compatibility aliases
    SAFETY = "EMERGENCY_STOP"
    DIALOGUE = "DIRECT_DIALOGUE_INTENT"
    GESTURE = "GESTURE_INTENT"
    VISUAL_PERSON = "VISUAL_TRACKING"


class TargetSelectorType(str, Enum):
    """Selector method for explicit gaze intents."""
    CURRENT_SPEAKER = "CURRENT_SPEAKER"        # Direct attention to currently/recently speaking user
    TARGET_ID = "TARGET_ID"                    # Direct attention to specific tracked target ID
    ABSOLUTE_YAW = "ABSOLUTE_YAW"              # Direct attention to specific angle in robot frame
    RELATIVE_DIRECTION = "RELATIVE_DIRECTION"  # Direct attention relative to head (LEFT, RIGHT, CENTER)


# =============================================================================
# 1. MEASUREMENTS (Passive Observations - Never Directly Move Motors)
# =============================================================================

@dataclass
class AudioMeasurement:
    """Raw acoustic directional measurement from microphone array perception."""
    timestamp: float
    valid: bool
    vad: bool
    raw_azimuth_deg: float = 0.0          # Reported by array [0..359] clockwise
    relative_azimuth_deg: float = 0.0     # Head-relative bearing [-180..+180] REP-103
    body_azimuth_deg: float = 0.0         # Robot base coordinate bearing [-180..+180]
    elevation_deg: float = 0.0
    confidence: float = 0.0               # [0.0..1.0] from PSR and energy
    rms: float = 0.0                      # Frame acoustic energy RMS
    peak: float = 0.0                     # Peak amplitude
    snr_db: float = 0.0                   # Signal-to-noise ratio estimate
    frame_id: str = "mic_link"


# Backward compatibility alias
AudioObservation = AudioMeasurement


@dataclass
class FilteredAudioState:
    """Filtered acoustic state estimate with estimated angular velocity."""
    timestamp: float
    valid: bool
    azimuth_deg: float = 0.0              # Filtered body azimuth [-180..+180]
    angular_velocity_deg_s: float = 0.0   # Estimated angular rate (dθ/dt)
    confidence: float = 0.0               # Motion-compensated confidence [0.0..1.0]
    variance: float = 1.0                 # Kalman estimation variance
    is_outlier: bool = False              # True if sample was gated as an isolated outlier
    motion_attenuated: bool = False       # True if confidence was attenuated due to head motion


@dataclass
class VisualMeasurement:
    """Single-frame visual detection from camera (OAK-D Lite)."""
    timestamp: float
    valid: bool
    target_id: Optional[str] = None
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) in pixels
    u_norm: float = 0.0                   # Normalized image coordinates [-1.0..+1.0]
    v_norm: float = 0.0
    depth_m: float = 0.0                  # Metric stereo depth
    pos_3d_camera: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x_opt, y_opt, z_opt) in meters
    camera_azimuth_deg: float = 0.0       # Angle relative to optical axis [-36°..+36°]
    camera_elevation_deg: float = 0.0
    body_azimuth_deg: float = 0.0         # Transformed to robot base coordinate frame
    confidence: float = 0.0               # Detection confidence [0.0..1.0]
    eyes_visible: bool = False            # True if facial landmarks / eyes confirmed
    eye_contact: bool = False             # True if looking directly at robot
    head_yaw_deg: float = 0.0             # User's estimated head pose yaw
    emotion: str = "neutral"              # Facial emotion
    person_name: Optional[str] = None     # Recognized name
    is_known: bool = False
    frame_id: str = "oak_rgb_camera_optical_frame"


# Backward compatibility alias
VisualObservation = VisualMeasurement


@dataclass
class VisualTargetTrack:
    """Persistent 3D visual track with temporal Kalman filtering and coasting."""
    target_id: str
    pos_3d: Tuple[float, float, float]    # (x, y, z) in robot base frame
    vel_3d: Tuple[float, float, float]    # (vx, vy, vz) in m/s
    body_azimuth_deg: float
    body_elevation_deg: float
    distance_m: float
    confidence: float
    tracking_state: TrackingState
    last_seen_time: float
    age_frames: int = 1
    missed_frames: int = 0
    emotion: str = "neutral"
    person_name: Optional[str] = None
    is_known: bool = False
    eye_contact: bool = False


# =============================================================================
# 2. TARGET STATE (Candidate & Active Multimodal Tracking State)
# =============================================================================

@dataclass
class FusedTarget:
    """Unified audio-visual target representation."""
    target_id: str
    modality: Modality
    body_azimuth_deg: float
    body_elevation_deg: float
    distance_m: float
    confidence: float
    is_speaking: bool
    eye_contact: bool
    person_name: Optional[str]
    is_known: bool
    timestamp: float
    tracking_state: TrackingState
    audio_confidence: float = 0.0
    visual_confidence: float = 0.0


@dataclass
class TargetState:
    """Complete target management state snapshot."""
    active_target: Optional[FusedTarget]
    candidate_targets: List[FusedTarget] = field(default_factory=list)
    timestamp: float = 0.0


# =============================================================================
# 3. BEHAVIOR INTENTS (Explicit Cognitive Desires)
# =============================================================================

@dataclass
class ExplicitGazeIntent:
    """Explicit user-commanded gaze intent (e.g. 'Astro bana dön')."""
    selector: TargetSelectorType = TargetSelectorType.CURRENT_SPEAKER
    target_id: Optional[str] = None
    target_yaw_deg: Optional[float] = None
    confidence: float = 1.0
    timestamp: float = 0.0
    expiry_time: float = 0.0
    valid: bool = True
    reason: str = "EXPLICIT_USER_COMMAND"


@dataclass
class DialogueGazeIntent:
    """Dialogue gaze intent generated by AI brain conversational system."""
    target_yaw_deg: float
    confidence: float = 0.90
    timestamp: float = 0.0
    expiry_time: float = 0.0
    valid: bool = True
    reason: str = "AI_DIALOGUE_INTERACTION"


@dataclass
class GestureGazeIntent:
    """Social head gesture sequence intent (nod, shake, scan)."""
    gesture_name: str
    target_yaw_deg: float
    confidence: float = 1.0
    timestamp: float = 0.0
    valid: bool = True
    reason: str = "SOCIAL_GESTURE"


@dataclass
class SafetyGazeIntent:
    """Emergency stop or hardware lock safety intent."""
    is_locked: bool = False
    is_sleeping: bool = False
    target_yaw_deg: float = 0.0
    confidence: float = 1.0
    timestamp: float = 0.0
    valid: bool = True
    reason: str = "SAFETY_LOCK"


@dataclass
class AudioEventCounters:
    """Telemetry counters for raw vs valid audio event auditing."""
    raw_audio_events: int = 0
    accepted_audio_events: int = 0
    rejected_audio_events: int = 0
    invalid_angle_events: int = 0
    stale_audio_events: int = 0
    audio_target_births: int = 0


@dataclass
class AttentionDecision:
    """Typed decision output from AttentionArbiterCore."""
    owner: PrioritySource
    target_id: Optional[str]
    target_yaw_deg: float
    confidence: float
    reason: str
    timestamp: float
    is_preemption: bool = False
    preempted_target_id: Optional[str] = None


@dataclass
class GazeTarget:
    """High-level behavioral gaze target produced by Gaze Policy."""
    target_yaw_deg: float
    target_pitch_deg: float = 0.0
    confidence: float = 1.0
    target_id: Optional[str] = None
    owner: PrioritySource = PrioritySource.IDLE
    gaze_state: GazeStateEnum = GazeStateEnum.IDLE
    timestamp: float = 0.0
    reason: str = "DEFAULT"


@dataclass
class GazeCommand:
    """Authoritative output from Gaze Manager to Motion Planner."""
    target_yaw_deg: float
    target_pitch_deg: float = 0.0
    priority_source: PrioritySource = PrioritySource.IDLE
    gaze_state: GazeStateEnum = GazeStateEnum.IDLE
    active_target_id: Optional[str] = None
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class TrajectoryPoint:
    """Interpolated smooth motion setpoint for low-level controller."""
    timestamp: float
    position_deg: float
    velocity_deg_s: float = 0.0
    acceleration_deg_s2: float = 0.0
    jerk_deg_s3: float = 0.0
    is_settled: bool = False


@dataclass
class HeadFeedback:
    """Actual closed-loop telemetry feedback from hardware MCU."""
    timestamp: float
    actual_yaw_deg: float
    actual_velocity_deg_s: float = 0.0
    target_yaw_deg: float = 0.0
    encoder_ticks: int = 0
    motor_pwm: int = 0
    is_stalled: bool = False
    is_limited: bool = False
    watchdog_ok: bool = True
    mcu_alive: bool = True
