#!/usr/bin/env python3
"""The gaze pipeline, driven directly instead of through ROS topics.

Every stage here — perception, fusion, target management, the state machine, the
motion planner — is the object the ROS node uses, imported from astro_base.gaze.
Only the wiring is local: roughly forty lines that hand one stage's output to the
next, in place of a graph of publishers and subscribers.

That matters for more than tidiness. The hard question all through this stack has
been whether a misbehaviour lives in the algorithms or in the transport, and the
transport has been where most of them lived: DDS dropping seven of every ten camera
frames, YAML keys silently ignored, a bytes payload costing 45 ms a frame. Running
the identical brain with none of that answers the question directly. If tracking is
clean here and ragged under ROS, the fault is the plumbing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import core_path  # noqa: F401
from astro_base.gaze.audio_filter import AudioFilterCore  # noqa: E402
from astro_base.gaze.audio_perception import AudioPerceptionCore  # noqa: E402
from astro_base.gaze.coordinate_frames import (  # noqa: E402
    CalibrationConfig,
    CoordinateTransformer,
)
from astro_base.gaze.gaze_state_machine import SocialGazeFSM  # noqa: E402
from astro_base.gaze.motion_planner import MotionPlannerCore  # noqa: E402
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore  # noqa: E402
from astro_base.gaze.spatial_memory import EpistemicSpatialMemory  # noqa: E402
from astro_base.gaze.target_manager import TargetManagerCore  # noqa: E402
from astro_base.gaze.types import GazeStateEnum, PrioritySource  # noqa: E402
from astro_base.gaze.visual_perception import VisualPerceptionCore  # noqa: E402
from astro_base.gaze.visual_tracker import VisualTrackerCore  # noqa: E402


@dataclass
class Detection:
    """One face box, in pixels, as the detector reports it."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    detector_source: Optional[str] = "vision_json"


@dataclass
class GazeResult:
    """What the pipeline decided this cycle."""
    target_yaw_deg: float
    gaze_state: GazeStateEnum
    owner: PrioritySource
    target_id: Optional[str]
    confidence: float
    head_angle_deg: float
    face_bearings_deg: Tuple[float, ...] = ()
    command_source: str = "VISUAL"
    target_source: str = "CAMERA"
    active_target_at_command: str = "NONE"
    active_track_at_command: str = "NONE"
    command_generation_reason: str = "NONE"
    head_feedback_deg: float = 0.0
    head_feedback_age_ms: float = 0.0
    head_feedback_source: str = "NONE"


# A detection whose publisher reports no confidence: over the target manager's 0.40
# hold threshold, under its 0.75 acquisition one, so an unscored frame alone cannot
# seize the head. Mirrors the ROS node.
UNSCORED_CONFIDENCE = 0.65

# The calibration the ROS side reads. This program used the dataclass defaults and
# never opened it, which quietly broke the promise the folder is built on: the same
# brain, the same settings. It matters most for the microphone array's mounting
# offset — `audio.yaw_offset_deg` is where a measured rotation gets written, and it
# could not reach here at all.
DEFAULT_CALIBRATION_PATH = (Path(__file__).resolve().parent.parent
                            / "ros2_ws" / "src" / "astro_base" / "config"
                            / "calibration_params.yaml")


def _load_calibration(path=None) -> CalibrationConfig:
    """Reads the shared calibration, falling back to defaults if it is not there.

    A missing file degrades rather than stops: this program has to stay runnable at
    a desk, and a diagnostic tool that refuses to start is not a diagnostic tool.
    """
    candidate = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    try:
        return CalibrationConfig.load_yaml(str(candidate))
    except (OSError, ValueError):
        return CalibrationConfig()


class GazeTracker:
    """Holds the shared gaze objects and steps them once per frame."""

    def __init__(self, calibration: Optional[CalibrationConfig] = None,
                 calibration_path=None, **kwargs):
        self.calib = calibration or _load_calibration(calibration_path)
        self.transformer = CoordinateTransformer(self.calib)
        self.spatial_memory = EpistemicSpatialMemory()

        self.visual_perception = VisualPerceptionCore(transformer=self.transformer)
        self.visual_tracker = VisualTrackerCore(transformer=self.transformer)
        self.audio_perception = AudioPerceptionCore(transformer=self.transformer)
        self.audio_filter = AudioFilterCore()
        self.fusion = AudioVisualFusionCore(spatial_memory=self.spatial_memory)
        self.target_manager = TargetManagerCore()
        self.fsm = SocialGazeFSM(
            min_limit_deg=self.calib.head.min_angle_deg,
            max_limit_deg=self.calib.head.max_angle_deg,
            spatial_memory=self.spatial_memory,
        )
        self.planner = MotionPlannerCore(
            min_limit_deg=self.calib.head.min_angle_deg,
            max_limit_deg=self.calib.head.max_angle_deg,
        )

        self.head_angle_deg: float = 0.0
        self.head_velocity_deg_s: float = 0.0
        self.head_feedback_missing: bool = True
        self._last_head_time: Optional[float] = None
        self._latest_audio = None
        self._latest_tracks: List = []

    def step(
        self,
        faces: Sequence[Detection],
        frame_size: Tuple[int, int],
        doa_deg: Optional[float],
        measured_head_deg: Optional[float],
        timestamp: float,
        speech=None,
        is_robot_speaking: bool = False,
    ) -> GazeResult:
        """Runs one cycle: perception, fusion, arbitration, motion.

        `speech` is the SpeechDetector's verdict for the window the bearing came
        from. Only a bearing carried by human speech is allowed to steer: a passing
        car and a steady buzz are both loud and both persistent, so neither energy
        nor bearing stability separates them from a talker — but harmonic structure
        and syllable-rate modulation do. Without a verdict the bearing is ignored
        rather than trusted, because trusting it is the measured failure: in a
        130 s run the head swung between the limits for 90 s chasing noise.

        `is_robot_speaking` hoparlör çalarken True olur. Hoparlör mikrofonun
        yanında; robot konuştuğu anda güçlü ve harmonik bir kerteriz üretilir ve
        o kerteriz her zaman hoparlörü gösterir. Konuşma filtresi bunu elemez —
        robotun sesi de konuşmadır.
        """
        if measured_head_deg is not None:
            new_angle = float(measured_head_deg)
            if self._last_head_time is not None and timestamp > self._last_head_time:
                dt = timestamp - self._last_head_time
                self.head_velocity_deg_s = (new_angle - self.head_angle_deg) / dt
            else:
                self.head_velocity_deg_s = 0.0
            self.head_angle_deg = new_angle
            self.head_feedback_missing = False
            self._last_head_time = timestamp
        else:
            self.head_feedback_missing = True

        if doa_deg is not None and speech is not None and speech.is_speech:
            self._ingest_audio(doa_deg, timestamp, float(speech.confidence),
                               is_robot_speaking)

        self._ingest_vision(faces, frame_size, timestamp)

        fused = self.fusion.fuse(self._latest_audio, self._latest_tracks, timestamp)
        target_state = self.target_manager.update(fused, timestamp)

        command = self.fsm.update(
            target_state=target_state,
            actual_head_yaw_deg=self.head_angle_deg,
            timestamp=timestamp,
            actual_head_vel_deg_s=self.head_velocity_deg_s,
        )

        if command.gaze_state == GazeStateEnum.IDLE:
            self.audio_filter.reset()
            self._latest_audio = None

        # With no encoder, assume the head went where it was told rather than that it
        # sits at zero: assuming zero makes a person centred after a turn compute back
        # to zero, which drives the head to centre and parks it. The planner's
        # rate-limited integration is the model; the raw command would settle instantly
        # and lie to the state machine about having arrived.
        trajectory = self.planner.plan_step(
            gaze_cmd=command,
            actual_pos_deg=None if self.head_feedback_missing else self.head_angle_deg,
            timestamp=timestamp,
        )
        if self.head_feedback_missing:
            self.head_angle_deg = float(trajectory.position_deg)
            self.head_velocity_deg_s = float(trajectory.velocity_deg_s)
            self._last_head_time = timestamp

        return GazeResult(
            target_yaw_deg=float(command.target_yaw_deg),
            gaze_state=command.gaze_state,
            owner=command.priority_source,
            target_id=command.active_target_id,
            confidence=float(command.confidence),
            head_angle_deg=self.head_angle_deg,
            face_bearings_deg=tuple(t.body_azimuth_deg for t in self._latest_tracks),
        )

    def _ingest_audio(self, doa_deg: float, timestamp: float, confidence: float,
                      is_robot_speaking: bool = False) -> None:
        """Feeds one bearing in, carrying the speech verdict's confidence.

        `process_raw_doa` defaults to 0.85, which is what the ROS node hands it when
        the ReSpeaker's own DSP reports an angle. Here the number was standing in for
        a measurement nobody made: the estimator's own confidence term was tried and
        found useless — across four acoustic conditions it stayed between 0.40 and
        0.46, so it ranks loudness, not direction quality. The speech score does
        discriminate, so it is what travels.
        """
        observation = self.audio_perception.process_raw_doa(
            raw_doa_deg=float(doa_deg),
            timestamp=timestamp,
            actual_head_yaw_deg=self.head_angle_deg,
            confidence=confidence,
            is_robot_speaking=is_robot_speaking,
        )
        self._latest_audio = self.audio_filter.filter_observation(
            obs=observation, head_velocity_deg_s=self.head_velocity_deg_s
        )

    def _ingest_vision(
        self, faces: Sequence[Detection], frame_size: Tuple[int, int], timestamp: float
    ) -> None:
        width, height = frame_size
        observations = []
        for face in faces:
            observations.append(
                self.visual_perception.process_detection(
                    x=face.x, y=face.y, w=face.w, h=face.h,
                    depth_m=self._estimate_distance(face.w, width),
                    timestamp=timestamp,
                    actual_head_yaw_deg=self.head_angle_deg,
                    frame_width=width, frame_height=height,
                    confidence=face.confidence if face.confidence else UNSCORED_CONFIDENCE,
                )
            )
        self._latest_tracks = self.visual_tracker.update(
            observations=observations,
            timestamp=timestamp,
            actual_head_yaw_deg=self.head_angle_deg,
        )

    @staticmethod
    def _estimate_distance(box_width_px: int, frame_width_px: int) -> float:
        """Rough range from apparent face width — a head is about 16 cm across.

        Only the bearing steers the head; distance gates the social zone, so a
        monocular estimate is enough and keeps this runnable on any webcam.
        """
        if box_width_px <= 0:
            return 1.5
        focal_px = frame_width_px * 0.8
        return float(min(4.0, max(0.3, (0.16 * focal_px) / box_width_px)))
