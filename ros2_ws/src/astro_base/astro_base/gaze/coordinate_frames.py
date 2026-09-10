"""Coordinate Frames, Kinematic Transformations, and Calibration for ASTRO Gaze.

Defines mathematical conversions between:
  - `oak_rgb_camera_optical_frame` (REP-103: +Z forward, +X right, +Y down)
  - `mic_link` (ReSpeaker 4-mic circular array frame)
  - `head_link` / `head_yaw_link` (Robot head pan frame)
  - `base_link` (Robot mobile chassis base frame)
  - `world` / `odom` (Global navigation frame)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import yaml

from astro_base.gaze.angle_math import clamp_deg, wrap_deg


@dataclass
class HeadCalibration:
    """Head mechanical joint calibration parameters."""
    zero_offset_deg: float = 0.0
    # 440 tick / 170 derece: boyun ±85 dönüyor. Kalibrasyon dosyası okunmadığında
    # da bundan daha çekingen olmamalı, yoksa kafa erişebileceği kişide duruyor.
    min_angle_deg: float = -85.0
    max_angle_deg: float = 85.0
    ticks_per_deg: float = 1.5000









@dataclass
class AudioCalibration:
    """Microphone array mounting and calibration parameters."""
    yaw_offset_deg: float = 0.0
    invert: bool = True  # ReSpeaker measures clockwise; REP-103 is CCW (positive=left)


@dataclass
class CameraCalibration:
    """OAK-D Lite camera mounting and optical calibration parameters."""
    yaw_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    hfov_deg: float = 72.0
    vfov_deg: float = 53.0
    focal_length_px: float = 512.0  # Approx for 640x480 resolution


@dataclass
class CalibrationConfig:
    """Unified system calibration configuration."""
    head: HeadCalibration = field(default_factory=HeadCalibration)
    audio: AudioCalibration = field(default_factory=AudioCalibration)
    camera: CameraCalibration = field(default_factory=CameraCalibration)

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationConfig":
        if "ros__parameters" in data:
            data = data["ros__parameters"]
        elif "/**" in data and isinstance(data["/**"], dict) and "ros__parameters" in data["/**"]:
            data = data["/**"]["ros__parameters"]

        cfg = cls()
        if "head" in data:
            h = data["head"]

            cfg.head = HeadCalibration(
                zero_offset_deg=float(h.get("zero_offset_deg", 0.0)),
                min_angle_deg=float(h.get("min_angle_deg", -90.0)),
                max_angle_deg=float(h.get("max_angle_deg", 90.0)),
                ticks_per_deg=float(h.get("ticks_per_deg", 2.5882)),
            )
        if "audio" in data:
            a = data["audio"]
            cfg.audio = AudioCalibration(
                yaw_offset_deg=float(a.get("yaw_offset_deg", 0.0)),
                invert=bool(a.get("invert", True)),
            )
        if "camera" in data:
            c = data["camera"]
            cfg.camera = CameraCalibration(
                yaw_offset_deg=float(c.get("yaw_offset_deg", 0.0)),
                pitch_offset_deg=float(c.get("pitch_offset_deg", 0.0)),
                hfov_deg=float(c.get("hfov_deg", 72.0)),
                vfov_deg=float(c.get("vfov_deg", 53.0)),
                focal_length_px=float(c.get("focal_length_px", 512.0)),
            )
        return cfg

    @classmethod
    def load_yaml(cls, yaml_path: str) -> "CalibrationConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> "CalibrationConfig":
        return cls.load_yaml(yaml_path)


    def to_dict(self) -> dict:
        return {
            "head": {
                "zero_offset_deg": self.head.zero_offset_deg,
                "min_angle_deg": self.head.min_angle_deg,
                "max_angle_deg": self.head.max_angle_deg,
                "ticks_per_deg": self.head.ticks_per_deg,
            },
            "audio": {
                "yaw_offset_deg": self.audio.yaw_offset_deg,
                "invert": self.audio.invert,
            },
            "camera": {
                "yaw_offset_deg": self.camera.yaw_offset_deg,
                "pitch_offset_deg": self.camera.pitch_offset_deg,
                "hfov_deg": self.camera.hfov_deg,
                "vfov_deg": self.camera.vfov_deg,
                "focal_length_px": self.camera.focal_length_px,
            },
        }

    def save_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class CoordinateTransformer:
    """Transforms sensor observations between Camera, Microphone, Head, and Base frames."""

    def __init__(self, calib: Optional[CalibrationConfig] = None):
        self.calib = calib or CalibrationConfig()

    def raw_audio_doa_to_head_bearing(self, raw_doa_deg: float) -> float:
        """Converts raw circular ReSpeaker DOA (0..359° clockwise) to head-relative bearing.

        In REP-103 convention:
          - 0° = Straight ahead
          - +90° = Left
          - -90° = Right
          - ±180° = Directly behind
        """
        raw = (raw_doa_deg + self.calib.audio.yaw_offset_deg) % 360.0
        if raw <= 180.0:
            yaw = raw
        else:
            yaw = raw - 360.0

        if self.calib.audio.invert:
            yaw = -yaw

        return wrap_deg(yaw)

    def audio_head_bearing_to_body_yaw(
        self,
        head_relative_bearing_deg: float,
        actual_head_yaw_deg: float
    ) -> float:
        """Transforms head-relative acoustic bearing into absolute robot body yaw frame."""
        return wrap_deg(actual_head_yaw_deg + head_relative_bearing_deg)

    def raw_audio_to_body_yaw(
        self,
        raw_doa_deg: float,
        actual_head_yaw_deg: float
    ) -> float:
        """Full pipeline: Raw ReSpeaker DOA -> Robot body yaw."""
        head_rel = self.raw_audio_doa_to_head_bearing(raw_doa_deg)
        return self.audio_head_bearing_to_body_yaw(head_rel, actual_head_yaw_deg)

    def camera_pixel_to_optical_angles(
        self,
        u_px: float,
        v_px: float,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float]:
        """Converts 2D pixel coordinates (u, v) into optical angles relative to camera axis.

        Returns:
          (cam_azimuth_deg, cam_elevation_deg)
          - cam_azimuth_deg: positive=Left, negative=Right (in robot yaw sense)
          - cam_elevation_deg: positive=Up, negative=Down
        """
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        norm_u = (u_px - cx) / cx  # [-1.0..+1.0], +1 is right
        norm_v = (v_px - cy) / cy  # [-1.0..+1.0], +1 is down

        # Image right (+norm_u) corresponds to negative robot yaw (turn right)
        half_hfov = self.calib.camera.hfov_deg / 2.0
        half_vfov = self.calib.camera.vfov_deg / 2.0

        azimuth = float(-norm_u * half_hfov + self.calib.camera.yaw_offset_deg)
        elevation = float(-norm_v * half_vfov + self.calib.camera.pitch_offset_deg)

        return azimuth, elevation

    def camera_bearing_to_body_yaw(
        self,
        cam_azimuth_deg: float,
        actual_head_yaw_deg: float
    ) -> float:
        """Transforms camera-relative azimuth into absolute robot body yaw."""
        return wrap_deg(actual_head_yaw_deg + cam_azimuth_deg)

    def camera_point_to_body_frame(
        self,
        pos_3d_cam: Tuple[float, float, float],
        actual_head_yaw_deg: float
    ) -> Tuple[float, float, float]:
        """Transforms 3D optical camera coordinates (x_opt, y_opt, z_opt) to robot base frame (x, y, z).

        Optical frame (REP-103):
          x_opt: Right
          y_opt: Down
          z_opt: Forward

        Head frame:
          x_head = z_opt + 0.06 (camera forward offset)
          y_head = -x_opt (left is +Y)
          z_head = -y_opt + 0.02 (up is +Z)

        Base frame (rotated by actual_head_yaw_deg about Z):
          x_base = x_head * cos(theta) - y_head * sin(theta)
          y_base = x_head * sin(theta) + y_head * cos(theta)
          z_base = z_head + 0.21 (head height)
        """
        x_opt, y_opt, z_opt = pos_3d_cam

        x_head = z_opt + 0.06
        y_head = -x_opt
        z_head = -y_opt + 0.02

        theta_rad = math.radians(actual_head_yaw_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        x_base = x_head * cos_t - y_head * sin_t
        y_base = x_head * sin_t + y_head * cos_t
        z_base = z_head + 0.21

        return float(x_base), float(y_base), float(z_base)
