"""Pure Visual Perception Core for ASTRO Gaze System.

Processes visual face/person detections and stereo depth maps to produce metric 3D
spatial observations and gaze geometry.
"""

import math
from typing import List, Optional, Tuple
import numpy as np

from astro_base.gaze.angle_math import wrap_deg
from astro_base.gaze.coordinate_frames import CoordinateTransformer
from astro_base.gaze.types import VisualObservation


class VisualPerceptionCore:
    """Extracts 3D spatial bearings, direct eye contact, and emotion from vision detections."""

    def __init__(
        self,
        transformer: Optional[CoordinateTransformer] = None,
        min_confidence: float = 0.50,
        direct_gaze_max_yaw_deg: float = 22.0,
        social_zone_min_dist_m: float = 0.35,
        social_zone_max_dist_m: float = 3.50,
    ):
        self.transformer = transformer or CoordinateTransformer()
        self.min_confidence = min_confidence
        self.direct_gaze_max_yaw_deg = direct_gaze_max_yaw_deg
        self.social_zone_min_dist_m = social_zone_min_dist_m
        self.social_zone_max_dist_m = social_zone_max_dist_m

    def pixel_and_depth_to_3d_camera(
        self,
        u_px: float,
        v_px: float,
        depth_m: float,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> Tuple[float, float, float]:
        """Converts pixel coordinate (u, v) and stereo depth into metric 3D optical camera frame coordinates (x_opt, y_opt, z_opt).

        Optical frame (REP-103):
          x_opt: Right (+X)
          y_opt: Down (+Y)
          z_opt: Forward (+Z)
        """
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        focal_px = self.transformer.calib.camera.focal_length_px

        z_opt = max(0.1, depth_m)
        x_opt = (u_px - cx) * z_opt / focal_px
        y_opt = (v_px - cy) * z_opt / focal_px

        return float(x_opt), float(y_opt), float(z_opt)

    def process_detection(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        depth_m: float,
        timestamp: float,
        actual_head_yaw_deg: float = 0.0,
        frame_width: int = 640,
        frame_height: int = 480,
        confidence: float = 0.80,
        eyes_visible: bool = True,
        head_yaw_deg: float = 0.0,
        emotion: str = "neutral",
        person_name: Optional[str] = None,
        is_known: bool = False,
        cam_azimuth_deg: Optional[float] = None,
    ) -> VisualObservation:
        """Processes a single bounding box detection into a rich VisualObservation."""
        center_u = x + (w / 2.0)
        center_v = y + (h / 2.0)

        if cam_azimuth_deg is not None:
            cam_azimuth = float(cam_azimuth_deg)
            half_vfov = self.transformer.calib.camera.vfov_deg / 2.0
            norm_v = (center_v - (frame_height / 2.0)) / (frame_height / 2.0)
            cam_elevation = float(-norm_v * half_vfov + self.transformer.calib.camera.pitch_offset_deg)
            z_opt = max(0.1, depth_m)
            x_opt = -z_opt * math.tan(math.radians(cam_azimuth))
            y_opt = z_opt * math.tan(math.radians(-cam_elevation))
            pos_3d_cam = (float(x_opt), float(y_opt), float(z_opt))
            norm_u = -cam_azimuth / (self.transformer.calib.camera.hfov_deg / 2.0)
        else:
            # 1. 3D Camera coordinates
            pos_3d_cam = self.pixel_and_depth_to_3d_camera(
                center_u, center_v, depth_m, frame_width, frame_height
            )

            # 2. Camera-relative azimuth and elevation angles
            cam_azimuth, cam_elevation = self.transformer.camera_pixel_to_optical_angles(
                center_u, center_v, frame_width, frame_height
            )
            norm_u = (center_u - (frame_width / 2.0)) / (frame_width / 2.0)
            norm_v = (center_v - (frame_height / 2.0)) / (frame_height / 2.0)

        # 3. Transform to robot base body frame
        body_yaw = self.transformer.camera_bearing_to_body_yaw(cam_azimuth, actual_head_yaw_deg)

        # 4. Direct eye contact verification
        in_social_zone = (self.social_zone_min_dist_m <= depth_m <= self.social_zone_max_dist_m)
        direct_eye_contact = (
            eyes_visible
            and (abs(head_yaw_deg) <= self.direct_gaze_max_yaw_deg)
            and in_social_zone
        )

        is_valid = (confidence >= self.min_confidence) and (depth_m > 0.1)

        norm_u = (center_u - (frame_width / 2.0)) / (frame_width / 2.0)
        norm_v = (center_v - (frame_height / 2.0)) / (frame_height / 2.0)

        return VisualObservation(
            timestamp=timestamp,
            valid=is_valid,
            bbox=(int(x), int(y), int(w), int(h)),
            u_norm=round(float(norm_u), 3),
            v_norm=round(float(norm_v), 3),
            depth_m=round(float(depth_m), 3),
            pos_3d_camera=pos_3d_cam,
            camera_azimuth_deg=round(float(cam_azimuth), 1),
            camera_elevation_deg=round(float(cam_elevation), 1),
            body_azimuth_deg=round(float(body_yaw), 1),
            confidence=round(float(confidence), 2),
            eyes_visible=eyes_visible,
            eye_contact=direct_eye_contact,
            head_yaw_deg=round(float(head_yaw_deg), 1),
            emotion=emotion,
            person_name=person_name,
            is_known=is_known,
        )
