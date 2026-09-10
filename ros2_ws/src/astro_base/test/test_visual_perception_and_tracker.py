"""Unit tests for Visual Perception Core and 3D Spatial Visual Tracker."""

import math
import unittest
import numpy as np

from astro_base.gaze.types import TrackingState, VisualObservation
from astro_base.gaze.visual_perception import VisualPerceptionCore
from astro_base.gaze.visual_tracker import VisualTrackerCore


class TestVisualPerception(unittest.TestCase):
    def setUp(self):
        self.perc = VisualPerceptionCore(
            min_confidence=0.50,
            direct_gaze_max_yaw_deg=22.0,
            social_zone_min_dist_m=0.35,
            social_zone_max_dist_m=3.50,
        )

    def test_center_pixel_projects_to_optical_center(self):
        """Center pixel (320, 240) in 640x480 frame at 1.5m depth projects to (0, 0, 1.5)."""
        x_opt, y_opt, z_opt = self.perc.pixel_and_depth_to_3d_camera(
            u_px=320.0, v_px=240.0, depth_m=1.5, frame_width=640, frame_height=480
        )
        self.assertAlmostEqual(x_opt, 0.0, places=3)
        self.assertAlmostEqual(y_opt, 0.0, places=3)
        self.assertAlmostEqual(z_opt, 1.5, places=3)

    def test_direct_eye_contact_criteria(self):
        """Direct eye contact requires eyes visible, low head yaw, and social distance."""
        # 1. Direct frontal gaze in social zone -> True
        obs1 = self.perc.process_detection(
            x=280, y=200, w=80, h=80, depth_m=1.2, timestamp=1.0,
            confidence=0.85, eyes_visible=True, head_yaw_deg=5.0
        )
        self.assertTrue(obs1.valid)
        self.assertTrue(obs1.eye_contact)

        # 2. Turned head (yaw = 35°) -> False
        obs2 = self.perc.process_detection(
            x=280, y=200, w=80, h=80, depth_m=1.2, timestamp=1.0,
            confidence=0.85, eyes_visible=True, head_yaw_deg=35.0
        )
        self.assertFalse(obs2.eye_contact)

        # 3. Too far away (depth = 4.5m, outside social zone) -> False
        obs3 = self.perc.process_detection(
            x=280, y=200, w=80, h=80, depth_m=4.5, timestamp=1.0,
            confidence=0.85, eyes_visible=True, head_yaw_deg=0.0
        )
        self.assertFalse(obs3.eye_contact)


class TestVisualTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = VisualTrackerCore(
            gating_distance_m=0.85,
            coasting_timeout_s=0.70,
        )

    def test_track_promotion_and_coasting(self):
        """Tests promotion to TRACKING on 2nd hit, COASTING on miss, and re-acquisition."""
        t = 1.0

        # Frame 1: Initial observation -> DETECTED
        obs1 = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(0.0, 0.0, 1.5),
            depth_m=1.5, confidence=0.80
        )
        tracks1 = self.tracker.update([obs1], timestamp=t)
        self.assertEqual(len(tracks1), 1)
        self.assertEqual(tracks1[0].tracking_state, TrackingState.DETECTED)
        tid = tracks1[0].target_id

        # Frame 2: Second observation -> Promoted to TRACKING
        t += 0.05
        obs2 = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(0.02, 0.0, 1.5),
            depth_m=1.5, confidence=0.85
        )
        tracks2 = self.tracker.update([obs2], timestamp=t)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].target_id, tid)
        self.assertEqual(tracks2[0].tracking_state, TrackingState.TRACKING)

        # Frame 3: Missing detection (drop out) -> COASTING
        t += 0.05
        tracks3 = self.tracker.update([], timestamp=t)
        self.assertEqual(len(tracks3), 1)
        self.assertEqual(tracks3[0].target_id, tid)
        self.assertEqual(tracks3[0].tracking_state, TrackingState.COASTING)

        # Frame 4: Target reappears -> Returns to TRACKING under same ID
        t += 0.05
        obs4 = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(0.03, 0.0, 1.5),
            depth_m=1.5, confidence=0.85
        )
        tracks4 = self.tracker.update([obs4], timestamp=t)
        self.assertEqual(len(tracks4), 1)
        self.assertEqual(tracks4[0].target_id, tid)
        self.assertEqual(tracks4[0].tracking_state, TrackingState.TRACKING)

        # Frame 5: Target missing past coast timeout (>0.70s) -> Track is LOST / purged
        t += 0.80
        tracks5 = self.tracker.update([], timestamp=t)
        self.assertEqual(len(tracks5), 0)

    def test_multi_person_tracking(self):
        """Tests simultaneous tracking of multiple distinct people."""
        t = 1.0

        # Person A (Left side: x = -0.6m) and Person B (Right side: x = +0.6m)
        obs_a = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(-0.6, 0.0, 1.5),
            depth_m=1.5, confidence=0.80, person_name="Alice"
        )
        obs_b = VisualObservation(
            timestamp=t, valid=True, pos_3d_camera=(0.6, 0.0, 1.8),
            depth_m=1.8, confidence=0.85, person_name="Bob"
        )

        tracks = self.tracker.update([obs_a, obs_b], timestamp=t)
        self.assertEqual(len(tracks), 2)
        # Verify 2 distinct IDs were assigned
        tids = {tr.target_id for tr in tracks}
        self.assertEqual(len(tids), 2)

    def test_direct_cam_azimuth_bearing_projection(self):
        """Verify direct cam_azimuth_deg overrides pixel normalization and yields correct body yaw."""
        core = VisualPerceptionCore()
        # Head at +17.0°, person in camera view at -14.0° (to the right of optical axis)
        obs = core.process_detection(
            x=320, y=240, w=60, h=60, depth_m=1.5,
            timestamp=1.0, actual_head_yaw_deg=17.0,
            confidence=0.85, cam_azimuth_deg=-14.0,
        )
        self.assertAlmostEqual(obs.camera_azimuth_deg, -14.0, delta=0.5)
        self.assertAlmostEqual(obs.body_azimuth_deg, 3.0, delta=0.5)

    def test_smooth_walking_person_tracking(self):
        """Verify walking person continuously maintains track identity and smooth body azimuth."""
        t = 1.0
        core = VisualPerceptionCore()
        # Person walking from -20° to +20° in body frame across 10 steps
        prev_tid = None
        for step in range(10):
            # Head stationary at 0°
            angle = -20.0 + step * 4.0
            obs = core.process_detection(
                x=320, y=240, w=60, h=60, depth_m=1.5,
                timestamp=t, actual_head_yaw_deg=0.0,
                confidence=0.85, cam_azimuth_deg=angle,
            )
            tracks = self.tracker.update([obs], timestamp=t)
            self.assertEqual(len(tracks), 1)
            if prev_tid is None:
                prev_tid = tracks[0].target_id
            else:
                self.assertEqual(tracks[0].target_id, prev_tid, "Track ID must persist during continuous walk")
            t += 0.05


if __name__ == "__main__":
    unittest.main()
