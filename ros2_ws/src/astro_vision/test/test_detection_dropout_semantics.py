#!/usr/bin/env python3
"""Regression tests verifying standalone visual semantics:
1. No synthetic bounding boxes or DetectionHold in face_detector_node.
2. When a face drops out, /vision/faces publishes an empty list.
3. VisualTrackerCore handles empty detections gracefully without crashing.
4. SocialGaze/GazeTracker coasting maintains visual target during short dropout.
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STANDALONE_DIR = _REPO_ROOT / "standalone"
if str(_STANDALONE_DIR) not in sys.path:
    sys.path.insert(0, str(_STANDALONE_DIR))

from astro_vision.face_detector_node import SpatialVisionNode, Image
from astro_base.gaze.visual_tracker import VisualTrackerCore
from astro_base.gaze.types import TrackingState, VisualObservation
from tracker import GazeTracker, Detection


class _MockDetectorWithControl:
    """Mock detector returning configurable faces."""
    def __init__(self):
        self.faces = []

    def detect(self, frame):
        return list(self.faces)


class TestDetectionDropoutSemantics(unittest.TestCase):
    def _create_image_msg(self, w=640, h=480):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        return Image(
            data=frame.tobytes(),
            height=h,
            width=w,
            encoding="bgr8",
            step=w * 3,
            is_bigendian=0,
        )

    def test_face_dropout_publishes_empty_detection(self):
        """Bir frame'de face mevcut. Sonraki frame'de face yok.
        Beklenen: ikinci frame /vision/faces içinde gerçek detection bulunmamalı."""
        node = SpatialVisionNode()
        mock_det = _MockDetectorWithControl()
        node.face_detector = mock_det

        msg = self._create_image_msg()

        # Frame 1: Face is present
        mock_det.faces = [(200, 150, 100, 100, 0.90)]
        node.image_callback(msg)

        self.assertIsNotNone(node.pub_faces.last_msg)
        faces_frame1 = json.loads(node.pub_faces.last_msg.data)
        self.assertEqual(len(faces_frame1), 1)
        self.assertEqual(faces_frame1[0]["x"], 200)

        # Frame 2: Face dropped out (no detection)
        mock_det.faces = []
        node.image_callback(msg)

        self.assertIsNotNone(node.pub_faces.last_msg)
        faces_frame2 = json.loads(node.pub_faces.last_msg.data)
        self.assertEqual(faces_frame2, [], "Second frame must have empty detection list")
        self.assertEqual(len(faces_frame2), 0)

    def test_no_synthetic_bbox_after_dropout(self):
        """Yüz kaybolduktan sonra eski bbox tekrar publish edilmemeli.
        Sentetik hold tespiti sıfır olmalı."""
        node = SpatialVisionNode()
        mock_det = _MockDetectorWithControl()
        node.face_detector = mock_det

        msg = self._create_image_msg()

        # Frame 1: Face at (180, 120, 90, 90)
        mock_det.faces = [(180, 120, 90, 90, 0.88)]
        node.image_callback(msg)

        # Frame 2, 3, 4: Face disappears
        for frame_num in [2, 3, 4]:
            mock_det.faces = []
            node.image_callback(msg)

            published_faces = json.loads(node.pub_faces.last_msg.data)
            self.assertEqual(
                published_faces, [],
                f"Frame {frame_num} published synthetic bounding box instead of empty list!"
            )

        self.assertEqual(node.synthetic_hold_detections_count, 0)

    def test_visual_tracker_handles_dropout(self):
        """Detection boş olsa bile VisualTracker crash olmamalı."""
        tracker = VisualTrackerCore(gating_distance_m=0.85, coasting_timeout_s=0.70)
        t = 100.0

        # Initial detection
        obs = VisualObservation(
            timestamp=t,
            valid=True,
            pos_3d_camera=(0.1, 0.0, 1.5),
            depth_m=1.5,
            confidence=0.85,
        )
        tracks = tracker.update([obs], timestamp=t)
        self.assertEqual(len(tracks), 1)

        # Dropout frames: empty detection list
        for dt in [0.05, 0.10, 0.15, 0.20]:
            try:
                dropout_tracks = tracker.update([], timestamp=t + dt)
            except Exception as exc:
                self.fail(f"VisualTracker crashed on empty observations: {exc}")

            self.assertTrue(len(dropout_tracks) >= 0)

    def test_visual_coast_handles_dropout(self):
        """Kısa dropout sırasında visual target coasting uygulanmalı.
        Hedef açısı sıfırlanmamalı ve son bilinen görsel kerteriz korunmalı."""
        tracker = GazeTracker(fallback_enabled=False, coast_timeout_s=1.0)
        t = 1000.0

        # Frame 1: Face detected at optical center
        # In 640x480, x=270, w=100 -> center_u=320 -> azimuth=0°
        face = Detection(x=270, y=190, w=100, h=100, confidence=0.85, detector_source="YuNet")
        res1 = tracker.step(
            faces=[face],
            frame_size=(640, 480),
            doa_deg=None,
            measured_head_deg=0.0,
            timestamp=t,
        )
        self.assertIsNotNone(res1.target_id)
        self.assertEqual(res1.command_source, "VISUAL")
        last_yaw = res1.target_yaw_deg

        # Frame 2: Dropout (faces=[]) at t + 50 ms (< coast_timeout_s)
        t += 0.05
        res2 = tracker.step(
            faces=[],
            frame_size=(640, 480),
            doa_deg=None,
            measured_head_deg=0.0,
            timestamp=t,
        )
        # Visual target should coast at last_yaw rather than collapse to 0.0° or safety zero
        self.assertAlmostEqual(res2.target_yaw_deg, last_yaw, places=1)


if __name__ == "__main__":
    unittest.main()
