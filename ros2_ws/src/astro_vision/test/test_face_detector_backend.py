#!/usr/bin/env python3
"""Tests for which face detector the vision nodes actually run.

Haar cascades hold a face fine while it faces the camera and fall apart the moment
it does not. Measured over 240-frame scenes built from an enrolled photo:

    condition            Haar (1.08, bridged)      YuNet
    still, gentle sway   100%                      100%   (8.4 ms vs 3.9 ms)
    head tilted 22 deg    90.0%, worst gap 462 ms  100%   (4.4 ms)
    turning to profile    85.8%, worst gap 792 ms   90.4%, worst gap 165 ms

Every Haar failure is a pose the person is going to strike constantly while a robot
tracks them, and YuNet costs half as much. The model already ships with the repo —
scripts/install_face_models.sh fetches it for the SFace recogniser — so detection
was running on the weaker of two detectors that were both already installed.
"""

import unittest
from pathlib import Path

import numpy as np

from astro_vision.detection_quality import (
    CONFIDENCE_FLOOR,
    HaarFaceDetector,
    YuNetFaceDetector,
    create_face_detector,
)


class _StubCascade:
    def detectMultiScale(self, image, **kwargs):
        return [(10, 20, 30, 40)]


class TestBackendSelection(unittest.TestCase):
    def test_yunet_is_used_when_its_model_is_installed(self, ):
        detector = create_face_detector(
            model_dir=Path.home() / ".astro" / "models", haar_cascade=_StubCascade()
        )

        self.assertIsInstance(detector, YuNetFaceDetector)

    def test_haar_is_the_fallback_when_the_model_is_missing(self):
        detector = create_face_detector(
            model_dir=Path("/nonexistent/models"), haar_cascade=_StubCascade()
        )

        self.assertIsInstance(detector, HaarFaceDetector)

    def test_the_fallback_still_produces_scored_detections(self):
        detector = create_face_detector(
            model_dir=Path("/nonexistent/models"), haar_cascade=_StubCascade()
        )

        faces = detector.detect(np.zeros((480, 640, 3), np.uint8))

        self.assertEqual(faces, [(10, 20, 30, 40, CONFIDENCE_FLOOR)])


class TestYuNetDetections(unittest.TestCase):
    def setUp(self):
        model = Path.home() / ".astro" / "models" / "yunet.onnx"
        if not model.exists():
            self.skipTest(f"{model} not installed; run scripts/install_face_models.sh")
        self.detector = YuNetFaceDetector(model)

    def test_an_empty_scene_yields_no_faces(self):
        self.assertEqual(self.detector.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_a_real_face_is_found_and_carries_the_models_own_score(self):
        photo = Path("ros2_ws/src/astro_vision/data/known_faces/cevdet_yilmaz/cevdet_yilmaz.jpg")
        if not photo.exists():
            self.skipTest(f"{photo} missing")
        import cv2

        frame = np.full((480, 640, 3), 50, np.uint8)
        face = cv2.imread(str(photo))
        face = cv2.resize(face, (int(190 * face.shape[1] / face.shape[0]), 190))
        frame[145:335, 250:250 + face.shape[1]] = face

        faces = self.detector.detect(frame)

        self.assertEqual(len(faces), 1)
        x, y, w, h, confidence = faces[0]
        self.assertGreater(confidence, 0.8, "YuNet reports its own score, not a mapped one")
        self.assertGreater(w, 40)
        self.assertTrue(240 < x < 460, f"box should sit over the pasted face, got x={x}")

    def test_the_frame_size_may_change_between_calls(self):
        """The node resizes for detection, so the input size is not fixed at setup."""
        self.detector.detect(np.zeros((480, 640, 3), np.uint8))
        self.detector.detect(np.zeros((240, 320, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
