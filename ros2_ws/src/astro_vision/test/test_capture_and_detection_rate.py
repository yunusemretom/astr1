#!/usr/bin/env python3
"""Rate and freshness tests for the webcam -> face detection path.

Face tracking was lagging by roughly a quarter second, and profiling showed the
cost was not detection: a loaded 640x480 frame costs ~13.5 ms on this class of CPU,
which would sustain ~74 Hz. The latency came from two throttles and a permanently
full driver buffer, so these tests pin the rates, the buffer depth and the cadence
plumbing rather than the detector's speed.
"""

import json
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from astro_vision.face_detector_node import SpatialVisionNode
from astro_vision.webcam_publisher_node import WebcamPublisherNode


class _FakeCapture:
    """Records property writes the way cv2.VideoCapture applies them to V4L2."""

    def __init__(self, *_args, **_kwargs):
        self.props = {}
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


class TestWebcamPublisherKeepsFramesFresh(unittest.TestCase):
    """A 30 FPS camera polled at 15 Hz leaves every frame stale in the V4L2 queue.

    OpenCV buffers four frames by default, so the backlog never drains and read()
    returns in 0.26 ms with an image up to ~133 ms old — before detection starts.
    Consuming at the camera's own rate is what drains it; shrinking the queue is not.
    """

    def test_capture_runs_at_the_camera_rate_by_default(self):
        node = WebcamPublisherNode(capture_factory=_FakeCapture)

        self.assertEqual(node.fps, 30.0)

    def test_the_driver_queue_depth_is_left_alone(self):
        """Shrinking it to 1 was measured to cost a third of the frame rate on this
        camera (30 -> 20 FPS): the driver can no longer overlap DMA with userspace.
        Freshness comes from consuming at the camera's rate, not from a short queue.
        """
        node = WebcamPublisherNode(capture_factory=_FakeCapture)

        self.assertNotIn(cv2.CAP_PROP_BUFFERSIZE, node.cap.props)

    def test_the_requested_rate_is_pushed_down_to_the_driver(self):
        node = WebcamPublisherNode(capture_factory=_FakeCapture)

        self.assertEqual(node.cap.props[cv2.CAP_PROP_FPS], 30.0)


class TestFaceDetectorCadence(unittest.TestCase):
    def test_every_frame_is_processed_by_default(self):
        """The hardcoded `% 2` skip halved an already-halved stream down to 7.5 Hz."""
        node = SpatialVisionNode()

        self.assertEqual(node.process_every_n, 1)
        self.assertEqual([n for n in range(1, 7) if node._should_process(n)], [1, 2, 3, 4, 5, 6])

    def test_the_cadence_parameter_is_declared_so_the_config_can_reach_it(self):
        """camera_params.yaml sets process_every_n on a node that never declared it."""
        node = SpatialVisionNode()

        self.assertIn("process_every_n", node._declared_parameters)

    def test_a_configured_cadence_selects_every_nth_frame(self):
        node = SpatialVisionNode()
        node.process_every_n = 3

        self.assertEqual([n for n in range(1, 7) if node._should_process(n)], [3, 6])

    def test_the_declared_detector_tuning_reaches_the_cascade(self):
        """scale_factor / min_neighbors / min_size were read, then never used."""
        node = SpatialVisionNode()
        node.scale_factor, node.min_neighbors, node.min_size = 1.2, 7, 48

        self.assertEqual(
            node._face_detect_kwargs(),
            {"scaleFactor": 1.2, "minNeighbors": 7, "minSize": (48, 48)},
        )


class TestDetectionIsStableFrameToFrame(unittest.TestCase):
    """Intermittent detection came from pose, not from frame rate.

    Measured over 240-frame scenes built from an enrolled photo, the Haar cascade
    held a face perfectly while it faced the camera and collapsed the moment it did
    not: 90.0% at 22 degrees of roll (worst gap 462 ms) and 85.8% turning to profile
    (worst gap 792 ms). YuNet held 100% and 90.4% on the same scenes at half the
    cost, and a three-frame bridge closes what it still drops.
    """

    def test_detection_runs_on_the_pose_robust_model_when_it_is_installed(self):
        """The cascade holds a face only while it faces the camera: 90.0% at 22
        degrees of roll with a 462 ms gap, 85.8% turning to profile with a 792 ms gap.
        YuNet holds 100% and 90.4% on those scenes at roughly half the cost, and its
        model already ships for the SFace recogniser.
        """
        from astro_vision.detection_quality import YuNetFaceDetector

        if not (Path.home() / ".astro" / "models" / "yunet.onnx").exists():
            self.skipTest("yunet.onnx not installed; run scripts/install_face_models.sh")
        node = SpatialVisionNode()

        self.assertIsInstance(node.face_detector, YuNetFaceDetector)

    def test_detection_hold_is_removed_for_standalone_semantics(self):
        """DetectionHold is removed so that dropouts publish empty detections,
        preventing synthetic bounding box re-projection and positive feedback runaway."""
        node = SpatialVisionNode()

        self.assertFalse(hasattr(node, "detection_hold"))
        self.assertEqual(node.synthetic_hold_detections_count, 0)

    def test_the_dropped_fallback_cascade_is_really_gone(self):
        """alt2 recovered 1 frame in 600 across two scenes while costing 13.6 ms per
        run, and with YuNet leading it has nothing left to recover.
        """
        node = SpatialVisionNode()

        self.assertFalse(hasattr(node, "face_alt_cascade"))
        self.assertNotIn("fallback_every_n", node._declared_parameters)


class TestAFrameFlowsThroughTheNode(unittest.TestCase):
    """Guards the plumbing the cadence work rewired: kwargs, skip test, publishing."""

    def _image_msg(self, frame):
        from astro_vision.face_detector_node import Image

        return Image(
            data=frame.tobytes(), height=frame.shape[0], width=frame.shape[1],
            encoding="bgr8", step=frame.shape[1] * 3, is_bigendian=0,
        )

    def test_a_processed_frame_publishes_a_face_list(self):
        node = SpatialVisionNode()
        frame = np.zeros((480, 640, 3), np.uint8)

        node.image_callback(self._image_msg(frame))

        self.assertIsNotNone(node.pub_faces.last_msg)
        self.assertEqual(json.loads(node.pub_faces.last_msg.data), [])

    def test_a_skipped_frame_publishes_nothing(self):
        node = SpatialVisionNode()
        node.process_every_n = 2
        frame = np.zeros((480, 640, 3), np.uint8)

        node.image_callback(self._image_msg(frame))

        self.assertIsNone(node.pub_faces.last_msg, "frame 1 of 2 must be skipped")


class TestConfigMatchesWhatTheNodesDeclare(unittest.TestCase):
    """camera_params.yaml was setting keys no node ever declared.

    ROS silently drops an override for an undeclared parameter, so the file read as
    documentation of behaviour that did not exist — process_every_n: 3 sat next to a
    hardcoded `% 2`, and nothing flagged the contradiction.
    """

    def _config(self) -> dict:
        import yaml

        path = os.path.join(
            os.path.dirname(__file__), "..", "config", "camera_params.yaml"
        )
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_every_face_detector_key_is_declared_by_the_node(self):
        configured = set(self._config()["face_detector_node"]["ros__parameters"])
        declared = set(SpatialVisionNode()._declared_parameters)

        self.assertEqual(configured - declared, set())

    def test_every_webcam_key_is_declared_by_the_node(self):
        configured = set(self._config()["webcam_publisher_node"]["ros__parameters"])
        declared = set(WebcamPublisherNode(capture_factory=_FakeCapture)._declared_parameters)

        self.assertEqual(configured - declared, set())

    def test_the_configured_capture_rate_matches_the_camera(self):
        self.assertEqual(self._config()["webcam_publisher_node"]["ros__parameters"]["fps"], 30.0)

    def test_the_configured_cadence_processes_every_frame(self):
        self.assertEqual(
            self._config()["face_detector_node"]["ros__parameters"]["process_every_n"], 1
        )


if __name__ == "__main__":
    unittest.main()
