#!/usr/bin/env python3
"""Tests for the debug session recorder.

Intermittent tracking could not be reproduced off the robot, so the recorder exists
to bring one run back for offline study. One file has to answer three questions at
once: what the camera saw, what the detector called a face, and what the gaze stack
was actually tracking — because "the head did not turn" splits into a detection
failure, an arbitration failure and an actuator failure, and only the last two are
distinguishable from the commanded-versus-actual angle.
"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from astro_vision.session_recorder import (
    NotEnoughDiskSpace,
    draw_status,
    SessionRecorder,
    default_dds_profile,
    format_status_lines,
    require_free_space,
)


GAZE = {
    "gaze_state": "TRACKING",
    "attention_owner": "VISUAL_TRACKING",
    "active_target_id": "person_1",
    "target_confidence": 0.93,
    "desired_yaw_deg": 9.4,
    "actual_yaw_deg": 7.1,
}


class TestStatusOverlay(unittest.TestCase):
    def test_the_strip_names_the_state_owner_and_target(self):
        first, _ = format_status_lines(GAZE)

        self.assertIn("TRACKING", first)
        self.assertIn("VISUAL_TRACKING", first)
        self.assertIn("person_1", first)
        self.assertIn("0.93", first)

    def test_the_strip_shows_commanded_against_actual_angle(self):
        """The line that separates an arbitration failure from an actuator one."""
        _, second = format_status_lines(GAZE)

        self.assertIn("+9.4", second)
        self.assertIn("+7.1", second)

    def test_a_run_without_the_gaze_node_still_records(self):
        first, second = format_status_lines(None)

        self.assertIn("gaze", first.lower())
        self.assertEqual(second, "")

    def test_an_idle_owner_is_reported_verbatim_rather_than_hidden(self):
        """A box on screen while the owner reads IDLE is the whole diagnosis."""
        first, _ = format_status_lines({**GAZE, "attention_owner": "IDLE", "active_target_id": None})

        self.assertIn("IDLE", first)


class TestOverlayFitsTheFrame(unittest.TestCase):
    """The first recording truncated the strip at the frame edge, losing the
    confidence value — the one number that says whether a visible box was strong
    enough to own the head.
    """

    def _rightmost_lit_column(self, frame, band_top):
        band = frame[band_top:, :]
        lit = np.where(band.max(axis=(0, 2)) > 40)[0]
        return int(lit.max()) if len(lit) else 0

    def test_a_long_status_line_is_not_cut_off_at_the_edge(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        lines = (
            "HOLDING_ATTENTION  owner=VISUAL_TRACKING  hedef=person_1  conf=0.93",
            "istenen +12.4  ->  gercek +11.8",
        )

        out = draw_status(frame, lines)

        self.assertLess(
            self._rightmost_lit_column(out, 480 - 54), 639,
            "the strip must fit inside the frame instead of running off it",
        )

    def test_a_narrow_frame_still_fits_its_text(self):
        out = draw_status(np.zeros((240, 320, 3), np.uint8), ("X" * 80, "Y" * 60))

        self.assertLess(self._rightmost_lit_column(out, 240 - 54), 319)


class TestTransportProfile(unittest.TestCase):
    """/vision/face_image is a 900 KB topic, and the default 208 KB UDP receive
    buffer drops most of it: measured 2.1 Hz against the 30 Hz the detector was
    publishing. camera.launch.py already puts the publisher on shared memory, so a
    recorder that does not match it records a tenth of the run without saying so.
    """

    def test_the_shared_memory_profile_is_found_where_the_package_installs_it(self):
        path = default_dds_profile()

        if path is None:
            self.skipTest("astro_vision not installed; nothing to resolve")
        self.assertTrue(path.endswith("fastdds_shm.xml"))
        self.assertTrue(os.path.exists(path))


class TestDiskGuard(unittest.TestCase):
    def test_a_full_disk_is_refused_before_any_frame_is_written(self):
        with self.assertRaises(NotEnoughDiskSpace):
            require_free_space(tempfile.gettempdir(), min_free_bytes=10 ** 15)

    def test_a_disk_with_room_is_accepted(self):
        require_free_space(tempfile.gettempdir(), min_free_bytes=1024)


class _Img:
    def __init__(self, frame):
        self.data = frame.tobytes()
        self.height, self.width = frame.shape[:2]
        self.encoding = "bgr8"
        self.step = frame.shape[1] * 3
        self.is_bigendian = 0


class TestRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.frame = np.full((480, 640, 3), 60, np.uint8)

    def test_every_received_frame_reaches_the_video(self):
        recorder = SessionRecorder(output_dir=self.tmp.name)
        for _ in range(5):
            recorder._on_image(_Img(self.frame))
        recorder.close()

        cap = cv2.VideoCapture(recorder.video_path)
        frames = 0
        while cap.read()[0]:
            frames += 1
        cap.release()
        self.assertEqual(frames, 5)

    def test_the_status_strip_is_burned_into_the_recorded_frame(self):
        recorder = SessionRecorder(output_dir=self.tmp.name)
        recorder._on_gaze(type("M", (), {"data": __import__("json").dumps(GAZE)})())

        recorder._on_image(_Img(self.frame))
        recorder.close()

        cap = cv2.VideoCapture(recorder.video_path)
        ok, written = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertGreater(
            int(np.abs(written.astype(int) - self.frame.astype(int)).sum()), 0,
            "the recorded frame must carry the overlay, not just the raw image",
        )

    def test_the_recording_lands_in_its_own_timestamped_directory(self):
        recorder = SessionRecorder(output_dir=self.tmp.name)
        recorder._on_image(_Img(self.frame))
        recorder.close()

        self.assertTrue(os.path.exists(recorder.video_path))
        self.assertTrue(recorder.video_path.endswith(".avi"))
        self.assertNotEqual(os.path.dirname(recorder.video_path), self.tmp.name)


if __name__ == "__main__":
    unittest.main()
