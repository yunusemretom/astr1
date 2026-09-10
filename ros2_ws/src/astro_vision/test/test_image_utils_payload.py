#!/usr/bin/env python3
"""Tests for how frames are handed to rclpy.

Profiling the live pipeline showed 45 ms per frame inside bgr_to_imgmsg while the
publish itself cost 0.08 ms and the frame's own tobytes() cost 0.04 ms. The time
went into rclpy validating a `bytes` payload element by element in pure Python —
921,600 isinstance checks per frame — which capped the whole vision stack. Handing
it an array.array('B') instead takes the fast path and skips that validation.
"""

import time
import unittest

import numpy as np

from std_msgs.msg import Header

from astro_vision.image_utils import bgr_to_imgmsg, imgmsg_to_bgr


def _Header():
    """The real message type: rclpy validates the field, and that validation is the
    very cost these tests exist to keep out of the per-frame path."""
    header = Header()
    header.frame_id = "camera_link"
    return header


class TestImagePayloadUsesTheFastPath(unittest.TestCase):
    def setUp(self):
        self.frame = (np.arange(480 * 640 * 3, dtype=np.uint64) % 251).astype(np.uint8)
        self.frame = self.frame.reshape(480, 640, 3)

    def test_a_frame_is_converted_in_well_under_a_frame_period(self):
        """rclpy's uint8[] setter returns immediately for an array.array and otherwise
        runs `all(isinstance(v, int) ...)` and `all(0 <= v < 256 ...)` over every
        element. Both land on the same stored array.array, so only the clock tells
        them apart: 0.04 ms via the fast path against 45 ms through the checks.
        """
        bgr_to_imgmsg(self.frame, _Header())  # warm

        start = time.perf_counter()
        for _ in range(5):
            bgr_to_imgmsg(self.frame, _Header())
        per_frame_ms = (time.perf_counter() - start) / 5 * 1000

        self.assertLess(
            per_frame_ms, 5.0,
            f"conversion took {per_frame_ms:.1f} ms; a 30 FPS budget is 33 ms in total",
        )

    def test_the_payload_still_round_trips_to_the_same_image(self):
        msg = bgr_to_imgmsg(self.frame, _Header())
        msg.encoding = "bgr8"

        np.testing.assert_array_equal(imgmsg_to_bgr(msg), self.frame)

    def test_the_header_fields_still_describe_the_frame(self):
        msg = bgr_to_imgmsg(self.frame, _Header())

        self.assertEqual((msg.height, msg.width), (480, 640))
        self.assertEqual(msg.encoding, "bgr8")
        self.assertEqual(msg.step, 640 * 3)
        self.assertEqual(len(msg.data), 480 * 640 * 3)


if __name__ == "__main__":
    unittest.main()
