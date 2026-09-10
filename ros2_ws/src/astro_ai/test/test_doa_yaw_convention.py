#!/usr/bin/env python3
"""The DOA -> body yaw convention must be identical everywhere.

action_manager converts a ReSpeaker bearing and publishes the result to
/head/target_yaw, which head_tracker_node accepts as a TURN_TO_SOUND command and
locks in for 15 seconds at the highest non-safety priority. If the two converters
disagree on sign, an LLM-triggered turn_to_sound sends the head to the mirror image
of where the tracker would look, and holds it there.
"""

import os
import sys
import unittest

test_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(test_dir, "..", "astro_ai")))
sys.path.insert(
    0, os.path.abspath(os.path.join(test_dir, "..", "..", "astro_base", "astro_base"))
)

from action_manager import circular_doa_to_yaw  # noqa: E402
from head_tracker_node import HEAD_TRACKER_DEFAULTS, doa_to_robot_yaw  # noqa: E402


class TestDoaYawConvention(unittest.TestCase):
    def test_action_manager_matches_head_tracker_for_every_quadrant(self):
        for raw_doa in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
            with self.subTest(doa=raw_doa):
                expected = doa_to_robot_yaw(
                    raw_doa,
                    offset_deg=HEAD_TRACKER_DEFAULTS["doa_offset_deg"],
                    invert=HEAD_TRACKER_DEFAULTS["doa_invert"],
                )
                self.assertAlmostEqual(
                    circular_doa_to_yaw(raw_doa),
                    expected,
                    places=3,
                    msg=f"DOA {raw_doa} derece icin iki donusturucu ayrisiyor",
                )


if __name__ == "__main__":
    unittest.main()
