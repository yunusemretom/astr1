#!/usr/bin/env python3
"""Tests for bridging the single-frame misses a Haar cascade makes.

Measured on a 300-frame scene built from an enrolled photo: with the cascade's own
output published raw, a face that is plainly present is reported on only 82% of
frames, leaving 41 separate dropouts. Most are one or two frames long — the cascade
losing a face it finds again immediately — and holding the last detection across
those closes every one of them.

A held detection is not a fresh one, so its confidence decays: the gaze stack may
keep a lock through the gap (hold threshold 0.40) but must not newly acquire a
target from a frame where nothing was actually seen (acquisition threshold 0.75).
"""

import unittest

from astro_vision.detection_quality import CONFIDENCE_CEILING, DetectionHold


FACE = (100, 120, 80, 80, CONFIDENCE_CEILING)


class TestDetectionHold(unittest.TestCase):
    def test_a_fresh_detection_passes_through_untouched(self):
        hold = DetectionHold(hold_frames=2)

        self.assertEqual(hold.update([FACE]), [FACE])

    def test_a_single_frame_miss_is_bridged(self):
        hold = DetectionHold(hold_frames=2)
        hold.update([FACE])

        bridged = hold.update([])

        self.assertEqual(len(bridged), 1)
        self.assertEqual(bridged[0][:4], FACE[:4])

    def test_a_bridged_detection_reports_a_decayed_confidence(self):
        hold = DetectionHold(hold_frames=2, decay=0.75)
        hold.update([FACE])

        first = hold.update([])[0]
        second = hold.update([])[0]

        self.assertLess(first[4], FACE[4])
        self.assertLess(second[4], first[4])

    def test_a_bridged_detection_cannot_acquire_a_new_target(self):
        """Below 0.75 the target manager will hold a lock but not start one."""
        hold = DetectionHold(hold_frames=3, decay=0.75)
        hold.update([FACE])

        self.assertLess(hold.update([])[0][4], 0.75)

    def test_the_hold_expires_instead_of_inventing_a_person(self):
        hold = DetectionHold(hold_frames=2)
        hold.update([FACE])

        hold.update([])
        hold.update([])

        self.assertEqual(hold.update([]), [])

    def test_nothing_is_held_before_anything_was_ever_seen(self):
        hold = DetectionHold(hold_frames=2)

        self.assertEqual(hold.update([]), [])

    def test_a_new_detection_restarts_the_budget(self):
        hold = DetectionHold(hold_frames=1)
        hold.update([FACE])
        hold.update([])          # budget spent
        hold.update([FACE])      # seen again

        self.assertEqual(len(hold.update([])), 1)

    def test_a_zero_budget_disables_bridging(self):
        hold = DetectionHold(hold_frames=0)
        hold.update([FACE])

        self.assertEqual(hold.update([]), [])


if __name__ == "__main__":
    unittest.main()
