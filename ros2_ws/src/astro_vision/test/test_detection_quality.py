#!/usr/bin/env python3
"""Tests for mapping Haar cascade stage weights onto a detection confidence.

The gaze stack arbitrates on confidence: below 0.50 an observation is not even
valid, below 0.75 it cannot acquire the head on its own. A cascade that reports
only a bounding box leaves all of that inert, so the stage weight it already
computes has to be carried across.
"""

import unittest

from astro_vision.detection_quality import (
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    HAAR_WEIGHT_CEILING,
    detect_faces_with_confidence,
    haar_level_weight_to_confidence,
)


class _ScoringCascade:
    """Stands in for a modern OpenCV cascade, which reports per-detection weights."""

    def __init__(self, rects, weights):
        self._rects, self._weights = rects, weights

    def detectMultiScale3(self, image, **kwargs):
        return self._rects, tuple(0 for _ in self._rects), self._weights

    def detectMultiScale(self, image, **kwargs):
        raise AssertionError("the scoring path must be preferred when it exists")


class _LegacyCascade:
    """Stands in for a build exposing only the unscored entry point."""

    def __init__(self, rects):
        self._rects = rects

    def detectMultiScale(self, image, **kwargs):
        return self._rects


class TestHaarLevelWeightToConfidence(unittest.TestCase):
    def test_a_weight_at_or_below_the_floor_maps_to_the_confidence_floor(self):
        self.assertEqual(haar_level_weight_to_confidence(0.0), CONFIDENCE_FLOOR)
        self.assertEqual(haar_level_weight_to_confidence(-3.0), CONFIDENCE_FLOOR)

    def test_a_weight_at_or_above_the_ceiling_maps_to_the_confidence_ceiling(self):
        self.assertEqual(haar_level_weight_to_confidence(HAAR_WEIGHT_CEILING), CONFIDENCE_CEILING)
        self.assertEqual(haar_level_weight_to_confidence(40.0), CONFIDENCE_CEILING)

    def test_the_mapping_is_monotonic_between_the_bounds(self):
        weights = [0.5, 1.5, 2.5, 3.5, 4.5]
        confidences = [haar_level_weight_to_confidence(w) for w in weights]
        self.assertEqual(confidences, sorted(confidences))
        self.assertEqual(len(set(confidences)), len(confidences))

    def test_a_marginal_detection_stays_below_the_gaze_stack_validity_gate(self):
        """VisualPerceptionCore drops anything under 0.50 outright."""
        self.assertLess(haar_level_weight_to_confidence(0.4), 0.50)

    def test_a_strong_detection_clears_the_acquisition_threshold(self):
        """TargetManagerCore needs 0.75 before a target may own the head."""
        self.assertGreaterEqual(haar_level_weight_to_confidence(4.0), 0.75)

    def test_a_missing_weight_falls_back_to_the_floor(self):
        """Older OpenCV builds return no weights; silence must not read as quality."""
        self.assertEqual(haar_level_weight_to_confidence(None), CONFIDENCE_FLOOR)


class TestDetectFacesWithConfidence(unittest.TestCase):
    def test_each_detection_carries_the_confidence_of_its_stage_weight(self):
        cascade = _ScoringCascade(
            rects=[(10, 20, 30, 40), (50, 60, 70, 80)],
            weights=(HAAR_WEIGHT_CEILING, 0.0),
        )

        faces = detect_faces_with_confidence(cascade, image=None, scaleFactor=1.12)

        self.assertEqual(
            faces,
            [(10, 20, 30, 40, CONFIDENCE_CEILING), (50, 60, 70, 80, CONFIDENCE_FLOOR)],
        )

    def test_a_cascade_without_stage_weights_falls_back_to_the_floor(self):
        cascade = _LegacyCascade(rects=[(10, 20, 30, 40)])

        faces = detect_faces_with_confidence(cascade, image=None)

        self.assertEqual(faces, [(10, 20, 30, 40, CONFIDENCE_FLOOR)])

    def test_no_detections_yields_no_faces(self):
        self.assertEqual(detect_faces_with_confidence(_ScoringCascade([], ()), image=None), [])
        self.assertEqual(detect_faces_with_confidence(_LegacyCascade([]), image=None), [])


if __name__ == "__main__":
    unittest.main()
