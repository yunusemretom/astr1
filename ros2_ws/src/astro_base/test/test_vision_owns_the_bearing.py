#!/usr/bin/env python3
"""While a face is visible, vision decides where to look — audio only decides who.

Fusion used to average the two bearings, so a wandering DOA dragged the commanded
angle around a face that had not moved. Measured against a face held at +20
degrees, a DOA at +45 pulled the target to +25.1 — and the gaze deadband is 3
degrees, so acoustic noise alone was enough to command head motion, over and over,
while the person stood still. That is the head "struggling" to hold a face.

The project's own validation puts visual bearing error at 0.40 degrees and filtered
DOA at 3.23. Averaging a coarse source into a precise one can only degrade it, so
vision owns the angle whenever there is a face. Audio keeps the job it is actually
good at: saying which of several people is talking.
"""

import unittest

from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.types import (
    FilteredAudioState,
    Modality,
    TrackingState,
    VisualTargetTrack,
)


def _track(azimuth: float, confidence: float = 0.90, target_id: str = "person_1", t: float = 100.0):
    return VisualTargetTrack(
        target_id=target_id, pos_3d=(1.5, 0.0, 0.0), vel_3d=(0.0, 0.0, 0.0),
        body_azimuth_deg=azimuth, body_elevation_deg=0.0, distance_m=1.5,
        confidence=confidence, tracking_state=TrackingState.TRACKING, last_seen_time=t,
    )


def _audio(azimuth: float, confidence: float = 0.70, t: float = 100.0):
    return FilteredAudioState(timestamp=t, valid=True, azimuth_deg=azimuth, confidence=confidence)


class TestBearingComesFromVisionAlone(unittest.TestCase):
    def setUp(self):
        self.fusion = AudioVisualFusionCore()
        self.t = 100.0

    def test_a_speaking_face_is_aimed_at_where_it_was_seen(self):
        out = self.fusion.fuse(_audio(35.0), [_track(20.0)], self.t)

        self.assertEqual(out[0].body_azimuth_deg, 20.0)

    def test_doa_noise_does_not_move_the_target_at_all(self):
        """The bearing has to be identical across the DOA's whole spread, not merely
        close: anything over the 3 degree gaze deadband commands the head to move."""
        bearings = {
            self.fusion.fuse(_audio(float(doa)), [_track(20.0)], self.t)[0].body_azimuth_deg
            for doa in range(5, 41, 5)
        }

        self.assertEqual(bearings, {20.0})

    def test_a_loud_close_talker_still_does_not_pull_the_aim(self):
        out = self.fusion.fuse(_audio(42.0, confidence=1.0), [_track(20.0)], self.t)

        self.assertEqual(out[0].body_azimuth_deg, 20.0)


class TestAudioStillSaysWhoIsTalking(unittest.TestCase):
    def setUp(self):
        self.fusion = AudioVisualFusionCore()
        self.t = 100.0

    def test_a_face_with_matching_sound_is_marked_as_speaking(self):
        out = self.fusion.fuse(_audio(25.0), [_track(20.0)], self.t)

        self.assertEqual(out[0].modality, Modality.FUSED)
        self.assertTrue(out[0].is_speaking)

    def test_a_silent_face_is_not_marked_as_speaking(self):
        out = self.fusion.fuse(None, [_track(20.0)], self.t)

        self.assertEqual(out[0].modality, Modality.VISION)
        self.assertFalse(out[0].is_speaking)

    def test_the_acoustic_confidence_is_still_reported(self):
        """Telemetry and turn-taking both read it; only the angle is vision's."""
        out = self.fusion.fuse(_audio(25.0, confidence=0.62), [_track(20.0)], self.t)

        self.assertAlmostEqual(out[0].audio_confidence, 0.62, places=2)

    def test_the_speaking_face_outranks_a_silent_one(self):
        out = self.fusion.fuse(_audio(25.0), [_track(20.0), _track(-40.0, target_id="person_2")], self.t)

        self.assertEqual(out[0].target_id, "person_1")
        self.assertTrue(out[0].is_speaking)


class TestAudioStillAimsWhenNothingIsVisible(unittest.TestCase):
    """Out of frame, a coarse bearing is the only bearing there is."""

    def test_sound_with_no_face_produces_an_audio_target_at_the_heard_angle(self):
        out = AudioVisualFusionCore().fuse(_audio(55.0), [], 100.0)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].modality, Modality.AUDIO)
        self.assertEqual(out[0].body_azimuth_deg, 55.0)


if __name__ == "__main__":
    unittest.main()
