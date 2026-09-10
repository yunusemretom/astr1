#!/usr/bin/env python3
"""Sound from behind may claim the head, but only if it keeps saying the same thing.

The acoustic envelope rejected anything past 75 degrees outright, so someone
calling from behind the robot could never get a response. That gate was added for a
real reason — reverberation off a rear wall used to slam the neck into its
mechanical limit — so simply widening it would bring that back.

A reflection is scattered and momentary; a person talking is persistent and stays
put. Requiring the far zone to repeat a consistent bearing separates the two.

The far edge is geometry, not preference: the neck reaches 85 degrees and the
camera sees 36 either side of where it points, so 121 degrees is the furthest a
source can be and still be brought into view by turning the head. Past that the
robot would turn, see nothing, lose the target and swing back — which is the
behaviour being fixed, not a new feature. Facing someone at a true 180 degrees
needs the base to rotate, which this stack does not do.
"""

import unittest

from astro_base.gaze.audio_perception import AudioPerceptionCore, PersistentBearingGate


class TestPersistentBearingGate(unittest.TestCase):
    def setUp(self):
        self.gate = PersistentBearingGate(required_hits=4, tolerance_deg=20.0, expiry_s=1.5)

    def test_a_single_report_is_not_enough(self):
        self.assertFalse(self.gate.admits(100.0, timestamp=0.0))

    def test_a_bearing_that_repeats_is_admitted(self):
        for i in range(3):
            self.assertFalse(self.gate.admits(100.0, timestamp=i * 0.1))

        self.assertTrue(self.gate.admits(100.0, timestamp=0.3))

    def test_small_wander_still_counts_as_the_same_source(self):
        """A talker's DOA jitters by a few degrees; that is not a different person."""
        for bearing, t in ((100.0, 0.0), (108.0, 0.1), (95.0, 0.2)):
            self.gate.admits(bearing, timestamp=t)

        self.assertTrue(self.gate.admits(103.0, timestamp=0.3))

    def test_scattered_reflections_never_accumulate(self):
        """Reverb arrives from a different direction each time, which is the tell."""
        admitted = [
            self.gate.admits(bearing, timestamp=i * 0.1)
            for i, bearing in enumerate([100.0, 140.0, 95.0, 160.0, 110.0, 145.0])
        ]

        self.assertEqual(admitted, [False] * 6)

    def test_a_stale_streak_expires_rather_than_carrying_over(self):
        for i in range(3):
            self.gate.admits(100.0, timestamp=i * 0.1)

        self.assertFalse(self.gate.admits(100.0, timestamp=5.0))

    def test_once_admitted_it_keeps_being_admitted_while_it_persists(self):
        for i in range(4):
            self.gate.admits(100.0, timestamp=i * 0.1)

        self.assertTrue(self.gate.admits(101.0, timestamp=0.5))


class TestTheEnvelopeReachesWhatTheNeckCanReach(unittest.TestCase):
    def setUp(self):
        self.perception = AudioPerceptionCore()

    def _observe(self, raw_doa: float, times: int = 1, start: float = 0.0):
        obs = None
        for i in range(times):
            obs = self.perception.process_raw_doa(
                raw_doa_deg=raw_doa, timestamp=start + i * 0.1,
                actual_head_yaw_deg=0.0, confidence=0.85,
            )
        return obs

    def test_the_near_field_is_still_accepted_immediately(self):
        """Nothing about ordinary conversation should now need to repeat itself."""
        self.assertTrue(self._observe(30.0).valid)

    def test_a_lone_report_from_behind_is_refused(self):
        self.assertFalse(self._observe(260.0).valid)

    def test_someone_calling_persistently_from_behind_is_accepted(self):
        self.assertTrue(self._observe(260.0, times=5).valid)

    def test_sound_the_head_could_never_bring_into_view_stays_refused(self):
        """At 180 the neck cannot help; turning would show the robot nothing."""
        self.assertFalse(self._observe(180.0, times=10).valid)

    def test_a_rejected_bearing_is_counted_for_telemetry(self):
        before = self.perception.counters.invalid_angle_events

        self._observe(180.0, times=3)

        self.assertGreater(self.perception.counters.invalid_angle_events, before)


if __name__ == "__main__":
    unittest.main()
