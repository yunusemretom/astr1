#!/usr/bin/env python3
"""Directional regression tests for `AudioPerceptionCore.process_frame`.

The 180-degree `gcc_phat` sign fix (mirrored head turning away from whoever was
speaking) was applied to two copies of the same algorithm:
`ros2_ws/src/astro_audio/astro_audio/doa_estimator.py` (tested directionally by
`TestTheAngleIsNotMirrored` in `astro_audio/test/test_doa_estimator.py`) and
`ros2_ws/src/astro_base/astro_base/gaze/audio_perception.py` (this file's
subject). Only the first had a directional test; this is the copy
`social_gaze_node` actually runs on the robot, so it needed the same coverage
independently -- a fix applied to one copy and verified only on the other is
not verified at all.
"""

import unittest

import numpy as np

from astro_base.gaze.audio_perception import AudioPerceptionCore


def shifted_channels(right_lead_samples: int = 0, front_lead_samples: int = 0,
                     length: int = 2048, seed: int = 3,
                     amplitude: float = 5000.0) -> np.ndarray:
    """Four channels, integer-sample-shifted, with the physics spelled out.

    Copied from `shifted_channels` in
    `ros2_ws/src/astro_audio/test/test_doa_estimator.py` rather than imported --
    tests in one package must not reach into another package's test tree, and
    the two `process_frame`/`estimate_from_multichannel_pcm` implementations
    take channel arrays in the same (front, right, back, left) convention, so
    the same synthetic-scene generator applies to both.

    One rule, non-negotiable: the microphone nearer a source hears it EARLIER.
    Earlier means sampling from further ALONG the base array (a later index),
    not a shift applied after the fact -- the doa_estimator test's docstring
    explains at length how getting this backwards let two sign errors cancel
    and hide a real bug for years.

        right_lead_samples > 0  -> source on the right (right mic hears first)
        front_lead_samples > 0  -> source in front (front mic hears first)

    Channel order matches `AudioPerceptionCore.process_frame`'s convention:
    0 front, 1 right, 2 back, 3 left.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, length * 3) * amplitude
    origin = length

    def take(lead: int) -> np.ndarray:
        return base[origin + lead: origin + lead + length]

    return np.stack([
        take(+front_lead_samples),   # 0 front
        take(+right_lead_samples),   # 1 right
        take(-front_lead_samples),   # 2 back
        take(-right_lead_samples),   # 3 left
    ])


# Full pair-distance delay (~4.01 samples at 16 kHz for the 0.086 m pair) --
# large enough that the direction is unambiguous, matching the astro_audio
# test's own choice of lead.
FULL_LEAD_SAMPLES = 4


class TestDirectionIsNotMirrored(unittest.TestCase):
    """Right-hears-first must read as a right-side bearing, left the mirror,
    front near zero.

    `process_frame` reports two angles: `raw_azimuth_deg` (0..360, clockwise in
    the ReSpeaker's own frame: 0=front, 90=right, 180=back, 270=left) and
    `relative_azimuth_deg` (REP-103 head frame, where the module's own
    docstring states +90=left, -90=right). Both are computed and populated
    whenever the block clears the energy/VAD gate, independent of whether the
    acoustic-envelope gate (`_bearing_is_admissible`, +-75 degrees instantly,
    wider with persistence) admits the bearing as `valid` -- a +-90 degree
    bearing needs four consistent hits to become `valid`, which is a
    reverberation-vs-real-speaker distinction, not a direction-finding one. So
    these tests check the two azimuth fields directly rather than `valid`,
    per the review note: weakening the direction assertions to chase `valid`
    would test persistence, not whether the bearing is mirrored.

    A fresh core per assertion avoids any persistence-gate state carrying over
    between right/left/front cases.
    """

    def _process(self, **shift_kwargs):
        core = AudioPerceptionCore()
        channels = shifted_channels(**shift_kwargs)
        return core.process_frame(channels, timestamp=1.0)

    def test_right_hears_first_gives_a_right_side_bearing(self):
        obs = self._process(right_lead_samples=FULL_LEAD_SAMPLES)

        self.assertAlmostEqual(
            obs.raw_azimuth_deg, 90.0, delta=1.0,
            msg=f"sag mikrofon once duydu ama ham acimut {obs.raw_azimuth_deg} "
                "-- ReSpeaker cercevesinde sag 90 olmali")
        self.assertLess(
            obs.relative_azimuth_deg, 0.0,
            msg=f"sag mikrofon once duydu ama bas-bagimli kerteriz "
                f"{obs.relative_azimuth_deg} (-90 = sag olmali, docstring'in "
                "kendi sozu)")

    def test_left_hears_first_mirrors_the_right_case(self):
        obs = self._process(right_lead_samples=-FULL_LEAD_SAMPLES)

        self.assertAlmostEqual(
            obs.raw_azimuth_deg, 270.0, delta=1.0,
            msg=f"sol mikrofon once duydu ama ham acimut {obs.raw_azimuth_deg} "
                "-- ReSpeaker cercevesinde sol 270 olmali")
        self.assertGreater(
            obs.relative_azimuth_deg, 0.0,
            msg=f"sol mikrofon once duydu ama bas-bagimli kerteriz "
                f"{obs.relative_azimuth_deg} (+90 = sol olmali, docstring'in "
                "kendi sozu)")

    def test_front_hears_first_gives_a_near_zero_bearing(self):
        obs = self._process(front_lead_samples=FULL_LEAD_SAMPLES)

        self.assertAlmostEqual(
            obs.raw_azimuth_deg, 0.0, delta=1.0,
            msg=f"on mikrofon once duydu ama ham acimut {obs.raw_azimuth_deg}")
        self.assertAlmostEqual(
            obs.relative_azimuth_deg, 0.0, delta=1.0,
            msg=f"on mikrofon once duydu ama bas-bagimli kerteriz "
                f"{obs.relative_azimuth_deg}")

    def test_synthetic_scene_actually_clears_the_energy_and_vad_gates(self):
        """The scene has to be measured, not just assumed to be loud enough --
        a synthetic signal quiet enough to be gated out would make the three
        tests above pass vacuously (early-return defaults happen to be 0.0,
        which is indistinguishable from a genuine front bearing)."""
        obs = self._process(right_lead_samples=FULL_LEAD_SAMPLES)

        core = AudioPerceptionCore()
        self.assertGreaterEqual(
            obs.rms, core.min_rms_energy,
            "sentetik sahne enerji kapisini gecmiyor -- yukaridaki testler "
            "kapidan once erken donen varsayilan degerleri olcuyor olabilir")
        self.assertGreaterEqual(obs.peak, 900.0,
                                "sentetik sahne VAD tepe esigini gecmiyor")


if __name__ == "__main__":
    unittest.main()
