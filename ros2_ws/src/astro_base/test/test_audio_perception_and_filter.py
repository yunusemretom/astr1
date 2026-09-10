"""Unit tests for Audio Perception, GCC-PHAT, Outlier Gating, and Kalman Filtering."""

import math
import unittest
import numpy as np

from astro_base.gaze.angle_math import circular_distance_deg
from astro_base.gaze.audio_filter import (
    AudioFilterCore,
    CircularKalmanEstimator,
    CircularMedianFilter,
)
from astro_base.gaze.audio_perception import (
    AudioPerceptionCore,
    ReSpeaker4MicGeometry,
    gcc_phat,
)
from astro_base.gaze.motion_compensator import HeadMotionCompensator
from astro_base.gaze.types import AudioObservation


class TestAudioPerception(unittest.TestCase):
    def test_gcc_phat_zero_delay(self):
        fs = 16000
        n_samples = 960
        t = np.linspace(0, n_samples / fs, n_samples)
        # Synthetic 1kHz tone
        tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)

        tau, quality = gcc_phat(tone, tone, fs=fs, max_tau=0.001)
        self.assertAlmostEqual(tau, 0.0, places=4)
        self.assertGreater(quality, 0.5)

    def test_gcc_phat_known_shift(self):
        fs = 16000
        n_samples = 960
        # White noise with 2 sample shift
        np.random.seed(42)
        noise = np.random.randn(n_samples).astype(np.float32)
        shifted = np.roll(noise, 2)

        tau, quality = gcc_phat(noise, shifted, fs=fs, max_tau=0.001)
        expected_tau = 2.0 / fs  # 0.000125 s
        self.assertAlmostEqual(abs(tau), expected_tau, delta=1e-4)

    def test_audio_perception_core_speech_suppression(self):
        core = AudioPerceptionCore(self_speech_suppression_factor=0.10)
        np.random.seed(42)
        # Correlated loud speech source signal across 4 channels (frontal azimuth 0.0°)
        t = np.linspace(0, 960 / 16000, 960)
        source = (np.sin(2 * np.pi * 500 * t) * 3000.0).astype(np.int16)
        pcm = np.array([source, source, source, source], dtype=np.int16)

        # 1. Normal speech
        obs_normal = core.process_frame(pcm, timestamp=1.0, is_robot_speaking=False)
        self.assertTrue(obs_normal.valid)
        self.assertGreater(obs_normal.confidence, 0.3)

        # 2. When robot is speaking -> suppressed & marked invalid
        obs_speaking = core.process_frame(pcm, timestamp=1.1, is_robot_speaking=True)
        self.assertFalse(obs_speaking.valid)
        self.assertLess(obs_speaking.confidence, obs_normal.confidence)


class TestAudioFilterAndKalman(unittest.TestCase):
    def setUp(self):
        self.filter_core = AudioFilterCore(
            max_jump_deg=35.0,
            outlier_persistence_count=3,
            median_window_size=5,
        )

    def test_outlier_rejection_scenario(self):
        """Test sequence: [30, 31, 29, 87, 30, 31] -> 87° is rejected as an outlier."""
        angles = [30.0, 31.0, 29.0, 87.0, 30.0, 31.0]
        results = []

        t = 1.0
        for ang in angles:
            obs = AudioObservation(
                timestamp=t,
                valid=True,
                vad=True,
                body_azimuth_deg=ang,
                confidence=0.8,
            )
            state = self.filter_core.filter_observation(obs)
            results.append(state)
            t += 0.05

        # The 4th sample (87.0°) must be flagged as an outlier
        self.assertTrue(results[3].is_outlier)
        self.assertFalse(results[3].valid)
        # Heading should remain near ~30°
        self.assertLess(abs(results[3].azimuth_deg - 30.0), 5.0)

        # 5th and 6th samples return to valid tracking near 30°
        self.assertFalse(results[4].is_outlier)
        self.assertTrue(results[4].valid)
        self.assertLess(abs(results[5].azimuth_deg - 30.0), 3.0)

    def test_sustained_speaker_movement_accepted(self):
        """Test sequence: [30, 80, 81, 82] -> sustained switch accepted on 3rd consecutive frame."""
        angles = [30.0, 80.0, 81.0, 82.0]
        results = []

        t = 1.0
        for ang in angles:
            obs = AudioObservation(
                timestamp=t,
                valid=True,
                vad=True,
                body_azimuth_deg=ang,
                confidence=0.8,
            )
            state = self.filter_core.filter_observation(obs)
            results.append(state)
            t += 0.05

        # First two 80° samples are treated as candidate outliers
        self.assertTrue(results[1].is_outlier)
        self.assertTrue(results[2].is_outlier)
        # 3rd consecutive sample (82°) is accepted as genuine turn-taking/saccade
        self.assertFalse(results[3].is_outlier)
        self.assertTrue(results[3].valid)
        self.assertGreater(results[3].azimuth_deg, 70.0)

    def test_circular_wrap_kalman_smoothing(self):
        """Test crossing the ±180° boundary: [178, 179, -179, -178]."""
        self.filter_core.reset()
        angles = [178.0, 179.0, -179.0, -178.0]
        results = []

        t = 1.0
        for ang in angles:
            obs = AudioObservation(
                timestamp=t,
                valid=True,
                vad=True,
                body_azimuth_deg=ang,
                confidence=0.85,
            )
            state = self.filter_core.filter_observation(obs)
            results.append(state)
            t += 0.05

        for st in results:
            self.assertTrue(st.valid)
            self.assertFalse(st.is_outlier)
            # Distance to ±180° must stay <= 5°
            self.assertLessEqual(circular_distance_deg(st.azimuth_deg, 180.0), 5.0)

    def test_head_motion_attenuates_confidence(self):
        """Test that high head pan velocity decreases audio confidence."""
        compensator = HeadMotionCompensator(max_sens_velocity_deg_s=40.0)

        # 1. Stationary head (0 °/s)
        conf_stat, att_stat = compensator.compensate_confidence(0.8, 0.0, timestamp=1.0)
        self.assertEqual(conf_stat, 0.8)
        self.assertFalse(att_stat)

        # 2. Fast moving head (60 °/s)
        conf_fast, att_fast = compensator.compensate_confidence(0.8, 60.0, timestamp=1.0)
        self.assertLess(conf_fast, 0.3)
        self.assertTrue(att_fast)


if __name__ == "__main__":
    unittest.main()
