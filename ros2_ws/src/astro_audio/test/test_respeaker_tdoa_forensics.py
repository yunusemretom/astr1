#!/usr/bin/env python3
"""Unit tests for ReSpeaker raw pairwise cross-correlation and TDOA forensics.

Verifies:
1. Known synthetic delayed signals produce exact cross-correlation peak lags.
2. Pairwise lag detection on all 6 channel pairs.
3. Sample-to-microsecond conversion accuracy.
4. Consistency across simulated speaker positions (LEFT -> CENTER -> RIGHT).
5. Repeatability of stationary captures (CENTER_1 vs CENTER_2).
6. Parity between raw cross-correlation and GCC-PHAT TDOA.
"""

import math
import os
import sys
import unittest
import numpy as np

# Ensure repository paths
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", ".."))
scripts_dir = os.path.join(root_dir, "scripts")

for p in (root_dir, scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from respeaker_tdoa_forensics import (
    DEFAULT_MAX_LAG,
    HW_SAMPLE_RATE,
    PAIR_DEFINITIONS,
    calculate_channel_metrics,
    compute_hardware_gcc_phat_doa,
    compute_raw_cross_correlation,
    gcc_phat,
    microseconds_to_samples,
    normalized_correlation,
    samples_to_microseconds,
    simulate_station_pcm,
)


class TestRespeakerTDOAForensics(unittest.TestCase):
    """Test suite for raw TDOA cross-correlation forensics and metrics."""

    def setUp(self):
        self.fs = 16000
        self.rng = np.random.default_rng(12345)

    def test_sample_to_microsecond_conversion(self):
        """Verify sample-to-microsecond conversion formula: time_us = lag_samples / 16000 * 1e6."""
        # 1 sample @ 16kHz = 62.5 µs
        self.assertAlmostEqual(samples_to_microseconds(1.0, self.fs), 62.5, places=5)
        self.assertAlmostEqual(samples_to_microseconds(0.0, self.fs), 0.0, places=5)
        # 4 samples @ 16kHz = 250.0 µs (array diameter)
        self.assertAlmostEqual(samples_to_microseconds(4.0, self.fs), 250.0, places=5)
        self.assertAlmostEqual(samples_to_microseconds(-4.0, self.fs), -250.0, places=5)
        # 16 samples = 1000.0 µs
        self.assertAlmostEqual(samples_to_microseconds(16.0, self.fs), 1000.0, places=5)
        # Round trip
        self.assertAlmostEqual(microseconds_to_samples(250.0, self.fs), 4.0, places=5)

    def test_known_synthetic_delayed_signals(self):
        """Verify cross-correlation detects exact integer sample shifts on synthetic signals."""
        N = 8000
        base = self.rng.normal(0.0, 1.0, N + 200)

        test_delays = [0, 1, 2, 3, 4, -1, -2, -3, -4]
        for delay in test_delays:
            origin = 50
            sig_a = base[origin: origin + N]
            # When sig_b is delayed by +delay relative to sig_a:
            # sig_b[t] = sig_a[t - delay] -> sig_a leads sig_b -> lag = +delay
            sig_b = base[origin - delay: origin - delay + N]

            res = compute_raw_cross_correlation(sig_a, sig_b, fs=self.fs, max_lag_samples=16)

            self.assertEqual(
                res["peak_lag_samples"],
                delay,
                f"Expected peak lag of {delay} samples, got {res['peak_lag_samples']}",
            )
            self.assertAlmostEqual(res["normalized_correlation"], 1.0, places=2)
            expected_us = delay / 16000.0 * 1e6
            self.assertAlmostEqual(res["time_us"], expected_us, places=1)

    def test_pairwise_lag_detection_all_six_pairs(self):
        """Verify lag detection across all 6 physical mic pairs on a 4-channel synthetic array."""
        N = 8000
        base = self.rng.normal(0.0, 1.0, N + 100)
        origin = 50

        # Known relative delays per channel (0..3)
        # ch1 (Mic0): delay 0 (leads)
        # ch2 (Mic1): delay +2
        # ch3 (Mic2): delay +4 (lags)
        # ch4 (Mic3): delay +2
        delays = {1: 0, 2: 2, 3: 4, 4: 2}

        ch_data = {}
        for c in (1, 2, 3, 4):
            d = delays[c]
            ch_data[c] = base[origin - d: origin - d + N]

        # Verify all 6 pairs
        for c_a, c_b, pair_key, label in PAIR_DEFINITIONS:
            res = compute_raw_cross_correlation(ch_data[c_a], ch_data[c_b], fs=self.fs, max_lag_samples=16)
            expected_lag = delays[c_b] - delays[c_a]
            self.assertEqual(
                res["peak_lag_samples"],
                expected_lag,
                f"Pair {label} expected lag {expected_lag}, got {res['peak_lag_samples']}",
            )
            self.assertGreater(res["normalized_correlation"], 0.95)

    def test_repeatability_of_stationary_speaker(self):
        """Verify that repeated stationary captures produce identical TDOA (delta <= 1 sample)."""
        duration = 1.0
        # Capture CENTER twice with identical acoustic propagation
        pcm_1 = simulate_station_pcm("CENTER_1", duration_s=duration, seed=101)
        pcm_2 = simulate_station_pcm("CENTER_2", duration_s=duration, seed=202)

        res_fb_1 = compute_raw_cross_correlation(pcm_1[1], pcm_1[3], fs=self.fs, max_lag_samples=16)
        res_fb_2 = compute_raw_cross_correlation(pcm_2[1], pcm_2[3], fs=self.fs, max_lag_samples=16)

        res_rl_1 = compute_raw_cross_correlation(pcm_1[2], pcm_1[4], fs=self.fs, max_lag_samples=16)
        res_rl_2 = compute_raw_cross_correlation(pcm_2[2], pcm_2[4], fs=self.fs, max_lag_samples=16)

        delta_fb = abs(res_fb_1["peak_lag_samples_frac"] - res_fb_2["peak_lag_samples_frac"])
        delta_rl = abs(res_rl_1["peak_lag_samples_frac"] - res_rl_2["peak_lag_samples_frac"])

        self.assertLessEqual(delta_fb, 0.5, "Repeatability check failed on Front-Back axis!")
        self.assertLessEqual(delta_rl, 0.5, "Repeatability check failed on Right-Left axis!")

    def test_spatial_trend_left_center_right(self):
        """Verify that ch2-ch4 (Right-Left) lag varies monotonically as speaker moves LEFT -> CENTER -> RIGHT."""
        duration = 1.0
        pcm_left = simulate_station_pcm("LEFT", duration_s=duration, seed=301)
        pcm_center = simulate_station_pcm("CENTER", duration_s=duration, seed=302)
        pcm_right = simulate_station_pcm("RIGHT", duration_s=duration, seed=303)

        rl_left = compute_raw_cross_correlation(pcm_left[2], pcm_left[4], fs=self.fs)["peak_lag_samples_frac"]
        rl_center = compute_raw_cross_correlation(pcm_center[2], pcm_center[4], fs=self.fs)["peak_lag_samples_frac"]
        rl_right = compute_raw_cross_correlation(pcm_right[2], pcm_right[4], fs=self.fs)["peak_lag_samples_frac"]

        # Check monotonic progression
        is_monotonic = (rl_left < rl_center < rl_right) or (rl_left > rl_center > rl_right)
        self.assertTrue(is_monotonic, f"Expected monotonic shift, got Left={rl_left}, Center={rl_center}, Right={rl_right}")
        # Check variance > 2 samples across diameter
        self.assertGreater(abs(rl_right - rl_left), 2.0)

    def test_gcc_phat_parity_with_raw_cross_correlation(self):
        """Verify raw cross-correlation and GCC-PHAT both detect consistent peak lags on delayed signals."""
        N = 8000
        base = self.rng.normal(0.0, 1.0, N + 100)
        origin = 50

        for delay in [3, -3, 0]:
            sig_a = base[origin: origin + N]
            sig_b = base[origin - delay: origin - delay + N]

            raw_res = compute_raw_cross_correlation(sig_a, sig_b, fs=self.fs, max_lag_samples=16)
            tau, q = gcc_phat(sig_a.astype(np.float32), sig_b.astype(np.float32), fs=self.fs, max_tau=0.001)
            gcc_lag_smp = tau * self.fs

            self.assertAlmostEqual(raw_res["peak_lag_samples"], delay, places=0)
            self.assertAlmostEqual(gcc_lag_smp, delay, places=0)

    def test_channel_metrics_calculation(self):
        """Verify calculation of RMS, Peak, and Mean for a channel array."""
        data = np.array([100, -100, 200, -200], dtype=np.int16)
        metrics = calculate_channel_metrics(data)
        self.assertEqual(metrics["peak"], 200)
        self.assertAlmostEqual(metrics["mean"], 0.0, places=1)
        expected_rms = math.sqrt((100**2 + 100**2 + 200**2 + 200**2) / 4.0)
        self.assertAlmostEqual(metrics["rms"], round(expected_rms, 2), places=1)


if __name__ == "__main__":
    unittest.main()
