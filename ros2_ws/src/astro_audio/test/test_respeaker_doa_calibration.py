#!/usr/bin/env python3
"""Dedicated Unit & Acceptance Tests for ReSpeaker DOA Calibration and Circular Statistics.

Verifies:
1. Circular statistics boundary wrapping: [359°, 0°, 1°] resolves to ~0° mean and
   low variance (~1.0° stddev) rather than spurious linear mean (120°) and stddev (169°).
2. Format invariance: identical results for 0..360° vs -180..+180° representations.
3. Acoustic validity filtering: samples with low confidence or VAD=False are excluded from
   angular stats and reflected in valid_sample_ratio.
4. Edge cases: empty samples, single sample, noise-only samples.
5. End-to-end simulation: validates output table and JSON structure against required schema.
"""

import json
import math
import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure repository paths
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", ".."))
scripts_dir = os.path.join(root_dir, "scripts")

for p in (root_dir, scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibrate_respeaker_doa import (
    DOASample,
    PositionCalibrationResult,
    ReSpeakerHardwareCapture,
    compute_circular_stats,
    compute_hardware_gcc_phat_doa,
    run_calibration_session,
    run_simulated_capture,
)
from respeaker_device import RespeakerDeviceInfo


class TestCircularStatisticsAndDOACalibration(unittest.TestCase):

    def test_circular_cluster_around_zero_degrees(self):
        """CRITICAL TEST: Proves [359°, 0°, 1°] is treated as a tight cluster around 0°.

        Naive linear mean: (359 + 0 + 1) / 3 = 120.0° (completely wrong!)
        Naive linear stddev: std([359, 0, 1]) = 169.2° (huge false variance!)

        Circular statistics must report:
        - Mean: 0.0° (or 360.0° ≡ 0.0°)
        - Median: 0.0°
        - Sample stddev: 1.0° (low variance)
        """
        samples = [
            DOASample(timestamp=1.0, raw_doa_deg=359.0, confidence=0.85, vad=True),
            DOASample(timestamp=1.1, raw_doa_deg=0.0,   confidence=0.90, vad=True),
            DOASample(timestamp=1.2, raw_doa_deg=1.0,   confidence=0.88, vad=True),
        ]

        res = compute_circular_stats(samples, physical_deg=0.0, min_confidence=0.40)

        # 1. Mean must be 0.0°
        self.assertIsNotNone(res.mean_raw_deg)
        self.assertAlmostEqual(res.mean_signed_deg, 0.0, delta=0.05,
                               msg=f"Expected mean close to 0.0°, got {res.mean_signed_deg}°")
        # In [0, 360), 0.0° ≡ 360.0°
        self.assertTrue(abs(res.mean_raw_deg - 0.0) < 0.05 or abs(res.mean_raw_deg - 360.0) < 0.05)

        # 2. Median must be 0.0°
        self.assertAlmostEqual(res.median_signed_deg, 0.0, delta=0.05)

        # 3. Standard deviation must be tight (~1.0°), NOT > 100°
        self.assertIsNotNone(res.stddev_deg)
        self.assertAlmostEqual(res.stddev_deg, 1.0, delta=0.1,
                               msg=f"Expected stddev ~1.0°, got {res.stddev_deg}°")
        self.assertLess(res.stddev_deg, 5.0, "Stddev must not be inflated by 0°/360° boundary!")

        # 4. Range
        self.assertAlmostEqual(res.min_raw_deg, 359.0, delta=0.05)
        self.assertAlmostEqual(res.max_raw_deg, 1.0, delta=0.05)

        # 5. Validity ratio
        self.assertEqual(res.sample_count, 3)
        self.assertEqual(res.total_received, 3)
        self.assertAlmostEqual(res.valid_ratio, 1.0)
        self.assertTrue(res.is_stable)

    def test_linear_vs_circular_divergence_proof(self):
        """Demonstrates mathematical divergence between naive linear and circular stats."""
        raw_angles = [359.0, 0.0, 1.0]

        # Naive linear
        linear_mean = float(np.mean(raw_angles))
        linear_std = float(np.std(raw_angles, ddof=1))

        self.assertAlmostEqual(linear_mean, 120.0, delta=0.1)
        self.assertGreater(linear_std, 150.0)

        # Circular implementation
        samples = [DOASample(1.0 + i * 0.1, a, 0.8, True) for i, a in enumerate(raw_angles)]
        circ_res = compute_circular_stats(samples)

        self.assertAlmostEqual(circ_res.mean_signed_deg, 0.0, delta=0.05)
        self.assertAlmostEqual(circ_res.stddev_deg, 1.0, delta=0.05)

    def test_support_both_raw_formats(self):
        """Verifies circular statistics work identically for 0..360° and -180..+180° inputs."""
        # 1. 0..360° representation for a speaker at -30° (nominal 330°)
        samples_360 = [
            DOASample(1.0, 328.0, 0.85, True),
            DOASample(1.1, 330.0, 0.88, True),
            DOASample(1.2, 332.0, 0.84, True),
        ]
        res_360 = compute_circular_stats(samples_360, physical_deg=-30.0)

        # 2. -180..+180° representation for the same speaker (-30°)
        samples_pm180 = [
            DOASample(1.0, -32.0, 0.85, True),
            DOASample(1.1, -30.0, 0.88, True),
            DOASample(1.2, -28.0, 0.84, True),
        ]
        res_pm180 = compute_circular_stats(samples_pm180, physical_deg=-30.0)

        # Both must agree on the physical direction
        self.assertAlmostEqual(res_360.mean_signed_deg, -30.0, delta=0.05)
        self.assertAlmostEqual(res_pm180.mean_signed_deg, -30.0, delta=0.05)
        self.assertAlmostEqual(res_360.mean_raw_deg, 330.0, delta=0.05)
        self.assertAlmostEqual(res_pm180.mean_raw_deg, 330.0, delta=0.05)

        # Stddev must be identical (2.0°)
        self.assertAlmostEqual(res_360.stddev_deg, 2.0, delta=0.05)
        self.assertAlmostEqual(res_pm180.stddev_deg, 2.0, delta=0.05)

    def test_acoustic_validity_filtering_and_ratio(self):
        """Verifies filtering of invalid samples: low confidence and VAD=False."""
        samples = [
            DOASample(1.0, 45.0, 0.80, True),   # VALID
            DOASample(1.1, 46.0, 0.75, True),   # VALID
            DOASample(1.2, 44.0, 0.90, True),   # VALID
            DOASample(1.3, 180.0, 0.20, True),  # REJECT: low confidence (0.20 < 0.40)
            DOASample(1.4, 270.0, 0.85, False), # REJECT: VAD is False
            DOASample(1.5, 90.0, 0.10, False),  # REJECT: both low conf and VAD False
        ]

        res = compute_circular_stats(samples, physical_deg=45.0, min_confidence=0.40)

        # 3 out of 6 valid
        self.assertEqual(res.sample_count, 3)
        self.assertEqual(res.total_received, 6)
        self.assertAlmostEqual(res.valid_ratio, 0.50, delta=0.01)

        # Mean computed only from valid samples (45°, 46°, 44° -> 45.0°)
        self.assertAlmostEqual(res.mean_raw_deg, 45.0, delta=0.05)
        self.assertAlmostEqual(res.stddev_deg, 1.0, delta=0.05)
        # Average confidence of valid samples: (0.80 + 0.75 + 0.90) / 3 = 0.817
        self.assertAlmostEqual(res.confidence_mean, 0.817, delta=0.01)

    def test_empty_and_all_rejected_samples(self):
        """Verifies clean handling when no valid speech samples are present."""
        # Empty
        res_empty = compute_circular_stats([], physical_deg=0.0)
        self.assertEqual(res_empty.sample_count, 0)
        self.assertIsNone(res_empty.mean_raw_deg)
        self.assertIsNone(res_empty.stddev_deg)
        self.assertEqual(res_empty.valid_ratio, 0.0)
        self.assertFalse(res_empty.is_stable)

        # All rejected
        samples = [
            DOASample(1.0, 100.0, 0.15, False),
            DOASample(1.1, 200.0, 0.20, True),  # conf < 0.40
        ]
        res_rej = compute_circular_stats(samples, physical_deg=90.0, min_confidence=0.40)
        self.assertEqual(res_rej.sample_count, 0)
        self.assertEqual(res_rej.total_received, 2)
        self.assertIsNone(res_rej.mean_raw_deg)
        self.assertEqual(res_rej.valid_ratio, 0.0)

    def test_single_sample_handling(self):
        """Verifies single valid sample produces 0.0 stddev without division by zero."""
        samples = [DOASample(1.0, 75.0, 0.85, True)]
        res = compute_circular_stats(samples, physical_deg=75.0)

        self.assertEqual(res.sample_count, 1)
        self.assertAlmostEqual(res.mean_raw_deg, 75.0, delta=0.05)
        self.assertAlmostEqual(res.stddev_deg, 0.0, delta=0.05)

    def test_full_calibration_simulation_and_json_structure(self):
        """Tests end-to-end simulation across all 7 calibration positions and JSON export."""
        positions = [0.0, 30.0, 60.0, 90.0, -30.0, -60.0, -90.0]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_path = tf.name

        try:
            results = run_calibration_session(
                positions=positions,
                duration_s=1.0,
                min_confidence=0.40,
                output_path=tmp_path,
                simulate=True,
                non_interactive=True,
            )

            self.assertEqual(len(results), 7)
            for r in results:
                self.assertGreater(r.sample_count, 15)
                self.assertGreater(r.valid_ratio, 0.70)
                self.assertLess(r.stddev_deg, 5.0)
                self.assertTrue(r.is_stable)

            # Check JSON file
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("positions_deg", data)
            pos_list = data["positions_deg"]
            self.assertEqual(len(pos_list), 7)

            for entry in pos_list:
                # Required schema keys from user specification
                self.assertIn("physical_deg", entry)
                self.assertIn("mean_raw_deg", entry)
                self.assertIn("median_raw_deg", entry)
                self.assertIn("stddev_deg", entry)
                self.assertIn("confidence_mean", entry)
                self.assertIn("valid_ratio", entry)

                self.assertIsInstance(entry["physical_deg"], (int, float))
                self.assertIsInstance(entry["mean_raw_deg"], (int, float))
                self.assertIsInstance(entry["median_raw_deg"], (int, float))
                self.assertIsInstance(entry["stddev_deg"], (int, float))
                self.assertIsInstance(entry["confidence_mean"], (int, float))
                self.assertIsInstance(entry["valid_ratio"], (int, float))

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_direct_hardware_6ch_pcm_gcc_phat_extraction(self):
        """CRITICAL: Verifies hardware calibration extracts channels [1,2,3,4] from 6-ch PCM,

        runs direct GCC-PHAT without requiring ROS VAD or bringup stack, and returns valid DOA samples.
        """
        device_info = RespeakerDeviceInfo(
            alsa_device_string="hw:CARD=ArrayUAC10,DEV=0",
            device_index=2,
            device_name="ReSpeaker 4 Mic Array (UAC1.0)",
            card_id="ArrayUAC10",
            sample_rate=16000,
            sample_format="S16_LE",
            channels=6,
            mic_indices=(1, 2, 3, 4),
            is_valid_respeaker=True,
        )
        hw_capture = ReSpeakerHardwareCapture(device_info, min_confidence=0.0)
        hw_capture.start_recording()

        # Cardinal test directions: (name, lead_front, lead_right, expected_deg)
        cardinal_cases = [
            ("FRONT (0°)",  4,  0,   0.0),
            ("RIGHT (90°)", 0,  4,  90.0),
            ("BACK (180°)", -4, 0, 180.0),
            ("LEFT (270°)", 0, -4, 270.0),
        ]

        length = 640
        for name, lf, lr, expected_deg in cardinal_cases:
            rng = np.random.default_rng(42)
            base = rng.normal(0, 1, length * 3).astype(np.float32) * 5000.0
            mid = length
            get_lag = lambda lead: base[mid + lead : mid + lead + length]

            # Construct realistic 6-channel buffer where:
            # Ch 0: Uncorrelated beam noise (must be ignored)
            # Ch 1: Front mic
            # Ch 2: Right mic
            # Ch 3: Back mic
            # Ch 4: Left mic
            # Ch 5: Loopback channel (zeros)
            ch0_beam = np.random.default_rng(999).normal(0, 500, length).astype(np.float32)
            ch1_front = get_lag(+lf)
            ch2_right = get_lag(+lr)
            ch3_back = get_lag(-lf)
            ch4_left = get_lag(-lr)
            ch5_loop = np.zeros(length, dtype=np.float32)

            pcm_6ch = np.stack([ch0_beam, ch1_front, ch2_right, ch3_back, ch4_left, ch5_loop]).astype(np.int16)

            # Process block via ReSpeakerHardwareCapture
            sample = hw_capture.process_pcm_block(pcm_6ch, timestamp=100.0)

            # Verify sample is valid and DOA matches expected direction
            self.assertIsNotNone(sample, f"{name}: Failed to return DOASample from 6-ch PCM")
            self.assertAlmostEqual(sample.raw_doa_deg, expected_deg, delta=3.0,
                                   msg=f"{name}: expected {expected_deg}°, got {sample.raw_doa_deg}°")
            self.assertGreater(sample.confidence, 0.0)
            self.assertTrue(sample.vad)

        # Retrieve recorded samples
        recorded = hw_capture.stop_recording()
        self.assertEqual(len(recorded), 4)

        # Verify statistics in hardware mode (require_vad=False)
        stats = compute_circular_stats(recorded, physical_deg=0.0, min_confidence=0.0, require_vad=False)
        self.assertEqual(stats.sample_count, 4)
        self.assertGreater(stats.valid_ratio, 0.99)


if __name__ == "__main__":
    unittest.main()
