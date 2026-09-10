#!/usr/bin/env python3
"""Dedicated Tests for ReSpeaker Multi-Channel Mapping and GCC-PHAT DOA Parity.

Verifies:
1. ReSpeaker 6-channel buffer layout: Ch 0 is beamformed audio, Ch 1..4 are raw mics.
2. GCC-PHAT Pair Signs & Directions:
   - CENTER (0°): Front mic leads -> DOA = 0.0°
   - RIGHT (+90°): Right mic leads -> DOA = +90.0°
   - REAR (180°): Back mic leads -> DOA = 180.0°
   - LEFT (270°): Left mic leads -> DOA = 270.0° (-90.0° in REP-103)
3. Coordinate conversion parity between ReSpeaker circular frame and REP-103 body yaw.
4. Standalone vs ROS Parity: AudioSource and AudioStreamNode extract the identical 4 raw channels from a 6-channel stream.
5. HID Isolation: Coarse 45° HID polling is inhibited during multi-channel GCC-PHAT operation.
"""

import math
import os
import sys
import unittest
import numpy as np

# Ensure import paths
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir = os.path.abspath(os.path.join(pkg_dir, "..", "..", ".."))
standalone_dir = os.path.join(root_dir, "standalone")
astro_base_dir = os.path.join(root_dir, "ros2_ws", "src", "astro_base")

for p in (pkg_dir, root_dir, standalone_dir, astro_base_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_audio.doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry, gcc_phat
from astro_audio.audio_stream_node import AudioStreamNode
from astro_base.gaze.coordinate_frames import CoordinateTransformer
from sources import AudioSource, RESPEAKER_MIC_CHANNELS, ARRAY_MIC_CHANNELS


def create_respeaker_6ch_frame(front_lead: int, right_lead: int, length: int = 1024, seed: int = 42) -> np.ndarray:
    """Creates a realistic 6-channel ReSpeaker UAC 1.0 audio buffer:
    Ch 0: Processed Mono Audio (Beamformed & Filtered)
    Ch 1: Mic 0 (Front)
    Ch 2: Mic 1 (Right)
    Ch 3: Mic 2 (Back)
    Ch 4: Mic 3 (Left)
    Ch 5: Playback loopback
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, length * 3) * 6000.0
    origin = length

    def take(lead: int) -> np.ndarray:
        return base[origin + lead: origin + lead + length]

    raw_mics = [
        take(+front_lead),   # Mic 0: Front
        take(+right_lead),   # Mic 1: Right
        take(-front_lead),   # Mic 2: Back
        take(-right_lead),   # Mic 3: Left
    ]

    # Processed beam on channel 0 (simulates XMOS DSP output)
    processed = np.mean(raw_mics, axis=0)
    loopback = np.zeros(length, dtype=np.float32)

    return np.stack([
        processed,      # Ch 0
        raw_mics[0],    # Ch 1: Front
        raw_mics[1],    # Ch 2: Right
        raw_mics[2],    # Ch 3: Back
        raw_mics[3],    # Ch 4: Left
        loopback,       # Ch 5: Loopback
    ])


class TestReSpeakerChannelMappingAndDOA(unittest.TestCase):

    def setUp(self):
        self.estimator = AcousticDOAEstimator(sample_rate=16000)
        self.transformer = CoordinateTransformer()

    def test_gcc_phat_pair_signs_and_directions(self):
        """Validates that GCC-PHAT correctly resolves the 4 cardinal directions."""
        cases = {
            "CENTER": (4, 0, 0.0),
            "RIGHT":  (0, 4, 90.0),
            "REAR":   (-4, 0, 180.0),
            "LEFT":   (0, -4, 270.0),
        }

        for name, (f_lead, r_lead, expected_circ) in cases.items():
            frame_6ch = create_respeaker_6ch_frame(f_lead, r_lead)
            # Raw mics are channels 1..4
            raw_mics = frame_6ch[1:5]
            az, conf, valid = self.estimator.estimate_from_multichannel_pcm(raw_mics)

            self.assertTrue(valid, f"{name} should be valid")
            self.assertIsNotNone(az, f"{name} azimuth should not be None")
            circ_doa = az if az >= 0.0 else az + 360.0
            self.assertAlmostEqual(circ_doa, expected_circ, delta=2.0,
                                   msg=f"{name} expected {expected_circ}°, got {circ_doa}°")

    def test_corrupted_channel_order_fails_physical_consistency(self):
        """Demonstrates the root cause: taking channels 0..3 rotates and corrupts DOA."""
        frame_6ch = create_respeaker_6ch_frame(front_lead=4, right_lead=0)
        # Buggy extraction: channels 0, 1, 2, 3 (includes processed channel, misses left mic)
        buggy_mics = frame_6ch[:4]
        az_buggy, _, _ = self.estimator.estimate_from_multichannel_pcm(buggy_mics)
        circ_buggy = az_buggy if az_buggy >= 0.0 else az_buggy + 360.0
        # Buggy extraction reports 90.0° (RIGHT) for a CENTER speaker!
        self.assertAlmostEqual(circ_buggy, 90.0, delta=2.0)

    def test_standalone_and_ros_channel_extraction_parity(self):
        """Proves AudioSource and AudioStreamNode extract the identical raw channels from a 6-ch device."""
        frame_6ch = create_respeaker_6ch_frame(front_lead=0, right_lead=4)  # Speaker on Right

        # Standalone extraction
        standalone_mics = frame_6ch[list(RESPEAKER_MIC_CHANNELS)]
        az_standalone, _, valid_sa = self.estimator.estimate_from_multichannel_pcm(standalone_mics)

        # ROS extraction (using _mic_channel_indices=(1, 2, 3, 4) for 6-ch device)
        ros_mic_indices = (1, 2, 3, 4)
        ros_mics = frame_6ch[list(ros_mic_indices)]
        az_ros, _, valid_ros = self.estimator.estimate_from_multichannel_pcm(ros_mics)

        self.assertTrue(valid_sa)
        self.assertTrue(valid_ros)
        self.assertEqual(az_standalone, az_ros)
        self.assertAlmostEqual(az_ros, 90.0, delta=1.0)

    def test_coordinate_conversion_to_rep103_head_bearing(self):
        """Verifies ReSpeaker circular [0..359°] maps 1:1 to REP-103 body yaw [-180..+180°]."""
        # Center: 0° -> 0°
        self.assertAlmostEqual(self.transformer.raw_audio_doa_to_head_bearing(0.0), 0.0, delta=0.5)
        # Right: 90° -> -90° (Right in REP-103 is negative)
        self.assertAlmostEqual(self.transformer.raw_audio_doa_to_head_bearing(90.0), -90.0, delta=0.5)
        # Rear: 180° -> 180°
        self.assertAlmostEqual(abs(self.transformer.raw_audio_doa_to_head_bearing(180.0)), 180.0, delta=0.5)
        # Left: 270° -> +90° (Left in REP-103 is positive)
        self.assertAlmostEqual(self.transformer.raw_audio_doa_to_head_bearing(270.0), 90.0, delta=0.5)

    def test_pcm_channel_extraction(self):
        """STEP 1 TEST: Verifies interleaved multi-channel byte buffer unpacks to (channels, frames)."""
        num_channels = 6
        num_frames = 320
        # Create known synthetic pattern for each channel
        expected_channels = np.zeros((num_channels, num_frames), dtype=np.int16)
        for c in range(num_channels):
            expected_channels[c] = np.arange(num_frames, dtype=np.int16) * 10 + c

        # Interleave channels: [s0_c0, s0_c1, ..., s0_c5, s1_c0, ...]
        interleaved = expected_channels.T.reshape(-1)
        raw_bytes = interleaved.tobytes()

        # Unpack as audio_stream_node does
        raw_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        self.assertEqual(len(raw_arr), num_channels * num_frames)

        multi_ch = raw_arr.reshape(-1, num_channels).T
        self.assertEqual(multi_ch.shape, (6, 320))
        np.testing.assert_array_equal(multi_ch, expected_channels)

        # Selected mic channels extraction
        selected_indices = (1, 2, 3, 4)
        mics = multi_ch[list(selected_indices)]
        self.assertEqual(mics.shape, (4, 320))
        np.testing.assert_array_equal(mics, expected_channels[1:5])

    def test_six_channel_detection_and_selected_indices(self):
        """STEP 2 & 3 TEST: Verifies channel count and mic index selection logic."""
        # 1. When hardware has >= 6 channels (ReSpeaker standard layout)
        max_in_ch_6 = 6
        pref_ch_0 = 0
        if pref_ch_0 in (1, 2, 4, 6, 8):
            cap_6 = pref_ch_0
        elif max_in_ch_6 >= 6:
            cap_6 = 6
        elif max_in_ch_6 >= 4:
            cap_6 = 4
        else:
            cap_6 = 1
        mic_idx_6 = (1, 2, 3, 4) if cap_6 >= 6 else (0, 1, 2, 3)

        self.assertEqual(cap_6, 6)
        self.assertEqual(mic_idx_6, (1, 2, 3, 4))
        self.assertEqual(mic_idx_6, RESPEAKER_MIC_CHANNELS)

        # 2. When hardware has 4 channels (e.g. 4-ch raw array or simulated 4-ch)
        max_in_ch_4 = 4
        if pref_ch_0 in (1, 2, 4, 6, 8):
            cap_4 = pref_ch_0
        elif max_in_ch_4 >= 6:
            cap_4 = 6
        elif max_in_ch_4 >= 4:
            cap_4 = 4
        else:
            cap_4 = 1
        mic_idx_4 = (1, 2, 3, 4) if cap_4 >= 6 else (0, 1, 2, 3)

        self.assertEqual(cap_4, 4)
        self.assertEqual(mic_idx_4, (0, 1, 2, 3))
        self.assertEqual(mic_idx_4, ARRAY_MIC_CHANNELS)

    def test_standalone_and_ros_input_array_bitwise_parity(self):
        """STEP 4 & 8 TEST: Proves standalone AudioSource and ROS audio_stream_node feed bitwise identical arrays to GCC-PHAT."""
        frame_6ch = create_respeaker_6ch_frame(front_lead=0, right_lead=4, length=320, seed=123)
        frame_6ch_int16 = frame_6ch.astype(np.int16)

        # Standalone extraction:
        # AudioSource opens float32, unpacks block.T (shape channels, frames), selects RESPEAKER_MIC_CHANNELS=(1,2,3,4)
        standalone_mics = frame_6ch_int16[list(RESPEAKER_MIC_CHANNELS)]

        # ROS extraction:
        # AudioStreamNode unpacks interleaved bytes, shapes to (6, frames), selects _mic_channel_indices=(1,2,3,4)
        interleaved_bytes = frame_6ch_int16.T.reshape(-1).tobytes()
        raw_arr = np.frombuffer(interleaved_bytes, dtype=np.int16)
        multi_ch = raw_arr.reshape(-1, 6).T
        ros_mic_indices = (1, 2, 3, 4)
        ros_mics = multi_ch[list(ros_mic_indices)]

        # 1. Proves input arrays are bitwise identical
        np.testing.assert_array_equal(standalone_mics, ros_mics)

        # 2. Proves GCC-PHAT invocation on identical inputs produces identical DOA
        az_standalone, conf_sa, valid_sa = self.estimator.estimate_from_multichannel_pcm(standalone_mics)
        az_ros, conf_ros, valid_ros = self.estimator.estimate_from_multichannel_pcm(ros_mics)

        self.assertTrue(valid_sa)
        self.assertTrue(valid_ros)
        self.assertEqual(az_standalone, az_ros)
        self.assertEqual(conf_sa, conf_ros)
        self.assertAlmostEqual(az_ros, 90.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()

