"""ASTRO V1 — Phase 1 Acceptance & Verification Test Suite.

Verifies:
  1. AudioStreamNode 4-channel hardware configuration & GCC-PHAT DOA readiness
  2. AcousticDOAEstimator multi-channel TDOA calculation
  3. Vision multi-class emotion detection (happy, surprised, focused, neutral)
  4. Multi-person parsing and AttentionManager focus selection in AstroRealtimeNode
"""

import json
import math
import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure test import paths
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [
    os.path.join(pkg_root, "astro_ai"),
    os.path.join(pkg_root, "astro_ai", "astro_ai"),
    os.path.join(pkg_root, "astro_audio"),
    os.path.join(pkg_root, "astro_audio", "astro_audio"),
    os.path.join(pkg_root, "astro_vision"),
    os.path.join(pkg_root, "astro_vision", "astro_vision"),
    os.path.join(pkg_root, "astro_base"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_ai.brain.social_brain import SocialBrain
from astro_ai.contracts.person_state import UnifiedPersonState, RelationshipRole
from astro_audio.doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry


class TestPhase1AudioDOA(unittest.TestCase):
    def test_01_doa_estimator_initialization_and_geometry(self):
        """Verify ReSpeaker 4-mic geometry is configured for radius 0.043m."""
        estimator = AcousticDOAEstimator(sample_rate=16000)
        self.assertIsNotNone(estimator)
        self.assertAlmostEqual(ReSpeakerGeometry.RADIUS_M, 0.043, places=3)
        self.assertEqual(ReSpeakerGeometry.SAMPLE_RATE, 16000)

    def test_02_doa_estimator_multichannel_synthetic_signal(self):
        """Verify estimate_from_multichannel_pcm runs on synthetic 4-channel 16kHz audio."""
        estimator = AcousticDOAEstimator(sample_rate=16000)
        # Create 50ms of 1kHz sine wave across 4 channels (800 samples)
        t = np.linspace(0, 0.05, 800, endpoint=False)
        sig = (np.sin(2 * np.pi * 1000 * t) * 10000).astype(np.int16)
        
        # Channel 0 reference, Channels 1-3 with small time delays
        ch0 = sig
        ch1 = np.roll(sig, 2)
        ch2 = np.roll(sig, 4)
        ch3 = np.roll(sig, 2)
        multi_ch = np.stack([ch0, ch1, ch2, ch3])

        azimuth_deg, conf, valid = estimator.estimate_from_multichannel_pcm(multi_ch)
        self.assertIsInstance(azimuth_deg, (float, int, type(None)))
        self.assertIsInstance(conf, float)
        self.assertIsInstance(valid, bool)


class TestPhase1VisionEmotion(unittest.TestCase):
    def test_03_facial_emotion_multi_class_logic(self):
        """Verify _detect_facial_emotion identifies happy, surprised, focused, and neutral."""
        from astro_vision.face_detector_node import SpatialVisionNode

        # Create dummy instance
        with patch.object(SpatialVisionNode, "__init__", return_value=None):
            node = SpatialVisionNode()
            node.smile_cascade = MagicMock()

            # 1. Smile -> happy
            node.smile_cascade.detectMultiScale.return_value = [(10, 10, 20, 20)]
            dummy_face = np.ones((100, 100), dtype=np.uint8) * 128
            self.assertEqual(node._detect_facial_emotion(dummy_face, 100, 100), "happy")

            # 2. No smile, low yaw + eyes -> focused
            node.smile_cascade.detectMultiScale.return_value = []
            self.assertEqual(
                node._detect_facial_emotion(dummy_face, 100, 100, yaw=2.0, eyes_found=True),
                "focused"
            )

            # 3. No smile, yaw > 15 -> neutral
            self.assertEqual(
                node._detect_facial_emotion(dummy_face, 100, 100, yaw=20.0, eyes_found=True),
                "neutral"
            )

            # 4. Open mouth contrast -> surprised
            surprised_face = np.ones((100, 100), dtype=np.uint8) * 150
            # Dark oral cavity in mouth region (lower half)
            surprised_face[60:90, 35:65] = 20
            self.assertEqual(
                node._detect_facial_emotion(surprised_face, 100, 100, yaw=15.0, eyes_found=False),
                "surprised"
            )


class TestPhase1MultiPersonAttention(unittest.TestCase):
    def test_04_attention_manager_selects_active_speaker_over_silent(self):
        """Verify AttentionManager prioritizes speaking person over closer silent person."""
        brain = SocialBrain()
        p1 = UnifiedPersonState(person_id="p1", name="Misafir", distance_m=1.0, is_speaking=False)
        p2 = UnifiedPersonState(person_id="p2", name="Baran", distance_m=1.8, is_speaking=True, is_known=True)

        chosen, score = brain.attention_manager.select_focus_target([p1, p2])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.name, "Baran")

    def test_05_attention_manager_selects_looking_over_diverted_gaze(self):
        """Verify AttentionManager prioritizes direct gaze when neither is speaking."""
        brain = SocialBrain()
        p1 = UnifiedPersonState(person_id="p1", name="Ali", distance_m=1.2, is_looking_at_robot=True)
        p2 = UnifiedPersonState(person_id="p2", name="Veli", distance_m=1.2, is_looking_at_robot=False)

        chosen, score = brain.attention_manager.select_focus_target([p1, p2])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.name, "Ali")

    @patch.dict(os.environ, {"ASTRO_TEST_MODE": "1", "OPENAI_API_KEY": ""})
    def test_06_realtime_node_on_faces_runs_attention_selection(self):
        """Verify AstroRealtimeNode._on_faces updates world model and selects focus target."""
        from astro_ai.astro_realtime_node import AstroRealtimeNode

        node = AstroRealtimeNode()
        self.assertIsNotNone(node.social_brain)

        # Send multi-person faces JSON
        faces_payload = [
            {"name": "Misafir", "distance_m": 2.5, "looking_at_robot": False, "yaw_deg": 35.0},
            {"name": "Baran", "is_known": True, "distance_m": 1.2, "looking_at_robot": True, "yaw_deg": 0.0},
        ]
        msg = MagicMock()
        msg.data = json.dumps(faces_payload)

        node._on_faces(msg)
        # Baran should be selected as focus due to familiarity + proximity + direct gaze
        self.assertEqual(node._active_person_name, "Baran")
        self.assertAlmostEqual(node._user_distance, 1.2)
        self.assertTrue(node._looking_at_robot)


if __name__ == "__main__":
    unittest.main()
