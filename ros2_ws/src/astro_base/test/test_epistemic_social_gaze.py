#!/usr/bin/env python3
"""Tests for Epistemic Situational Awareness & Social Scene Grounding in ASTRO Robot."""

import math
import os
import sys
import time
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from astro_base.gaze.attention_arbiter import AttentionArbiterCore
from astro_base.gaze.gaze_state_machine import SocialGazeFSM
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.spatial_memory import EpistemicSpatialMemory
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    ExplicitGazeIntent,
    FilteredAudioState,
    FusedTarget,
    GazeStateEnum,
    Modality,
    PrioritySource,
    TargetSelectorType,
    TargetState,
    TrackingState,
    VisualTargetTrack,
)


class TestEpistemicSocialGaze(unittest.TestCase):
    def setUp(self):
        self.spatial_mem = EpistemicSpatialMemory(
            person_memory_ttl_s=15.0,
            reverb_suppression_ttl_s=10.0,
            reverb_sector_width_deg=25.0,
        )
        self.fusion = AudioVisualFusionCore(
            spatial_gate_deg=25.0,
            spatial_memory=self.spatial_mem,
        )
        self.target_manager = TargetManagerCore(
            acquisition_threshold=0.75,
            hold_threshold=0.40,
            target_lost_timeout_s=1.0,
        )
        self.arbiter = AttentionArbiterCore(
            min_limit_deg=-75.0,
            max_limit_deg=75.0,
            spatial_memory=self.spatial_mem,
        )
        self.fsm = SocialGazeFSM(
            deadband_deg=2.5,
            idle_return_timeout_s=15.0,
            min_limit_deg=-75.0,
            max_limit_deg=75.0,
            settling_persistence_required=1,
            arbiter=self.arbiter,
            spatial_memory=self.spatial_mem,
        )

    def test_negative_evidence_suppresses_reverb_zone(self):
        """Invariant: After scanning an empty acoustic angle without finding a face, the angle is suppressed."""
        now = 100.0

        # 1. Acoustic sound arrives from empty wall at +55.0 deg
        audio = FilteredAudioState(azimuth_deg=55.0, confidence=0.85, valid=True, timestamp=now)
        candidates = self.fusion.fuse(audio, [], timestamp=now)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].body_azimuth_deg, 55.0)

        # 2. FSM orients to +55.0 deg
        target_state = self.target_manager.update(candidates, timestamp=now)
        cmd = self.fsm.update(target_state, actual_head_yaw_deg=0.0, timestamp=now)
        self.assertEqual(self.fsm.state, GazeStateEnum.ORIENTING)

        # 3. Head arrives at +55.0 deg and settles -> enters ACQUIRING
        now += 1.0
        cmd = self.fsm.update(target_state, actual_head_yaw_deg=55.0, actual_head_vel_deg_s=0.0, timestamp=now)
        self.assertEqual(self.fsm.state, GazeStateEnum.ACQUIRING)

        # 4. In ACQUIRING, no visual face appears -> times out after 0.8s
        now += 0.9
        empty_target_state = self.target_manager.update([], timestamp=now)
        cmd = self.fsm.update(empty_target_state, actual_head_yaw_deg=55.0, actual_head_vel_deg_s=0.0, timestamp=now)

        # Invariant: Negative evidence registered! +55.0 deg must now be a Reverb Zone!
        self.assertTrue(self.spatial_mem.is_acoustic_reverb_zone(55.0, timestamp=now))
        self.assertTrue(self.spatial_mem.is_acoustic_reverb_zone(50.0, timestamp=now))

        # 5. A new acoustic sound from the exact same wall (+55.0 deg) arrives
        now += 0.2
        reverb_audio = FilteredAudioState(azimuth_deg=55.0, confidence=0.85, valid=True, timestamp=now)
        new_candidates = self.fusion.fuse(reverb_audio, [], timestamp=now)

        # Invariant: Fusion MUST reject the reverb zone!
        self.assertEqual(len(new_candidates), 0, "Acoustic reverb zone was not suppressed!")

    def test_spatial_person_memory_persistence(self):
        """Invariant: Confirmed person is remembered for TTL and anchors loose acoustic DOA."""
        now = 200.0

        # 1. Camera observes person at +12.0 deg
        vt = VisualTargetTrack(
            target_id="person_baran",
            pos_3d=(0.3, 1.4, 0.0),
            vel_3d=(0.0, 0.0, 0.0),
            body_azimuth_deg=12.0,
            body_elevation_deg=0.0,
            distance_m=1.5,
            confidence=0.90,
            last_seen_time=now,
            person_name="Baran",
            tracking_state=TrackingState.TRACKING,
        )
        candidates = self.fusion.fuse(None, [vt], timestamp=now)
        self.assertEqual(len(candidates), 1)

        # Invariant: Person registered in spatial memory
        self.assertAlmostEqual(self.spatial_mem.get_most_likely_person_location(now), 12.0)

        # 2. Camera blinks / drops out for 3 seconds
        now += 3.0
        self.assertAlmostEqual(self.spatial_mem.get_most_likely_person_location(now), 12.0)

        # 3. Loose acoustic speech arrives at +18.0 deg (close to Baran at +12.0 deg)
        audio = FilteredAudioState(azimuth_deg=18.0, confidence=0.80, valid=True, timestamp=now)
        fused = self.fusion.fuse(audio, [], timestamp=now)
        self.assertEqual(len(fused), 1)
        # Invariant: Snapped to known person location!
        self.assertAlmostEqual(fused[0].body_azimuth_deg, 12.0)

    def test_hearing_speech_in_empty_space_reorients_to_known_human(self):
        """Invariant: If robot is looking at empty space when ACQUIRING fails, it reorients to known human."""
        now = 300.0

        # 1. Robot knows Baran is at +15.0 deg
        self.spatial_mem.register_person_observation(
            person_id="person_baran",
            bearing_deg=15.0,
            confidence=0.90,
            timestamp=now,
        )

        # 2. Stray acoustic reflection sent robot to +65.0 deg
        now += 1.0
        self.fsm.state = GazeStateEnum.ACQUIRING
        self.fsm.target_yaw_deg = 65.0
        self.fsm._state_entry_time = now

        # 3. ACQUIRING times out at +65.0 deg after 0.85s
        now += 0.85
        empty_target_state = TargetState(active_target=None, candidate_targets=[])
        cmd = self.fsm.update(empty_target_state, actual_head_yaw_deg=65.0, timestamp=now)

        # Invariant: Instead of falling to 0.0 deg neutral, robot reorients to known person at +15.0 deg!
        self.assertEqual(self.fsm.state, GazeStateEnum.ORIENTING)
        self.assertAlmostEqual(self.fsm.target_yaw_deg, 15.0)
        self.assertEqual(self.fsm.last_transition_reason, "REORIENT_TO_KNOWN_HUMAN")

    def test_explicit_command_resolves_via_spatial_memory(self):
        """Invariant: 'Astro bana don' resolves to known person even when active DOA is weak/zero."""
        now = 400.0

        # 1. Person registered at +22.0 deg in spatial memory
        self.spatial_mem.register_person_observation(
            person_id="person_baran",
            bearing_deg=22.0,
            confidence=0.88,
            timestamp=now - 2.0,
        )

        # 2. User gives explicit command: selector=CURRENT_SPEAKER
        intent = ExplicitGazeIntent(
            selector=TargetSelectorType.CURRENT_SPEAKER,
            confidence=1.0,
            timestamp=now,
            reason="explicit_speech_command",
            valid=True,
        )

        empty_target_state = TargetState(active_target=None, candidate_targets=[])
        decision = self.arbiter.arbitrate(
            target_state=empty_target_state,
            explicit_intent=intent,
            actual_head_yaw_deg=0.0,
            timestamp=now,
        )

        # Invariant: AttentionArbiter resolves straight to +22.0 deg via spatial memory!
        self.assertEqual(decision.owner, PrioritySource.EXPLICIT_USER_GAZE)
        self.assertAlmostEqual(decision.target_yaw_deg, 22.0)
        self.assertEqual(decision.reason, "EXPLICIT_SPATIAL_MEMORY_PERSON")


if __name__ == '__main__':
    unittest.main()
