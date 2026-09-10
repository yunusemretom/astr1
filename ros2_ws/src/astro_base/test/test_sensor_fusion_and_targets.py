"""Unit tests for Audio-Visual Sensor Fusion and Target Management Core."""

import unittest
from astro_base.gaze.sensor_fusion import AudioVisualFusionCore
from astro_base.gaze.target_manager import TargetManagerCore
from astro_base.gaze.types import (
    FilteredAudioState,
    FusedTarget,
    Modality,
    TrackingState,
    VisualTargetTrack,
)


class TestAudioVisualFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = AudioVisualFusionCore(
            spatial_gate_deg=25.0,
            audio_freshness_half_life_s=0.80,
            vision_freshness_half_life_s=1.20,
        )

    def test_spatial_consistency_fused_target(self):
        """Audio at 35° and Vision at 33° (within 25° gate) fuse into a single FUSED target."""
        t = 1.0
        audio = FilteredAudioState(
            timestamp=t, valid=True, azimuth_deg=35.0, confidence=0.80
        )
        visual = [
            VisualTargetTrack(
                target_id="person_1", pos_3d=(1.0, 0.6, 0.0), vel_3d=(0.0, 0.0, 0.0),
                body_azimuth_deg=33.0, body_elevation_deg=0.0, distance_m=1.17,
                confidence=0.85, tracking_state=TrackingState.TRACKING,
                last_seen_time=t
            )
        ]

        fused = self.fusion.fuse(audio, visual, timestamp=t)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].modality, Modality.FUSED)
        self.assertTrue(fused[0].is_speaking)
        self.assertAlmostEqual(fused[0].body_azimuth_deg, 33.5, delta=1.0)
        self.assertGreater(fused[0].confidence, 0.70)

    def test_spatial_separation_distinct_targets(self):
        """Audio at +70° and Vision at -20° (>25° gate) remain separate candidate targets."""
        t = 1.0
        audio = FilteredAudioState(
            timestamp=t, valid=True, azimuth_deg=70.0, confidence=0.80
        )
        visual = [
            VisualTargetTrack(
                target_id="person_1", pos_3d=(1.5, -0.5, 0.0), vel_3d=(0.0, 0.0, 0.0),
                body_azimuth_deg=-20.0, body_elevation_deg=0.0, distance_m=1.58,
                confidence=0.85, tracking_state=TrackingState.TRACKING,
                last_seen_time=t
            )
        ]

        fused = self.fusion.fuse(audio, visual, timestamp=t)
        self.assertEqual(len(fused), 2)
        # One Vision target (silent person) and one Audio target (off-camera speaker)
        modalities = {f.modality for f in fused}
        self.assertIn(Modality.VISION, modalities)
        self.assertIn(Modality.AUDIO, modalities)


class TestTargetManager(unittest.TestCase):
    def setUp(self):
        self.tm = TargetManagerCore(
            acquisition_threshold=0.75,
            hold_threshold=0.40,
            target_lost_timeout_s=1.0,
            min_attention_dwell_s=2.50,
            turn_taking_min_dwell_s=0.80,
            turn_taking_min_angle_deg=20.0,
        )

    def test_hysteresis_acquisition_and_hold(self):
        """Test target requires ≥0.75 to acquire, but holds down to ≥0.40."""
        t = 1.0

        # 1. Target with 0.70 confidence -> Does NOT acquire
        cand1 = [
            FusedTarget(
                target_id="person_1", modality=Modality.VISION, body_azimuth_deg=20.0,
                body_elevation_deg=0.0, distance_m=1.5, confidence=0.70,
                is_speaking=False, eye_contact=False, person_name=None, is_known=False,
                timestamp=t, tracking_state=TrackingState.TRACKING
            )
        ]
        state1 = self.tm.update(cand1, timestamp=t)
        self.assertIsNone(state1.active_target)

        # 2. Target reaches 0.80 confidence -> Acquires!
        cand2 = [
            FusedTarget(
                target_id="person_1", modality=Modality.FUSED, body_azimuth_deg=20.0,
                body_elevation_deg=0.0, distance_m=1.5, confidence=0.80,
                is_speaking=True, eye_contact=True, person_name=None, is_known=False,
                timestamp=t, tracking_state=TrackingState.TRACKING
            )
        ]
        state2 = self.tm.update(cand2, timestamp=t)
        self.assertIsNotNone(state2.active_target)
        self.assertEqual(state2.active_target.target_id, "person_1")

        # 3. Target confidence drops to 0.50 (above hold 0.40) -> Remains active!
        t += 0.5
        cand3 = [
            FusedTarget(
                target_id="person_1", modality=Modality.VISION, body_azimuth_deg=20.0,
                body_elevation_deg=0.0, distance_m=1.5, confidence=0.50,
                is_speaking=False, eye_contact=False, person_name=None, is_known=False,
                timestamp=t, tracking_state=TrackingState.TRACKING
            )
        ]
        state3 = self.tm.update(cand3, timestamp=t)
        self.assertIsNotNone(state3.active_target)
        self.assertEqual(state3.active_target.target_id, "person_1")

        # 4. Target confidence drops to 0.20 for > 1.0s -> Target is dropped / cleared
        t += 1.2
        cand4 = [
            FusedTarget(
                target_id="person_1", modality=Modality.VISION, body_azimuth_deg=20.0,
                body_elevation_deg=0.0, distance_m=1.5, confidence=0.20,
                is_speaking=False, eye_contact=False, person_name=None, is_known=False,
                timestamp=t, tracking_state=TrackingState.TRACKING
            )
        ]
        state4 = self.tm.update(cand4, timestamp=t)
        self.assertIsNone(state4.active_target)

    def test_turn_taking_speaker_switch(self):
        """Test active speaker switch when a new speaker speaks from a distinct angle (>20°) for >0.8s."""
        t = 1.0

        # Initial active speaker: Speaker A at 10°
        spk_a = FusedTarget(
            target_id="spk_a", modality=Modality.FUSED, body_azimuth_deg=10.0,
            body_elevation_deg=0.0, distance_m=1.5, confidence=0.85,
            is_speaking=True, eye_contact=True, person_name="Alice", is_known=True,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )
        self.tm.update([spk_a], timestamp=t)
        self.assertEqual(self.tm.active_target.target_id, "spk_a")

        # New Speaker B starts speaking at 45° (separation = 35° > 20°)
        spk_b = FusedTarget(
            target_id="spk_b", modality=Modality.FUSED, body_azimuth_deg=45.0,
            body_elevation_deg=0.0, distance_m=1.8, confidence=0.85,
            is_speaking=True, eye_contact=True, person_name="Bob", is_known=True,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

        # After 0.3s (less than turn_taking_min_dwell 0.8s) -> stays on Speaker A
        t += 0.3
        st_mid = self.tm.update([spk_a, spk_b], timestamp=t)
        self.assertEqual(st_mid.active_target.target_id, "spk_a")

        # After 0.9s of sustained speech from Speaker B -> switches to Speaker B!
        t += 0.9
        st_final = self.tm.update([spk_a, spk_b], timestamp=t)
        self.assertEqual(st_final.active_target.target_id, "spk_b")


    def _audio(self, conf, t, az=40.0):
        return FusedTarget(
            target_id="audio_speaker_1", modality=Modality.AUDIO, body_azimuth_deg=az,
            body_elevation_deg=0.0, distance_m=1.8, confidence=conf,
            is_speaking=True, eye_contact=False, person_name=None, is_known=False,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

    def _vision(self, conf, t, az=10.0, tid="person_1"):
        return FusedTarget(
            target_id=tid, modality=Modality.VISION, body_azimuth_deg=az,
            body_elevation_deg=0.0, distance_m=1.5, confidence=conf,
            is_speaking=False, eye_contact=False, person_name=None, is_known=False,
            timestamp=t, tracking_state=TrackingState.TRACKING
        )

    def test_sound_is_acquired_when_nothing_is_seen(self):
        """A voice with nobody visible must move the head, not be ignored.

        An audio-only target's confidence is DOA confidence times a freshness that
        halves every 0.4 s, so it effectively never clears the 0.75 visual bar. Held to
        that bar the robot sits still through a voice it can plainly hear — which is
        what the recording showed: audio present, owner=IDLE, head parked.
        """
        tm = TargetManagerCore()
        state = tm.update([self._audio(0.55, 1.0)], timestamp=1.0)

        self.assertIsNotNone(state.active_target)
        self.assertEqual(state.active_target.modality, Modality.AUDIO)

    def test_sound_below_the_audio_bar_is_still_ignored(self):
        """The lower bar is a bar, not an open door — faint bearings stay out."""
        tm = TargetManagerCore()
        state = tm.update([self._audio(0.30, 1.0)], timestamp=1.0)

        self.assertIsNone(state.active_target)

    def test_a_visible_face_still_wins_over_sound(self):
        """Sound is a fallback. It must not outrank a face that cleared the visual bar,
        or the head would turn away from someone it can see toward a reflection."""
        tm = TargetManagerCore()
        state = tm.update(
            [self._vision(0.85, 1.0, az=10.0), self._audio(0.60, 1.0, az=70.0)],
            timestamp=1.0,
        )

        self.assertIsNotNone(state.active_target)
        self.assertEqual(state.active_target.modality, Modality.VISION)

    def test_weak_face_does_not_block_the_audio_fallback(self):
        """A face below the visual bar leaves nothing acquired, so sound should still
        get its turn rather than the weak face suppressing it."""
        tm = TargetManagerCore()
        state = tm.update(
            [self._vision(0.50, 1.0, az=10.0), self._audio(0.60, 1.0, az=70.0)],
            timestamp=1.0,
        )

        self.assertIsNotNone(state.active_target)
        self.assertEqual(state.active_target.modality, Modality.AUDIO)


if __name__ == "__main__":
    unittest.main()
