"""Audio-Visual Sensor Fusion Engine.

Implements:
  1. Spatial Consistency Gating between acoustic DOA and visual face tracks
  2. Temporal Freshness-Decay Confidence Weighting
  3. Dynamic Multimodal Fusion (Fused, Vision-Only, Audio-Only fallback modes)
  4. Association Scoring and Multi-Candidate Generation
"""

import math
from typing import List, Optional, Tuple

from astro_base.gaze.angle_math import (
    circular_distance_deg,
    circular_mean_deg,
    wrap_deg,
)
from astro_base.gaze.spatial_memory import EpistemicSpatialMemory
from astro_base.gaze.types import (
    FilteredAudioState,
    FusedTarget,
    Modality,
    TrackingState,
    VisualTargetTrack,
)


class AudioVisualFusionCore:
    """Multimodal Sensor Fusion combining acoustic sound localization and 3D visual face tracking."""

    def __init__(
        self,
        spatial_gate_deg: float = 25.0,
        audio_freshness_half_life_s: float = 0.40,
        vision_freshness_half_life_s: float = 1.20,
        audio_weight_base: float = 0.40,
        vision_weight_base: float = 1.0,
        fallback_enabled: bool = True,
        spatial_memory: Optional[EpistemicSpatialMemory] = None,
    ):
        self.spatial_gate_deg = spatial_gate_deg
        self.audio_half_life = audio_freshness_half_life_s
        self.vision_half_life = vision_freshness_half_life_s
        self.audio_weight_base = audio_weight_base
        self.vision_weight_base = vision_weight_base
        self.fallback_enabled = fallback_enabled
        self.spatial_memory = spatial_memory or EpistemicSpatialMemory()

    def compute_freshness(self, elapsed_s: float, half_life_s: float) -> float:
        """Calculates exponential temporal freshness weight [0.0..1.0]."""
        if elapsed_s <= 0.0:
            return 1.0
        decay_constant = math.log(2.0) / max(0.01, half_life_s)
        return float(math.exp(-decay_constant * elapsed_s))

    def fuse(
        self,
        audio_state: Optional[FilteredAudioState],
        visual_tracks: List[VisualTargetTrack],
        timestamp: float,
    ) -> List[FusedTarget]:
        """Fuses audio perception and visual tracks into unified FusedTargets.

        Association Strategy:
          - Spatial Consistency: Compares audio azimuth with visual track azimuth.
          - If within spatial_gate_deg (e.g. ≤25°), sound is associated with the visual face -> FUSED.
          - Unassociated visual tracks -> VISION modality (silent person).
          - Unassociated audio sound -> AUDIO modality (candidate speaker outside FOV / coarse cue).
        """
        fused_targets: List[FusedTarget] = []
        matched_audio = False

        has_valid_audio = (
            audio_state is not None
            and audio_state.valid
            and audio_state.confidence > 0.10
        )

        audio_freshness = 0.0
        audio_eff_conf = 0.0
        if has_valid_audio:
            # Epistemic Reverb Suppression: Reject acoustic bearings proven to be empty walls
            if self.spatial_memory.is_acoustic_reverb_zone(audio_state.azimuth_deg, timestamp):
                has_valid_audio = False
            else:
                elapsed_aud = max(0.0, timestamp - audio_state.timestamp)
                audio_freshness = self.compute_freshness(elapsed_aud, self.audio_half_life)
                audio_eff_conf = audio_state.confidence * audio_freshness
                if audio_eff_conf < 0.35 or audio_freshness < 0.35:
                    has_valid_audio = False

        # 1. Process all visual tracks and associate matching sound
        for vt in visual_tracks:
            elapsed_vis = max(0.0, timestamp - vt.last_seen_time)
            vis_freshness = self.compute_freshness(elapsed_vis, self.vision_half_life)
            vis_eff_conf = vt.confidence * vis_freshness * self.vision_weight_base

            # Register/update confirmed person in epistemic spatial memory
            if vt.confidence >= 0.40:
                self.spatial_memory.register_person_observation(
                    person_id=vt.target_id,
                    bearing_deg=vt.body_azimuth_deg,
                    confidence=vt.confidence,
                    timestamp=timestamp,
                    distance_m=vt.distance_m,
                    person_name=vt.person_name,
                )

            # Check spatial consistency with audio
            is_associated = False
            if has_valid_audio and not matched_audio:
                dist_deg = circular_distance_deg(audio_state.azimuth_deg, vt.body_azimuth_deg)
                if dist_deg <= self.spatial_gate_deg:
                    is_associated = True
                    matched_audio = True

            if is_associated:
                # FUSED target: Face + Active Voice.
                # Vision owns the angle; audio only says that this face is the one
                # talking. Averaging the two used to drag the aim off a motionless
                # face — a DOA at +45 pulled a face held at +20 out to +25.1, past
                # the 3 degree gaze deadband, so acoustic noise alone kept commanding
                # head motion. This project's own validation puts visual bearing
                # error at 0.40 degrees against 3.23 for filtered DOA, so blending
                # the coarse source in could only ever degrade the precise one.
                fused_yaw_val = vt.body_azimuth_deg

                # Combined confidence
                combined_conf = min(1.0, vis_eff_conf + 0.30 * audio_eff_conf)

                fused_targets.append(
                    FusedTarget(
                        target_id=vt.target_id,
                        modality=Modality.FUSED,
                        body_azimuth_deg=round(fused_yaw_val, 1),
                        body_elevation_deg=vt.body_elevation_deg,
                        distance_m=vt.distance_m,
                        confidence=round(combined_conf, 2),
                        is_speaking=True,
                        eye_contact=vt.eye_contact,
                        person_name=vt.person_name,
                        is_known=vt.is_known,
                        timestamp=timestamp,
                        tracking_state=vt.tracking_state,
                        audio_confidence=round(audio_state.confidence, 2),
                        visual_confidence=round(vt.confidence, 2),
                    )
                )
            else:
                # VISION-Only target: Face detected, but not currently speaking
                fused_targets.append(
                    FusedTarget(
                        target_id=vt.target_id,
                        modality=Modality.VISION,
                        body_azimuth_deg=vt.body_azimuth_deg,
                        body_elevation_deg=vt.body_elevation_deg,
                        distance_m=vt.distance_m,
                        confidence=round(vis_eff_conf, 2),
                        is_speaking=False,
                        eye_contact=vt.eye_contact,
                        person_name=vt.person_name,
                        is_known=vt.is_known,
                        timestamp=timestamp,
                        tracking_state=vt.tracking_state,
                        audio_confidence=0.0,
                        visual_confidence=round(vt.confidence, 2),
                    )
                )

        # 2. If valid audio was not associated to any existing visual face -> Create AUDIO target
        if has_valid_audio and not matched_audio and self.fallback_enabled:
            # Epistemic Grounding: Check if spatial memory knows a person nearby (within 35°)
            known_person_yaw = self.spatial_memory.get_most_likely_person_location(timestamp, max_age_s=15.0)
            target_bearing = audio_state.azimuth_deg
            if known_person_yaw is not None:
                if circular_distance_deg(audio_state.azimuth_deg, known_person_yaw) <= 35.0:
                    target_bearing = known_person_yaw

            # Sound from outside camera FOV or face temporarily occluded
            aud_target = FusedTarget(
                target_id="audio_speaker_1",
                modality=Modality.AUDIO,
                body_azimuth_deg=target_bearing,
                body_elevation_deg=0.0,
                distance_m=1.8,  # Default estimated social distance
                confidence=round(audio_eff_conf, 2),
                is_speaking=True,
                eye_contact=False,
                person_name=None,
                is_known=False,
                timestamp=timestamp,
                tracking_state=TrackingState.TRACKING,
                audio_confidence=round(audio_state.confidence, 2),
                visual_confidence=0.0,
            )
            fused_targets.append(aud_target)

        # Sort candidate targets: Fused > Vision > Audio, and by confidence
        def _sort_key(t: FusedTarget) -> Tuple[int, float]:
            mod_priority = {Modality.FUSED: 3, Modality.VISION: 2, Modality.AUDIO: 1, Modality.NONE: 0}
            return (mod_priority.get(t.modality, 0), t.confidence)

        fused_targets.sort(key=_sort_key, reverse=True)
        return fused_targets
