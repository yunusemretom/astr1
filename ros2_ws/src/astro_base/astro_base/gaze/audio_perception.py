"""Pure Audio Perception Core for ASTRO Gaze System.

Computes Direction of Arrival (DOA) from 4-channel raw audio using:
  1. Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
  2. Fractional-sample sub-sample interpolation
  3. Orthogonal mic-pair TDOA (Left-Right and Front-Back)
  4. Peak-to-Sidelobe Ratio (PSR) confidence scoring
  5. Adaptive background noise floor & dynamic VAD gating
  6. Self-speech suppression during robot TTS speech / playback
"""

import math
from typing import Optional, Tuple
import numpy as np

from astro_base.gaze.angle_math import wrap_deg
from astro_base.gaze.coordinate_frames import CoordinateTransformer
from astro_base.gaze.types import AudioEventCounters, AudioObservation


class ReSpeaker4MicGeometry:
    """Acoustic physical constants for ReSpeaker 4-Mic USB Array."""
    RADIUS_M = 0.043
    PAIR_DIST_M = 2.0 * RADIUS_M  # 0.086m between opposing mic pairs (0-2 and 1-3)
    SPEED_OF_SOUND_MPS = 343.0     # Speed of sound at 20°C in m/s
    SAMPLE_RATE_HZ = 16000
    MAX_TAU_S = PAIR_DIST_M / SPEED_OF_SOUND_MPS  # ~0.2507 ms (4.01 samples)


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int = 16000,
    max_tau: Optional[float] = None,
    interp: int = 16
) -> Tuple[float, float]:
    """Computes Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

    Returns:
      (tau_s, quality_0_to_1)
      - tau_s: Time delay in seconds (positive if refsig lags sig, negative if leads)
      - quality: Peak-to-Sidelobe Ratio normalized confidence proxy [0.0..1.0]
    """
    n = sig.shape[0] + refsig.shape[0]

    # Fast Fourier Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # Phase Transform (PHAT) whitening: 1 / |R|
    denom = np.abs(R)
    denom[denom < 1e-6] = 1e-6
    R_phat = R / denom

    # Inverse FFT with sub-sample interpolation
    cc = np.fft.irfft(R_phat, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(interp * n / 2)

    # Shift zero lag to center
    cc_windowed = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # Find peak index.
    #
    # İşaret: R = SIG·conj(REFSIG) ters dönüştürülünce, refsig sig'den D kadar
    # gecikmişse tepe -D'de çıkar. Docstring +D vaat ediyor ve aşağıdaki
    # `delta_x = -tau·c` satırları o vaade göre yazılmış; ikisi ayrışınca açı 180°
    # sabit hatayla, yani konuşanın tam tersine çıkıyordu. Aynı hatanın ikinci
    # kopyası astro_audio/doa_estimator.py içinde — ikisi birlikte düzeltildi.
    peak_idx = int(np.argmax(np.abs(cc_windowed)))
    shift = max_shift - peak_idx
    tau = shift / float(interp * fs)

    # Calculate Peak-to-Sidelobe Ratio (PSR)
    peak_val = float(np.max(np.abs(cc_windowed)))
    mean_val = float(np.mean(np.abs(cc_windowed)))
    std_val = float(np.std(np.abs(cc_windowed)))
    psr = (peak_val - mean_val) / max(1e-5, std_val)

    # Map PSR (typically 1.5 to 7.0) to [0.0, 1.0]
    quality = min(1.0, max(0.0, (psr - 1.5) / 5.0))

    return tau, quality


class PersistentBearingGate:
    """Admits a bearing only once it has repeated from roughly the same direction.

    Reverberation off a wall arrives from a scattered direction each time; a person
    talking stays put. Counting consecutive consistent reports separates them, which
    is what makes it safe to listen behind the robot at all — the rear envelope was
    closed precisely because single reflections used to drive the neck into its
    mechanical limit.
    """

    def __init__(
        self,
        required_hits: int = 4,
        tolerance_deg: float = 20.0,
        expiry_s: float = 1.5,
    ):
        self.required_hits = required_hits
        self.tolerance_deg = tolerance_deg
        self.expiry_s = expiry_s
        self._bearing: Optional[float] = None
        self._hits: int = 0
        self._last_time: float = 0.0

    def reset(self) -> None:
        self._bearing = None
        self._hits = 0
        self._last_time = 0.0

    def admits(self, bearing_deg: float, timestamp: float) -> bool:
        """Counts this report and says whether the direction has earned a turn yet."""
        bearing = wrap_deg(bearing_deg)
        expired = (self._bearing is None) or (timestamp - self._last_time > self.expiry_s)
        consistent = (
            self._bearing is not None
            and abs(wrap_deg(bearing - self._bearing)) <= self.tolerance_deg
        )

        if expired or not consistent:
            self._bearing = bearing
            self._hits = 1
        else:
            # Track the source as it drifts rather than pinning the first report.
            self._bearing = wrap_deg(self._bearing + 0.5 * wrap_deg(bearing - self._bearing))
            self._hits += 1

        self._last_time = timestamp
        return self._hits >= self.required_hits


class AudioPerceptionCore:
    """Core acoustic processor computing directional observations and energy features."""

    def __init__(
        self,
        transformer: Optional[CoordinateTransformer] = None,
        min_rms_energy: float = 350.0,
        vad_threshold: float = 450.0,
        min_confidence: float = 0.40,
        self_speech_suppression_factor: float = 0.15,
        # Sohbet konisi: burada ses anında kabul edilir.
        max_acoustic_envelope_deg: float = 75.0,
        # Kafa limiti (85) + kamera yarı-FOV (36): bir kaynağın kafayı çevirerek
        # kadraja sokulabileceği en uzak açı. Ötesinde kafa dönse de kişi görünmez,
        # robot boşluğa bakıp hedefi kaybeder — düzeltilmeye çalışılan davranış bu.
        # Gerçek arka (121-180) gövdenin dönmesini ister; bu yığında yok.
        max_reachable_bearing_deg: float = 121.0,
    ):
        self.transformer = transformer or CoordinateTransformer()
        self.min_rms_energy = min_rms_energy
        self.vad_threshold = vad_threshold
        self.min_confidence = min_confidence
        self.self_speech_suppression_factor = self_speech_suppression_factor
        self.max_acoustic_envelope_deg = max_acoustic_envelope_deg
        self.max_reachable_bearing_deg = max_reachable_bearing_deg
        # Sohbet konisinin dışı ısrar ister; içi eskisi gibi anında geçer.
        self.rear_gate = PersistentBearingGate()

        self.noise_floor_rms = 120.0
        self.max_tau = ReSpeaker4MicGeometry.MAX_TAU_S
        self.counters = AudioEventCounters()

    def _bearing_is_admissible(
        self, rel_bearing: float, body_yaw: float, timestamp: float
    ) -> bool:
        """Conversation cone passes at once; the reachable margin must persist.

        Persistence is tracked in the body frame, not head-relative: a person standing
        behind the robot keeps the same body bearing while the head sweeps around.
        """
        if abs(rel_bearing) <= self.max_acoustic_envelope_deg:
            return True
        if abs(rel_bearing) > self.max_reachable_bearing_deg:
            return False
        return self.rear_gate.admits(body_yaw, timestamp)

    def process_frame(
        self,
        pcm_channels: np.ndarray,
        timestamp: float,
        actual_head_yaw_deg: float = 0.0,
        is_robot_speaking: bool = False,
        is_playback_active: bool = False,
    ) -> AudioObservation:
        """Processes 4-channel audio frame and produces a directional observation.

        Args:
          pcm_channels: (4, N) or (N, 4) numpy array of int16/float32 PCM samples:
                        Ch 0: Front, Ch 1: Right, Ch 2: Back, Ch 3: Left
          timestamp: Current monotonic timestamp
          actual_head_yaw_deg: Current physical head yaw angle (degrees)
          is_robot_speaking: True if TTS engine is actively speaking
          is_playback_active: True if audio playback is active
        """
        if pcm_channels is None or pcm_channels.size == 0:
            return AudioObservation(timestamp=timestamp, valid=False, vad=False)

        # Normalize array orientation to (4, N)
        if pcm_channels.ndim == 2 and pcm_channels.shape[0] > pcm_channels.shape[1]:
            pcm_channels = pcm_channels.T

        if pcm_channels.shape[0] < 4:
            return AudioObservation(timestamp=timestamp, valid=False, vad=False)

        float_pcm = pcm_channels.astype(np.float32)
        mono_mix = np.mean(float_pcm, axis=0)

        # 1. Compute acoustic energy features
        rms = float(np.sqrt(np.mean(mono_mix ** 2)))
        peak = float(np.max(np.abs(mono_mix)))

        # 2. Adaptive background noise tracking
        if not is_robot_speaking and not is_playback_active and rms < 300.0:
            self.noise_floor_rms = 0.96 * self.noise_floor_rms + 0.04 * rms

        # 3. Dynamic Voice Activity Detection (VAD)
        dynamic_vad_thresh = max(self.vad_threshold, self.noise_floor_rms * 2.5)
        vad_active = (rms >= dynamic_vad_thresh) and (peak >= 900.0)

        # Signal-to-noise ratio in dB
        snr_db = 20.0 * math.log10(max(1.0, rms) / max(1.0, self.noise_floor_rms))

        # Check basic energy gate
        if rms < self.min_rms_energy or not vad_active:
            return AudioObservation(
                timestamp=timestamp,
                valid=False,
                vad=vad_active,
                rms=rms,
                peak=peak,
                snr_db=snr_db,
                confidence=0.0
            )

        mic_front = float_pcm[0]
        mic_right = float_pcm[1]
        mic_back = float_pcm[2]
        mic_left = float_pcm[3]

        # 4. TDOA Computation via GCC-PHAT
        # Left-Right Pair: Mic 3 (Left) vs Mic 1 (Right)
        tau_lr, q_lr = gcc_phat(
            mic_left, mic_right,
            fs=ReSpeaker4MicGeometry.SAMPLE_RATE_HZ,
            max_tau=self.max_tau
        )

        # Front-Back Pair: Mic 2 (Back) vs Mic 0 (Front)
        tau_fb, q_fb = gcc_phat(
            mic_back, mic_front,
            fs=ReSpeaker4MicGeometry.SAMPLE_RATE_HZ,
            max_tau=self.max_tau
        )

        # 5. Spatial angle resolution
        # delta_x > 0 when sound is on the right (+X in robot plane)
        # delta_y > 0 when sound is in front (+Y in robot plane)
        delta_x = -tau_lr * ReSpeaker4MicGeometry.SPEED_OF_SOUND_MPS
        delta_y = -tau_fb * ReSpeaker4MicGeometry.SPEED_OF_SOUND_MPS
        if abs(delta_x) < 1e-7:
            delta_x = 0.0
        if abs(delta_y) < 1e-7:
            delta_y = 0.0

        # Azimuth angle in ReSpeaker circular frame [0..359°]
        raw_azimuth = math.degrees(math.atan2(delta_x, delta_y))
        raw_azimuth_0_360 = (raw_azimuth + 360.0) % 360.0

        # Quality & Confidence calculation
        avg_quality = (q_lr + q_fb) / 2.0
        energy_factor = min(1.0, max(0.1, (rms - self.min_rms_energy) / 1200.0))
        confidence = float(min(1.0, max(0.0, 0.65 * avg_quality + 0.35 * energy_factor)))

        # 6. Self-Speech Suppression
        self.counters.raw_audio_events += 1
        if is_robot_speaking or is_playback_active:
            self.counters.stale_audio_events += 1
            confidence *= self.self_speech_suppression_factor

        # 7. Coordinate transformations
        rel_bearing = self.transformer.raw_audio_doa_to_head_bearing(raw_azimuth_0_360)
        body_yaw = self.transformer.audio_head_bearing_to_body_yaw(rel_bearing, actual_head_yaw_deg)

        # Strict conversational acoustic envelope gate (±75° relative to head)
        # Sohbet konisi anında geçer; dışı, yankıyı gerçek konuşmacıdan ayırmak
        # için ısrar ister (bkz. _bearing_is_admissible).
        in_acoustic_fov = self._bearing_is_admissible(rel_bearing, body_yaw, timestamp)
        if not in_acoustic_fov:
            self.counters.invalid_angle_events += 1
            confidence = 0.0

        is_valid = (confidence >= self.min_confidence) and (not is_robot_speaking) and (not is_playback_active) and in_acoustic_fov
        if is_valid:
            self.counters.accepted_audio_events += 1
        else:
            self.counters.rejected_audio_events += 1

        return AudioObservation(
            timestamp=timestamp,
            valid=is_valid,
            vad=vad_active and in_acoustic_fov,
            raw_azimuth_deg=round(raw_azimuth_0_360, 1),
            relative_azimuth_deg=round(rel_bearing, 1),
            body_azimuth_deg=round(body_yaw, 1),
            elevation_deg=0.0,
            confidence=round(confidence, 2),
            rms=round(rms, 1),
            peak=round(peak, 1),
            snr_db=round(snr_db, 1),
        )

    def process_raw_doa(
        self,
        raw_doa_deg: float,
        timestamp: float,
        actual_head_yaw_deg: float = 0.0,
        confidence: float = 0.85,
        is_robot_speaking: bool = False,
    ) -> AudioObservation:
        """Processes a pre-calculated raw DOA angle (e.g. from ReSpeaker onboard DSP)."""
        self.counters.raw_audio_events += 1
        rel_bearing = self.transformer.raw_audio_doa_to_head_bearing(raw_doa_deg)
        body_yaw = self.transformer.audio_head_bearing_to_body_yaw(rel_bearing, actual_head_yaw_deg)

        eff_conf = confidence
        if is_robot_speaking:
            self.counters.stale_audio_events += 1
            eff_conf *= self.self_speech_suppression_factor

        # Reject rear acoustic blindspot reflections (> 75° relative to head) so neck never slams to mechanical limits
        # Sohbet konisi anında geçer; dışı, yankıyı gerçek konuşmacıdan ayırmak
        # için ısrar ister (bkz. _bearing_is_admissible).
        in_acoustic_fov = self._bearing_is_admissible(rel_bearing, body_yaw, timestamp)
        if not in_acoustic_fov:
            self.counters.invalid_angle_events += 1
            eff_conf = 0.0

        is_valid = (eff_conf >= self.min_confidence) and (not is_robot_speaking) and in_acoustic_fov
        if is_valid:
            self.counters.accepted_audio_events += 1
        else:
            self.counters.rejected_audio_events += 1

        return AudioObservation(
            timestamp=timestamp,
            valid=is_valid,
            vad=is_valid,
            raw_azimuth_deg=round(raw_doa_deg, 1),
            relative_azimuth_deg=round(rel_bearing, 1),
            body_azimuth_deg=round(body_yaw, 1),
            elevation_deg=0.0,
            confidence=round(eff_conf, 2),
            rms=1500.0 if is_valid else 100.0,
            peak=3000.0 if is_valid else 200.0,
            snr_db=15.0 if is_valid else 0.0,
        )

