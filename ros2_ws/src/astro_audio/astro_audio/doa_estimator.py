#!/usr/bin/env python3
"""ASTRO V1 — Acoustic Direction of Arrival (DOA) Estimator.

Implements multi-channel acoustic DOA estimation for ReSpeaker 4-Mic USB Array
mounted vertically on ASTRO:
  - Board X axis = robot LEFT/RIGHT (raw ch1 <-> raw ch3 horizontal pair, baseline = 0.064m)
  - Board Y axis = robot UP/DOWN     (raw ch2 <-> raw ch4 vertical pair, baseline = 0.064m)
  - Horizontal angle estimated via arcsin(-tau * c / baseline) in range [-90.0°, +90.0°]
    where positive angle = robot RIGHT, negative angle = robot LEFT.
"""

import math
from typing import Optional, Tuple
import numpy as np


class ReSpeakerGeometry:
    """Microphone array geometry for ReSpeaker 4-Mic USB Array on ASTRO.

    Mounted vertically on ASTRO:
      - Board X axis = robot LEFT/RIGHT:
          raw ch1 = ODAS mic2 = (-0.032,  0.000, 0) -> Left
          raw ch3 = ODAS mic4 = (+0.032,  0.000, 0) -> Right
          ch1 <-> ch3 = HORIZONTAL LEFT/RIGHT pair (baseline = 0.064m)
      - Board Y axis = robot UP/DOWN:
          raw ch2 = ODAS mic3 = ( 0.000, -0.032, 0) -> Down
          raw ch4 = ODAS mic5 = ( 0.000, +0.032, 0) -> Up
          ch2 <-> ch4 = VERTICAL UP/DOWN pair (baseline = 0.064m)
    """
    HALF_SPACING_M = 0.032
    PAIR_DIST_M = 0.064
    SPEED_OF_SOUND_MPS = 343.0
    SAMPLE_RATE = 16000


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int = 16000,
    max_tau: Optional[float] = None,
    interp: int = 16
) -> Tuple[float, float]:
    """Computes Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
    
    Args:
        sig: First channel signal array
        refsig: Second channel signal array
        fs: Sampling rate in Hz
        max_tau: Maximum expected time delay in seconds (based on mic distance)
        interp: Interpolation factor for fractional-sample resolution
        
    Returns:
        tau: Time delay in seconds (positive if refsig lags sig, negative if refsig leads sig)
        quality: Normalized peak quality (confidence proxy, 0..1)
    """
    if sig is None or refsig is None or sig.size == 0 or refsig.size == 0:
        return 0.0, 0.0

    # Ensure inputs are finite arrays without modifying normal clean signals
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(refsig)):
        refsig = np.nan_to_num(refsig, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross-Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # Phase Transform weighting: 1 / |R| restricted to speech band (300 Hz <= f <= 3400 Hz)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    speech_mask = (freqs >= 300.0) & (freqs <= 3400.0)

    denom = np.abs(R)
    denom[~np.isfinite(denom)] = 1e-6
    denom[denom < 1e-6] = 1e-6

    R_phat = np.zeros_like(R, dtype=np.complex128)
    R_phat[speech_mask] = R[speech_mask] / denom[speech_mask]
    R_phat[~np.isfinite(R_phat)] = 0.0

    # Inverse FFT with interpolation for sub-sample precision
    cc = np.fft.irfft(R_phat, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(interp * n / 2)

    # Shift zero lag to center
    cc_windowed = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    
    # Peak index: positive peak argmax
    shift = max_shift - int(np.argmax(cc_windowed))
    tau = shift / float(interp * fs)

    # Calculate Peak-to-Sidelobe Ratio / normalized peak quality
    idx = int(np.argmax(cc_windowed))
    peak_val = float(cc_windowed[idx])
    mask_sidelobes = np.ones(len(cc_windowed), dtype=bool)
    mainlobe_halfwidth = interp
    mask_sidelobes[max(0, idx - mainlobe_halfwidth): min(len(cc_windowed), idx + mainlobe_halfwidth + 1)] = False

    if np.any(mask_sidelobes):
        sidelobes = np.abs(cc_windowed[mask_sidelobes])
        mean_val = float(np.mean(sidelobes))
        std_val = float(np.std(sidelobes))
    else:
        mean_val = float(np.mean(np.abs(cc_windowed)))
        std_val = float(np.std(np.abs(cc_windowed)))

    psr = (peak_val - mean_val) / max(1e-5, std_val)
    quality = min(1.0, max(0.0, (psr - 1.5) / 5.0))

    return tau, quality


class AcousticDOAEstimator:
    """Estimates acoustic sound azimuth from ReSpeaker multi-channel audio frames.

    Mounted vertically on ASTRO:
      - Horizontal pair: raw ch1 (Left) <-> raw ch3 (Right)
      - Vertical pair: raw ch2 (Down) <-> raw ch4 (Up)
      - Azimuth angle is computed solely from horizontal TDOA via asin(x), [-90.0°, +90.0°].
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_energy_threshold: float = 80.0,
        min_confidence: float = 0.40,
    ):
        self.sample_rate = sample_rate
        self.min_energy_threshold = min_energy_threshold
        self.min_confidence = min_confidence
        self.baseline = ReSpeakerGeometry.PAIR_DIST_M  # 0.064m
        self.max_tau = self.baseline / ReSpeakerGeometry.SPEED_OF_SOUND_MPS  # ~0.187ms

    def estimate_from_multichannel_pcm(
        self,
        pcm_channels: np.ndarray,
    ) -> Tuple[Optional[float], float, bool]:
        """Estimates sound horizontal direction from multi-channel audio buffer.

        Physical Model:
          Full 6-channel input:
            ch0 = processed audio
            ch1 = raw mic (ODAS mic2: Left,  -0.032m)
            ch2 = raw mic (ODAS mic3: Down,  -0.032m)
            ch3 = raw mic (ODAS mic4: Right, +0.032m)
            ch4 = raw mic (ODAS mic5: Up,    +0.032m)
            ch5 = playback
            raw = pcm_channels[1:5]
          4-channel raw input:
            raw = pcm_channels[:4]

          Horizontal pair:
            mic_horizontal_a = raw[0]  # ch1 (Left)
            mic_horizontal_b = raw[2]  # ch3 (Right)

        Returns:
            azimuth_deg: Estimated angle in degrees [-90.0°..+90.0°] (0°=center, +=right, -=left) or None
            confidence: Estimation confidence (0.0..1.0)
            valid: True if confidence meets threshold and energy is sufficient
        """
        if pcm_channels is None or pcm_channels.size == 0:
            return None, 0.0, False

        # Ensure shape is (channels, samples)
        if pcm_channels.ndim == 2:
            if pcm_channels.shape[0] > pcm_channels.shape[1]:
                pcm_channels = pcm_channels.T

        if pcm_channels.shape[0] < 4:
            return None, 0.0, False

        # Slicing: 6-channel vs 4-channel raw input
        if pcm_channels.shape[0] >= 6:
            raw = pcm_channels[1:5]
        else:
            raw = pcm_channels[:4]

        # Convert to float64 before energy calculation to prevent float32 square overflow
        raw = raw.astype(np.float64, copy=False)
        if not np.all(np.isfinite(raw)):
            raw = np.nan_to_num(raw, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        # Check frame energy (RMS) over raw mic channels (ch1..ch4) in float64
        mean_sq = np.mean(raw ** 2)
        rms = float(np.sqrt(mean_sq)) if np.isfinite(mean_sq) and mean_sq > 0.0 else 0.0
        if rms < self.min_energy_threshold:
            return None, 0.0, False

        # Horizontal pair: ch1 (Left) <-> ch3 (Right)
        mic_horizontal_a = raw[0]   # ch1
        mic_horizontal_b = raw[2]   # ch3

        tau_horizontal, q_horizontal = gcc_phat(
            mic_horizontal_a,
            mic_horizontal_b,
            fs=self.sample_rate,
            max_tau=self.max_tau,
        )

        # Vertical pair: raw[1] (Down) <-> raw[3] (Up) - never used in azimuth calculation
        mic_vertical_a = raw[1]   # ch2
        mic_vertical_b = raw[3]   # ch4
        tau_vertical, q_vertical = gcc_phat(
            mic_vertical_a,
            mic_vertical_b,
            fs=self.sample_rate,
            max_tau=self.max_tau,
        )

        # Horizontal angle calculation:
        # x = -tau_horizontal * c / baseline
        # Positive angle = robot RIGHT, Negative angle = robot LEFT
        x = -tau_horizontal * ReSpeakerGeometry.SPEED_OF_SOUND_MPS / self.baseline
        x = float(np.clip(x, -1.0, 1.0))
        raw_azimuth = math.degrees(math.asin(x))

        # Confidence: primary source is q_horizontal, with energy factor
        energy_factor = min(1.0, max(0.0, rms / 1500.0))
        confidence = round(float(q_horizontal * 0.7 + energy_factor * 0.3), 2)

        # Small consistency penalty if vertical TDOA exceeds physically possible baseline
        if abs(tau_vertical) * ReSpeakerGeometry.SPEED_OF_SOUND_MPS / self.baseline > 1.05:
            confidence = round(float(confidence * 0.9), 2)

        is_valid = (confidence >= self.min_confidence)
        if not is_valid:
            return None, confidence, False

        azimuth_deg = round(float(raw_azimuth), 1)
        return azimuth_deg, confidence, True
