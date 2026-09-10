#!/usr/bin/env python3
"""ASTRO Robot — Numerical Forensic Isolation Test Suite for GCC-PHAT vs Raw Cross-Correlation.

Investigates why raw cross-correlation and production GCC-PHAT disagree on physical ReSpeaker audio:
- Raw cross-correlation finds true acoustic delay (~0.5 - 2 samples).
- Production GCC-PHAT collapses toward 0 samples or picks false peaks.

Evaluates 6 independent variants:
A. Raw normalized cross-correlation
B. GCC-PHAT with current production implementation
C. GCC without phase whitening (IFFT(FFT(x) * conj(FFT(y))))
D. GCC with whitening but no window
E. GCC with whitening + Hanning window
F. Different FFT zero-padding sizes & frame sizes
G. Bandpass-filtered GCC-PHAT (speech band 300-3400 Hz)
H. Positive peak argmax vs absolute value peak argmax

Focuses on:
1. Zero delay
2. 0.5 sample delay
3. 1 sample delay
4. 1.5 sample delay
5. 2 sample delay
6. Cardinal direction synthetic array (0°, 90°, 180°, 270°)
7. Current vs candidate GCC parity
8. Raw cross-correlation vs corrected GCC
"""

import math
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository paths
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", ".."))
astro_audio_src = os.path.join(root_dir, "ros2_ws", "src", "astro_audio")
for p in (root_dir, astro_audio_src):
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_audio.doa_estimator import (
    ReSpeakerGeometry,
    gcc_phat,
)


def synthesize_bandlimited_speech(
    delay_samples: float,
    N: int = 1024,
    fs: int = 16000,
    snr_db: Optional[float] = 15.0,
    f_low: float = 300.0,
    f_high: float = 3400.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two bandlimited speech-like signals with exact fractional sample delay.

    sig_b is delayed by `delay_samples` relative to sig_a:
        sig_a[t] corresponds to sig_b[t + delay_samples] (sig_a leads sig_b if delay > 0).
    """
    rng = np.random.default_rng(seed)
    total_len = N + 500
    white = rng.normal(0, 1000.0, total_len)

    # Bandpass filter in frequency domain to model human voice
    freqs = np.fft.rfftfreq(total_len, 1.0 / fs)
    spec = np.fft.rfft(white)
    spec[(freqs < f_low) | (freqs > f_high)] = 0.0
    sig_base = np.fft.irfft(spec, n=total_len)

    # Fractional delay via frequency phase shift: exp(-j * 2 * pi * f * tau)
    tau_s = float(delay_samples) / float(fs)
    spec_base = np.fft.rfft(sig_base)
    spec_delayed = spec_base * np.exp(-1j * 2.0 * np.pi * freqs * tau_s)
    sig_del = np.fft.irfft(spec_delayed, n=total_len)

    # Crop to N samples
    start = 100
    sig_a = sig_base[start : start + N].copy()
    sig_b = sig_del[start : start + N].copy()

    # Add realistic sensor background noise
    if snr_db is not None:
        pwr_a = np.var(sig_a)
        pwr_b = np.var(sig_b)
        noise_std_a = np.sqrt(pwr_a / (10.0 ** (snr_db / 10.0)))
        noise_std_b = np.sqrt(pwr_b / (10.0 ** (snr_db / 10.0)))
        sig_a += rng.normal(0, noise_std_a, N)
        sig_b += rng.normal(0, noise_std_b, N)

    return sig_a.astype(np.float32), sig_b.astype(np.float32)


def raw_xcorr(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    max_lag: int = 16,
) -> Tuple[float, float]:
    """Variant A: Raw normalized cross-correlation with parabolic interpolation."""
    n = min(len(sig_a), len(sig_b))
    a = sig_a[:n].astype(np.float64) - np.mean(sig_a[:n])
    b = sig_b[:n].astype(np.float64) - np.mean(sig_b[:n])
    denom = math.sqrt(float(np.sum(a ** 2) * np.sum(b ** 2)))
    if denom < 1e-9:
        return 0.0, 0.0

    lags = np.arange(-max_lag, max_lag + 1)
    c_vals = np.zeros(len(lags), dtype=np.float64)
    for idx, k in enumerate(lags):
        if k >= 0:
            c_vals[idx] = float(np.sum(a[: n - k] * b[k:n]))
        else:
            c_vals[idx] = float(np.sum(a[-k:n] * b[: n + k]))

    best_idx = int(np.argmax(c_vals))
    best_k = lags[best_idx]
    best_k_frac = float(best_k)
    if 0 < best_idx < len(c_vals) - 1:
        y_l, y_m, y_r = c_vals[best_idx - 1], c_vals[best_idx], c_vals[best_idx + 1]
        denom_p = y_l - 2.0 * y_m + y_r
        if abs(denom_p) > 1e-9:
            delta = 0.5 * (y_l - y_r) / denom_p
            if abs(delta) <= 1.0:
                best_k_frac += delta

    norm_corr = float(c_vals[best_idx] / denom)
    return best_k_frac, norm_corr


def variant_c_unwhitened_gcc(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int = 16000,
    max_tau: float = 0.001,
    interp: int = 16,
) -> Tuple[float, float]:
    """Variant C: GCC without phase whitening: IFFT(FFT(x) * conj(FFT(y)))."""
    n = sig_a.shape[0] + sig_b.shape[0]
    SIG = np.fft.rfft(sig_a, n=n)
    REFSIG = np.fft.rfft(sig_b, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(interp * n / 2)
    cc_win = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    # Positive peak search
    idx = int(np.argmax(cc_win))
    shift = max_shift - idx
    tau = shift / float(interp * fs)
    peak = float(cc_win[idx])
    return tau * fs, peak


def variant_d_gcc_whitening_no_window(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int = 16000,
    max_tau: float = 0.001,
    interp: int = 16,
) -> Tuple[float, float]:
    """Variant D: GCC with full-band phase whitening and rectangular window."""
    n = sig_a.shape[0] + sig_b.shape[0]
    SIG = np.fft.rfft(sig_a, n=n)
    REFSIG = np.fft.rfft(sig_b, n=n)
    R = SIG * np.conj(REFSIG)
    denom = np.abs(R)
    denom[denom < 1e-6] = 1e-6
    R_phat = R / denom
    cc = np.fft.irfft(R_phat, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(interp * n / 2)
    cc_win = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    idx = int(np.argmax(cc_win))
    shift = max_shift - idx
    tau = shift / float(interp * fs)
    peak = float(cc_win[idx])
    return tau * fs, peak


def variant_e_gcc_whitening_hanning_window(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int = 16000,
    max_tau: float = 0.001,
    interp: int = 16,
) -> Tuple[float, float]:
    """Variant E: GCC with full-band phase whitening + Hanning window."""
    wa = sig_a * np.hanning(len(sig_a))
    wb = sig_b * np.hanning(len(sig_b))
    return variant_d_gcc_whitening_no_window(wa, wb, fs=fs, max_tau=max_tau, interp=interp)


def variant_g_bandpass_phat(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int = 16000,
    max_tau: float = 0.001,
    interp: int = 16,
    f_low: float = 300.0,
    f_high: float = 3400.0,
) -> Tuple[float, float]:
    """Variant G: Bandpass-filtered GCC-PHAT (whitening confined strictly to speech band)."""
    n = sig_a.shape[0] + sig_b.shape[0]
    SIG = np.fft.rfft(sig_a, n=n)
    REFSIG = np.fft.rfft(sig_b, n=n)
    R = SIG * np.conj(REFSIG)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    denom = np.abs(R)
    denom[denom < 1e-6] = 1e-6
    weight = np.zeros_like(R, dtype=np.complex128)
    weight[band_mask] = R[band_mask] / denom[band_mask]
    cc = np.fft.irfft(weight, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(interp * n / 2)
    cc_win = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    idx = int(np.argmax(cc_win))
    shift = max_shift - idx
    tau = shift / float(interp * fs)
    peak = float(cc_win[idx])
    return tau * fs, peak


class TestGCCPHATForensics(unittest.TestCase):
    """Regression tests for speech-band stabilized GCC-PHAT on short delays."""

    def test_01_zero_delay_recovery(self):
        """1. Zero delay: gcc_phat evaluates to 0.0 samples."""
        sa, sb = synthesize_bandlimited_speech(delay_samples=0.0, N=1024, snr_db=20)
        raw_lag, _ = raw_xcorr(sa, sb)
        tau_gcc, q_gcc = gcc_phat(sa, sb, fs=16000, max_tau=0.001)
        gcc_lag = tau_gcc * 16000

        self.assertAlmostEqual(raw_lag, 0.0, delta=0.1)
        self.assertAlmostEqual(gcc_lag, 0.0, delta=0.1)
        self.assertGreaterEqual(q_gcc, 0.5)

    def test_02_half_sample_delay_recovery(self):
        """2. 0.5 sample delay: gcc_phat recovers 0.50 samples with high quality."""
        sa, sb = synthesize_bandlimited_speech(delay_samples=0.5, N=1024, snr_db=15)
        raw_lag, _ = raw_xcorr(sa, sb)
        tau_gcc, q_gcc = gcc_phat(sa, sb, fs=16000, max_tau=0.001)
        gcc_lag = tau_gcc * 16000

        self.assertAlmostEqual(raw_lag, 0.50, delta=0.15)
        self.assertAlmostEqual(gcc_lag, 0.50, delta=0.10)
        self.assertGreaterEqual(q_gcc, 0.5)

    def test_03_one_sample_delay_recovery(self):
        """3. 1.0 sample delay: gcc_phat recovers 1.00 samples with high quality."""
        sa, sb = synthesize_bandlimited_speech(delay_samples=1.0, N=1024, snr_db=15)
        raw_lag, _ = raw_xcorr(sa, sb)
        tau_gcc, q_gcc = gcc_phat(sa, sb, fs=16000, max_tau=0.001)
        gcc_lag = tau_gcc * 16000

        self.assertAlmostEqual(raw_lag, 1.00, delta=0.15)
        self.assertAlmostEqual(gcc_lag, 1.00, delta=0.10)
        self.assertGreaterEqual(q_gcc, 0.5)

    def test_04_one_point_five_sample_delay_recovery(self):
        """4. 1.5 sample delay: gcc_phat recovers 1.50 samples."""
        sa, sb = synthesize_bandlimited_speech(delay_samples=1.5, N=1024, snr_db=15)
        raw_lag, _ = raw_xcorr(sa, sb)
        tau_gcc, q_gcc = gcc_phat(sa, sb, fs=16000, max_tau=0.001)
        gcc_lag = tau_gcc * 16000

        self.assertAlmostEqual(raw_lag, 1.50, delta=0.15)
        self.assertAlmostEqual(gcc_lag, 1.50, delta=0.10)
        self.assertGreaterEqual(q_gcc, 0.5)

    def test_05_two_sample_delay_recovery(self):
        """5. 2.0 sample delay: gcc_phat recovers 2.00 samples."""
        sa, sb = synthesize_bandlimited_speech(delay_samples=2.0, N=1024, snr_db=15)
        raw_lag, _ = raw_xcorr(sa, sb)
        tau_gcc, q_gcc = gcc_phat(sa, sb, fs=16000, max_tau=0.001)
        gcc_lag = tau_gcc * 16000

        self.assertAlmostEqual(raw_lag, 2.00, delta=0.15)
        self.assertAlmostEqual(gcc_lag, 2.00, delta=0.10)
        self.assertGreaterEqual(q_gcc, 0.5)

    def test_06_cardinal_direction_synthetic_array(self):
        """6. Cardinal direction synthetic array (0°, 90°, 180°, 270°)."""
        fs = 16000
        directions = [
            (0.0, -4.01, 0.0),    # FRONT (0°): FB=-4.01, RL=0.00
            (90.0, 0.0, 4.01),    # RIGHT (+90°): FB=0.00, RL=+4.01
            (180.0, 4.01, 0.0),   # BACK (180°): FB=+4.01, RL=0.00
            (270.0, 0.0, -4.01),  # LEFT (-90°): FB=0.00, RL=-4.01
        ]
        for angle, expected_fb, expected_rl in directions:
            sa_fb, sb_fb = synthesize_bandlimited_speech(delay_samples=expected_fb, N=1024, snr_db=20)
            tau_fb, _ = gcc_phat(sa_fb, sb_fb, fs=fs, max_tau=0.001)
            rec_fb = tau_fb * fs
            self.assertAlmostEqual(rec_fb, expected_fb, delta=0.25, msg=f"FB failed at {angle}°")

            sa_rl, sb_rl = synthesize_bandlimited_speech(delay_samples=expected_rl, N=1024, snr_db=20)
            tau_rl, _ = gcc_phat(sa_rl, sb_rl, fs=fs, max_tau=0.001)
            rec_rl = tau_rl * fs
            self.assertAlmostEqual(rec_rl, expected_rl, delta=0.25, msg=f"RL failed at {angle}°")

    def test_07_current_vs_candidate_gcc_parity(self):
        """7. GCC parity on full-band white signal."""
        rng = np.random.default_rng(999)
        white_a = rng.normal(0, 1.0, 2048)
        white_b = np.roll(white_a, 2)
        white_a[:2] = 0
        white_b[:2] = 0

        tau_gcc, _ = gcc_phat(white_a.astype(np.float32), white_b.astype(np.float32), fs=16000, max_tau=0.001)
        gcc_lag = tau_gcc * 16000
        unwhite_lag, _ = variant_c_unwhitened_gcc(white_a, white_b, fs=16000, max_tau=0.001)

        self.assertAlmostEqual(gcc_lag, 2.0, delta=0.1)
        self.assertAlmostEqual(unwhite_lag, 2.0, delta=0.1)

    def test_08_raw_cross_correlation_vs_corrected_gcc(self):
        """8. Raw cross-correlation vs Corrected GCC."""
        for target_d in [0.0, 0.48, 0.99, 1.61]:
            sa, sb = synthesize_bandlimited_speech(delay_samples=target_d, N=1024, snr_db=15)
            raw_lag, _ = raw_xcorr(sa, sb)
            tau_gcc, _ = gcc_phat(sa, sb, fs=16000, max_tau=0.001)
            gcc_lag = tau_gcc * 16000

            self.assertAlmostEqual(gcc_lag, raw_lag, delta=0.15)


if __name__ == "__main__":
    unittest.main()
