#!/usr/bin/env python3
"""ASTRO Robot — ReSpeaker Raw Pairwise Cross-Correlation & TDOA Forensic Analyzer.

Measures raw pairwise cross-correlation / TDOA directly from the actual ReSpeaker PCM
to determine whether the 4 physical microphone channels contain expected physical time delays.

Target Hardware Configuration:
    Device:       hw:CARD=ArrayUAC10,DEV=0 (ReSpeaker 4 Mic Array v2.0, USB Audio)
    Sample Rate:  16000 Hz
    Format:       S16_LE
    Channels:     6 channels
    Mic Channels: ch1 (Mic 0 Front), ch2 (Mic 1 Right), ch3 (Mic 2 Back), ch4 (Mic 3 Left)

Measured Microphone Pairs (all 6 combinations):
    1. ch1 ↔ ch2 (Front ↔ Right)
    2. ch1 ↔ ch3 (Front ↔ Back)  <-- Critical Front-Back Axis
    3. ch1 ↔ ch4 (Front ↔ Left)
    4. ch2 ↔ ch3 (Right ↔ Back)
    5. ch2 ↔ ch4 (Right ↔ Left)  <-- Critical Right-Left Axis
    6. ch3 ↔ ch4 (Back ↔ Left)

Lag Convention:
    For pair (ch_a, ch_b):
        lag > 0: ch_a leads ch_b (ch_b is delayed relative to ch_a; sound reached ch_a first).
        lag < 0: ch_b leads ch_a (ch_a is delayed relative to ch_b; sound reached ch_b first).
        lag = 0: sound arrived at ch_a and ch_b simultaneously.
    time_us = lag_samples / 16000 * 1e6

Outputs:
    - Human-readable summary table (ch1-ch3 lag, ch2-ch4 lag, confidence across positions)
    - Repeatability check for stationary speaker (e.g. CENTER_1 vs CENTER_2)
    - Raw Cross-Correlation vs existing GCC-PHAT TDOA comparison
    - CSV export: config/respeaker_tdoa_forensics.csv

Diagnostic Tool ONLY. Does NOT modify production runtime code.
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout/stderr for UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

# Ensure script directory is on sys.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

from respeaker_device import (
    REQUIRED_CHANNELS,
    REQUIRED_MIC_CHANNELS,
    REQUIRED_SAMPLE_FORMAT,
    REQUIRED_SAMPLE_RATE,
    RESPEAKER_ALSA_DEVICE,
    RESPEAKER_CARD_ID,
    RespeakerDeviceInfo,
    resolve_respeaker_capture_device,
    validate_respeaker_device,
)

DEFAULT_POSITIONS = ["CENTER_1", "CENTER_2", "RIGHT", "BACK", "LEFT"]
DEFAULT_DURATION_S = 6.0
DEFAULT_OUTPUT_CSV = "config/respeaker_tdoa_forensics.csv"
HW_SAMPLE_RATE = REQUIRED_SAMPLE_RATE
DEFAULT_MAX_LAG = 16  # Max lag in samples (16 samples = 1000 µs @ 16kHz, diameter is 250 µs)

# Opposing mic pair distance: 2 * 43mm = 86mm
RESPEAKER_RADIUS_M = 0.043
RESPEAKER_PAIR_DIST_M = 0.086
SPEED_OF_SOUND_MPS = 343.0
MAX_PHYSICAL_TAU_S = RESPEAKER_PAIR_DIST_M / SPEED_OF_SOUND_MPS  # ~0.0002507 s (4.01 samples)

# Pairwise channel definitions using physical microphone channels (1-indexed into 6-channel PCM)
# ch1: Mic 0 (Front, 0°)
# ch2: Mic 1 (Right, +90°)
# ch3: Mic 2 (Back, 180°)
# ch4: Mic 3 (Left, 270° / -90°)
PAIR_DEFINITIONS = [
    (1, 2, "ch1_ch2", "Front-Right (ch1-ch2)"),
    (1, 3, "ch1_ch3", "Front-Back  (ch1-ch3)"),  # Critical
    (1, 4, "ch1_ch4", "Front-Left  (ch1-ch4)"),
    (2, 3, "ch2_ch3", "Right-Back  (ch2-ch3)"),
    (2, 4, "ch2_ch4", "Right-Left  (ch2-ch4)"),  # Critical
    (3, 4, "ch3_ch4", "Back-Left   (ch3-ch4)"),
]


def samples_to_microseconds(samples: float, sample_rate: int = HW_SAMPLE_RATE) -> float:
    """Converts a lag measured in samples to microseconds.

    Formula:
        time_us = lag_samples / 16000 * 1e6
    """
    return float(samples) / float(sample_rate) * 1e6


def microseconds_to_samples(time_us: float, sample_rate: int = HW_SAMPLE_RATE) -> float:
    """Converts a lag measured in microseconds to samples."""
    return float(time_us) * float(sample_rate) / 1e6


def normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates zero-lag normalized correlation coefficient r in [-1.0, 1.0]."""
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    xd = xf - np.mean(xf)
    yd = yf - np.mean(yf)
    denom = float(np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2)))
    return float(np.sum(xd * yd) / denom) if denom > 1e-9 else 0.0


def compute_raw_cross_correlation(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int = HW_SAMPLE_RATE,
    max_lag_samples: int = DEFAULT_MAX_LAG,
) -> Dict[str, Any]:
    """Computes raw pairwise cross-correlation between two signals.

    Lag Convention:
        lag > 0: sig_a leads sig_b (sig_b is delayed relative to sig_a).
        lag < 0: sig_b leads sig_a (sig_a is delayed relative to sig_b).
        lag = 0: signals arrive simultaneously.

    Returns dict containing:
        - peak_lag_samples: Integer peak lag in samples
        - peak_lag_samples_frac: Refined peak lag via parabolic interpolation
        - peak_correlation: Raw cross-correlation value at peak
        - normalized_correlation: Normalized correlation rho in [-1.0, 1.0] at peak
        - time_us: Peak lag converted to microseconds
        - lags: 1D array of evaluated lag shifts
        - correlation_values: 1D array of cross-correlation values
    """
    n = min(len(sig_a), len(sig_b))
    if n == 0:
        return {
            "peak_lag_samples": 0,
            "peak_lag_samples_frac": 0.0,
            "peak_correlation": 0.0,
            "normalized_correlation": 0.0,
            "time_us": 0.0,
            "lags": np.array([0]),
            "correlation_values": np.array([0.0]),
        }

    a = sig_a[:n].astype(np.float64) - np.mean(sig_a[:n])
    b = sig_b[:n].astype(np.float64) - np.mean(sig_b[:n])

    energy_a = float(np.sum(a ** 2))
    energy_b = float(np.sum(b ** 2))
    norm_denom = math.sqrt(energy_a * energy_b) if (energy_a > 1e-9 and energy_b > 1e-9) else 1e-9

    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=int)
    c_vals = np.zeros(len(lags), dtype=np.float64)

    for idx, k in enumerate(lags):
        if k >= 0:
            # b is shifted by +k, testing if a aligns with b delayed by k
            c_vals[idx] = float(np.sum(a[: n - k] * b[k:n]))
        else:
            # k < 0, a is shifted by -k, testing if b aligns with a delayed by -k
            c_vals[idx] = float(np.sum(a[-k:n] * b[: n + k]))

    # Find peak of correlation
    best_idx = int(np.argmax(c_vals))
    best_k = int(lags[best_idx])
    peak_corr = float(c_vals[best_idx])
    norm_corr = float(peak_corr / norm_denom) if norm_denom > 1e-9 else 0.0
    norm_corr = max(-1.0, min(1.0, norm_corr))

    # Parabolic sub-sample refinement around peak
    best_k_frac = float(best_k)
    if 0 < best_idx < len(c_vals) - 1:
        y_left = c_vals[best_idx - 1]
        y_mid = c_vals[best_idx]
        y_right = c_vals[best_idx + 1]
        denom = y_left - 2.0 * y_mid + y_right
        if abs(denom) > 1e-9:
            delta = 0.5 * (y_left - y_right) / denom
            if abs(delta) <= 1.0:
                best_k_frac = float(best_k) + float(delta)

    time_us = samples_to_microseconds(best_k_frac, sample_rate=fs)

    return {
        "peak_lag_samples": best_k,
        "peak_lag_samples_frac": round(best_k_frac, 3),
        "peak_correlation": peak_corr,
        "normalized_correlation": round(norm_corr, 4),
        "time_us": round(time_us, 2),
        "lags": lags,
        "correlation_values": c_vals,
    }


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int = HW_SAMPLE_RATE,
    max_tau: Optional[float] = None,
    interp: int = 16,
) -> Tuple[float, float]:
    """Computes Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

    Existing implementation from production/calibration pipeline.
    Returns:
        (tau_seconds, quality_confidence)
    """
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    denom = np.abs(R)
    denom[denom < 1e-6] = 1e-6
    R_phat = R / denom

    cc = np.fft.irfft(R_phat, n=interp * n)
    max_shift = int(interp * fs * max_tau) if max_tau else int(interp * n / 2)
    cc_windowed = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    shift = max_shift - int(np.argmax(np.abs(cc_windowed)))
    tau = shift / float(interp * fs)

    peak_val = float(np.max(np.abs(cc_windowed)))
    mean_val = float(np.mean(np.abs(cc_windowed)))
    std_val = float(np.std(np.abs(cc_windowed)))
    psr = (peak_val - mean_val) / max(1e-5, std_val)
    quality = min(1.0, max(0.0, (psr - 1.5) / 5.0))
    return tau, quality


def compute_hardware_gcc_phat_doa(
    mics_4ch: np.ndarray,
    sample_rate: int = HW_SAMPLE_RATE,
) -> Tuple[Optional[float], float, float, float]:
    """Computes DOA and pair TDOA from 4 microphone channels using current GCC-PHAT.

    ReSpeaker 4-Mic Circular Array geometry:
        mics_4ch[0]: Mic 0 (Front, 0 deg)
        mics_4ch[1]: Mic 1 (Right, +90 deg)
        mics_4ch[2]: Mic 2 (Back, 180 deg)
        mics_4ch[3]: Mic 3 (Left, 270 deg / -90 deg)

    Returns:
        (azimuth_deg_360, confidence, tau_fb_s, tau_lr_s)
    """
    if mics_4ch is None or mics_4ch.shape[0] < 4:
        return None, 0.0, 0.0, 0.0

    if mics_4ch.shape[0] > mics_4ch.shape[1]:
        mics_4ch = mics_4ch.T

    speed_of_sound = SPEED_OF_SOUND_MPS
    max_tau = MAX_PHYSICAL_TAU_S

    mic_front = mics_4ch[0].astype(np.float32)
    mic_right = mics_4ch[1].astype(np.float32)
    mic_back = mics_4ch[2].astype(np.float32)
    mic_left = mics_4ch[3].astype(np.float32)

    if float(np.max(np.abs(mics_4ch))) < 1e-3:
        return None, 0.0, 0.0, 0.0

    # Current GCC-PHAT calls as in production / calibrate_respeaker_doa.py:
    # Pair 1: Mic 3 (Left) vs Mic 1 (Right) -> Left-Right axis
    tau_lr, q_lr = gcc_phat(mic_left, mic_right, fs=sample_rate, max_tau=max_tau)
    # Pair 2: Mic 2 (Back) vs Mic 0 (Front) -> Back-Front axis
    tau_fb, q_fb = gcc_phat(mic_back, mic_front, fs=sample_rate, max_tau=max_tau)

    delta_x = -tau_lr * speed_of_sound
    delta_y = -tau_fb * speed_of_sound

    raw_azimuth = math.degrees(math.atan2(delta_x, delta_y))
    raw_doa_360 = raw_azimuth if raw_azimuth >= 0.0 else raw_azimuth + 360.0
    if abs(raw_doa_360) < 1e-5 or abs(raw_doa_360 - 360.0) < 1e-5:
        raw_doa_360 = 0.0

    confidence = round(float((q_lr + q_fb) / 2.0), 3)
    return round(raw_doa_360, 2), confidence, tau_fb, tau_lr


def calculate_channel_metrics(channel_data: np.ndarray) -> Dict[str, float]:
    """Calculates RMS, Peak, Mean, and Zero-Crossing Rate for a 1D audio array."""
    cf = channel_data.astype(np.float64)
    rms = float(np.sqrt(np.mean(cf ** 2)))
    peak = int(np.max(np.abs(channel_data)))
    mean = float(np.mean(cf))
    return {
        "rms": round(rms, 2),
        "peak": peak,
        "mean": round(mean, 2),
    }


def record_physical_pcm(
    device_idx: Any,
    channels: int = REQUIRED_CHANNELS,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = HW_SAMPLE_RATE,
) -> np.ndarray:
    """Records raw multi-channel int16 PCM from hardware device."""
    if not HAS_SOUNDDEVICE or sd is None:
        raise RuntimeError("sounddevice is not available in this environment.")

    total_samples = int(duration_s * sample_rate)
    recording = sd.rec(
        frames=total_samples,
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=device_idx,
        blocking=True,
    )
    # Shape returned: (channels, total_samples)
    return recording.T


def simulate_station_pcm(
    position: str,
    channels: int = 6,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = HW_SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """Generates realistic synthetic 6-channel PCM with exact acoustic delays.

    Physical delay model:
        Array diameter = 86mm, c = 343m/s, max delay = ~4.0 samples at 16kHz.
        CENTER / FRONT (0°): Front leads Back by +4 samples. Left & Right coincident (0 samples).
        RIGHT (+90°): Right leads Left by +4 samples. Front & Back coincident (0 samples).
        BACK (180°): Back leads Front by +4 samples (Front lags Back by -4 samples).
        LEFT (-90° / 270°): Left leads Right by +4 samples (Right lags Left by -4 samples).
        CENTER_1 and CENTER_2: Identical acoustic delay to test repeatability.
    """
    total_samples = int(duration_s * sample_rate)
    rng = np.random.default_rng(seed)
    base_signal = rng.normal(0.0, 1800.0, total_samples + 200)

    pos_upper = position.upper()
    if "CENTER" in pos_upper or "FRONT" in pos_upper:
        # Front leads Back by 4 samples: mic0 early (0), mic2 late (4)
        lags = {0: 0, 1: 2, 2: 4, 3: 2}
    elif "RIGHT" in pos_upper:
        # Right leads Left by 4 samples: mic1 early (0), mic3 late (4)
        lags = {0: 2, 1: 0, 2: 2, 3: 4}
    elif "BACK" in pos_upper or "REAR" in pos_upper:
        # Back leads Front by 4 samples: mic2 early (0), mic0 late (4)
        lags = {0: 4, 1: 2, 2: 0, 3: 2}
    elif "LEFT" in pos_upper:
        # Left leads Right by 4 samples: mic3 early (0), mic1 late (4)
        lags = {0: 2, 1: 4, 2: 2, 3: 0}
    else:
        lags = {0: 0, 1: 0, 2: 0, 3: 0}

    origin = 50
    m0 = base_signal[origin + lags[0]: origin + lags[0] + total_samples] + rng.normal(0.0, 15.0, total_samples)
    m1 = base_signal[origin + lags[1]: origin + lags[1] + total_samples] + rng.normal(0.0, 15.0, total_samples)
    m2 = base_signal[origin + lags[2]: origin + lags[2] + total_samples] + rng.normal(0.0, 15.0, total_samples)
    m3 = base_signal[origin + lags[3]: origin + lags[3] + total_samples] + rng.normal(0.0, 15.0, total_samples)

    ch0_processed = (m0 + m1 + m2 + m3) / 4.0
    ch5_loopback = rng.normal(0.0, 5.0, total_samples)

    arr = np.stack([ch0_processed, m0, m1, m2, m3, ch5_loopback[:total_samples]])
    return arr.astype(np.int16)


def run_tdoa_forensics(
    positions: List[str],
    duration_s: float,
    output_csv: str,
    device_override: Optional[Any] = None,
    max_lag_samples: int = DEFAULT_MAX_LAG,
    simulate: bool = False,
    non_interactive: bool = False,
) -> None:
    """Runs pairwise cross-correlation / TDOA forensic measurement across positions."""
    print("=" * 80)
    print(" ASTRO ReSpeaker Raw Cross-Correlation & TDOA Forensic Analyzer")
    print("=" * 80)

    # 1. Device resolution & validation
    device_info = resolve_respeaker_capture_device(
        allow_simulation=simulate,
        device_override=device_override,
    )
    if not validate_respeaker_device(device_info):
        sys.exit(1)

    print(f"Device:       {device_info.alsa_device_string} ({device_info.device_name})")
    print(f"Sample Rate:  {device_info.sample_rate} Hz | Format: {device_info.sample_format} | Channels: {device_info.channels}")
    print(f"Mics (1..4):  ch1=Front, ch2=Right, ch3=Back, ch4=Left")
    print(f"Duration:     {duration_s:.1f} seconds per position")
    print(f"Positions:    {', '.join(positions)}")
    print(f"Max Lag:      ±{max_lag_samples} samples (±{samples_to_microseconds(max_lag_samples):.1f} µs)")
    print(f"Output CSV:   {output_csv}")
    print("=" * 80)

    summary_table_rows: List[Dict[str, Any]] = []
    detailed_csv_rows: List[Dict[str, Any]] = []

    for pos in positions:
        print("\n" + "#" * 80)
        print(f" POSITION: {pos}")
        print("#" * 80)

        pos_upper = pos.upper()
        if "CENTER" in pos_upper or "FRONT" in pos_upper:
            guidance = "Place stationary speaker ~1 meter in FRONT (0°) of ReSpeaker."
        elif "RIGHT" in pos_upper:
            guidance = "Place stationary speaker ~1 meter to the RIGHT (+90°) of ReSpeaker."
        elif "BACK" in pos_upper or "REAR" in pos_upper:
            guidance = "Place stationary speaker ~1 meter directly BEHIND (180°) ReSpeaker."
        elif "LEFT" in pos_upper:
            guidance = "Place stationary speaker ~1 meter to the LEFT (-90° / 270°) of ReSpeaker."
        else:
            guidance = f"Place stationary speaker ~1 meter from ReSpeaker at {pos}."

        print(f"Placement: {guidance}")

        if not non_interactive:
            try:
                input(f"Press ENTER when ready to record {duration_s:.1f}s at {pos} (speak/play audio continuously)... ")
            except (EOFError, KeyboardInterrupt):
                print("\nForensic measurement aborted by user.")
                break
        else:
            print(f"[Auto-mode] Capturing for {duration_s:.1f}s...")

        if simulate:
            seed = int(abs(hash(pos)) % (2 ** 31))
            pcm = simulate_station_pcm(
                position=pos,
                channels=device_info.channels,
                duration_s=duration_s,
                sample_rate=device_info.sample_rate,
                seed=seed,
            )
        else:
            pcm = record_physical_pcm(
                device_idx=device_info.device_index,
                channels=device_info.channels,
                duration_s=duration_s,
                sample_rate=device_info.sample_rate,
            )

        num_ch, num_samples = pcm.shape
        actual_dur = num_samples / device_info.sample_rate
        print(f"Captured: {actual_dur:.2f}s ({num_samples} samples), {num_ch} channels")

        ch_data = {
            1: pcm[1],
            2: pcm[2],
            3: pcm[3],
            4: pcm[4],
        }

        ch_metrics: Dict[int, Dict[str, float]] = {}
        for c in (1, 2, 3, 4):
            ch_metrics[c] = calculate_channel_metrics(ch_data[c])

        print("\n--- Per-Channel Signal Quality ---")
        for c in (1, 2, 3, 4):
            m = ch_metrics[c]
            label = {1: "ch1 (Front)", 2: "ch2 (Right)", 3: "ch3 (Back)", 4: "ch4 (Left)"}[c]
            print(f"  {label:<15} RMS={m['rms']:7.1f}  Peak={m['peak']:5d}  Mean={m['mean']:+6.1f}")

        mics_4ch = np.stack([ch_data[1], ch_data[2], ch_data[3], ch_data[4]])
        gcc_azimuth, gcc_conf, tau_fb, tau_lr = compute_hardware_gcc_phat_doa(
            mics_4ch=mics_4ch,
            sample_rate=device_info.sample_rate,
        )

        pair_results: Dict[str, Dict[str, Any]] = {}
        print("\n--- Pairwise Cross-Correlation & TDOA (All 6 Pairs) ---")
        print(f"  {'Pair':<24} {'Raw Lag (smp)':<16} {'Raw Time (µs)':<16} {'Peak Corr':<12} {'Norm Corr':<12} {'GCC Lag (smp)':<16} {'GCC Time (µs)':<16}")
        print("  " + "-" * 116)

        for c_a, c_b, pair_key, pair_label in PAIR_DEFINITIONS:
            raw_res = compute_raw_cross_correlation(
                sig_a=ch_data[c_a],
                sig_b=ch_data[c_b],
                fs=device_info.sample_rate,
                max_lag_samples=max_lag_samples,
            )

            gcc_tau, gcc_quality = gcc_phat(
                sig=ch_data[c_a].astype(np.float32),
                refsig=ch_data[c_b].astype(np.float32),
                fs=device_info.sample_rate,
                max_tau=MAX_PHYSICAL_TAU_S,
            )
            gcc_lag_smp = gcc_tau * device_info.sample_rate
            gcc_time_us = gcc_tau * 1e6

            pair_results[pair_key] = {
                "pair_key": pair_key,
                "pair_label": pair_label,
                "ch_a": f"ch{c_a}",
                "ch_b": f"ch{c_b}",
                "raw_lag_samples": raw_res["peak_lag_samples"],
                "raw_lag_samples_frac": raw_res["peak_lag_samples_frac"],
                "raw_time_us": raw_res["time_us"],
                "raw_peak_corr": raw_res["peak_correlation"],
                "raw_norm_corr": raw_res["normalized_correlation"],
                "gcc_tau_s": gcc_tau,
                "gcc_lag_samples": round(gcc_lag_smp, 2),
                "gcc_time_us": round(gcc_time_us, 2),
                "gcc_quality": round(gcc_quality, 3),
                "ch_a_rms": ch_metrics[c_a]["rms"],
                "ch_a_peak": ch_metrics[c_a]["peak"],
                "ch_a_mean": ch_metrics[c_a]["mean"],
                "ch_b_rms": ch_metrics[c_b]["rms"],
                "ch_b_peak": ch_metrics[c_b]["peak"],
                "ch_b_mean": ch_metrics[c_b]["mean"],
            }

            detailed_csv_rows.append({
                "position": pos,
                "pair": pair_key,
                "pair_label": pair_label,
                "ch_a": f"ch{c_a}",
                "ch_b": f"ch{c_b}",
                "raw_lag_samples": raw_res["peak_lag_samples"],
                "raw_lag_samples_frac": raw_res["peak_lag_samples_frac"],
                "raw_time_us": raw_res["time_us"],
                "raw_peak_corr": f"{raw_res['peak_correlation']:.1f}",
                "raw_norm_corr": raw_res["normalized_correlation"],
                "gcc_tau_s": f"{gcc_tau:.7f}",
                "gcc_lag_samples": round(gcc_lag_smp, 2),
                "gcc_time_us": round(gcc_time_us, 2),
                "gcc_quality": round(gcc_quality, 3),
                "gcc_azimuth_deg": gcc_azimuth,
                "gcc_confidence": gcc_conf,
                "ch_a_rms": ch_metrics[c_a]["rms"],
                "ch_a_peak": ch_metrics[c_a]["peak"],
                "ch_a_mean": ch_metrics[c_a]["mean"],
                "ch_b_rms": ch_metrics[c_b]["rms"],
                "ch_b_peak": ch_metrics[c_b]["peak"],
                "ch_b_mean": ch_metrics[c_b]["mean"],
            })

            raw_lag_str = f"{raw_res['peak_lag_samples_frac']:+6.2f} smp"
            raw_time_str = f"{raw_res['time_us']:+7.1f} µs"
            gcc_lag_str = f"{gcc_lag_smp:+6.2f} smp"
            gcc_time_str = f"{gcc_time_us:+7.1f} µs"

            print(f"  {pair_label:<24} {raw_lag_str:<16} {raw_time_str:<16} {raw_res['peak_correlation']:<12.1e} "
                  f"{raw_res['normalized_correlation']:<12.4f} {gcc_lag_str:<16} {gcc_time_str:<16}")

        fb = pair_results["ch1_ch3"]
        rl = pair_results["ch2_ch4"]

        summary_table_rows.append({
            "position": pos,
            "fb_raw_lag_samples": fb["raw_lag_samples_frac"],
            "fb_raw_time_us": fb["raw_time_us"],
            "fb_norm_corr": fb["raw_norm_corr"],
            "rl_raw_lag_samples": rl["raw_lag_samples_frac"],
            "rl_raw_time_us": rl["raw_time_us"],
            "rl_norm_corr": rl["raw_norm_corr"],
            "gcc_azimuth": gcc_azimuth,
            "gcc_conf": gcc_conf,
            "fb_gcc_lag_samples": fb["gcc_lag_samples"],
            "rl_gcc_lag_samples": rl["gcc_lag_samples"],
        })

    print("\n" + "=" * 90)
    print(" REQUIRED PHYSICAL TDOA SUMMARY TABLE")
    print("=" * 90)
    print(f" {'POSITION':<12} {'ch1-ch3 lag (Front-Back)':<26} {'ch2-ch4 lag (Right-Left)':<26} {'Confidence':<12} {'GCC Azimuth'}")
    print("-" * 90)

    for r in summary_table_rows:
        fb_str = f"{r['fb_raw_lag_samples']:+5.2f} smp ({r['fb_raw_time_us']:+6.1f} µs)"
        rl_str = f"{r['rl_raw_lag_samples']:+5.2f} smp ({r['rl_raw_time_us']:+6.1f} µs)"
        conf_str = f"{r['gcc_conf']:.2f}"
        az_str = f"{r['gcc_azimuth']:.1f}°" if r['gcc_azimuth'] is not None else "NONE"
        print(f" {r['position']:<12} {fb_str:<26} {rl_str:<26} {conf_str:<12} {az_str}")

    print("=" * 90)

    print("\n" + "=" * 90)
    print(" CRITICAL TEST: REPEATABILITY VERIFICATION (STATIONARY SPEAKER)")
    print("=" * 90)

    center_runs = [r for r in summary_table_rows if "CENTER" in r["position"].upper() or "FRONT" in r["position"].upper()]
    if len(center_runs) >= 2:
        r1, r2 = center_runs[0], center_runs[1]
        delta_fb = abs(r1["fb_raw_lag_samples"] - r2["fb_raw_lag_samples"])
        delta_rl = abs(r1["rl_raw_lag_samples"] - r2["rl_raw_lag_samples"])
        delta_fb_us = abs(r1["fb_raw_time_us"] - r2["fb_raw_time_us"])
        delta_rl_us = abs(r1["rl_raw_time_us"] - r2["rl_raw_time_us"])

        print(f"Comparing {r1['position']} vs {r2['position']}:")
        print(f"  Front-Back (ch1-ch3) lag delta: {delta_fb:.2f} samples ({delta_fb_us:.1f} µs)")
        print(f"  Right-Left (ch2-ch4) lag delta: {delta_rl:.2f} samples ({delta_rl_us:.1f} µs)")

        is_repeatable = (delta_fb <= 1.2) and (delta_rl <= 1.2)
        if is_repeatable:
            print("  ✓ REPEATABILITY: PASSED (TDOA is physically stable for a stationary speaker).")
            print("    -> Confirms the raw PCM capture is consistent and repeatable across captures.")
        else:
            print("  ⚠️ REPEATABILITY: FAILED (Radically different lags observed for the same position).")
            print("    -> PROBLEM IS UPSTREAM OF ANGLE CONVERSION: Capture buffers, ALSA overrun/slip, or hardware clock jitter.")
    else:
        print("  Notice: Only one CENTER capture provided. To run repeatability test, capture CENTER_1 and CENTER_2.")

    print("\n" + "=" * 90)
    print(" SPATIAL TREND VERIFICATION (LEFT -> CENTER -> RIGHT)")
    print("=" * 90)

    row_map = {r["position"].upper(): r for r in summary_table_rows}
    left_r = next((v for k, v in row_map.items() if "LEFT" in k), None)
    center_r = next((v for k, v in row_map.items() if "CENTER" in k or "FRONT" in k), None)
    right_r = next((v for k, v in row_map.items() if "RIGHT" in k), None)

    if left_r and center_r and right_r:
        rl_left = left_r["rl_raw_lag_samples"]
        rl_center = center_r["rl_raw_lag_samples"]
        rl_right = right_r["rl_raw_lag_samples"]

        print(f"  LEFT:   ch2-ch4 lag = {rl_left:+5.2f} samples ({left_r['rl_raw_time_us']:+6.1f} µs)")
        print(f"  CENTER: ch2-ch4 lag = {rl_center:+5.2f} samples ({center_r['rl_raw_time_us']:+6.1f} µs)")
        print(f"  RIGHT:  ch2-ch4 lag = {rl_right:+5.2f} samples ({right_r['rl_raw_time_us']:+6.1f} µs)")

        increasing = (rl_left < rl_center < rl_right)
        decreasing = (rl_left > rl_center > rl_right)
        has_variance = abs(rl_right - rl_left) >= 1.5

        if (increasing or decreasing) and has_variance:
            print("  ✓ SPATIAL TREND: Monotonic variation confirmed as speaker moves LEFT -> CENTER -> RIGHT.")
            if increasing:
                print("    Sign convention: RIGHT produces positive ch2-ch4 lag, LEFT produces negative lag.")
            else:
                print("    Sign convention: LEFT produces positive ch2-ch4 lag, RIGHT produces negative lag.")
        elif not has_variance:
            print("  ⚠️ SPATIAL TREND: ch2-ch4 lag does NOT vary with speaker position (|Right - Left| < 1.5 smp).")
            print("    -> PROBLEM IN HARDWARE CAPTURE: Mic signals are identical, mono downmixed, or mis-mapped.")
        else:
            print("  ⚠️ SPATIAL TREND: Non-monotonic or erratic variation detected across Left -> Center -> Right.")
    else:
        print("  Notice: Include LEFT, CENTER, and RIGHT positions to verify spatial trend.")

    print("\n" + "=" * 90)
    print(" GCC-PHAT VS RAW CROSS-CORRELATION COMPARISON")
    print("=" * 90)

    agreements: List[bool] = []
    for r in summary_table_rows:
        diff_fb = abs(r["fb_raw_lag_samples"] - r["fb_gcc_lag_samples"])
        diff_rl = abs(r["rl_raw_lag_samples"] - r["rl_gcc_lag_samples"])
        agrees = (diff_fb <= 1.2) and (diff_rl <= 1.2)
        agreements.append(agrees)
        status_tag = "AGREE" if agrees else "DISAGREE"
        print(f"  {r['position']:<12}: Front-Back (Raw={r['fb_raw_lag_samples']:+5.2f} smp, GCC={r['fb_gcc_lag_samples']:+5.2f} smp, diff={diff_fb:.2f}) | "
              f"Right-Left (Raw={r['rl_raw_lag_samples']:+5.2f} smp, GCC={r['rl_gcc_lag_samples']:+5.2f} smp, diff={diff_rl:.2f}) -> [{status_tag}]")

    print("\n--- Diagnostic Decision Tree Conclusion ---")
    all_agree = all(agreements) if agreements else False
    if not all_agree:
        print("  [DIAGNOSIS]: Raw cross-correlation and GCC-PHAT DISAGREE.")
        print("  -> ROOT CAUSE: GCC-PHAT implementation / phase whitening / windowing is corrupting the peak.")
    else:
        if left_r and right_r and abs(left_r["rl_raw_lag_samples"] - right_r["rl_raw_lag_samples"]) < 1.5:
            print("  [DIAGNOSIS]: Raw cross-correlation and GCC-PHAT AGREE, BUT DO NOT VARY with speaker position.")
            print("  -> ROOT CAUSE: Raw microphone signals / hardware geometry / capture path is the problem.")
            print("     (Signals on ch1..4 do not contain distinct physical acoustic propagation delays).")
        else:
            print("  [DIAGNOSIS]: Raw cross-correlation and GCC-PHAT AGREE and VARY CONSISTENTLY with speaker position.")
            print("  -> ROOT CAUSE: Upstream acoustic delays are physically valid.")
            print("     Problem is isolated downstream in angle conversion / coordinate frame mapping / arctan sign.")
    print("=" * 90)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "position",
            "pair",
            "pair_label",
            "ch_a",
            "ch_b",
            "raw_lag_samples",
            "raw_lag_samples_frac",
            "raw_time_us",
            "raw_peak_corr",
            "raw_norm_corr",
            "gcc_tau_s",
            "gcc_lag_samples",
            "gcc_time_us",
            "gcc_quality",
            "gcc_azimuth_deg",
            "gcc_confidence",
            "ch_a_rms",
            "ch_a_peak",
            "ch_a_mean",
            "ch_b_rms",
            "ch_b_peak",
            "ch_b_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_csv_rows)

    print(f"\n[FILE SAVED] Raw TDOA forensics exported to: {output_csv}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASTRO ReSpeaker Raw Cross-Correlation & TDOA Forensic Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=DEFAULT_POSITIONS,
        help="Physical speaker positions to measure",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Recording window duration in seconds per position",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for output CSV file",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Explicit audio input device index override",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help="Maximum cross-correlation search lag in samples",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulated physical test with synthetic delayed signals",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without waiting for user ENTER keypresses",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tdoa_forensics(
        positions=args.positions,
        duration_s=args.duration,
        output_csv=args.output,
        device_override=args.device,
        max_lag_samples=args.max_lag,
        simulate=args.simulate,
        non_interactive=args.non_interactive,
    )
