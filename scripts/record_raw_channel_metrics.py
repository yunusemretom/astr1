#!/usr/bin/env python3
"""ASTRO Robot — Forensic Raw Microphone Channel Telemetry & Physical Validator.

Forensically inspects raw ReSpeaker channels before GCC-PHAT to establish:
1. Exact device capture format (ALSA device, hw/card, rate, format, channel count, frame size).
2. Raw per-channel signal telemetry (RMS, peak, mean, zero-crossing rate) across all channels (0..5).
3. Channel distinctness: Normalized pairwise correlation among channels 1..4.
4. Physical channel ordering: Verification of mic arrival leads per cardinal position.
5. Exports metrics to CSV for analysis.

Usage:
    python3 scripts/record_raw_channel_metrics.py
    python3 scripts/record_raw_channel_metrics.py --duration 3.0 --output config/raw_channel_metrics.csv
    python3 scripts/record_raw_channel_metrics.py --simulate --non-interactive

Diagnostic tool ONLY. Does NOT modify production runtime.
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

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

from respeaker_device import (
    RESPEAKER_ALSA_DEVICE,
    RESPEAKER_CARD_ID,
    REQUIRED_CHANNELS,
    REQUIRED_MIC_CHANNELS,
    REQUIRED_SAMPLE_FORMAT,
    REQUIRED_SAMPLE_RATE,
    RespeakerDeviceInfo,
    resolve_respeaker_capture_device,
    validate_respeaker_device,
)

DEFAULT_POSITIONS = ["CENTER", "FRONT", "RIGHT", "BACK", "LEFT"]
DEFAULT_DURATION_S = 10.0
DEFAULT_OUTPUT_CSV = "config/raw_channel_metrics.csv"
HW_SAMPLE_RATE = REQUIRED_SAMPLE_RATE
HW_BLOCK_SIZE = 320


def normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates zero-lag normalized correlation coefficient r in [-1.0, 1.0]."""
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    xd = xf - np.mean(xf)
    yd = yf - np.mean(yf)
    denom = float(np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2)))
    return float(np.sum(xd * yd) / denom) if denom > 1e-9 else 0.0


def compute_tdoa_lag_samples(sig: np.ndarray, refsig: np.ndarray, max_lag: int = 16) -> int:
    """Computes cross-correlation lag shift in samples between two signals.

    Positive lag indicates refsig lags sig (sig leads).
    """
def compute_tdoa_lag_samples(sig: np.ndarray, refsig: np.ndarray, max_lag: int = 16) -> int:
    """Computes cross-correlation lag shift in samples between two signals.

    Positive lag indicates refsig lags sig (sig leads).
    """
    # Use up to first 4096 samples for fast lag estimation
    n = min(len(sig), len(refsig), 4096)
    s = sig[:n].astype(np.float64) - np.mean(sig[:n])
    r = refsig[:n].astype(np.float64) - np.mean(refsig[:n])

    best_lag = 0
    best_corr = -1e12
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            val = float(np.sum(s[:n + lag] * r[-lag:n]))
        elif lag > 0:
            val = float(np.sum(s[lag:n] * r[:n - lag]))
        else:
            val = float(np.sum(s * r))
        if val > best_corr:
            best_corr = val
            best_lag = lag
    return best_lag


def calculate_channel_metrics(channel_data: np.ndarray) -> Dict[str, float]:
    """Calculates RMS, Peak, Mean, and Zero-Crossing Rate for a 1D audio array."""
    cf = channel_data.astype(np.float64)
    rms = float(np.sqrt(np.mean(cf ** 2)))
    peak = int(np.max(np.abs(channel_data)))
    mean = float(np.mean(cf))
    zcr = float(np.mean(np.diff(np.signbit(cf)) != 0)) if len(cf) > 1 else 0.0
    return {
        "rms": round(rms, 2),
        "peak": peak,
        "mean": round(mean, 2),
        "zcr": round(zcr, 4),
    }


def record_physical_pcm(
    device_idx: int,
    channels: int,
    duration_s: float,
    sample_rate: int = HW_SAMPLE_RATE,
) -> np.ndarray:
    """Records raw multi-channel int16 PCM from hardware device."""
    if not HAS_SOUNDDEVICE or sd is None:
        raise RuntimeError("sounddevice is not available")

    total_samples = int(duration_s * sample_rate)
    recording = sd.rec(
        frames=total_samples,
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=device_idx,
        blocking=True,
    )
    # Shape: (channels, total_samples)
    return recording.T


def simulate_physical_pcm(
    position: str,
    channels: int = 6,
    duration_s: float = 3.0,
    sample_rate: int = HW_SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """Generates realistic synthetic multi-channel audio for physical positions.

    Simulates:
    - Ch 0: Processed beamformed mono
    - Ch 1: Front Mic 0 (leads on FRONT/CENTER)
    - Ch 2: Right Mic 1 (leads on RIGHT)
    - Ch 3: Back Mic 2 (leads on BACK)
    - Ch 4: Left Mic 3 (leads on LEFT)
    - Ch 5: Loopback (quiet / zero)
    """
    total_samples = int(duration_s * sample_rate)
    rng = np.random.default_rng(seed)
    base_signal = rng.normal(0.0, 1500.0, total_samples + 100)

    # Lead offsets in samples (+ leads, - lags)
    lags = {
        "CENTER": (0, 0, 0, 0),
        "FRONT":  (+4, 0, -4, 0),
        "RIGHT":  (0, +4, 0, -4),
        "BACK":   (-4, 0, +4, 0),
        "LEFT":   (0, -4, 0, +4),
    }.get(position.upper(), (0, 0, 0, 0))

    origin = 50
    # Add slight channel-independent noise to represent physical acoustic paths
    m1 = base_signal[origin + lags[0]: origin + lags[0] + total_samples] + rng.normal(0.0, 30.0, total_samples)
    m2 = base_signal[origin + lags[1]: origin + lags[1] + total_samples] + rng.normal(0.0, 30.0, total_samples)
    m3 = base_signal[origin + lags[2]: origin + lags[2] + total_samples] + rng.normal(0.0, 30.0, total_samples)
    m4 = base_signal[origin + lags[3]: origin + lags[3] + total_samples] + rng.normal(0.0, 30.0, total_samples)

    ch0_processed = (m1 + m2 + m3 + m4) / 3.0
    ch5_loopback = rng.normal(0.0, 5.0, total_samples)

    arr = np.stack([ch0_processed, m1, m2, m3, m4, ch5_loopback[:total_samples]])
    return arr.astype(np.int16)


def run_raw_channel_investigation(
    positions: List[str],
    duration_s: float,
    output_csv: str,
    device_override: Optional[Any] = None,
    simulate: bool = False,
    non_interactive: bool = False,
) -> None:
    """Runs the controlled raw channel investigation across all positions."""
    print("=" * 60)
    print(" ASTRO ReSpeaker Raw Channel Investigation & Telemetry")
    print("=" * 60)

    # STEP 1 & 2: Device discovery & strict validation
    device_info = resolve_respeaker_capture_device(
        allow_simulation=simulate,
        device_override=device_override,
    )
    if not validate_respeaker_device(device_info):
        sys.exit(1)

    csv_detailed_rows: List[Dict[str, Any]] = []
    csv_summary_rows: List[Dict[str, Any]] = []

    for pos in positions:
        print("\n" + "-" * 60)
        print(f"POSITION: {pos}")
        print(f"Place speaker approximately 1 meter from ReSpeaker at {pos}.")

        if not non_interactive:
            try:
                input(f"Press ENTER when ready to capture {duration_s:.1f}s at {pos} (speak/play audio continuously)... ")
            except (EOFError, KeyboardInterrupt):
                print("\nInvestigation aborted by user.")
                break
        else:
            print(f"[Auto-mode] Capturing for {duration_s:.1f}s...")

        print(f"Recording {duration_s:.1f}s of raw audio from {device_info.alsa_device_string}...")
        if simulate:
            pcm = simulate_physical_pcm(
                position=pos,
                channels=device_info.channels,
                duration_s=duration_s,
                sample_rate=device_info.sample_rate,
            )
        else:
            pcm = record_physical_pcm(
                device_idx=device_info.device_index,
                channels=device_info.channels,
                duration_s=duration_s,
                sample_rate=device_info.sample_rate,
            )

        num_ch, num_samples = pcm.shape
        print(f"\n[STEP 3: PROVEN PCM SHAPE]")
        print(f"  pcm_shape:              {pcm.shape} (channels, samples)")
        print(f"  capture_channels:       {num_ch}")
        print(f"  duration_recorded:      {num_samples / device_info.sample_rate:.2f} seconds ({num_samples} frames)")

        # STEP 2: Per-channel telemetry
        print(f"\n[STEP 2: RAW CHANNEL TELEMETRY]")
        ch_metrics: Dict[int, Dict[str, float]] = {}
        for c in range(num_ch):
            metrics = calculate_channel_metrics(pcm[c])
            ch_metrics[c] = metrics
            print(f"  ch{c}  RMS={metrics['rms']:8.1f}  Peak={metrics['peak']:5d}  "
                  f"Mean={metrics['mean']:+6.1f}  ZCR={metrics['zcr']:.4f}")
            csv_detailed_rows.append({
                "position": pos,
                "channel": f"ch{c}",
                "rms": metrics["rms"],
                "peak": metrics["peak"],
                "mean": metrics["mean"],
                "zcr": metrics["zcr"],
            })

        # Explicit Requirement: Print ch1..ch4 RMS and pairwise correlations
        print(f"\n[RAW MICROPHONE CHANNELS (1..4)]")
        for mic_idx in (1, 2, 3, 4):
            if mic_idx < num_ch:
                print(f"  ch{mic_idx} RMS: {ch_metrics[mic_idx]['rms']:.1f}")

        # Check non-zero raw mic channels
        mics_rms = [ch_metrics[i]["rms"] for i in (1, 2, 3, 4) if i < num_ch]
        if any(r < 10.0 for r in mics_rms):
            print("  ⚠️ ALERT: One or more raw microphone channels report near-zero RMS (< 10.0)!")
        else:
            print("  ✓ Verified non-zero raw microphone channels (RMS > 10.0).")

        # STEP 5: Channel Distinctness (Normalized Correlations among channels 1..4)
        corrs: Dict[str, float] = {}
        if num_ch >= 5:
            print(f"\n[STEP 5: PAIRWISE CORRELATIONS (CHANNELS 1..4)]")
            pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            for c_a, c_b in pairs:
                r_val = normalized_correlation(pcm[c_a], pcm[c_b])
                corrs[f"corr_{c_a}{c_b}"] = round(r_val, 4)
                print(f"  corr(ch{c_a}, ch{c_b}) = {r_val:.4f}")

            # Check if any pair is exactly identical (1.0000)
            identical_pairs = [k for k, v in corrs.items() if abs(v - 1.0) < 1e-6]
            if identical_pairs:
                print(f"  ⚠️ WARNING: Identical/duplicated channels detected: {identical_pairs}!")
            else:
                print("  ✓ Channels exhibit physical separation (no identical duplications).")

            # STEP 6: Physical Channel Order & TDOA Lead Analysis
            print(f"\n[STEP 6: PHYSICAL CHANNEL ORDER & TDOA LEADS]")
            lag_fb = compute_tdoa_lag_samples(pcm[1], pcm[3])  # Ch 1 (Front) vs Ch 3 (Back)
            lag_rl = compute_tdoa_lag_samples(pcm[2], pcm[4])  # Ch 2 (Right) vs Ch 4 (Left)
            print(f"  Front-Back Pair (ch1 vs ch3) lag: {lag_fb:+d} samples "
                  f"({'ch1 leads (Front)' if lag_fb > 0 else 'ch3 leads (Back)' if lag_fb < 0 else 'coincident'})")
            print(f"  Right-Left Pair (ch2 vs ch4) lag: {lag_rl:+d} samples "
                  f"({'ch2 leads (Right)' if lag_rl > 0 else 'ch4 leads (Left)' if lag_rl < 0 else 'coincident'})")

        summary_entry = {
            "position": pos,
            "ch0_rms": ch_metrics.get(0, {}).get("rms", 0.0),
            "ch0_peak": ch_metrics.get(0, {}).get("peak", 0),
            "ch1_rms": ch_metrics.get(1, {}).get("rms", 0.0),
            "ch1_peak": ch_metrics.get(1, {}).get("peak", 0),
            "ch2_rms": ch_metrics.get(2, {}).get("rms", 0.0),
            "ch2_peak": ch_metrics.get(2, {}).get("peak", 0),
            "ch3_rms": ch_metrics.get(3, {}).get("rms", 0.0),
            "ch3_peak": ch_metrics.get(3, {}).get("peak", 0),
            "ch4_rms": ch_metrics.get(4, {}).get("rms", 0.0),
            "ch4_peak": ch_metrics.get(4, {}).get("peak", 0),
            "ch5_rms": ch_metrics.get(5, {}).get("rms", 0.0),
            "ch5_peak": ch_metrics.get(5, {}).get("peak", 0),
            **corrs,
        }
        csv_summary_rows.append(summary_entry)

    # Save to CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["position", "channel", "rms", "peak", "mean", "zcr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_detailed_rows)

    summary_csv = output_csv.replace(".csv", "_summary.csv")
    if csv_summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_summary_rows)

    print("\n" + "=" * 60)
    print(" INVESTIGATION COMPLETE")
    print(f"  Detailed per-channel CSV saved to: {output_csv}")
    print(f"  Position summary CSV saved to:     {summary_csv}")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASTRO ReSpeaker Raw Channel Diagnostic and Telemetry Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=DEFAULT_POSITIONS,
        help="Physical positions to test",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Recording window in seconds per position",
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
        help="Explicit audio input device index",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulated physical test without hardware",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without waiting for user ENTER keypresses",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_raw_channel_investigation(
        positions=args.positions,
        duration_s=args.duration,
        output_csv=args.output,
        device_override=args.device,
        simulate=args.simulate,
        non_interactive=args.non_interactive,
    )
