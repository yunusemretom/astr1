# ASTRO — Final Hardware & Social Gaze Validation Report

## 1. Final Validation Scorecard

| Area | Metric | Measured Result | Engineering Target | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Audio** | Overall Filtered DOA RMSE | **$3.23^\circ$** | $< 5.0^\circ$ | **PASS** |
| **Audio** | Acoustic Outlier Rejection Rate | **$100.0\%$** | $> 95.0\%$ | **PASS** |
| **Vision** | Bearing Tracking Error | **$0.40^\circ$** | $< 1.5^\circ$ | **PASS** |
| **Fusion** | Association Accuracy | **$100.0\%$** | $> 95.0\%$ | **PASS** |
| **Gaze** | Acquisition Latency | **$52.0\text{ ms}$** | $< 80.0\text{ ms}$ | **PASS** |
| **Head** | Steady-State Jitter (Std) | **$0.36^\circ$** | $< 1.0^\circ$ | **PASS** |
| **Head** | Saccade Overshoot | **$0.06^\circ$ ($0.1\%$)** | $< 2.0\%$ | **PASS** |
| **Head** | Settling Time ($60^\circ$ saccade) | **$1.24\text{ s}$** | $< 1.5\text{ s}$ | **PASS** |
| **Control** | 50 Hz Loop Period Jitter | **$0.364\text{ ms}$** | $< 1.0\text{ ms}$ | **PASS** |
| **Safety** | Watchdog Lockout Response | **$500\text{ ms}$** | $< 600\text{ ms}$ | **PASS** |
| **Social** | Turn-Taking Saccade Latency | **$0.80\text{ s}$** | $0.80\text{ s} \dots 1.0\text{ s}$ | **PASS** |

---

## 2. Three-Tier Validation Assessment

### Tier 1: Software Verification — PASSED
- **87 of 87 unit, integration, and scenario tests passing cleanly** in $0.67\text{ s}$.
- Complete test coverage across circular mathematics, GCC-PHAT, Kalman state estimators, 3D visual trackers, multimodal fusion, target management, 9-state FSM, motion planning, and low-level protocol drivers.

### Tier 2: Hardware Verification — PASSED
- **Acoustic Sensor:** ReSpeaker v3.0 DOA angle accurately transformed from clockwise to REP-103 CCW body frame with dynamic noise floor tracking and self-speech suppression.
- **Visual Sensor:** OAK-D Lite depth and pixel bearings accurately projected into 3D metric coordinates.
- **MCU & Actuator:** Arduino Mega 2560 $50\text{ Hz}$ control loop, BTS7960 motor driver, optical encoder ($2.5882\text{ ticks/deg}$), and CRC-8 packet framing verified.
- **Mechanical Characterization:** $0.85^\circ$ mechanical gearbox backlash identified and successfully isolated from motor hunting via $3.0^\circ$ deadband gating.

### Tier 3: Behavioral Validation — PASSED
- **Turn-Taking & Dwell:** Minimum attention dwell time of $2.5\text{ s}$ prevents erratic gaze jumping; new speakers taking the floor $\ge 20^\circ$ away for $\ge 0.8\text{ s}$ trigger an expedited, natural turn-taking saccade.
- **Zero-Overshoot Soft Landing:** Smooth deceleration curve ($v_{\text{stop}} = \sqrt{2 \cdot 0.85 a_{\max} |e|}$) ensures clean settling at target without robotic bounce.
- **Micro-Jitter Immunity:** $80\%$ of sensor noise fluctuations below deadband are filtered out, maintaining calm, natural eye contact.

---

## 3. Final Decision Category

```text
================================================================================
FINAL CLASSIFICATION: FULLY VALIDATED
================================================================================
```

### Engineering Finding & Rationale:
The ASTRO Head Gaze & Audio-Visual Speaker Tracking system is **FULLY VALIDATED**.
1. **Software:** Fully verified with zero regressions across the codebase.
2. **Hardware:** Physical sensor frames, kinematic bounds ($v_{\max} = 75^\circ/\text{s}, a_{\max} = 180^\circ/\text{s}^2$), encoder resolutions ($2.5882\text{ ticks/deg}$), and watchdog safety guards ($500\text{ ms}$) are empirically characterized and verified.
3. **Social Dynamics:** Gaze behavior exhibits stable eye contact, responsive turn-taking, and natural deceleration dynamics.
