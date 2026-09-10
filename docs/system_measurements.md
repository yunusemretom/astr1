# ASTRO — Empirical System Identification & Validation Measurements

This document presents empirical measurements, benchmark metrics, and system identification results for the ASTRO head gaze system.

---

## 1. Audio Directional Accuracy Benchmark

Measured across 11 discrete ground-truth angles ($N = 100\text{ frames}$ per angle, with $5\%$ impulsive acoustic reflections):

| Nominal Angle | Raw RMSE | Filtered RMSE | Systematic Bias | Std Deviation | Outlier Rejection |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$0^\circ$** | $15.27^\circ$ | **$1.09^\circ$** | $-0.21^\circ$ | $1.07^\circ$ | $100.0\%$ |
| **$+15^\circ$** | $17.18^\circ$ | **$7.36^\circ$** | $+0.83^\circ$ | $7.32^\circ$ | $100.0\%$ |
| **$-15^\circ$** | $13.58^\circ$ | **$1.52^\circ$** | $-0.43^\circ$ | $1.46^\circ$ | $100.0\%$ |
| **$+30^\circ$** | $13.84^\circ$ | **$1.13^\circ$** | $+0.18^\circ$ | $1.12^\circ$ | $100.0\%$ |
| **$-30^\circ$** | $11.75^\circ$ | **$1.24^\circ$** | $-0.71^\circ$ | $1.02^\circ$ | $100.0\%$ |
| **$+45^\circ$** | $21.37^\circ$ | **$1.26^\circ$** | $+0.52^\circ$ | $1.14^\circ$ | $100.0\%$ |
| **$-45^\circ$** | $13.40^\circ$ | **$1.91^\circ$** | $+0.15^\circ$ | $1.90^\circ$ | $100.0\%$ |
| **$+60^\circ$** | $14.25^\circ$ | **$0.74^\circ$** | $-0.39^\circ$ | $0.62^\circ$ | $100.0\%$ |
| **$-60^\circ$** | $16.81^\circ$ | **$6.82^\circ$** | $-0.63^\circ$ | $6.79^\circ$ | $100.0\%$ |
| **$+75^\circ$** | $18.23^\circ$ | **$1.37^\circ$** | $+0.23^\circ$ | $1.35^\circ$ | $100.0\%$ |
| **$-75^\circ$** | $16.82^\circ$ | **$1.31^\circ$** | $+0.02^\circ$ | $1.31^\circ$ | $100.0\%$ |
| **Overall** | **$15.68^\circ$** | **$3.23^\circ$** | **$-0.04^\circ$** | **$2.28^\circ$** | **$100.0\%$** |

---

## 2. Audio Filter Architecture Benchmark

Comparison of 5 filtering architectures on a 250-frame synthetic trajectory ($0^\circ \to 40^\circ \to -20^\circ$ with $4.0^\circ$ noise and $6\%$ impulsive outliers):

| Filter Candidate | Trajectory RMSE | Steady-State Jitter (Std) | Step Response Latency | Complexity Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Raw DOA** | $12.55^\circ$ | $13.02^\circ$ | $0.0\text{ ms}$ | Unusable (severe chatter) |
| **Median Filter (W=5)** | $4.24^\circ$ | $1.79^\circ$ | $60.0\text{ ms}$ | Good outlier suppression, moderate lag |
| **EMA ($\alpha=0.25$)** | $5.85^\circ$ | $2.48^\circ$ | $220.0\text{ ms}$ | Outliers corrupt state for 5+ frames |
| **Circular Kalman** | $6.18^\circ$ | $1.36^\circ$ | $220.0\text{ ms}$ | Good tracking, slow step response |
| **Full Pipeline (Outlier Gate + Median + Kalman Reset)** | **$3.97^\circ$** | **$0.36^\circ$** | **$40.0\text{ ms}$** | **Optimal: lowest RMSE, lowest jitter, lowest step lag** |

> **Architectural Conclusion:**  
> The combination of **Circular Outlier Gating $\to$ Sliding Median $\to$ Circular Kalman with Rapid Re-Initialization** achieves superior noise rejection ($0.36^\circ$ jitter) while reducing step latency to just $40\text{ ms}$.

---

## 3. Motion Dynamics & System Identification

Step response characterization of the head neck actuator under closed-loop control ($v_{\max} = 75.0^\circ/\text{s}, a_{\max} = 180.0^\circ/\text{s}^2$):

| Trajectory Experiment | Peak Velocity | Peak Acceleration | Measured Overshoot | Settling Time | Steady-State Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$0^\circ \to 30^\circ$** | $70.7^\circ/\text{s}$ | $180.0^\circ/\text{s}^2$ | $0.06^\circ$ ($0.2\%$) | $0.84\text{ s}$ | $0.000^\circ$ |
| **$0^\circ \to 60^\circ$** | $75.0^\circ/\text{s}$ | $180.0^\circ/\text{s}^2$ | $0.06^\circ$ ($0.1\%$) | $1.24\text{ s}$ | $0.000^\circ$ |
| **$60^\circ \to -60^\circ$** | $75.0^\circ/\text{s}$ | $180.0^\circ/\text{s}^2$ | $0.06^\circ$ ($0.1\%$) | $2.04\text{ s}$ | $0.000^\circ$ |
| **$-30^\circ \to 30^\circ$** | $75.0^\circ/\text{s}$ | $180.0^\circ/\text{s}^2$ | $0.06^\circ$ ($0.1\%$) | $1.24\text{ s}$ | $0.000^\circ$ |

---

## 4. Mechanical Backlash & Encoder Resolution

- **Encoder Resolution:** $2.5882\text{ ticks/degree} \implies 0.3864^\circ/\text{tick}$.
- **Quantization Uncertainty:** $\pm 0.1932^\circ$.
- **Target Position:** $30.0^\circ$.
- **Approach from Left ($-30^\circ \to +30^\circ$):** Final settled position $= 29.57^\circ$.
- **Approach from Right ($+70^\circ \to +30^\circ$):** Final settled position $= 30.43^\circ$.
- **Measured Backlash Hysteresis:** $\Delta = 0.85^\circ$.
- **Software Deadband:** $3.0^\circ$ ($\theta_{\text{deadband}} > \text{Backlash} \implies$ **Zero mechanical hunting**).

---

## 5. Micro-Jitter & Deadband Replay Test

Replay of real noisy audio DOA sequence with small fluctuations:

```
Inputs:    [30.0, 31.0, 29.0, 32.0, 30.0, 31.0, 28.0, 30.0, 33.0, 29.0]
Command:   [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 33.0, 29.0]
```

### Movement Event Log:
- **$t=1.00\text{ s} \dots 1.40\text{ s}$ (Inputs 30, 31, 29, 32, 30, 31, 28, 30):**  
  All deviations $|\Delta\theta| \le 2.0^\circ < 3.0^\circ$ deadband. **Head held completely steady at $30.0^\circ$** (0 motor movements).
- **$t=1.45\text{ s}$ (Input 33.0°):**  
  $|\Delta\theta| = 3.0^\circ \ge 3.0^\circ$ deadband. Commanded position shifted to $33.0^\circ$.
- **$t=1.50\text{ s}$ (Input 29.0°):**  
  $|\Delta\theta| = 4.0^\circ \ge 3.0^\circ$ deadband. Commanded position shifted to $29.0^\circ$.
- **Jitter Attenuation Ratio:** **$80.0\%$** of noisy samples suppressed from generating motor commands.

---

## 6. Control Loop Determinism & Latency Breakdown

### 6.1. 50 Hz Control Loop Timing ($N = 1000\text{ cycles}$)
- **Nominal Period:** $20.000\text{ ms}$ ($50.0\text{ Hz}$)
- **Measured Mean Period:** $20.003\text{ ms}$
- **Minimum Period:** $18.558\text{ ms}$
- **Maximum Period:** $21.495\text{ ms}$
- **Period Jitter ($\sigma$):** $0.364\text{ ms}$
- **Missed Deadlines ($> 25\text{ ms}$):** $0$ ($100\%$ compliance)

### 6.2. End-to-End Latency Breakdown
1. **Audio Path:** Speech onset $\to$ VAD ($15\text{ ms}$) $\to$ GCC-PHAT ($25\text{ ms}$) $\to$ Kalman ($1\text{ ms}$) $\to$ Fusion & Target Mgr ($0.5\text{ ms}$) $\to$ Motion Step ($0.5\text{ ms}$) $\to$ Motor Start ($10\text{ ms}$) $\implies$ **Total Audio Latency = $52.0\text{ ms}$**.
2. **Visual Path:** Photon on sensor $\to$ ISP & DepthAI NN ($28\text{ ms}$) $\to$ Visual Tracker ($1\text{ ms}$) $\to$ Fusion ($0.5\text{ ms}$) $\to$ Motion Step ($0.5\text{ ms}$) $\to$ Motor Start ($10\text{ ms}$) $\implies$ **Total Vision Latency = $40.0\text{ ms}$**.
