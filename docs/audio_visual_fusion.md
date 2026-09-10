# ASTRO — Audio-Visual Perception, Spatial Tracking & Multimodal Fusion

## 1. Acoustic Perception Core & Circular Filtering

### 1.1. GCC-PHAT TDOA Localization
The microphone array consists of 4 microphones on a circular boundary ($R = 43\text{ mm}$, $f_s = 16000\text{ Hz}$).
Phase-Transform Generalized Cross-Correlation (GCC-PHAT) calculates time difference of arrival (TDOA) $\tau$:

$$R_{xy}^{\text{PHAT}}(f) = \frac{X(f) \cdot Y^*(f)}{|X(f) \cdot Y^*(f)|}$$
$$\tau = \arg\max_t \mathcal{F}^{-1}\left\{R_{xy}^{\text{PHAT}}(f)\right\}$$

### 1.2. Dynamic VAD & Self-Voice Suppression
- **Dynamic VAD:** Signal is declared voice active if $\text{RMS} \ge \max(500.0, 2.5 \cdot \text{RMS}_{\text{noise}})$ and peak $\ge 900.0$.
- **Self-Speech Suppression:** When robot TTS is active (`is_robot_speaking = True`), audio confidence is multiplied by $0.15$ and marked invalid to prevent chasing own echo.
- **Head Motion Self-Noise Attenuation:** When head velocity $\omega_{head} > 25.0^\circ/\text{s}$, audio confidence is exponentially attenuated:
  $$C_{\text{comp}} = C_{\text{raw}} \cdot \max\left(0.10, 1.0 - \frac{\omega_{head} - 25.0}{75.0 - 25.0}\right)$$

### 1.3. 2-State Circular Kalman State Estimator
State vector $\mathbf{x} = [\theta, \dot{\theta}]^T$. Measurement updates calculate angular innovation with circular wrapping:
$$y = \text{wrap\_deg}(z - \hat{\theta})$$

---

## 2. 3D Visual Perception & Spatial Tracking

### 2.1. Direct Eye Contact Verification
Direct social eye contact is verified when:
1. Both eyes are visible in face detection.
2. Estimated head pose yaw angle $|\theta_{yaw}| \le 22.0^\circ$.
3. Person is within the social proximity zone ($0.35\text{ m} \le d \le 3.50\text{ m}$).

### 2.2. 6-State 3D Constant-Velocity Kalman Tracker
Tracks multiple individuals in 3D base coordinates $(x, y, z)$:
$$\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$$

- **Data Association:** Greedy 3D Euclidean distance bipartite matching below $0.85\text{ m}$ gating radius.
- **Temporal Coasting:** Unobserved tracks coast via Kalman prediction for up to $0.70\text{ s}$ before being marked `LOST`.

---

## 3. Multimodal Audio-Visual Fusion

### 3.1. Spatial Consistency Gating
When an acoustic bearing $\theta_{\text{audio}}$ and visual face track $\theta_{\text{visual}}$ satisfy:
$$\text{circular\_distance\_deg}(\theta_{\text{audio}}, \theta_{\text{visual}}) \le 25.0^\circ$$

The cues are fused into a unified multimodal target (`Modality.FUSED`):
$$\theta_{\text{fused}} = \text{circular\_mean\_deg}\left([\theta_{\text{visual}}, \theta_{\text{audio}}], [w_{\text{vis}} \cdot 1.5, w_{\text{aud}} \cdot 0.5]\right)$$
$$C_{\text{fused}} = \min(1.0, w_{\text{vis}} + 0.30 \cdot w_{\text{aud}})$$

### 3.2. Exponential Temporal Freshness Decay
Confidence weights decay exponentially when observations pause:
$$w(t) = C \cdot \exp\left(-\frac{\ln 2}{\tau_{\text{half\_life}}} \cdot \Delta t\right)$$
Where $\tau_{\text{audio}} = 0.80\text{ s}$ and $\tau_{\text{vision}} = 1.20\text{ s}$.

### 3.3. Dual-Threshold Hysteresis & Turn-Taking
- **Acquisition Threshold ($C \ge 0.75$):** Required to acquire a new active speaker.
- **Hold Threshold ($C \ge 0.40$):** Maintains current target lock even if speech pauses.
- **Turn-Taking Preemption:** A new speaker located $\ge 20.0^\circ$ away who sustains speech for $\ge 0.80\text{ s}$ smoothly captures active gaze attention.
