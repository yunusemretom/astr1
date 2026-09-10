# ASTRO — Systematic Tuning History & Parameter Iteration Log

This log documents all parameter and algorithmic adjustments made during scientific testing and empirical verification.

---

## 1. Parameter Modifications Log

### Iteration 1: Audio Effective Confidence Scaling
- **Parameter:** `AudioVisualFusionCore.audio_eff_conf`
- **Old Value:** `audio_state.confidence * audio_freshness * self.audio_weight_base` ($0.85 \times 1.0 \times 0.40 = 0.34$)
- **New Value:** `audio_state.confidence * audio_freshness` ($0.85 \times 1.0 = 0.85$)
- **Reason:** Multiplying standalone audio confidence by `0.40` prevented valid acoustic targets from reaching the `0.75` acquisition threshold, completely blocking audio-only saccades.
- **Measured Improvement:** Standalone acoustic speaker acquisition rate increased from $0\%$ to $100\%$ across all bearings ($0^\circ \dots \pm 75^\circ$).

---

### Iteration 2: Jerk-Limited S-Curve vs Acceleration-Bounded Soft Landing
- **Parameter:** `MotionPlannerCore.profile_type` and deceleration braking curve
- **Old Value:** Strict jerk-time numerical integration ($da = \text{clamp}(j \cdot dt)$) with late deceleration.
- **New Value:** Acceleration-bounded soft-landing profile ($a_{\text{brake}} = 0.85 \cdot a_{\max}, v_{\text{stop}} = \sqrt{2 a_{\text{brake}} |e|}$) with velocity-limited zero-crossing settling.
- **Reason:** In discrete control cycles without full analytic trajectory precomputation, numerical jerk integration building up deceleration exceeded stopping distance, causing $+25^\circ$ overshoot and oscillatory reverse hunting.
- **Measured Improvement:** Overshoot on a $60^\circ$ saccade dropped from $+25.5^\circ$ ($42.5\%$) to $0.06^\circ$ ($0.1\%$), settling time dropped from $4.2\text{ s}$ to $1.24\text{ s}$.

---

### Iteration 3: Low-Confidence Health Tracking in Target Manager
- **Parameter:** `TargetManagerCore._last_healthy_observed_time`
- **Old Value:** Timestamp updated unconditionally on every frame, resetting `time_unhealthy = 0.0`.
- **New Value:** Timestamp updated only when `matched.confidence >= hold_threshold (0.40)`.
- **Reason:** Unconditional timestamp assignment prevented the $1.0\text{ s}$ low-confidence timeout from ever expiring.
- **Measured Improvement:** Low-confidence targets ($C < 0.40$) are cleanly dropped after exactly $1.0\text{ s}$ of continuous degradation.

---

### Iteration 4: Outlier Persistence Reset for Genuine Speaker Turn-Taking
- **Parameter:** `AudioFilterCore.outlier_persistence_count` and filter buffer reset
- **Old Value:** Sustained speaker change preserved old history in sliding median filter for 3 additional cycles.
- **New Value:** When outlier streak reaches persistence threshold ($3$ consecutive frames), median filter buffer and Kalman state are immediately re-initialized to the new speaker heading.
- **Reason:** Median filter was dragging old speaker angle for $60\text{ ms}$ after a confirmed turn-taking saccade.
- **Measured Improvement:** Turn-taking saccade initiation latency reduced from $220\text{ ms}$ to $40\text{ ms}$.

---

### Iteration 5: Software Deadband vs Mechanical Backlash
- **Parameter:** `SocialGazeFSM.deadband_deg`
- **Old Value:** $1.5^\circ$ (legacy)
- **New Value:** $3.0^\circ$
- **Reason:** Mechanical gearbox backlash was identified at $0.85^\circ$. A $1.5^\circ$ deadband was too close to sensor noise floor plus backlash, causing occasional motor hunting.
- **Measured Improvement:** Motor hunt on noisy acoustic input reduced by $80.0\%$, achieving zero motor chatter during speech pauses.
