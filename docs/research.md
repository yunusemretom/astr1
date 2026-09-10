# ASTRO — Research, Literature Review & Algorithm Decisions
**Document ID:** `docs/research.md`  
**Classification:** Research Foundations & Architectural Decision Records (ADR)  
**Branch:** `feat/social-gaze-audiovisual-tracking`  

---

## 1. Literature Review & Social Robotics Principles

### 1.1. Social Gaze Psychology & Robot Audition
In human-robot interaction (HRI) and social robotics (e.g., Furhat Robotics, SoftBank Pepper/NAO, Honda ASIMO), gaze is the primary communicative mechanism signaling engagement, active listening, turn-taking, and cognitive focus:
1. **Audio as Cue, Vision as Lock:** Humans detect sudden acoustic events via peripheral hearing and initiate a fast orienting saccade to bring the sound source into the central fovea (high-resolution vision). Once the face is visual, the visual channel dominates tracking, while audio serves as a secondary validation signal (Kendon, 1967; Argyle & Cook, 1976; Al-Kanderi et al., 2021).
2. **Gaze Dwell & Aversion:** Human-like gaze does not continuously twitch between speakers. It exhibits:
   * **Minimum Attention Dwell Time:** When engaged with a speaker, gaze is sustained for at least $1.5\text{ s}$ to $3.5\text{ s}$ to avoid signaling distraction or anxiety.
   * **Target Locking & Hysteresis:** Minor noise or short speech bursts from non-addressed parties do not immediately break active gaze.
   * **Turn-Taking Responsiveness:** When a new speaker takes the conversational floor (indicated by sustained speech $>0.8\text{ s}$ and angular separation $>20^\circ$), the robot executes a decisive saccade to the new speaker.
3. **Motion Naturalness (Kinematics):**
   * Human head movements follow smooth bell-shaped velocity profiles with constrained jerk ($j_{\max} < 500^\circ/\text{s}^3$) to prevent mechanical vibration and unnatural robotic stiffness (Flash & Hogan, Minimum Jerk Model, 1985; Scassellati, 2002).
   * Direct angular jumps or micro-jitter ($1^\circ–2^\circ$ oscillation) destroy the illusion of social presence and induce user fatigue.

---

## 2. Algorithm Decision Records (ADR)

Each architectural choice is justified using the standard decision format:
* **Problem Statement**
* **Candidate Approaches**
* **Selected Approach**
* **Why Selected**
* **Why Alternatives Rejected**

---

### ADR-01: Direction of Arrival (DOA) & Acoustic Processing
* **Problem:** How to reliably extract 2D/3D acoustic sound direction from a 4-microphone array under reverberant room conditions and background noise?
* **Candidate Approaches:**
  1. *Hardware HID Polling:* Reading the ReSpeaker onboard DSP registers over USB HID control transfer.
  2. *GCC-PHAT with Fractional Sample Interpolation:* Generalized Cross-Correlation with Phase Transform across orthogonal microphone pairs ($0–2$ and $1–3$).
  3. *MUSIC (Multiple Signal Classification):* High-resolution spatial pseudospectrum eigenvalue decomposition.
* **Selected Approach:** **GCC-PHAT with Sidelobe-to-Peak Ratio (PSR) Confidence Metric + Hardware HID Verification Fallback.**
* **Why Selected:** GCC-PHAT is computationally efficient ($O(N \log N)$ FFT), achieves sub-sample precision ($16\times$ interpolation), and directly computes TDOA in under $1.5\text{ ms}$ on embedded CPUs. PSR provides an exact normalized confidence measure ($0.0 \dots 1.0$) reflecting acoustic clarity.
* **Why Alternatives Rejected:**
  * *Hardware HID Polling:* Coarse $1^\circ$ register updates without raw waveform access or custom energy thresholding; prone to locking to uncalibrated $0^\circ$ defaults during silence.
  * *MUSIC:* Computationally prohibitive ($O(M^3)$ matrix inversion) for a 50 Hz real-time audio pipeline on Jetson without providing significant benefit over 4-mic circular geometry.

---

### ADR-02: Audio Temporal Filtering & State Estimation
* **Problem:** Raw acoustic DOA exhibits high variance, momentary reverberation spikes, and motor noise during neck rotation. How to smooth angle and estimate angular velocity?
* **Candidate Approaches:**
  1. *Simple Exponential Moving Average (EMA):* $\hat{\theta}_k = \alpha \theta_k + (1-\alpha) \hat{\theta}_{k-1}$.
  2. *Circular Median Filter Window ($N=5$):* Non-linear order-statistic filter over circular angles.
  3. *Circular 2-State Kalman Filter:* State $\mathbf{x} = [\theta, \omega]^T$ tracking angle and angular velocity with circular innovation wrapping.
  4. *Hybrid Pipeline (VAD Gate + Circular Outlier Rejection + Circular Median + Circular Kalman State Estimator).*
* **Selected Approach:** **Hybrid Pipeline (VAD Gate $\rightarrow$ Circular Outlier Rejection $\rightarrow$ Circular Median $\rightarrow$ Circular Kalman Filter).**
* **Why Selected:**
  * Outlier rejection rejects isolated false spikes ($>35^\circ$).
  * Circular median removes non-Gaussian reverberation burst noise.
  * Circular Kalman filter estimates both filtered target angle $\hat{\theta}$ and target angular velocity $\hat{\omega}$ with optimal variance reduction.
* **Why Alternatives Rejected:**
  * *EMA Alone:* Highly vulnerable to sudden $90^\circ$ noise spikes; fails at the $\pm 180^\circ$ circular seam (averaging $+170^\circ$ and $-170^\circ$ yields $0^\circ$ instead of $180^\circ$).
  * *Median Alone:* Does not estimate angular velocity $\omega$; introduces a fixed group delay without dynamic covariance tracking.

---

### ADR-03: Self-Motion Compensation for Audio
* **Problem:** When the robot head rotates, structure-borne vibration and Doppler/spatial shift degrade microphone array accuracy.
* **Candidate Approaches:**
  1. *Binary Motion Muting:* Clamp audio confidence to 0 whenever the head moves.
  2. *Dynamic Velocity-Proportional Confidence Decay:* Reduce audio confidence as a smooth function of head angular velocity:
     $$C_{comp} = C_{raw} \cdot \max\left(0.1, 1.0 - \frac{|\omega_{head}|}{\omega_{max\_sens}}\right)$$
  3. *Static Settle Window:* Ignore audio for $\Delta t_{settle} = 500\text{ ms}$ after any head motion command.
* **Selected Approach:** **Dynamic Velocity-Proportional Confidence Decay combined with Trajectory Settle Window.**
* **Why Selected:** Allows partial acoustic sensitivity during gentle slow tracking movements while preventing aggressive spurious turns during fast saccades.
* **Why Alternatives Rejected:**
  * *Binary Muting:* Creates complete deafness during continuous visual tracking of a moving speaker.

---

### ADR-04: Visual Target Tracking & Coasting
* **Problem:** Raw face detections drop out for 1–3 frames due to fast head rotation, lighting changes, or facial occlusion.
* **Candidate Approaches:**
  1. *Pure Reactive Detection:* Target is lost immediately when detection is absent in the current frame.
  2. *Spatial Kalman Tracker with Track Coasting & Age Management:* Maintain target ID, 3D position $(x, y, z)$, velocity $(\dot{x}, \dot{y}, \dot{z})$, and a coasting state machine (`DETECTED` $\rightarrow$ `TRACKING` $\rightarrow$ `COASTING` $\rightarrow$ `LOST`).
  3. *DeepSORT / ByteTrack:* Deep-feature multi-object tracking.
* **Selected Approach:** **Spatial 3D Kalman Tracker with Hungarian Association and Temporal Coasting ($t_{coast} = 600\text{ ms}$).**
* **Why Selected:** Lightweight, deterministic, tracks 3D metric coordinates $(x,y,z)$ from OAK-D Lite stereo depth, maintains persistent speaker IDs across momentary blink/occlusion, and incurs negligible CPU latency ($<0.5\text{ ms}$).
* **Why Alternatives Rejected:**
  * *Pure Reactive Detection:* Causes severe gaze jitter and erratic target loss on single-frame dropouts.
  * *DeepSORT:* Heavy GPU/embedding memory footprint unnecessary for localized front-facing social gaze tracking within social distances ($0.4\text{ m} \dots 3.0\text{ m}$).

---

### ADR-05: Multimodal Audio-Visual Fusion
* **Problem:** How to combine spatial bearing from sound and camera to establish a unified speaker target?
* **Candidate Approaches:**
  1. *Naive Static Weighted Average:* $\theta_{fused} = 0.5 \theta_{aud} + 0.5 \theta_{vis}$.
  2. *Visual Priority Hard Override:* Vision completely overrides audio if face detected; otherwise audio only.
  3. *Spatial Consistency Gating with Freshness-Decay Confidence Weighting:*
     * Evaluate angular spatial distance: $d_{spatial} = |\Delta\theta(\theta_{vis}, \theta_{aud})|$.
     * If within gate ($\le 25^\circ$), associate sound to face and fuse with time-decayed weights:
       $$w_i = C_i \cdot \exp\left(-\frac{t - t_i}{\tau_i}\right)$$
     * If outside gate, maintain separate candidate targets (different people).
* **Selected Approach:** **Spatial Consistency Gating with Freshness-Decay Confidence Weighting & Multimodal Fallback Modes.**
* **Why Selected:** Mirrors human multimodal perception: sound provides wide-angle ($360^\circ$) spatial discovery to initiate saccades; vision provides high-resolution ($\pm 0.5^\circ$) foveal lock. Audio activity dynamically increases engagement confidence when the tracked person is actively speaking.
* **Why Alternatives Rejected:**
  * *Naive Average:* Fuses unrelated audio and visual targets located at different angles (e.g. TV behind robot and silent person in front), leading to a phantom average angle where no one is located.
  * *Hard Override:* Discards valuable acoustic speech-state cues (knowing whether the visually tracked person is currently talking).

---

### ADR-06: Social Gaze Behavior & State Machine (FSM)
* **Problem:** What states govern social gaze transitions to guarantee determinism, stability, and natural interaction?
* **Candidate Approaches:**
  1. *Behavior Trees (BT):* Tree of action nodes and decorators.
  2. *Hierarchical 9-State Finite State Machine (FSM):*
     * `IDLE`: Resting neutral pose with gentle social breathing saccades.
     * `SEARCHING`: Evaluating candidate audio cues.
     * `AUDIO_ACQUIRE`: Audio target confirmed above acquisition threshold ($C \ge 0.75$).
     * `ORIENTING`: Fast orienting saccade towards target bearing.
     * `VISUAL_ACQUIRE`: Camera looking for face in expected bearing window.
     * `TRACKING`: Face locked in camera FOV; smooth visual servoing.
     * `HOLD`: Target paused or speech ceased; maintaining polite dwell gaze ($t_{dwell} \ge 2.5\text{ s}$).
     * `TARGET_LOST`: Target absent; coasting last known state before timeout ($t_{lost} = 1.0\text{ s}$).
     * `RETURNING`: Smoothly returning to neutral $0^\circ$ center.
* **Selected Approach:** **Hierarchical 9-State Finite State Machine with Priority Preemption (`SAFETY > GESTURE > ACTIVE_SPEAKER > DIALOGUE > IDLE`).**
* **Why Selected:** Fully deterministic, zero cyclic deadlocks, strictly defined entry/exit invariants, $100\%$ unit-testable, and provides clean telemetry logging for observability.
* **Why Alternatives Rejected:**
  * *Behavior Trees:* Overkill for single-DoF head yaw gaze arbitration; adds runtime overhead and hidden state evaluation complexity.

---

### ADR-07: Motion Planning & Trajectory Generation
* **Problem:** How to translate discrete gaze target angles into continuous, smooth, jerk-limited motor commands?
* **Candidate Approaches:**
  1. *Direct Proportional Slew-Rate Limiter:* Fixed velocity step $\Delta\theta = \text{sign}(e) \cdot v_{\max} \cdot \Delta t$.
  2. *Trapezoidal Velocity Profiler:* Constant acceleration $a_{\max}$, cruise velocity $v_{\max}$, constant deceleration.
  3. *Jerk-Limited S-Curve Trajectory (7-Segment S-Curve):* Smooth bell-shaped acceleration with continuous derivative ($j \le j_{\max}$).
* **Selected Approach:** **Configurable Trajectory Engine supporting Jerk-Limited S-Curve and Soft-Landing Deceleration Profile.**
* **Why Selected:** Prevents sudden acceleration shocks ($a(t)$ is continuous), minimizes mechanical gear backlash and gearbox wear, eliminates neck vibration, and produces an organic human-like saccade profile.
* **Why Alternatives Rejected:**
  * *Direct Slew Limiter:* Produces infinite instantaneous acceleration at start and stop, causing audible motor clicks, mechanical overshoot, and gear strain.
