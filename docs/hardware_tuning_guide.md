# ASTRO — Hardware Calibration & Parameter Tuning Guide

This practical manual explains how to calibrate and fine-tune the ASTRO head gaze system on physical hardware.

---

## 1. Physical Calibration Procedures

### 1.1. Head Neck Zero Calibration
1. Align the head physically facing straight forward along the robot base chassis.
2. In `config/calibration_params.yaml`, ensure `head.zero_offset_deg` is `0.0`.
3. If mechanical zero is slightly off, measure angular error with a digital protractor and record the delta in `zero_offset_deg`.
4. Verify software limits:
   ```yaml
   head:
     min_angle_deg: -90.0
     max_angle_deg: 90.0
     ticks_per_deg: 2.5882
   ```

### 1.2. Camera Optical Calibration
1. Place an AprilTag or face target directly in front of the robot chassis at $1.5\text{ m}$ distance, $0^\circ$ azimuth.
2. Observe `/gaze/active_target` or raw camera detections.
3. If the detected visual azimuth is offset from $0^\circ$, adjust `camera.yaw_offset_deg` in `config/calibration_params.yaml`:
   $$\text{yaw\_offset\_deg} = -\text{observed\_azimuth\_deg}$$

### 1.3. Microphone Array Alignment
1. Stand at $+45.0^\circ$ (Left of the robot) and speak clearly.
2. Echo the filtered audio state:
   ```bash
   ros2 topic echo /gaze/state
   ```
3. Check `target_yaw_deg`. If sound is detected at $-45.0^\circ$ (Right instead of Left), ensure `audio.invert: true`.
4. If there is a constant angular bias, adjust `audio.yaw_offset_deg`.

---

## 2. Dynamic Performance Parameter Tuning

All algorithmic parameters are configured in `config/social_gaze_params.yaml`:

### 2.1. Motion & Saccade Dynamics
- **`max_velocity_deg_s` (Default: `75.0`):** Set between `60.0` (gentle social) and `120.0` (rapid alert).
- **`max_acceleration_deg_s2` (Default: `180.0`):** Controls responsiveness. Higher values yield crisper starts; lower values give smoother, softer motion.
- **`soft_landing_zone_deg` (Default: `15.0`):** Deceleration braking zone. Increase if the head overshoots during high-speed saccades.
- **`gaze_deadband_deg` (Default: `3.0`):** Deadband filter. If the head jitters when looking at a stationary speaker, increase to `3.5` or `4.0`.

### 2.2. Social Attention & Turn-Taking
- **`min_attention_dwell_s` (Default: `2.50`):** Minimum time robot maintains gaze on a conversational partner before looking away.
- **`turn_taking_min_dwell_s` (Default: `0.80`):** Duration a new speaker must sustain voice to trigger an active turn-taking saccade.
- **`spatial_gate_deg` (Default: `25.0`):** Maximum angular error between voice DOA and camera face to fuse into a single speaker.

---

## 3. Diagnostic & Verification Commands

### Launching the System
```bash
# Launch both hardware serial bridge and social gaze controller
ros2 launch astro_base social_gaze.launch.py

# Launch social gaze in headless simulation / test mode
ros2 launch astro_base social_gaze.launch.py launch_serial_bridge:=false
```

### Monitoring Gaze State
```bash
ros2 topic echo /gaze/state
```
Example JSON payload output:
```json
{
  "fsm_state": "TRACKING",
  "priority": "ACTIVE_SPEAKER",
  "target_yaw_deg": 28.5,
  "planned_pos_deg": 28.4,
  "planned_vel_deg_s": 1.2,
  "actual_pos_deg": 28.2,
  "active_target_id": "person_1",
  "is_speaking": true,
  "is_settled": true
}
```

### Triggering Social Gestures via CLI
```bash
# Nod (Yes / Approval)
ros2 topic pub --once /behavior/gesture std_msgs/msg/String "{data: 'nod'}"

# Shake (No / Disapproval)
ros2 topic pub --once /behavior/gesture std_msgs/msg/String "{data: 'shake'}"

# Recenter
ros2 topic pub --once /behavior/gesture std_msgs/msg/String "{data: 'center'}"
```
