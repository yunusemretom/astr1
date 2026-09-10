# ASTRO Social Gaze Manager & FSM Synchronization

## 1. Overview
The Gaze Manager arbitrates sensory inputs (active speaker, visual face track, dialogue intent, gestures, safety lock) and manages the behavioral state transitions of the robot head.

## 2. Spatial Angle Distinction
The system maintains strict semantic separation between three core spatial representations:
1. **`desired_yaw_deg` (Target)**: The perceptual ground-truth azimuth of the active speaker or interlocutor (e.g. $-35.0^\circ$).
2. **`planned_yaw_deg` (Trajectory)**: The instantaneous continuous setpoint produced by the 50 Hz jerk-limited S-curve motion planner (e.g. $-9.97^\circ$).
3. **`actual_yaw_deg` (Feedback)**: The true physical head orientation read from the quadrature optical encoder via the microcontroller telemetry stream (e.g. $-3.86^\circ$).

## 3. FSM Completion & Settling Window Condition
The FSM prohibits premature state transitions (e.g., `ORIENTING -> HOLD` or `ORIENTING -> VISUAL_ACQUIRE`) based on virtual setpoints alone. An orienting saccade is strictly completed only when:

$$\left|\text{actual\_yaw\_deg} - \text{target\_yaw\_deg}\right| \le 2.5^\circ$$
$$\text{AND} \quad \left|\text{actual\_vel\_deg\_s}\right| \le 3.0^\circ/\text{s}$$
$$\text{AND} \quad \text{Condition persists for } N \ge 3 \text{ consecutive cycles } (60\text{ ms})$$

### Failsafe Timeout
If mechanical obstruction prevents settling within $3.0\text{ s}$, the FSM engages timeout protection to prevent deadlock.

## 4. ROS 2 Interfaces
- **`/gaze/state`** (`astro_base/msg/GazeStatus`): Canonical typed state message.
- **`/gaze/debug`** (`std_msgs/msg/String`): JSON debug telemetry.
- **`/head/state`** (`astro_base/msg/HeadState`): Physical actuator & encoder telemetry.
