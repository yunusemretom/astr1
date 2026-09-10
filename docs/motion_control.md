# ASTRO — Motion Planning & Closed-Loop Head Control

## 1. Kinematic Constraints & Trajectory Generation

The motion planning engine (`MotionPlannerCore`) converts high-level discrete behavioral gaze goals into smooth, acceleration-bounded trajectories evaluated synchronously at $50\text{ Hz}$ ($\Delta t = 20\text{ ms}$).

### Kinematic Parameter Bounds:
- **Maximum Velocity ($v_{\max}$):** $75.0^\circ/\text{s}$
- **Maximum Acceleration ($a_{\max}$):** $180.0^\circ/\text{s}^2$
- **Maximum Jerk ($j_{\max}$):** $360.0^\circ/\text{s}^3$
- **Software Mechanical Limits:** $[-90.0^\circ, +90.0^\circ]$

---

## 2. Zero-Overshoot Braking & Soft-Landing Curve

To prevent overshoot and oscillations when approaching a target angle, the motion planner calculates maximum permissible braking velocity as a function of remaining error $e = \theta_{target} - \theta_{current}$:

$$v_{\text{stop}}(e) = \sqrt{2 \cdot a_{\text{brake}} \cdot |e|}, \quad \text{where } a_{\text{brake}} = 0.85 \cdot a_{\max}$$

The target velocity is bounded by:
$$v_{\text{desired}} = \text{sign}(e) \cdot \min\left(v_{\max}, v_{\text{stop}}(e), v_{\text{soft}}(e)\right)$$

Where $v_{\text{soft}}(e)$ provides an organic deceleration slope within the proximity zone ($|e| \le 15.0^\circ$):
$$v_{\text{soft}}(e) = \max\left(2.5, v_{\max} \cdot \frac{|e|}{\theta_{\text{soft}}}\right)$$

### Acceleration Slew Rate Limiting:
$$\Delta v = \text{clamp}\left(v_{\text{desired}} - v_{\text{current}}, -a_{\max}\Delta t, a_{\max}\Delta t\right)$$
$$v_{\text{current}} = \text{clamp}\left(v_{\text{current}} + \Delta v, -v_{\max}, v_{\max}\right)$$
$$\theta_{\text{current}} = \text{clamp}\left(\theta_{\text{current}} + v_{\text{current}}\Delta t, -90.0^\circ, 90.0^\circ\right)$$

### Settling Criteria:
The trajectory settles cleanly at the target when:
$$|e| \le 0.25^\circ \quad \text{and} \quad |v_{\text{current}}| < 2.0^\circ/\text{s}$$

---

## 3. Low-Level Arduino Protocol & Closed-Loop Control

### 3.1. Protocol v2.0 Binary Framing
Messages between ROS 2 host and Arduino Mega 2560 are framed using CRC-8-ATM checksum:

```
[0xAA] [0x55] [LEN] [MSG_ID] [PAYLOAD (0..255 B)] [CRC-8]
```

- **`MSG_HEAD_CMD (0x03)`**: Payload is 4-byte IEEE-754 `float32` commanded head yaw in degrees.
- **`MSG_HEARTBEAT (0x01)`**: Payload is 4-byte sequence `uint32`.
- **`MSG_HEARTBEAT_ACK (0x13)`**: Echoes sequence ID back to host.

### 3.2. Closed-Loop Position PID in Firmware
- **Control Rate:** $50\text{ Hz}$ (Timer5 PWM on BTS7960 driver)
- **Encoder Resolution:** $2.5882\text{ ticks/deg}$
- **Hardware Watchdog:** $500\text{ ms}$ (motors disabled if host heartbeat drops)
- **Hardware Stall Guard:** If PWM $> 35$ and encoder movement $< 2\text{ ticks}$ for $> 1.5\text{ s}$, stall flag is raised and power is cut.
