# ASTRO Head Control & Actuator Feedback

## 1. Hardware Actuation & Telemetry Pipeline

```text
[ Gaze Manager ]
       ↓ desired target (-35.0°)
[ Motion Planner (50 Hz) ]
       ↓ planned trajectory (-9.97°)
[ Serial Bridge ]
       ↓ 0x03 MSG_HEAD_CMD
[ Arduino Mega 2560 (AstroFirmware.ino) ]
       ↓ PID (Kp=3.8, Kd=1.2) + BTS7960 Motor Driver
[ Physical Head Actuator & Quadrature Optical Encoder ]
       ↓ 440 ticks / 170° (2.5882 ticks/deg)
[ 0x11 MSG_ENCODER_TICKS (16 bytes) ]
       ↓ dl (4B), dr (4B), head_ticks (4B), dt_us (4B)
[ Serial Bridge & /head/state (HeadState.msg) ]
       ↓ actual_yaw_deg (-34.9°), actual_vel_deg_s (0.1°/s), encoder_valid=True
[ SocialGazeFSM.update ]
```

## 2. HeadState ROS 2 Message Structure
```text
std_msgs/Header header
float64 position_deg          # Actual physical angle from encoder
float64 velocity_deg_s        # Filtered physical angular rate
float64 target_position_deg   # Commanded target setpoint
bool moving                   # True if |velocity| > 1.0 deg/s
bool at_target                # True if settled within tolerance for >=3 cycles
bool enabled                  # Motor power and safety gate state
bool watchdog_healthy         # Serial heartbeat and MCU health
bool encoder_valid            # True if real hardware encoder feedback stream active
uint8 fault_code              # 0=OK, 1=Watchdog, 2=Stall, 3=Limit
```

## 3. ReSpeaker DOA vs REP-103 Coordinate Convention
- **ReSpeaker Hardware Register**: Reports raw azimuth in range $[0 \dots 359^\circ]$ **clockwise** ($0^\circ = \text{Front}, 90^\circ = \text{Right}, 270^\circ = \text{Left}$).
- **ROS Standard (REP-103)**: Right-hand coordinate system with $+Z$ up ($+ = \text{Left (CCW)}, - = \text{Right (CW)}$).
- **Coordinate Transformer**: Converts raw hardware angle via `invert=true`:
  $$\text{bearing}_{\text{REP-103}} = -\text{raw\_azimuth}_{\text{CW}}$$
  - Example: Sound at Right: ReSpeaker reports $+35^\circ \implies \text{REP-103 bearing} = -35.0^\circ$ (Robot turns Right).
  - Example: Sound at Left: ReSpeaker reports $325^\circ \implies \text{REP-103 bearing} = +35.0^\circ$ (Robot turns Left).
