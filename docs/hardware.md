# ASTRO — Hardware Truth & Ground Truth Specifications
**Document ID:** `docs/hardware.md`  
**Classification:** Hardware Verification & Mechanical Constraint Specification  
**Branch:** `feat/social-gaze-audiovisual-tracking`  

---

## 1. Physical Hardware Inventory & Verification Status

Every claim in this document is tagged with its empirical verification level:
* `[VERIFIED FACT]`: Explicitly confirmed by hardware datasheet, source code, and working test bench.
* `[OBSERVED IN REPOSITORY]`: Discovered in existing codebase or configs, but depends on physical calibration.
* `[ENGINEERING ASSUMPTION]`: Deduced from standard physical robotics constraints.
* `[RESEARCH FINDING]`: Established from academic literature or industrial social robotics benchmarks.
* `[UNKNOWN]`: Unverified or physically unmeasured parameter requiring hardware calibration.

---

## 2. ReSpeaker Mic Array v3.0 / USB 4-Mic Array

* **Microphone Topology:** 4 circular MEMS microphones placed symmetrically at radius $R = 43.0\text{ mm}$ ($0.043\text{ m}$) `[VERIFIED FACT]`.
* **Opposing Pair Spacing:** $2 \times R = 86.0\text{ mm}$ ($0.086\text{ m}$) between Mic 0–2 (Front–Back) and Mic 1–3 (Right–Left) `[VERIFIED FACT]`.
* **Audio Sampling Rate:** 16,000 Hz (16 kHz), 16-bit signed integer linear PCM `[VERIFIED FACT]`.
* **Speed of Sound Constant ($c$):** $343.0\text{ m/s}$ at $20^\circ\text{C}$ in air `[VERIFIED FACT]`.
* **Maximum Acoustic Time Delay ($\tau_{\max}$):**
  $$\tau_{\max} = \frac{d}{c} = \frac{0.086\text{ m}}{343.0\text{ m/s}} \approx 0.2507\text{ ms} \quad (4.01\text{ samples at } 16\text{ kHz}) \quad \text{[VERIFIED FACT]}$$
* **USB Vendor ID / Product ID:** VID `0x2886`, PID `0x0018` `[VERIFIED FACT]`.
* **On-Board Processing:** Hardware AEC (Acoustic Echo Cancellation), BF (Beamforming), VAD, and DOA register via USB HID control transfer `[VERIFIED FACT]`.
* **Mounting Point:** Rigidly fastened to robot `head_link` (`mic_joint` in `astro.urdf.xacro` at $z = +0.10\text{ m}$ relative to head center) `[VERIFIED FACT]`.
* **Consequence of Head Mounting:** ReSpeaker rotates with the robot head. Therefore, raw DOA is **relative to the head**, requiring composition with actual head yaw $\theta_{head}$ to determine the world/body frame sound azimuth `[VERIFIED FACT]`.
* **Acoustic Noise Artifacts:** Head DC motor structure-borne vibration introduces acoustic noise during yaw motion, necessitating velocity-based confidence gating `[VERIFIED FACT]`.

---

## 3. Luxonis OAK-D Lite (Spatial AI RGB-D Camera)

* **Sensors:** 4K RGB color camera + Stereo Depth pairs (Left/Right mono OV7251 cameras) `[VERIFIED FACT]`.
* **Horizontal Field of View (HFOV):** $\approx 72.0^\circ$ ($\pm 36.0^\circ$ half-angle from optical center) `[VERIFIED FACT]`.
* **Vertical Field of View (VFOV):** $\approx 53.0^\circ$ `[VERIFIED FACT]`.
* **Stereo Baseline:** $75.0\text{ mm}$ ($0.075\text{ m}$) `[VERIFIED FACT]`.
* **Effective Depth Range:** $0.20\text{ m} \le z \le 8.0\text{ m}$ (Optimal social interaction range: $0.40\text{ m} \le z \le 2.50\text{ m}$) `[VERIFIED FACT]`.
* **Optical Frame Convention (REP-103):** $+Z$ forward along optical axis, $+X$ to the right of image sensor, $+Y$ downwards `[VERIFIED FACT]`.
* **Mounting Point:** Rigidly fastened to `head_link` (`oak_joint` at $x = +0.06\text{ m}, z = +0.02\text{ m}$) `[VERIFIED FACT]`.
* **Processing Capabilities:** On-board Myriad X VPU capable of running MobileNet-SSD / Face Detection / Stereo Depth disparity at 30 FPS `[VERIFIED FACT]`.

---

## 4. Microcontroller, Motor Driver & Head Actuator

* **Microcontroller:** Arduino Mega 2560 (Microchip ATmega2560 @ 16 MHz, 5V logic) `[VERIFIED FACT]`.
* **Host Serial Interface:** UART0 via CH340/CH341 USB bridge, standard baud rate: $115,200\text{ baud}$, 8N1 `[VERIFIED FACT]`.
* **Debug Telemetry Interface:** UART2 (Serial2 on Pins 16/17 TX2/RX2) @ $115,200\text{ baud}$ for forensic diagnosis without interfering with UART0 binary stream `[VERIFIED FACT]`.
* **Head Motor Driver:** BTS7960 43A High-Power H-Bridge, driven via hardware PWM on Timer5 (Pins 44/45, phase-correct 31.37 kHz PWM) `[VERIFIED FACT]`.
* **Head Actuator Type:** Brushed DC Gearmotor with incremental quadrature encoder `[VERIFIED FACT]`.
* **Encoder Resolution / Calibration:**
  $$\text{Calibrated scale factor} = \frac{440\text{ ticks}}{170.0^\circ} = 2.5882\text{ ticks/degree} \quad \text{[VERIFIED FACT]}$$
* **Mechanical Limits:**
  * Software safety range: $\theta_{\min} = -90.0^\circ$, $\theta_{\max} = +90.0^\circ$ ($180.0^\circ$ total reachable arc) `[VERIFIED FACT]`.
  * Mechanical hard stops exist beyond $\pm 90^\circ$; no limit switches installed `[VERIFIED FACT]`.
  * Homing strategy: Robot power-up position is defined as $\theta = 0.0^\circ$ (Center) `[VERIFIED FACT]`.
* **Firmware Control Loop Rate:** $50.0\text{ Hz}$ ($\Delta t = 20\text{ ms}$) `[VERIFIED FACT]`.
* **Firmware Head PID Gains:** $K_p = 2.0$, $K_d = 0.05$, Feedforward Stiction PWM Base $= 35$, Max PWM limit $= 150/255$ `[VERIFIED FACT]`.
* **Stall Protection:** If PWM is applied and encoder movement $< 2\text{ ticks}$ for $> 1500\text{ ms}$, motor PWM is clamped to 0 and target is reset to current ticks `[VERIFIED FACT]`.
* **Hardware Watchdog:** AVR Hardware Watchdog timer set to 2.0s; host communication watchdog disables motor outputs if no heartbeat or command received within $500\text{ ms}$ `[VERIFIED FACT]`.

---

## 5. Serial Binary Packet Protocol (Protocol v2.0)

* **Packet Structure:**
  ```text
  [SOF1: 0xAA] [SOF2: 0x55] [LEN: 1+N] [MSG_ID: 1 Byte] [PAYLOAD: N Bytes] [CRC8: 1 Byte]
  ```
* **CRC-8 Algorithm:** CRC-8-ATM, polynomial $0x07$, initial value $0x00$ `[VERIFIED FACT]`.
* **Message ID Map:**
  * `0x01` (`HEARTBEAT`): Host $\rightarrow$ MCU (`uint32 seq`) `[VERIFIED FACT]`.
  * `0x02` (`WHEEL_CMD`): Host $\rightarrow$ MCU (`float32 left_rpm`, `float32 right_rpm`) `[VERIFIED FACT]`.
  * `0x03` (`HEAD_CMD`): Host $\rightarrow$ MCU (`float32 angle_deg`) `[VERIFIED FACT]`.
  * `0x10` (`IMU_DATA`): MCU $\rightarrow$ Host (`float32[6] ax,ay,az,gx,gy,gz`, `uint32 micros`) `[OBSERVED IN REPOSITORY]`.
  * `0x11` (`ENCODER_TICKS`): MCU $\rightarrow$ Host (`int32 dl`, `int32 dr`, `uint32 dt_us`) `[VERIFIED FACT]`.
  * `0x12` (`DIAGNOSTICS`): MCU $\rightarrow$ Host (`uint16 vbat_mV`, `int16 mcu_temp_cX100`, `uint32 flags`) `[VERIFIED FACT]`.
  * `0x13` (`HEARTBEAT_ACK`): MCU $\rightarrow$ Host (`uint32 seq`) `[VERIFIED FACT]`.

---

## 6. Coordinate Frames & Transformations (REP-103)

```text
[world / odom]
       │
       ▼ (Mobile base odometry & localization)
  [base_link]
       │
       ▼ (Revolute yaw joint: z = +0.21m, yaw = θ_head ∈ [-90°, +90°])
 [head_yaw_link]
       │
       ▼ (Fixed pitch joint)
  [head_link]
   ┌───┴────────────────────────┐
   │                            │
   ▼ (Fixed: [0.06, 0, 0.02])   ▼ (Fixed: [0, 0, 0.10])
[oak_link]                 [mic_link]
   │
   ▼ (Fixed: rpy = [-90°, 0, -90°])
[oak_rgb_camera_optical_frame]
```

### Azimuth / Angle Definitions:
* **Robot Body Frame (`base_link`):**
  * Yaw $0.0^\circ$ = Straight ahead ($+X$).
  * Yaw $+90.0^\circ$ = Left ($+Y$).
  * Yaw $-90.0^\circ$ = Right ($-Y$).
  * Yaw $\pm 180.0^\circ$ = Straight behind ($-X$).
* **Camera Optical Frame (`oak_rgb_camera_optical_frame`):**
  * $+Z$ = Forward into scene.
  * $+X$ = Camera Right $\implies$ negative robot yaw relative to optical axis.
  * $+Y$ = Camera Down $\implies$ negative robot pitch relative to optical axis.
  * Azimuth: $\alpha_{cam} = \text{atan2}(-x_{opt}, z_{opt})$.
  * Elevation: $\beta_{cam} = \text{atan2}(-y_{opt}, z_{opt})$.
* **Microphone Frame (`mic_link`):**
  * Rigidly rotates with `head_link`.
  * Sound DOA is reported relative to head front: $\theta_{rel}$.
  * Absolute body sound azimuth: $\theta_{body} = \text{wrap\_deg}(\theta_{head} + \theta_{rel})$.
