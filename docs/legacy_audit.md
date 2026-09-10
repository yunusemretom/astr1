# ASTRO — Legacy Head Gaze & Audio-Visual Tracking Audit
**Document ID:** `docs/legacy_audit.md`  
**Classification:** Engineering Audit & Architectural Transition Report  
**Branch:** `feat/social-gaze-audiovisual-tracking`  

---

## 1. Executive Summary & Audit Scope

This document provides a comprehensive, ground-truth audit of the legacy perception, head control, and audio-visual tracking subsystems across the ASTRO repository.

The legacy implementation in `ros2_ws/src/astro_base/astro_base/head_tracker_node.py` (998 lines) suffered from architectural coupling, mixed abstraction levels, and direct sensor-to-actuator control loops. This audit establishes what components will be discarded, what low-level hardware interfaces and utilities can be reused, and the structural deficiencies that necessitated a greenfield rebuild.

---

## 2. Environment & Toolchain Audit

* **ROS 2 Distribution:** ROS 2 Humble Hawksbill `[VERIFIED FACT]`
* **Operating System Target:** Ubuntu 22.04 LTS (x86_64 and aarch64 Jetson Orin Nano) `[VERIFIED FACT]` / Windows (Local Development) `[OBSERVED IN REPOSITORY]`
* **Python Runtime:** Python 3.10 / 3.12 compatible `[VERIFIED FACT]`
* **C++ Standard:** C++17 / GCC 11 `[OBSERVED IN REPOSITORY]`
* **Microcontroller Target:** Arduino Mega 2560 (ATmega2560 @ 16 MHz), compiled with Arduino IDE / PlatformIO `[VERIFIED FACT]`
* **Core Build Tool:** `colcon` with `ament_cmake` / `ament_python` `[VERIFIED FACT]`

---

## 3. Legacy Package Analysis & Component Status

| Package | Files Inspected | Legacy Role | Status for Greenfield Architecture |
|---|---|---|---|
| `astro_base` | `head_tracker_node.py` | Monolithic head tracking, gaze, VAD, DOA, fusion, slew-rate limiter | ❌ **DISCARD COMPLETELY.** Monolithic anti-pattern; replaced by modular 7-layer pipeline. |
| `astro_base` | `serial_bridge.py` | Binary packet bridge to Arduino Mega (protocol v2.0) | 🟡 **REFACTOR & MODERNIZE.** CRC8 packet framing, heartbeat, and serial I/O are verified; add actual head state streaming. |
| `astro_base` | `diff_drive_node.py` | Base differential drive kinematics & odometry | 🟢 **RETAIN AS-IS.** Independent base drive. |
| `astro_base` | `msg/HeadCmd.msg`, `WheelCmd.msg` | Basic ROS messages (`angle_deg`) | 🟡 **EXPAND.** Add rich typed messages (`HeadState`, `AudioTarget`, `VisualTarget`, `FusedTarget`, `GazeStatus`). |
| `astro_audio` | `doa_estimator.py` | GCC-PHAT TDOA computation for 4-mic array | 🟢 **REUSE & ENHANCE.** Math is mathematically sound; integrate into pure `AudioPerceptionCore`. |
| `astro_audio` | `audio_capture_node.py` | USB sounddevice capture + HID DOA read | 🟡 **REUSE DRIVER LOGIC.** Clean interface to provide multichannel PCM and VAD. |
| `astro_vision` | `face_detector_node.py` | OpenCV Haar cascades + stereo depth + eye verification | 🟡 **REUSE SENSING LOGIC.** Separate detection from tracking; feed into pure `VisualPerceptionCore`. |
| `astro_vision` | `oak_perception_node.py` | DepthAI native pipeline for OAK-D Lite | 🟢 **REUSE AS HARDWARE DRIVER.** High-performance camera driver. |
| `astro_description` | `astro.urdf.xacro` | URDF kinodynamics & sensor TF tree | 🟢 **RETAIN TRUTH.** `head_yaw_joint` ($\pm 90^\circ$), `oak_link`, `mic_link` rigidly defined. |
| `arduino/AstroFirmware` | `AstroFirmware.ino`, `protocol.h` | 50 Hz PID position controller + watchdog | 🟢 **RETAIN MCU CONTROL.** Low-level PID and safety limits verified on hardware. |

---

## 4. Legacy Architecture Deficiencies & Anti-Patterns

### 4.1. Monolithic Node Anti-Pattern (`head_tracker_node.py`)
* **Problem:** A single 998-line node was responsible for:
  1. Subscribing to 12 disparate topics (`/audio/doa`, `/audio/mic_level`, `/audio/vad`, `/vision/head_yaw`, `/vision/faces`, `/scan`, `/tts/speaking`, `/robot/emotion`, `/head/gesture`, `/head/target_yaw`, `/head/safety`).
  2. Outlier rejection and circular consensus.
  3. Spatial association of visual faces and acoustic targets.
  4. Behavior state machine (`SocialGazeStateMachine`).
  5. Slew-rate trajectory generation.
  6. Rate limiting and directly publishing `/head_cmd`.
* **Impact:** High cognitive coupling, impossible unit test isolation, race conditions on shared thread locks, inability to independently benchmark or replace algorithms (e.g. testing Kalman vs EMA).

### 4.2. Reactive Sensor-to-Actuator Coupling
* **Problem:** Direct callback-driven setpoint updates occurred whenever sensor data arrived.
* **Impact:** Sensor noise and momentary DOA spikes caused micro-jitter and neck twitching.

### 4.3. Lack of Closed-Loop Feedback Integration
* **Problem:** `head_tracker_node.py` maintained an open-loop software estimate (`_estimated_yaw`) of head position and assumed the motor instantly tracked it or lagged by a constant. The actual physical encoder feedback remained on the microcontroller.
* **Impact:** When the physical neck lagged under load or was physically restrained, perception algorithms calculated frame transformations against an incorrect head angle.

### 4.4. Magic Numbers & Uncalibrated Thresholds
* **Problem:** Hardcoded magic values distributed throughout the Python source (`min_rms_threshold = 1600.0`, `consensus_tolerance_deg = 22.0`, `vision_max_correction_deg = 36.0`, `HEAD_TICKS_PER_DEG = 2.588`).
* **Impact:** System could not be tuned per physical robot instance without code edits.

---

## 5. Greenfield Rebuild Strategy

1. **Pure Core Modules (`astro_base.gaze.core.*`):**
   * Zero ROS dependencies (`rclpy` not imported).
   * 100% deterministic, testable via standard `pytest`.
   * Standardized floating-point math, circular angle handling, explicit matrix state estimations.
2. **Decoupled ROS 2 Node Architecture:**
   * Nodes serve purely as communication adapters and parameter injectors.
3. **Explicit Multi-Rate Pipeline:**
   * Audio Perception: Event/frame driven (~16–50 Hz).
   * Visual Perception: Camera frame driven (~15–30 Hz).
   * Multimodal Fusion & Target Management: Periodic state estimation (20–50 Hz).
   * Motion Planner: Smooth trajectory interpolation (50–100 Hz).
   * Low-Level MCU Controller: 50 Hz closed-loop position PID.
4. **Comprehensive Configuration & Telemetry:**
   * Structured YAML configuration for calibration, filtering, fusion, gaze FSM, and motion limits.
   * Telemetry publishing for RViz / PlotJuggler / ROS bag analysis.
