# ASTRO — Greenfield Social Robot Head Gaze System Architecture

## 1. Executive Summary

This document defines the greenfield architecture of the **ASTRO Social Robot Head Gaze & Audio-Visual Speaker Tracking System**. 

The architecture completely discards monolithic callback-driven control paradigms in favor of a strictly decoupled, feedback-controlled, and testable perception-control pipeline.

```mermaid
flowchart TD
    subgraph Sensors["1. Raw Sensors (Mounted on head_link)"]
        MicArray["ReSpeaker v3.0 (4-Mic Array, 16kHz)"]
        OakCam["OAK-D Lite (RGB + Stereo Depth)"]
        Encoders["Optical Head Encoder (2.588 ticks/deg)"]
    end

    subgraph Perception["2. Perception Core"]
        AudioPerc["AudioPerceptionCore<br/>(GCC-PHAT, Dynamic VAD, RMS, Self-Voice Suppression)"]
        VisualPerc["VisualPerceptionCore<br/>(3D Pinhole Backprojection, Eye Contact, Emotion)"]
    end

    subgraph Filtering["3. Filtering & State Estimation"]
        MotionComp["HeadMotionCompensator<br/>(Velocity-based confidence attenuation)"]
        AudioKalman["AudioFilterCore<br/>(Circular Outlier Gate, Median, 2-State Kalman)"]
        VisualTracker["VisualTrackerCore<br/>(6-State 3D Constant-Velocity Kalman Tracker)"]
    end

    subgraph FusionLayer["4. Multimodal Fusion & Target Management"]
        Fusion["AudioVisualFusionCore<br/>(Spatial Consistency Gate ≤25°, Freshness Decay)"]
        TargetMgr["TargetManagerCore<br/>(Dual Hysteresis [0.75 / 0.40], Attention Dwell ≥2.5s, Turn-Taking)"]
    end

    subgraph Decision["5. Behavioral Decision & Motion Planning"]
        GazeFSM["SocialGazeFSM<br/>(9-State Machine, Priority Arbiter: SAFETY > GESTURE > DIALOGUE > SPEAKER > IDLE)"]
        Planner["MotionPlannerCore<br/>(Smooth S-Curve/Trapezoidal Trajectory, Soft-Landing, Shortest Reachable Arc)"]
    end

    subgraph LowLevel["6. Actuator & Closed-Loop Control"]
        HeadCtrl["HeadControllerCore<br/>(Protocol v2.0 CRC8, 500ms Watchdog, Stall Detection)"]
        Arduino["Arduino Mega 2560 MCU<br/>(50 Hz Position PID, BTS7960 Motor Driver)"]
    end

    MicArray --> AudioPerc
    OakCam --> VisualPerc
    Encoders --> HeadCtrl

    AudioPerc --> AudioKalman
    MotionComp -.-> AudioKalman
    VisualPerc --> VisualTracker
    HeadCtrl -. Actual Velocity .-> MotionComp

    AudioKalman --> Fusion
    VisualTracker --> Fusion

    Fusion --> TargetMgr
    TargetMgr --> GazeFSM
    GazeFSM --> Planner
    HeadCtrl -. Actual Position .-> Planner

    Planner --> HeadCtrl
    HeadCtrl --> Arduino
```

---

## 2. Architectural Principles & Guarantees

1. **Separation of Concerns:**  
   Perception never directly writes to motor registers or publishes raw angles to hardware. Every stage has a defined data contract dataclass (`AudioObservation`, `FilteredAudioState`, `VisualTargetTrack`, `FusedTarget`, `TargetState`, `GazeCommand`, `TrajectoryPoint`, `HeadFeedback`).
2. **Deterministic 50 Hz Control Loop:**  
   Motor trajectories are evaluated synchronously at $50\text{ Hz}$ ($20\text{ ms}$ interval), isolating sensor callback jitter from physical motor driving.
3. **Circular Angle Mathematics:**  
   All angular calculations strictly handle the $180^\circ / -180^\circ$ circular branch seam using `wrap_deg`, `angular_diff_deg`, and `circular_mean_deg`.
4. **Natural Social Dynamics:**  
   Implements psychological attention dwell times ($\ge 2.5\text{ s}$), deadbands ($\ge 3.0^\circ$), and organic soft-landing deceleration profiles.
5. **Multi-Speaker Turn-Taking:**  
   Allows an expedited switch when a new speaker speaks distinctly ($\ge 20^\circ$ separation) for $\ge 0.80\text{ s}$.
6. **Safety & Self-Suppression:**  
   Automatic self-voice suppression when the robot is speaking, motion self-noise confidence attenuation during head turns, and hardware watchdog lockout if host-MCU communication is interrupted.
