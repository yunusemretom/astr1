"""ASTRO Social Robot Head Gaze & Audio-Visual Speaker Tracking System.

Clean Greenfield Architecture:
  - Sensing & Drivers
  - Audio Perception (GCC-PHAT, VAD, RMS, Self-Speech Gating)
  - Visual Perception & 3D Spatial Localization (OAK-D Lite RGB-D)
  - State Estimation & Temporal Filtering (Circular Outlier Gate, Median, Kalman)
  - Audio-Visual Sensor Fusion (Spatial Consistency Gating, Freshness Decay)
  - Target Management (Candidate/Active targets, Hysteresis, Multi-Speaker Turn Arbitration)
  - Social Gaze Manager (9-State FSM, Priority Arbitration, Deadband, Dwell)
  - Motion Planning (Jerk-Limited S-Curve & Trapezoidal Profiles, Soft-Landing)
  - Closed-Loop Low-Level Head Control (50 Hz Position PID, Safety Watchdog)
"""

__version__ = "2.0.0"

from astro_base.gaze.respeaker_localizer import ReSpeakerAudioLocalizer

__all__ = ["ReSpeakerAudioLocalizer"]
