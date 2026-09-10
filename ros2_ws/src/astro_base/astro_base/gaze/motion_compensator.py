"""Head Motion Compensation for Audio Sensing.

When the robot neck is in motion, structure-borne motor vibration reaches the microphone
capsules and degrades Direction of Arrival (DOA) accuracy. This module dynamically
attenuates acoustic confidence based on head angular velocity and a post-motion quiet settle window.
"""

from typing import Tuple


class HeadMotionCompensator:
    """Dynamically scales acoustic confidence based on physical head velocity and settle time."""

    def __init__(
        self,
        max_sens_velocity_deg_s: float = 40.0,
        settle_time_s: float = 0.45,
        min_confidence_floor: float = 0.10,
    ):
        self.max_sens_velocity_deg_s = max_sens_velocity_deg_s
        self.settle_time_s = settle_time_s
        self.min_confidence_floor = min_confidence_floor
        self._last_motion_time: float = 0.0

    def record_motion_event(self, timestamp: float) -> None:
        """Notifies the compensator that a physical head movement was executed."""
        self._last_motion_time = timestamp

    def is_in_settle_window(self, timestamp: float) -> bool:
        """Returns True if within the post-motion acoustic settle window."""
        return (timestamp - self._last_motion_time) < self.settle_time_s

    def compensate_confidence(
        self,
        raw_confidence: float,
        head_velocity_deg_s: float,
        timestamp: float,
    ) -> Tuple[float, bool]:
        """Calculates motion-compensated acoustic confidence.

        Returns:
          (compensated_confidence, was_attenuated)
        """
        if raw_confidence <= 0.0:
            return 0.0, False

        vel_abs = abs(head_velocity_deg_s)
        time_since_motion = timestamp - self._last_motion_time

        # 1. Velocity-based attenuation factor [min_floor..1.0]
        vel_ratio = min(1.0, vel_abs / max(1.0, self.max_sens_velocity_deg_s))
        vel_factor = max(self.min_confidence_floor, 1.0 - (0.85 * vel_ratio))

        # 2. Settle-window attenuation factor
        settle_factor = 1.0
        if time_since_motion < self.settle_time_s:
            progress = max(0.0, time_since_motion / max(0.001, self.settle_time_s))
            settle_factor = max(self.min_confidence_floor, 0.25 + 0.75 * progress)

        combined_scale = min(vel_factor, settle_factor)
        compensated = float(round(raw_confidence * combined_scale, 3))
        was_attenuated = (combined_scale < 0.95)

        return compensated, was_attenuated
