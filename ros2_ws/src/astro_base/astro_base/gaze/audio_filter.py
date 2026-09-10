"""Audio State Estimation & Filtering Pipeline.

Combines:
  1. Circular Outlier Rejection with Persistence Counter
  2. Sliding Window Circular Median Filter
  3. 2-State Circular Kalman Filter (Angle θ and Angular Velocity ω)
  4. Head Motion Velocity Compensation
"""

import collections
import math
from typing import Deque, Optional, Tuple
import numpy as np

from astro_base.gaze.angle_math import (
    angular_diff_deg,
    circular_distance_deg,
    wrap_deg,
)
from astro_base.gaze.motion_compensator import HeadMotionCompensator
from astro_base.gaze.types import AudioObservation, FilteredAudioState


class CircularMedianFilter:
    """Computes circular median over a sliding history window."""

    def __init__(self, window_size: int = 5):
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self._history: Deque[float] = collections.deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def update(self, angle_deg: float) -> float:
        self._history.append(wrap_deg(angle_deg))
        if len(self._history) == 1:
            return self._history[0]

        # The circular median minimizes the sum of circular distances to all samples
        best_angle = self._history[0]
        min_total_dist = float("inf")

        for candidate in self._history:
            total_dist = sum(circular_distance_deg(candidate, other) for other in self._history)
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_angle = candidate

        return best_angle


class CircularKalmanEstimator:
    """2-State Circular Kalman Filter tracking [Azimuth θ, Angular Velocity ω]."""

    def __init__(
        self,
        process_noise_q: float = 0.08,
        measurement_noise_r: float = 0.45,
    ):
        self.q = process_noise_q
        self.r = measurement_noise_r

        # State: [theta_deg, omega_deg_s]
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        # Covariance matrix P
        self.P = np.array([[10.0, 0.0], [0.0, 5.0]], dtype=np.float64)
        self.last_timestamp: Optional[float] = None
        self.initialized = False

    def reset(self, initial_angle_deg: float = 0.0) -> None:
        self.x = np.array([wrap_deg(initial_angle_deg), 0.0], dtype=np.float64)
        self.P = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        self.last_timestamp = None
        self.initialized = False

    def predict(self, dt: float) -> Tuple[float, float]:
        """Kalman Prediction Step."""
        dt = max(0.001, min(0.5, dt))

        # State transition F = [[1, dt], [0, 1]]
        theta_pred = wrap_deg(float(self.x[0] + self.x[1] * dt))
        omega_pred = float(self.x[1])
        self.x = np.array([theta_pred, omega_pred], dtype=np.float64)

        # Process noise covariance Q (discrete white noise acceleration model)
        q_theta = (dt ** 3 / 3.0) * self.q * 100.0
        q_omega = dt * self.q * 100.0
        q_cov = (dt ** 2 / 2.0) * self.q * 100.0
        Q = np.array([[q_theta, q_cov], [q_cov, q_omega]], dtype=np.float64)

        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.P = F @ self.P @ F.T + Q

        return float(self.x[0]), float(self.x[1])

    def update(self, measurement_deg: float, meas_variance: Optional[float] = None) -> Tuple[float, float]:
        """Kalman Measurement Update with Circular Residual Wrapping."""
        meas_deg = wrap_deg(measurement_deg)
        r_val = meas_variance if (meas_variance is not None and meas_variance > 0) else self.r

        # Innovation (residual) with circular wrapping
        # y = angular_diff(z, H @ x)
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        pred_theta = self.x[0]
        residual = angular_diff_deg(meas_deg, pred_theta)

        # Innovation covariance S = H P H^T + R
        S = float(self.P[0, 0] + r_val)
        # Kalman Gain K = P H^T / S
        K = np.array([[self.P[0, 0] / S], [self.P[1, 0] / S]], dtype=np.float64)

        # State update: x = x + K * y
        self.x[0] = wrap_deg(float(self.x[0] + K[0, 0] * residual))
        self.x[1] = float(self.x[1] + K[1, 0] * residual)

        # Covariance update: P = (I - K H) P
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        return float(self.x[0]), float(self.x[1])

    def step(self, measurement_deg: float, timestamp: float) -> Tuple[float, float, float]:
        """Executes full Predict + Update cycle given a new timestamped observation."""
        if not self.initialized or self.last_timestamp is None:
            self.x = np.array([wrap_deg(measurement_deg), 0.0], dtype=np.float64)
            self.last_timestamp = timestamp
            self.initialized = True
            return float(self.x[0]), float(self.x[1]), float(self.P[0, 0])

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        self.predict(dt)
        self.update(measurement_deg)

        return float(self.x[0]), float(self.x[1]), float(self.P[0, 0])


class AudioFilterCore:
    """Complete Audio State Estimator combining gating, median, and Kalman filters."""

    def __init__(
        self,
        max_jump_deg: float = 35.0,
        outlier_persistence_count: int = 3,
        median_window_size: int = 5,
        kalman_q: float = 0.08,
        kalman_r: float = 0.45,
        track_timeout_s: float = 1.5,
        motion_compensator: Optional[HeadMotionCompensator] = None,
    ):
        self.max_jump_deg = max_jump_deg
        self.outlier_persistence_count = outlier_persistence_count
        self.track_timeout_s = track_timeout_s
        self.motion_compensator = motion_compensator or HeadMotionCompensator()

        self.median_filter = CircularMedianFilter(window_size=median_window_size)
        self.kalman = CircularKalmanEstimator(process_noise_q=kalman_q, measurement_noise_r=kalman_r)

        self._last_accepted_angle: Optional[float] = None
        self._last_accepted_time: Optional[float] = None
        self._outlier_candidate_angle: Optional[float] = None
        self._outlier_candidate_time: Optional[float] = None
        self._outlier_streak: int = 0

    def reset(self) -> None:
        self.median_filter.reset()
        self.kalman.reset()
        self._last_accepted_angle = None
        self._last_accepted_time = None
        self._outlier_candidate_angle = None
        self._outlier_candidate_time = None
        self._outlier_streak = 0

    def filter_observation(
        self,
        obs: AudioObservation,
        head_velocity_deg_s: float = 0.0,
    ) -> FilteredAudioState:
        """Filters an incoming AudioObservation through the full estimation pipeline."""
        if not obs.valid or obs.confidence <= 0.0:
            return FilteredAudioState(
                timestamp=obs.timestamp,
                valid=False,
                azimuth_deg=self._last_accepted_angle or 0.0,
                angular_velocity_deg_s=0.0,
                confidence=0.0
            )

        raw_body_yaw = obs.body_azimuth_deg
        raw_conf = obs.confidence

        # 1. Motion Compensation
        comp_conf, was_attenuated = self.motion_compensator.compensate_confidence(
            raw_confidence=raw_conf,
            head_velocity_deg_s=head_velocity_deg_s,
            timestamp=obs.timestamp,
        )

        # 2. Outlier Gating with Track Expiry
        is_outlier = False
        time_since_accepted = (
            obs.timestamp - self._last_accepted_time
            if self._last_accepted_time is not None
            else float("inf")
        )
        if self._last_accepted_angle is not None and time_since_accepted <= self.track_timeout_s:
            jump = circular_distance_deg(raw_body_yaw, self._last_accepted_angle)
            if jump > self.max_jump_deg:
                # Check persistence streak for genuine speaker switches
                time_since_outlier = (
                    obs.timestamp - self._outlier_candidate_time
                    if self._outlier_candidate_time is not None
                    else float("inf")
                )
                if (
                    self._outlier_candidate_angle is not None
                    and time_since_outlier <= self.track_timeout_s
                    and circular_distance_deg(raw_body_yaw, self._outlier_candidate_angle) <= 15.0
                ):
                    self._outlier_streak += 1
                else:
                    self._outlier_candidate_angle = raw_body_yaw
                    self._outlier_candidate_time = obs.timestamp
                    self._outlier_streak = 1

                if self._outlier_streak < self.outlier_persistence_count:
                    # Reject as isolated outlier spike
                    is_outlier = True
                else:
                    # Sustained movement confirmed: accept new heading & reset sliding buffers
                    self._last_accepted_angle = raw_body_yaw
                    self._last_accepted_time = obs.timestamp
                    self._outlier_candidate_angle = None
                    self._outlier_candidate_time = None
                    self._outlier_streak = 0
                    self.median_filter.reset()
                    self.kalman.reset(initial_angle_deg=raw_body_yaw)
            else:
                self._last_accepted_angle = raw_body_yaw
                self._last_accepted_time = obs.timestamp
                self._outlier_candidate_angle = None
                self._outlier_candidate_time = None
                self._outlier_streak = 0
        else:
            # First observation or track expired: accept immediately as new sound track
            self._last_accepted_angle = raw_body_yaw
            self._last_accepted_time = obs.timestamp
            self._outlier_candidate_angle = None
            self._outlier_candidate_time = None
            self._outlier_streak = 0
            self.median_filter.reset()
            self.kalman.reset(initial_angle_deg=raw_body_yaw)

        if is_outlier:
            return FilteredAudioState(
                timestamp=obs.timestamp,
                valid=False,
                azimuth_deg=float(self.kalman.x[0]),
                angular_velocity_deg_s=float(self.kalman.x[1]),
                confidence=0.0,
                is_outlier=True,
                motion_attenuated=was_attenuated,
            )

        # 3. Sliding Median Filtering
        median_yaw = self.median_filter.update(raw_body_yaw)

        # 4. Circular Kalman State Estimation
        filt_theta, filt_omega, est_var = self.kalman.step(median_yaw, obs.timestamp)

        return FilteredAudioState(
            timestamp=obs.timestamp,
            valid=True,
            azimuth_deg=round(filt_theta, 1),
            angular_velocity_deg_s=round(filt_omega, 2),
            confidence=round(comp_conf, 2),
            variance=round(est_var, 3),
            is_outlier=False,
            motion_attenuated=was_attenuated,
        )
