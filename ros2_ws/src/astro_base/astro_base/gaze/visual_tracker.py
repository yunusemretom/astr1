"""3D Multi-Target Visual Tracking Pipeline.

Features:
  1. 6-State 3D Constant-Velocity Kalman Filter [x, y, z, vx, vy, vz]
  2. Bipartite Euclidean / Mahalanobis Spatial Data Association
  3. Track Lifecycle Management: DETECTED -> TRACKING -> COASTING -> LOST
  4. Temporal Coasting to survive momentary visual dropouts / blinks
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from astro_base.gaze.angle_math import wrap_deg
from astro_base.gaze.coordinate_frames import CoordinateTransformer
from astro_base.gaze.types import TrackingState, VisualObservation, VisualTargetTrack


class KalmanTrack3D:
    """3D Constant-Velocity Kalman Filter for tracking a single person."""

    def __init__(
        self,
        track_id: str,
        initial_pos_3d: Tuple[float, float, float],
        timestamp: float,
        obs: VisualObservation,
        process_noise_q: float = 0.50,
        measurement_noise_r: float = 0.15,
    ):
        self.track_id = track_id
        self.q = process_noise_q
        self.r = measurement_noise_r

        # State vector x: [x, y, z, vx, vy, vz]^T
        x0, y0, z0 = initial_pos_3d
        self.x = np.array([x0, y0, z0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Covariance matrix P (6x6)
        self.P = np.diag([0.2, 0.2, 0.3, 1.0, 1.0, 1.0]).astype(np.float64)

        self.last_update_time = timestamp
        self.last_seen_time = timestamp
        self.age_frames = 1
        self.hit_count = 1
        self.missed_frames = 0
        self.state = TrackingState.DETECTED

        # Visual metadata
        self.confidence = obs.confidence
        self.emotion = obs.emotion
        self.person_name = obs.person_name
        self.is_known = obs.is_known
        self.eye_contact = obs.eye_contact

    def predict(self, dt: float) -> Tuple[float, float, float]:
        """Kalman Prediction Step."""
        dt = max(0.001, min(0.5, dt))

        # State transition F = [[I3, dt*I3], [03, I3]]
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x = F @ self.x

        # Process noise covariance Q
        q_pos = (dt ** 3 / 3.0) * self.q
        q_vel = dt * self.q
        q_cov = (dt ** 2 / 2.0) * self.q

        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] = q_pos
            Q[i + 3, i + 3] = q_vel
            Q[i, i + 3] = q_cov
            Q[i + 3, i] = q_cov

        self.P = F @ self.P @ F.T + Q
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def update(self, obs_pos_3d: Tuple[float, float, float], obs: VisualObservation, timestamp: float) -> None:
        """Kalman Measurement Update Step."""
        z_meas = np.array(obs_pos_3d, dtype=np.float64)
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.eye(3, dtype=np.float64) * self.r

        # Innovation: y = z - H @ x
        residual = z_meas - H @ self.x

        # Innovation covariance: S = H P H^T + R
        S = H @ self.P @ H.T + R
        # Kalman Gain: K = P H^T S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ residual

        # Covariance update: P = (I - K H) P
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        self.last_update_time = timestamp
        self.last_seen_time = timestamp
        self.hit_count += 1
        self.missed_frames = 0

        # Promote to confirmed TRACKING after 2 hits
        if self.hit_count >= 2:
            self.state = TrackingState.TRACKING

        # Update metadata
        self.confidence = 0.7 * self.confidence + 0.3 * obs.confidence
        self.emotion = obs.emotion
        if obs.is_known:
            self.person_name = obs.person_name
            self.is_known = True
        self.eye_contact = obs.eye_contact

    def mark_missed(self, timestamp: float, coast_timeout_s: float = 0.70) -> None:
        """Marks track as unobserved in current frame; promotes to COASTING or LOST."""
        self.missed_frames += 1
        self.confidence = max(0.1, self.confidence * 0.85)

        time_since_seen = timestamp - self.last_seen_time
        if time_since_seen > coast_timeout_s:
            self.state = TrackingState.LOST
        else:
            self.state = TrackingState.COASTING

    def get_track_summary(self) -> VisualTargetTrack:
        """Generates a VisualTargetTrack summary dataclass."""
        x, y, z = float(self.x[0]), float(self.x[1]), float(self.x[2])
        vx, vy, vz = float(self.x[3]), float(self.x[4]), float(self.x[5])

        dist_m = float(math.sqrt(x ** 2 + y ** 2 + z ** 2))
        azimuth_deg = float(math.degrees(math.atan2(y, x)))
        elevation_deg = float(math.degrees(math.atan2(z, math.hypot(x, y)))) if dist_m > 0.1 else 0.0

        return VisualTargetTrack(
            target_id=self.track_id,
            pos_3d=(round(x, 3), round(y, 3), round(z, 3)),
            vel_3d=(round(vx, 3), round(vy, 3), round(vz, 3)),
            body_azimuth_deg=round(wrap_deg(azimuth_deg), 1),
            body_elevation_deg=round(elevation_deg, 1),
            distance_m=round(dist_m, 2),
            confidence=round(self.confidence, 2),
            tracking_state=self.state,
            last_seen_time=self.last_seen_time,
            age_frames=self.age_frames,
            missed_frames=self.missed_frames,
            emotion=self.emotion,
            person_name=self.person_name,
            is_known=self.is_known,
            eye_contact=self.eye_contact,
        )


class VisualTrackerCore:
    """Multi-target 3D spatial tracker managing persistence and coasting."""

    def __init__(
        self,
        transformer: Optional[CoordinateTransformer] = None,
        gating_distance_m: float = 0.85,
        coasting_timeout_s: float = 0.70,
    ):
        self.transformer = transformer or CoordinateTransformer()
        self.gating_distance_m = gating_distance_m
        self.coasting_timeout_s = coasting_timeout_s

        self.tracks: Dict[str, KalmanTrack3D] = {}
        self._next_track_idx = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_idx = 1

    def update(
        self,
        observations: List[VisualObservation],
        timestamp: float,
        actual_head_yaw_deg: float = 0.0,
    ) -> List[VisualTargetTrack]:
        """Updates all active tracks with the latest list of VisualObservations.

        Executes:
          1. Kalman Prediction for all active tracks
          2. 3D Spatial Association (greedy Euclidean bipartite matching)
          3. Measurement update for matched tracks
          4. Coasting update for missed tracks
          5. Initialization of new tracks for unassigned detections
          6. Purging of expired (LOST) tracks
        """
        # Filter only valid observations
        valid_obs = [o for o in observations if o.valid and o.depth_m > 0.1]

        # 1. Prediction step for all existing tracks
        for track in self.tracks.values():
            dt = timestamp - track.last_update_time
            track.predict(dt)
            track.age_frames += 1

        # Transform observations to 3D base coordinates
        obs_base_coords: List[Tuple[float, float, float]] = []
        for o in valid_obs:
            base_pt = self.transformer.camera_point_to_body_frame(
                o.pos_3d_camera, actual_head_yaw_deg
            )
            obs_base_coords.append(base_pt)

        # 2. Bipartite Matching (Greedy Association)
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_obs = set()

        if track_ids and obs_base_coords:
            # Build cost matrix (3D Euclidean distance)
            cost_matrix = np.zeros((len(track_ids), len(obs_base_coords)), dtype=np.float64)
            for i, tid in enumerate(track_ids):
                track = self.tracks[tid]
                tx, ty, tz = track.x[0], track.x[1], track.x[2]
                for j, (ox, oy, oz) in enumerate(obs_base_coords):
                    dist = math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2 + (tz - oz) ** 2)
                    cost_matrix[i, j] = dist

            # Greedy assignment of minimum distances below gating threshold
            while True:
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_val = cost_matrix[min_idx]
                if min_val > self.gating_distance_m or min_val == float("inf"):
                    break

                t_idx, o_idx = min_idx
                matched_tracks.add(track_ids[t_idx])
                matched_obs.add(o_idx)

                # Update the matched track
                self.tracks[track_ids[t_idx]].update(
                    obs_base_coords[o_idx], valid_obs[o_idx], timestamp
                )

                # Set row and column to infinity
                cost_matrix[t_idx, :] = float("inf")
                cost_matrix[:, o_idx] = float("inf")

        # 3. Unmatched tracks -> Mark missed / coasting
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.mark_missed(timestamp, self.coasting_timeout_s)

        # 4. Unmatched observations -> Initialize new candidate tracks
        for j, (ox, oy, oz) in enumerate(obs_base_coords):
            if j not in matched_obs:
                new_id = f"person_{self._next_track_idx}"
                self._next_track_idx += 1
                new_track = KalmanTrack3D(
                    track_id=new_id,
                    initial_pos_3d=(ox, oy, oz),
                    timestamp=timestamp,
                    obs=valid_obs[j],
                )
                self.tracks[new_id] = new_track

        # 5. Purge expired LOST tracks
        active_tids = [tid for tid, t in self.tracks.items() if t.state != TrackingState.LOST]
        self.tracks = {tid: self.tracks[tid] for tid in active_tids}

        # 6. Generate track summaries sorted by confidence / proximity
        summaries = [t.get_track_summary() for t in self.tracks.values()]
        summaries.sort(key=lambda tr: (-tr.confidence, tr.distance_m))

        return summaries
