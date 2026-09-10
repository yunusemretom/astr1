"""Motion Planning & Trajectory Generation Engine for ASTRO Head Gaze.

Features:
  1. Smooth Acceleration-Limited Trajectory Generation with Soft-Landing
  2. Strict Kinematic Constraints (v_max, a_max)
  3. Shortest Reachable Arc Wrapping with Mechanical Joint Limits [-90°, +90°]
  4. Organic Deceleration Braking Profile with Zero Overshoot
  5. Continuous Online Replanning without Velocity Discontinuities
"""

import math
from typing import Optional, Tuple

from astro_base.gaze.angle_math import (
    angular_diff_deg,
    clamp_deg,
    shortest_reachable_arc,
    wrap_deg,
)
from astro_base.gaze.types import GazeCommand, TrajectoryPoint


class MotionPlannerCore:
    """Computes smooth, acceleration-bounded kinematic trajectories for head pan actuator."""

    def __init__(
        self,
        # v and a are unchanged: they set how fast the head is, and the gaze scenarios
        # are timed against them. Smoothness is j's job, and j is what was broken —
        # accepted, stored, never consulted.
        #
        # 5400 is the tightest jerk the existing gaze scenarios still pass at. Measured
        # at 50 Hz: unlimited jerk lets acceleration swing its whole range inside one
        # 20 ms tick, which is the step a servo renders as a snap; 5400 spreads that
        # over 67 ms. Tighter is smoother still — 2700 gives 133 ms — but below 5400
        # test_holding_attention_persists starts failing, because jerk-limited motion
        # reaches its target later and the scenario is timed against motion that could
        # reverse instantly. That is a behavioural call to make against the real head,
        # not one to slip in behind a failing test.
        #
        # The shipped 360 was not a gentler choice, it was an unreachable one: at
        # a/j = 0.5 s a 60° move overshoots by 30° and never settles. It stayed harmless
        # only because max_jerk was accepted, stored, and never read.
        max_velocity_deg_s: float = 75.0,
        max_acceleration_deg_s2: float = 180.0,
        max_jerk_deg_s3: float = 5400.0,
        soft_landing_zone_deg: float = 15.0,
        min_limit_deg: float = -90.0,
        max_limit_deg: float = 90.0,
        profile_type: str = "trapezoidal",
    ):
        self.max_velocity = max_velocity_deg_s
        self.max_acceleration = max_acceleration_deg_s2
        self.max_jerk = max_jerk_deg_s3
        self.soft_landing_zone = soft_landing_zone_deg
        self.min_limit = min_limit_deg
        self.max_limit = max_limit_deg
        self.profile_type = profile_type.lower()

        # State tracking: position, velocity, acceleration
        self.current_pos = 0.0
        self.current_vel = 0.0
        self.current_acc = 0.0
        self.target_pos = 0.0
        self.last_update_time: Optional[float] = None

    def reset(self, initial_pos_deg: float = 0.0) -> None:
        self.current_pos = clamp_deg(initial_pos_deg, self.min_limit, self.max_limit)
        self.current_vel = 0.0
        self.current_acc = 0.0
        self.target_pos = self.current_pos
        self.last_update_time = None

    def set_target(self, target_deg: float) -> None:
        """Sets new goal position."""
        self.target_pos = clamp_deg(target_deg, self.min_limit, self.max_limit)

    def plan_step(
        self,
        gaze_cmd: GazeCommand,
        actual_pos_deg: Optional[float],
        timestamp: float,
    ) -> TrajectoryPoint:
        """Computes next trajectory point for the control cycle."""
        if self.last_update_time is None:
            if actual_pos_deg is not None:
                self.current_pos = clamp_deg(actual_pos_deg, self.min_limit, self.max_limit)
            self.current_vel = 0.0
            self.current_acc = 0.0
            self.last_update_time = timestamp
            is_init_settled = (abs(self.current_pos - gaze_cmd.target_yaw_deg) < 0.25)
            return TrajectoryPoint(
                timestamp=timestamp,
                position_deg=round(self.current_pos, 2),
                velocity_deg_s=0.0,
                acceleration_deg_s2=0.0,
                is_settled=is_init_settled,
            )

        dt = max(0.001, min(0.10, timestamp - self.last_update_time))
        self.last_update_time = timestamp

        # Closed-loop tracking synchronization: resync planner state only on severe physical desync / slip (>25.0°)
        if actual_pos_deg is not None and abs(self.current_pos - actual_pos_deg) > 25.0:
            self.current_pos = actual_pos_deg

        self.target_pos = clamp_deg(gaze_cmd.target_yaw_deg, self.min_limit, self.max_limit)

        # 1. Shortest reachable rotation arc respecting limits
        err_deg = shortest_reachable_arc(self.target_pos, self.current_pos, self.min_limit, self.max_limit)
        abs_err = abs(err_deg)


        # Settling check: settled only if both position error and velocity are minimal
        if abs_err < 0.25 and abs(self.current_vel) < 2.0:
            self.current_pos = self.target_pos
            self.current_vel = 0.0
            self.current_acc = 0.0
            return TrajectoryPoint(
                timestamp=timestamp,
                position_deg=round(self.current_pos, 2),
                velocity_deg_s=0.0,
                acceleration_deg_s2=0.0,
                is_settled=True,
            )

        # 2. Maximum reachable approach velocity with zero-overshoot braking curve.
        #
        # sqrt(2*a*d) is the answer when full deceleration is available the instant it
        # is asked for. Under a jerk limit it is not: deceleration ramps in over
        # a/j seconds, and the distance covered during that ramp is not free. Using the
        # instant-deceleration figure here overshoots by design — it commits to an
        # approach speed the brake cannot undo, and the result is the head sailing past
        # the target and hunting around it.
        #
        # While deceleration ramps in over t = a/j the head is still travelling at
        # nearly v, so the ramp costs about v·a/j of distance — not half that. Reserving
        # only half is what let a 35° command run out to 51°: the planner reached full
        # speed believing it could still stop, then discovered the brake needed more
        # room than it had left.
        #
        # Reserving it properly means solving d = v²/(2a) + v·a/j for v:
        #
        #     v = -a²/j + sqrt(a⁴/j² + 2·a·d)
        #
        # which collapses to sqrt(2*a*d) as j grows, so an unlimited-jerk planner keeps
        # exactly its old behaviour.
        a_brake = self.max_acceleration * 0.85
        if self.max_jerk > 0.0:
            # Acceleration still pushing toward the target has to be unwound before the
            # brake can even begin, so it lengthens the ramp. Acceleration already
            # braking does not — counting it too (an earlier attempt used |acc|) makes
            # the planner brake against its own brake and crawl the last third.
            pushing = self.current_acc * (1.0 if err_deg >= 0.0 else -1.0)
            swing = a_brake + max(0.0, pushing)
            ramp = (a_brake * swing) / self.max_jerk
            v_stop = -ramp + math.sqrt(ramp * ramp + 2.0 * a_brake * abs_err)
        else:
            v_stop = math.sqrt(2.0 * a_brake * abs_err)
        desired_v = math.copysign(min(self.max_velocity, v_stop), err_deg)

        # 3. The acceleration this tick's velocity gap asks for, capped at a_max.
        #
        # The gap is measured from where the velocity is *heading*, not where it is.
        # Under a jerk limit the planner cannot stop accelerating any faster than it
        # started: unwinding the current acceleration to zero takes |a|/j seconds and
        # adds a²/(2j) to the speed on the way. Comparing the braking curve against the
        # present velocity ignores that debt, so the planner keeps accelerating until
        # the curve says stop — by which point it is already too fast to stop in the
        # distance left. That is what sent a 10° command out to 19°.
        # The projection may only tighten the limit, never loosen it. While speeding up
        # it says "you will still gain this much, so stop now" — the point of it. While
        # braking it would say "you will have slowed by then, so brake less", and that
        # is a loan against distance the head is covering at its present speed, not its
        # future one. Taking whichever magnitude is larger keeps the first reading and
        # discards the second.
        projected_vel = self.current_vel
        if self.max_jerk > 0.0 and self.current_acc != 0.0:
            coast = (self.current_acc * self.current_acc) / (2.0 * self.max_jerk)
            candidate = self.current_vel + math.copysign(coast, self.current_acc)
            if abs(candidate) > abs(self.current_vel):
                projected_vel = candidate

        desired_acc = (desired_v - projected_vel) / dt
        desired_acc = max(-self.max_acceleration, min(self.max_acceleration, desired_acc))

        # 4. Jerk limiting: acceleration itself may only change so fast. Without this
        # the profile is acceleration-bounded but not jerk-bounded, so acceleration can
        # jump from +a_max to -a_max between two frames. That step is what a servo
        # renders as a snap, and it is the difference between "fast" and "abrupt".
        if self.max_jerk > 0.0:
            max_da = self.max_jerk * dt
            desired_acc = max(self.current_acc - max_da,
                              min(self.current_acc + max_da, desired_acc))

        self.current_acc = desired_acc
        self.current_vel += self.current_acc * dt

        # Velocity clamp
        self.current_vel = max(-self.max_velocity, min(self.max_velocity, self.current_vel))

        # 4. Position integration.
        #
        # A fixed 50 Hz step covers 1.5° at full speed, so the last tick before the
        # target steps past it and the next steps back — a small twitch at the end of
        # every movement. No amount of brake margin removes it: it is the integrator's
        # granularity, not the profile's. Landing exactly on the target when the step
        # would cross it costs nothing, because the braking curve has already brought
        # the speed down by the time the head is within one tick.
        step = self.current_vel * dt
        if abs(step) >= abs(err_deg) > 0.0:
            self.current_pos = self.target_pos
            self.current_vel = 0.0
            self.current_acc = 0.0
            return TrajectoryPoint(
                timestamp=timestamp,
                position_deg=round(self.current_pos, 2),
                velocity_deg_s=0.0,
                acceleration_deg_s2=0.0,
                is_settled=True,
            )

        self.current_pos += step
        self.current_pos = clamp_deg(self.current_pos, self.min_limit, self.max_limit)

        new_err = shortest_reachable_arc(self.target_pos, self.current_pos, self.min_limit, self.max_limit)
        is_settled = (abs(new_err) < 0.25 and abs(self.current_vel) < 2.0)
        if is_settled:
            self.current_pos = self.target_pos
            self.current_vel = 0.0
            self.current_acc = 0.0

        return TrajectoryPoint(
            timestamp=timestamp,
            position_deg=round(self.current_pos, 2),
            velocity_deg_s=round(self.current_vel, 2),
            acceleration_deg_s2=round(self.current_acc, 2),
            is_settled=is_settled,
        )
