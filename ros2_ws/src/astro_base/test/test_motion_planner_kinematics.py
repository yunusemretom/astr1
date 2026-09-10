"""Unit tests for Motion Planner & Trajectory Kinematics."""

import unittest
from astro_base.gaze.motion_planner import MotionPlannerCore
from astro_base.gaze.types import GazeCommand, GazeStateEnum, PrioritySource


class TestMotionPlannerKinematics(unittest.TestCase):
    def setUp(self):
        self.planner = MotionPlannerCore(
            # 360 °/s³ against 180 °/s² means a full acceleration reversal takes a
            # second, during which a 75 °/s head travels 75° — so that triple could not
            # stop inside a 60° move at all. It passed before only because max_jerk was
            # accepted and stored but never consulted.
            max_velocity_deg_s=75.0,
            max_acceleration_deg_s2=180.0,
            max_jerk_deg_s3=5400.0,
            soft_landing_zone_deg=12.0,
            min_limit_deg=-90.0,
            max_limit_deg=90.0,
            profile_type="s_curve",
        )

    def test_trajectory_obeys_velocity_and_acceleration_limits(self):
        """Tests that a 0° -> 60° saccade trajectory strictly adheres to v_max and a_max."""
        cmd = GazeCommand(target_yaw_deg=60.0, timestamp=1.0)
        self.planner.reset(initial_pos_deg=0.0)

        t = 1.0
        dt = 0.02  # 50 Hz control loop
        positions = []
        velocities = []
        accelerations = []

        for _ in range(150):  # Simulate 3.0 seconds
            pt = self.planner.plan_step(cmd, actual_pos_deg=None, timestamp=t)
            positions.append(pt.position_deg)
            velocities.append(pt.velocity_deg_s)
            accelerations.append(pt.acceleration_deg_s2)

            # Strict kinematic limits assertions
            self.assertLessEqual(abs(pt.velocity_deg_s), 75.0 + 0.1)
            self.assertLessEqual(abs(pt.acceleration_deg_s2), 180.0 + 1.0)

            if pt.is_settled:
                break
            t += dt

        # Final position must be settled at 60.0°
        self.assertAlmostEqual(positions[-1], 60.0, delta=0.25)
        self.assertAlmostEqual(velocities[-1], 0.0, delta=0.5)

    def test_mechanical_limit_clamping(self):
        """Tests that commanding an out-of-bounds target (120°) clamps smoothly to +90°."""
        cmd = GazeCommand(target_yaw_deg=120.0, timestamp=1.0)
        self.planner.reset(initial_pos_deg=0.0)

        t = 1.0
        for _ in range(150):
            pt = self.planner.plan_step(cmd, actual_pos_deg=None, timestamp=t)
            self.assertLessEqual(pt.position_deg, 90.0)
            t += 0.02
            if pt.is_settled:
                break

        self.assertAlmostEqual(self.planner.current_pos, 90.0, delta=0.25)

    def test_online_replanning_smooth_transition(self):
        """Tests changing target mid-flight (from +45° to -45°) reverses smoothly without velocity spikes."""
        cmd1 = GazeCommand(target_yaw_deg=45.0, timestamp=1.0)
        self.planner.reset(initial_pos_deg=0.0)

        t = 1.0
        # Run towards +45° for ~0.30s (head reaches velocity ~25°/s)
        for _ in range(15):
            self.planner.plan_step(cmd1, actual_pos_deg=None, timestamp=t)
            t += 0.02



        mid_vel = self.planner.current_vel
        self.assertGreater(mid_vel, 10.0)

        # Mid-flight target change to -45°
        cmd2 = GazeCommand(target_yaw_deg=-45.0, timestamp=t)
        prev_vel = mid_vel
        for _ in range(150):
            pt = self.planner.plan_step(cmd2, actual_pos_deg=None, timestamp=t)
            # Ensure velocity change per step is bounded by acceleration limit
            dv = abs(pt.velocity_deg_s - prev_vel)
            max_allowed_dv = (180.0 * 0.02) + 0.5  # a_max * dt + margin
            self.assertLessEqual(dv, max_allowed_dv)
            prev_vel = pt.velocity_deg_s
            t += 0.02
            if pt.is_settled:
                break


        self.assertAlmostEqual(self.planner.current_pos, -45.0, delta=0.25)

    def test_stationary_target_does_not_oscillate_or_limit_cycle(self):
        """CRITICAL REGRESSION: 0° -> -35° saccade must monotonically converge to -35° without -10° limit-cycle oscillation."""
        cmd = GazeCommand(target_yaw_deg=-35.0, timestamp=1.0)
        self.planner.reset(initial_pos_deg=0.0)

        t = 1.0
        dt = 0.02
        trajectory = []
        
        # Simulate physical motor lagging at 0° initially then slowly following
        simulated_actual = 0.0

        for step in range(120):
            # Actuator follows commanded position with realistic first-order mechanical lag
            simulated_actual += 0.20 * (self.planner.current_pos - simulated_actual)

            pt = self.planner.plan_step(cmd, actual_pos_deg=simulated_actual, timestamp=t)
            trajectory.append(pt.position_deg)

            # Monotonicity check during acceleration and cruise (until settling)
            if step > 0 and pt.position_deg > trajectory[step - 1] and pt.position_deg > -34.5:
                self.fail(
                    f"BUG DETECTED: Non-monotonic regression or reversal at step {step}: "
                    f"pos={pt.position_deg:.2f}° prev={trajectory[step-1]:.2f}° (Oscillation trapped around -10°!)"
                )

            if pt.is_settled:
                break
            t += dt

        # Verify convergence to -35.0°
        self.assertAlmostEqual(trajectory[-1], -35.0, delta=0.25)
        self.assertAlmostEqual(self.planner.current_vel, 0.0, delta=0.5)

    def test_plant_lag_closed_loop_convergence(self):
        """Tests that full closed-loop HIL simulation converges seamlessly to any target."""
        for target in [-35.0, +35.0, +60.0, -60.0]:
            planner = MotionPlannerCore(max_velocity_deg_s=75.0, max_acceleration_deg_s2=180.0)
            planner.reset(initial_pos_deg=0.0)
            cmd = GazeCommand(target_yaw_deg=target, timestamp=1.0)

            t = 1.0
            actual = 0.0
            for _ in range(150):
                actual += 0.25 * (planner.current_pos - actual)
                pt = planner.plan_step(cmd, actual_pos_deg=actual, timestamp=t)
                t += 0.02
                if pt.is_settled:
                    break

            self.assertAlmostEqual(planner.current_pos, target, delta=0.25)

    def test_acceleration_is_jerk_limited(self):
        """max_jerk must actually bound how fast acceleration changes.

        This is the regression guard for the defect this test file was written under:
        max_jerk was a constructor argument, was stored on the instance, and was never
        read. Everything below passed anyway, because nothing here looked at the rate
        acceleration changed — only at its magnitude.
        """
        cmd = GazeCommand(target_yaw_deg=70.0, timestamp=1.0)
        self.planner.reset(initial_pos_deg=-70.0)

        t, dt = 1.0, 0.02
        prev_acc = 0.0
        worst = 0.0
        for _ in range(300):
            pt = self.planner.plan_step(cmd, actual_pos_deg=None, timestamp=t)
            worst = max(worst, abs(pt.acceleration_deg_s2 - prev_acc) / dt)
            prev_acc = pt.acceleration_deg_s2
            if pt.is_settled:
                break
            t += dt

        # A little slack for the settling step, which zeroes acceleration outright.
        self.assertLessEqual(worst, 5400.0 * 1.2,
                             f"acceleration changed at {worst:.0f} deg/s^3, above max_jerk")

    def test_no_overshoot_across_move_sizes(self):
        """Short moves are the ones jerk limiting breaks first.

        A jerk-limited planner cannot start braking instantly, so the approach speed has
        to be capped for the distance actually left. Get that wrong and short commands
        overshoot hardest: before this was fixed a 10 deg command ran out past 19 deg,
        while a 60 deg one landed perfectly — which is why testing only large saccades
        hid it.
        """
        for target in (2.0, 5.0, 10.0, 20.0, 35.0, -35.0, 60.0, -60.0, 85.0):
            self.planner.reset(initial_pos_deg=0.0)
            cmd = GazeCommand(target_yaw_deg=target, timestamp=1.0)

            t, dt = 1.0, 0.02
            peak = 0.0
            settled = False
            for _ in range(400):
                pt = self.planner.plan_step(cmd, actual_pos_deg=None, timestamp=t)
                peak = max(peak, abs(pt.position_deg))
                if pt.is_settled:
                    settled = True
                    break
                t += dt

            self.assertTrue(settled, f"target {target} never settled")
            self.assertLessEqual(peak - abs(target), 0.25,
                                 f"target {target} overshot to {peak}")

    def test_head_is_nearly_stopped_when_it_arrives(self):
        """Arriving fast and being snapped to zero is not smoothness, just a hidden jolt.

        The planner lands exactly on target when the next step would cross it. That is
        only harmless if the braking curve has already taken the speed down by then —
        otherwise the twitch simply moves from the position trace into the velocity one.
        """
        for target in (5.0, 10.0, 20.0, 35.0, 60.0):
            self.planner.reset(initial_pos_deg=0.0)
            cmd = GazeCommand(target_yaw_deg=target, timestamp=1.0)

            t, dt = 1.0, 0.02
            last_vel = 0.0
            for _ in range(400):
                pt = self.planner.plan_step(cmd, actual_pos_deg=None, timestamp=t)
                if pt.is_settled:
                    break
                last_vel = pt.velocity_deg_s
                t += dt

            self.assertLessEqual(abs(last_vel), 5.0,
                                 f"target {target} arrived at {last_vel:.1f} deg/s")

    def test_unlimited_jerk_keeps_the_old_profile(self):
        """max_jerk <= 0 must fall back to the instant-deceleration braking curve.

        Anyone who wants the previous behaviour should be able to ask for it plainly
        rather than by picking a number large enough not to bind.
        """
        planner = MotionPlannerCore(
            max_velocity_deg_s=75.0, max_acceleration_deg_s2=180.0,
            max_jerk_deg_s3=0.0, min_limit_deg=-90.0, max_limit_deg=90.0,
        )
        planner.reset(initial_pos_deg=0.0)
        cmd = GazeCommand(target_yaw_deg=60.0, timestamp=1.0)

        t, dt = 1.0, 0.02
        settled = False
        for _ in range(300):
            pt = planner.plan_step(cmd, actual_pos_deg=None, timestamp=t)
            self.assertLessEqual(abs(pt.acceleration_deg_s2), 180.0 + 1.0)
            if pt.is_settled:
                settled = True
                break
            t += dt

        self.assertTrue(settled)


if __name__ == "__main__":
    unittest.main()

