#!/usr/bin/env python3
"""Head travel limits, and what happens when the encoder never reports back.

Two things make the head stop following a person who is still standing there.

The first is the soft limit. The firmware clamps at +/-180 deg and the calibration
file documents the encoder scale as 440 ticks over 170 deg of travel, so the
mechanism itself turns +/-85 — but the gaze stack clamped every command to +/-75
and simply stopped short of a person standing further round.

The second is worse because it is silent. Every bearing is computed as
`body_azimuth = actual_head_yaw + camera_azimuth`. When no encoder feedback
arrives, actual_head_yaw stays at 0 while the head physically turns, so a person
centred in the frame after a 20 deg turn is computed to be at 0 deg — and the head
is commanded straight back to centre, where it sits. Nothing in the pipeline said
anything was wrong; the node simply used its default.
"""

import os
import time
import unittest

import yaml

from astro_base.gaze.coordinate_frames import CalibrationConfig
from astro_base.social_gaze_node import SocialGazeNode


CALIBRATION = os.path.join(
    os.path.dirname(__file__), "..", "config", "calibration_params.yaml"
)


class TestHeadTravel(unittest.TestCase):
    """440 ticks / 170 deg is the range the encoder was actually characterised over."""

    DOCUMENTED_HALF_TRAVEL = 85.0

    def _head_config(self) -> dict:
        with open(CALIBRATION, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)["/**"]["ros__parameters"]["head"]

    def test_the_configured_limits_use_the_measured_travel(self):
        head = self._head_config()

        self.assertLessEqual(head["min_angle_deg"], -self.DOCUMENTED_HALF_TRAVEL)
        self.assertGreaterEqual(head["max_angle_deg"], self.DOCUMENTED_HALF_TRAVEL)

    def test_the_limits_stay_inside_what_the_firmware_accepts(self):
        """arduino/astro_firmware clamps to +/-180; never command past it."""
        head = self._head_config()

        self.assertGreaterEqual(head["min_angle_deg"], -180.0)
        self.assertLessEqual(head["max_angle_deg"], 180.0)

    def test_the_code_default_matches_the_configured_travel(self):
        """A node started without the calibration file must not be more timid."""
        head = CalibrationConfig().head

        self.assertLessEqual(head.min_angle_deg, -self.DOCUMENTED_HALF_TRAVEL)
        self.assertGreaterEqual(head.max_angle_deg, self.DOCUMENTED_HALF_TRAVEL)


class _Msg:
    def __init__(self, data):
        self.data = data


class TestMissingHeadFeedbackIsReported(unittest.TestCase):
    """Assuming 0 deg while the head is elsewhere collapses every bearing to centre."""

    def test_a_node_that_never_heard_from_the_encoder_says_so(self):
        node = SocialGazeNode()

        self.assertTrue(node.head_feedback_missing())

    def test_feedback_from_the_head_state_topic_clears_it(self):
        node = SocialGazeNode()

        node._on_head_state(type("S", (), {"position_deg": 12.0, "velocity_deg_s": 0.0})())

        self.assertFalse(node.head_feedback_missing())

    def test_feedback_from_joint_states_clears_it(self):
        node = SocialGazeNode()
        msg = type("J", (), {"name": ["head_yaw_joint"], "position": [0.2], "velocity": [0.0]})()

        node._on_joint_states(msg)

        self.assertFalse(node.head_feedback_missing())

    def test_the_control_cycle_warns_exactly_once_while_feedback_is_absent(self):
        """A per-cycle warning at 50 Hz would bury the log it needs to stand out in."""
        node = SocialGazeNode()
        warnings = []
        node.get_logger().warning = lambda msg, *a, **k: warnings.append(msg)

        for _ in range(120):
            node._control_cycle()

        self.assertEqual(len(warnings), 1)
        self.assertIn("geri besleme", warnings[0].lower())


class TestOpenLoopHeadEstimate(unittest.TestCase):
    """With no encoder, assume the head went where it was told — not that it is at 0.

    Assuming 0 makes every bearing collapse: the head turns 20 degrees, the person
    lands centred in the frame, and `0 + 0` puts them back at 0, so the command
    returns to centre and parks. The motion planner already integrates a rate- and
    acceleration-limited position toward each command, which is exactly a model of
    where the head should be, so that is what stands in for the missing encoder.
    """

    def _drive(self, node, yaw, cycles=4000):
        """plan_step clamps dt to a 1 ms floor, so the modelled head advances at least
        that much per cycle; 4000 of them is four seconds of travel."""
        node.fsm.set_dialogue_target(yaw_deg=yaw, duration_s=600.0, timestamp=time.monotonic())
        for _ in range(cycles):
            node._control_cycle()

    def test_the_estimate_follows_the_command_instead_of_sitting_at_zero(self):
        node = SocialGazeNode()

        self._drive(node, 40.0)

        self.assertGreater(node.actual_head_yaw_deg, 20.0)

    def test_a_real_encoder_reading_still_wins(self):
        node = SocialGazeNode()
        node._on_head_state(type("S", (), {"position_deg": 5.0, "velocity_deg_s": 0.0})())

        self._drive(node, 40.0)

        self.assertEqual(node.actual_head_yaw_deg, 5.0, "the encoder is the authority")

    def test_the_estimate_respects_the_head_limits(self):
        node = SocialGazeNode()

        self._drive(node, 400.0)

        self.assertLessEqual(node.actual_head_yaw_deg, node.calib.head.max_angle_deg)

    def test_the_planner_is_not_resynced_against_a_position_nobody_measured(self):
        """plan_step snaps its state to actual_pos_deg past 25 degrees of error; fed a
        fabricated 0 it dragged the modelled head back to centre every cycle."""
        node = SocialGazeNode()

        self._drive(node, 60.0)

        self.assertGreater(node.planner.current_pos, 25.0)


if __name__ == "__main__":
    unittest.main()
