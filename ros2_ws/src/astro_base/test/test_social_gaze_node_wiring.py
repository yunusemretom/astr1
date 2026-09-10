#!/usr/bin/env python3
"""Topic-level wiring tests for SocialGazeNode.

Every other astro_base test drives the gaze core classes directly, which leaves the
node's own input plumbing — the layer that decides which topic means what — with no
coverage at all. These tests drive the real ROS callbacks instead, because that is
where the defects live: a topic carrying one quantity being read as another, two
callbacks writing the same tracker, and a detector's silence about quality being
read as high confidence.
"""

import json
import unittest

from astro_base.social_gaze_node import SocialGazeNode


class _Msg:
    """Stand-in for a std_msgs message: the callbacks only ever read `.data`."""

    def __init__(self, data):
        self.data = data


def _face(**overrides) -> dict:
    """A face payload shaped like the one the astro_vision nodes actually publish."""
    face = {
        "x": 260, "y": 200, "width": 80, "height": 80,
        "distance_m": 1.6,
        "yaw_deg": 3.0,
        "looking_at_robot": True,
        "emotion": "neutral",
    }
    face.update(overrides)
    return face


class TestHeadlessHarness(unittest.TestCase):
    """The mock Node shim must be faithful enough to construct and run the node."""

    def test_node_constructs_with_its_declared_parameter_defaults(self):
        node = SocialGazeNode()

        self.assertEqual(node.fusion.spatial_gate_deg, 25.0)
        self.assertEqual(node.planner.max_velocity, 75.0)
        self.assertEqual(node.target_manager.min_attention_dwell_s, 2.50)

    def test_control_cycle_runs_and_publishes_a_head_command(self):
        node = SocialGazeNode()
        node.enable_actuator_output = True

        node._control_cycle()

        self.assertIsNotNone(node.pub_head_cmd_pos.last_msg)
        self.assertIsInstance(node.pub_head_cmd_pos.last_msg.data, float)


class TestVisionHeadYawIsNotABearing(unittest.TestCase):
    """`/vision/head_yaw` carries the *person's* head pose, not the bearing to them.

    All three astro_vision publishers fill it from `_estimate_head_yaw()`, which
    measures the eye midpoint's offset inside the face ROI. Reading it as a camera
    azimuth invents a target where nobody is standing — and when no eyes are found
    the detectors emit the literal sentinel 45.0, which would fabricate a
    high-confidence target 45 degrees off to the side.
    """

    def setUp(self):
        self.node = SocialGazeNode()

    def test_head_yaw_alone_never_fabricates_a_visual_track(self):
        self.node._on_vision_head_yaw(_Msg(45.0))

        self.assertEqual(self.node.visual_tracker.tracks, {})
        self.assertEqual(self.node.latest_visual_tracks, [])

    def test_head_yaw_does_not_starve_the_real_face_track(self):
        """The two topics arrive on the same frame; only one of them holds detections.

        Feeding both into the tracker makes the real face miss every other update,
        so its confidence decays while a phantom holds a constant one.
        """
        faces = _Msg(json.dumps([_face(x=20, distance_m=3.0)]))

        for _ in range(8):
            self.node._on_vision_json(faces)
            self.node._on_vision_head_yaw(_Msg(0.0))

        tracks = self.node.latest_visual_tracks
        self.assertEqual(len(tracks), 1, "the head-pose topic must not spawn a second track")
        self.assertEqual(tracks[0].missed_frames, 0, "the face was visible on every frame")
        self.assertGreater(tracks[0].body_azimuth_deg, 15.0, "the face sits off to the left")

    def test_head_yaw_supplies_the_head_pose_a_detection_omits(self):
        """Kept as an eye-contact cue: a face reported without its own yaw uses it."""
        self.node._on_vision_head_yaw(_Msg(40.0))
        face = _face()
        face.pop("yaw_deg")
        self.node._on_vision_json(_Msg(json.dumps([face])))

        self.assertFalse(
            self.node.latest_visual_tracks[0].eye_contact,
            "a head turned 40 degrees away is not making eye contact",
        )


class TestUnscoredDetectionsNeedCorroboration(unittest.TestCase):
    """A detector that reports no confidence was being credited with 0.85.

    That sits above the 0.75 acquisition threshold, so a single Haar frame — false
    positives included — seized the head outright and the dual-threshold hysteresis
    (0.75 acquire / 0.40 hold) never got to mean anything.
    """

    def setUp(self):
        self.node = SocialGazeNode()

    def _looking_at(self) -> str:
        self.node._control_cycle()
        active = self.node.target_manager.active_target
        return active.target_id if active else ""

    def test_unscored_detection_does_not_seize_attention(self):
        self.node._on_vision_json(_Msg(json.dumps([_face()])))

        self.assertEqual(self._looking_at(), "")

    def test_unscored_detection_is_still_tracked_and_held(self):
        """Below acquisition is not below hold: the person stays a live candidate."""
        self.node._on_vision_json(_Msg(json.dumps([_face()])))
        self.node._control_cycle()

        candidates = self.node.target_manager.candidate_targets
        self.assertEqual(len(candidates), 1)
        self.assertGreaterEqual(candidates[0].confidence, self.node.target_manager.hold_threshold)

    def test_unscored_detection_is_acquired_once_audio_corroborates_it(self):
        """Speech from the same bearing is the corroboration the score didn't carry."""
        self.node._on_vision_json(_Msg(json.dumps([_face()])))
        self.node._on_doa_deg(_Msg(350.0))  # ReSpeaker clockwise 350 deg -> +10 deg left

        self.assertNotEqual(self._looking_at(), "")

    def test_scored_detection_is_acquired_on_its_own(self):
        self.node._on_vision_json(_Msg(json.dumps([_face(confidence=0.92)])))

        self.assertNotEqual(self._looking_at(), "")


class TestSocialGazeForensicTelemetry(unittest.TestCase):
    """Verifies that SocialGazeNode produces complete forensic telemetry and logs causal chains."""

    def test_detection_and_command_forensic_chain(self):
        import io
        from unittest.mock import patch

        node = SocialGazeNode()
        face_data = _face(x=100, y=120, width=80, height=80, confidence=0.92, detector_source="yunet")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            node._on_vision_json(_Msg(json.dumps([face_data])))
            node._control_cycle()
            output = mock_stdout.getvalue()

        # 1. Detection telemetry verification (8 fields)
        self.assertTrue(len(node._latest_detections_telemetry) >= 1)
        det = node._latest_detections_telemetry[0]
        for field in ("frame_id", "timestamp", "bbox", "bearing", "confidence", "detector_source", "track_id", "track_state"):
            self.assertIn(field, det)
        self.assertEqual(det["detector_source"], "yunet")
        self.assertEqual(det["confidence"], 0.92)

        # 2. Forensic payload verification in node state
        self.assertIsNotNone(node.last_forensic_chain)
        forensic = node.last_forensic_chain

        # Target manager (3 fields)
        tm = forensic["target_manager"]
        for field in ("previous_active_target", "new_active_target", "reason"):
            self.assertIn(field, tm)

        # Attention decision (4 fields)
        att = forensic["attention"]
        for field in ("old_owner", "new_owner", "reason", "preempted_target"):
            self.assertIn(field, att)

        # Head command (5 fields)
        cmd = forensic["command"]
        for field in ("previous_target_yaw", "new_target_yaw", "command_source", "target_source", "reason"):
            self.assertIn(field, cmd)

        # 3. Debug topic includes forensic payload
        self.assertIsNotNone(node.pub_gaze_debug.last_msg)
        diag = json.loads(node.pub_gaze_debug.last_msg.data)
        self.assertIn("forensic", diag)
        self.assertEqual(diag["forensic"]["command"]["new_target_yaw"], cmd["new_target_yaw"])

        # 4. Causal chain output when target_yaw changes by > 3 degrees
        if forensic["delta_yaw"] > 3.0:
            self.assertIn("DETECTION → TRACK → TARGET → ATTENTION → COMMAND", output)
            self.assertIn("[FORENSIC CAUSAL CHAIN]", output)


if __name__ == "__main__":
    unittest.main()
