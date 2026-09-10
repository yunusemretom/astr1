#!/usr/bin/env python3
"""Regression tests for the DOA -> head yaw path.

Each test here pins down one defect found while tracing why the head drifts,
spins continuously, or jumps between the neck limits after running for a while.
"""

import os
import re
import sys
import time
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(pkg_dir, "..", "..", ".."))
astro_base_inner = os.path.join(pkg_dir, "astro_base")
for p in (pkg_dir, astro_base_inner):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from astro_base.head_tracker_node import (
        HeadTrackerNode,
        angular_diff_deg,
        doa_to_robot_yaw,
    )
except ImportError:
    from head_tracker_node import (
        HeadTrackerNode,
        angular_diff_deg,
        doa_to_robot_yaw,
    )

try:
    from astro_base.head_tracker_node import HEAD_TRACKER_DEFAULTS
except ImportError:
    try:
        from head_tracker_node import HEAD_TRACKER_DEFAULTS
    except ImportError:
        HEAD_TRACKER_DEFAULTS = None

LAUNCH_FILE = os.path.join(
    repo_root, "ros2_ws", "src", "astro_bringup", "launch", "base.launch.py"
)
FIRMWARE_FILE = os.path.join(
    repo_root, "arduino", "astro_firmware", "src", "main.cpp"
)
PARAMS_FILE = os.path.join(
    repo_root, "ros2_ws", "src", "astro_bringup", "config", "astro_params.yaml"
)


def setUpModule():
    try:
        import rclpy
        if not rclpy.ok():
            rclpy.init()
    except Exception:
        pass


def tearDownModule():
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


class MockMsg:
    def __init__(self, data):
        self.data = data


def make_node() -> HeadTrackerNode:
    """A head tracker parked at 0 deg, awake, hearing a loud steady voice."""
    try:
        import rclpy
        if not rclpy.ok():
            rclpy.init()
    except Exception:
        pass
    node = HeadTrackerNode()
    node.enabled = True
    node._is_sleeping = False
    node._is_speaking = False
    node._is_playback_active = False
    node._vad_active = True
    node._latest_rms = 3000.0
    node._ambient_rms = 120.0
    node.vision_fusion_enabled = False
    node.lidar_fusion_enabled = False
    node._vision_person_detected = False
    node._target_yaw = 0.0
    node._estimated_yaw = 0.0
    node._doa_history.clear()
    node._last_gaze_switch_time = 0.0
    return node


class TestShippedConfigurationIsApplied(unittest.TestCase):
    """The YAML in astro_bringup must actually reach the running node."""

    def test_params_yaml_key_matches_launched_node_name(self):
        launch_src = open(LAUNCH_FILE, encoding="utf-8").read()
        params_src = open(PARAMS_FILE, encoding="utf-8").read()

        m = re.search(
            r'executable="(?:head_tracker|social_gaze)"[^)]*?name="([^"]+)"', launch_src, re.S
        )
        self.assertIsNotNone(m, "social_gaze veya head_tracker Node blogu base.launch.py'da bulunamadi")
        launched_name = m.group(1)

        top_level_keys = set(re.findall(r"^([A-Za-z_/][\w/]*):", params_src, re.M))
        self.assertIn(
            launched_name,
            top_level_keys,
            f"astro_params.yaml '{launched_name}' node adiyla eslesmiyor; "
            f"parametreler sessizce yok sayilir. Mevcut anahtarlar: {sorted(top_level_keys)}",
        )

    def test_code_default_for_doa_invert_matches_shipped_yaml(self):
        self.assertIsNotNone(
            HEAD_TRACKER_DEFAULTS,
            "head_tracker_node tek bir HEAD_TRACKER_DEFAULTS kaynagi disari vermeli",
        )
        params_src = open(PARAMS_FILE, encoding="utf-8").read()
        m = re.search(r"^\s+doa_invert:\s*(\w+)", params_src, re.M)
        self.assertIsNotNone(m, "astro_params.yaml icinde doa_invert yok")
        yaml_value = m.group(1).lower() == "true"

        self.assertEqual(
            HEAD_TRACKER_DEFAULTS["doa_invert"],
            yaml_value,
            "Parametre yuklenemezse kod varsayilani devreye girer; ters isaret "
            "kafayi sesin ayna yonune cevirir.",
        )


class TestYawSignConvention(unittest.TestCase):
    def test_sound_from_the_right_yaws_like_the_look_right_gesture(self):
        # The node's own gesture table is the reference: positive yaw = left (REP-103).
        look_right = HeadTrackerNode.GESTURE_PROFILES["look_right"][0]
        self.assertLess(look_right, 0.0, "jest tablosu bozulmus")

        node = make_node()
        yaw = doa_to_robot_yaw(
            90.0, offset_deg=node.doa_offset_deg, invert=node.doa_invert
        )
        self.assertLess(
            yaw,
            0.0,
            "ReSpeaker 90 derece (sag) ile look_right jesti zit isaret veriyor: "
            "kafa sesin ayna yonune doner.",
        )


class TestConsensusAgainstBackSector(unittest.TestCase):
    def test_sound_from_behind_does_not_flip_between_neck_limits(self):
        """A source behind the robot straddles +-180 and must not split the cluster."""
        node = make_node()
        node.doa_invert = False
        node.consensus_tolerance_deg = 22.0
        # Hold the gaze dwell so no target is committed mid-run; this test is about what
        # lands in the consensus buffer, not about when the head decides to move.
        node._last_gaze_switch_time = time.monotonic()

        # Same physical source, +-5 deg of measurement noise around straight back.
        for doa in (175.0, 185.0, 175.0, 185.0, 175.0, 185.0):
            node._on_doa(MockMsg(doa))

        yaws = [y for _, y in node._doa_history]
        self.assertTrue(yaws, "DOA gecmisi bos")
        spread = max(abs(angular_diff_deg(y, yaws[0])) for y in yaws)
        self.assertLessEqual(
            spread,
            node.consensus_tolerance_deg,
            f"Arkadaki tek kaynak {spread:.0f} derecelik saciliyor: konsensus "
            f"gecmisine kirpilmis degerler yaziliyor. Ornekler: {yaws}",
        )


class TestSelfNoiseGate(unittest.TestCase):
    def test_doa_ignored_right_after_the_head_is_commanded_to_move(self):
        """Head motor noise reaches the head-mounted mic array; do not track during motion."""
        node = make_node()
        node._target_yaw = 60.0  # arbitrated elsewhere (gesture, turn_to_sound, LiDAR)
        node._last_update_time = time.monotonic() - 0.05
        node._control_loop()  # publishes /head_cmd -> the yaw motor starts running

        for _ in range(8):
            node._on_doa(MockMsg(45.0))

        self.assertEqual(
            len(node._doa_history),
            0,
            "Kafa hareket ederken gelen DOA ornekleri kabul edildi.",
        )


class TestVisionServoing(unittest.TestCase):
    def test_stale_vision_measurement_is_applied_only_once(self):
        """One face measurement must produce one correction, not one per control tick."""
        node = make_node()
        node.vision_fusion_enabled = True
        node.vision_gain = 0.35
        node._on_vision_head_yaw(MockMsg(10.0))
        node._vision_person_detected = True
        node._vision_last_seen_time = time.monotonic()

        for _ in range(40):  # 2 s at 20 Hz, no new vision measurement arrives
            node._last_update_time = time.monotonic() - 0.05
            node._control_loop()

        self.assertLessEqual(
            abs(node._target_yaw),
            5.0,
            f"Bayat gorsel olcum hedefi {node._target_yaw:.1f} dereceye tasidi; "
            f"tek olcum her dongude yeniden uygulaniyor.",
        )


class FakeClock:
    """Deterministic monotonic clock so a whole tracking session runs in microseconds."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def run_tracking_session(speaker_body_deg: float, seconds: float = 25.0):
    """Simulate a person standing still at a fixed BODY angle while the head tracks them.

    The mic array is bolted to head_link (URDF mic_joint), so what /doa reports is the
    bearing RELATIVE TO THE HEAD, not to the robot body. This helper reproduces that
    geometry honestly: every tick it recomputes the bearing from wherever the head
    currently points.
    """
    import astro_base.head_tracker_node as H  # already imported; grab the module object

    clock = FakeClock()
    real_time = H.time
    H.time = clock
    try:
        node = make_node()
        node.head_motion_settle_s = 0.0  # isolate the frame question from the noise gate
        node._last_update_time = clock.monotonic()
        node._last_speech_time = clock.monotonic()
        node._last_gaze_switch_time = 0.0

        trajectory = []
        for _ in range(int(seconds * 20)):
            # The Arduino runs its own position PID on whatever setpoint /head_cmd last
            # carried, so that value -- not the software slew trajectory -- is where the
            # neck actually is.
            head = getattr(node, "_last_published_cmd_yaw", None) or 0.0
            relative = angular_diff_deg(speaker_body_deg, head)
            raw_doa = (-relative) % 360.0  # ReSpeaker counts clockwise; doa_invert undoes it
            node._on_doa(MockMsg(raw_doa))
            node._control_loop()
            clock.advance(0.05)
            node._last_speech_time = clock.monotonic()
            trajectory.append(getattr(node, "_last_published_cmd_yaw", None) or 0.0)

        return node, trajectory
    finally:
        H.time = real_time


def worst_settled_error(speaker_body_deg: float, trajectory, tail_seconds: float = 6.0):
    """Largest pointing error over the tail of the run.

    Checking only the final sample would let an oscillating head pass whenever the run
    happens to end on a favourable tick, so measure the whole settled window instead.
    """
    tail = trajectory[-int(tail_seconds * 20):]
    return max(abs(angular_diff_deg(speaker_body_deg, yaw)) for yaw in tail)


class TestHeadActuallyFacesTheSpeaker(unittest.TestCase):
    """The mic rides on the head, so DOA is head-relative and must be composed, not assigned."""

    def test_head_settles_facing_a_stationary_speaker(self):
        speaker = 60.0
        node, traj = run_tracking_session(speaker)
        residual = worst_settled_error(speaker, traj)
        self.assertLessEqual(
            residual,
            node.deadband_deg,
            f"Kafa konusmaciya yerlesmedi: son 6 saniyede en kotu hata {residual:.1f} deg "
            f"(konusmaci={speaker:.1f} deg). "
            "Mikrofon head_link'e bagli oldugu icin DOA basa GORE olcum; mutlak "
            "govde acisi gibi atanirsa dongu yakinsamaz.",
        )

    def test_head_does_not_oscillate_once_it_has_found_the_speaker(self):
        """Assigning a head-relative bearing as an absolute target makes the head swing
        back to 0 the moment it arrives, then out again, forever."""
        speaker = -40.0
        node, traj = run_tracking_session(speaker, seconds=30.0)
        tail = traj[-120:]
        swing = max(tail) - min(tail)
        self.assertLessEqual(
            swing,
            node.deadband_deg,
            f"Kafa yerlestikten sonra hala {swing:.1f} deg genlikte gidip geliyor. "
            "Hedefe varinca DOA 0 okuyor, 0 mutlak hedef sanilip kafa one donuyor, "
            "sonra ses yine yandan geliyor -> sonsuz salinim.",
        )

    def test_speaker_behind_the_robot_is_reachable(self):
        speaker = 150.0
        node, traj = run_tracking_session(speaker, seconds=30.0)
        residual = worst_settled_error(speaker, traj)
        self.assertLessEqual(
            residual,
            node.deadband_deg,
            f"Arkadaki kaynak icin {residual:.1f} deg hata kaldi; boyun hareket "
            "sinirlari kafanin arkaya donmesine izin vermiyor.",
        )

    def test_turn_to_sound_bearing_is_relative_to_where_the_head_points(self):
        """/head/target_yaw carries a DOA-derived bearing, which is head-relative too."""
        node = make_node()
        node._estimated_yaw = 30.0
        node._target_yaw = 30.0
        node._on_target_yaw_cmd(MockMsg(20.0))  # "ses 20 deg solumda" (basa gore)
        self.assertAlmostEqual(
            node._target_yaw,
            50.0,
            delta=1.0,
            msg="Kafa 30 deg'de iken basa gore +20 deg gelen ses, govdede +50 deg'dedir. "
            "Dogrudan 20 deg'e gitmek kafayi kaynagin gerisinde birakir.",
        )


class TestHeadPoseReference(unittest.TestCase):
    """Composing a relative bearing needs the pose the FIRMWARE holds, not the software one."""

    def test_bearing_is_composed_onto_the_angle_the_arduino_was_given(self):
        node = make_node()
        # /head_cmd carries the full setpoint, so the Arduino PID is already driving to
        # 60 deg while the node's own slew trajectory is still crawling through 10 deg.
        node._last_published_cmd_yaw = 60.0
        node._estimated_yaw = 10.0

        self.assertAlmostEqual(
            node._acoustic_bearing_to_body_yaw(20.0),
            80.0,
            delta=0.5,
            msg="Kafa 60 deg'e surulurken basa gore +20 deg gelen ses govdede +80 deg'dedir. "
            "Yazilim yorungesi (_estimated_yaw) firmware'in gerisinde kaldigi icin ona "
            "eklemek gecici olarak yanlis hedef uretir.",
        )


class TestVisualServoingIsBounded(unittest.TestCase):
    """A face inside a 72 deg field of view can never justify walking the neck 120 deg."""

    CAMERA_HALF_FOV_DEG = 36.0

    def _run_with_stale_face(self, camera_azimuth_deg: float, seconds: float = 8.0):
        """The detector replays its last bounding box for a few frames when it drops a
        face (face_detector_node._last_known_face), so the same camera azimuth arrives
        over and over. Re-applying it must not walk the head any further each time."""
        import astro_base.head_tracker_node as H

        clock = FakeClock()
        real_time = H.time
        H.time = clock
        try:
            node = make_node()
            node.vision_fusion_enabled = True
            node._vision_person_detected = True
            node._vision_last_seen_time = clock.monotonic()
            node.head_motion_settle_s = 0.0
            node._vad_active = False  # vision alone drives the head here
            node._last_update_time = clock.monotonic()
            node._last_speech_time = clock.monotonic()

            targets = []
            for _ in range(int(seconds * 20)):
                node._vision_last_seen_time = clock.monotonic()
                node._last_speech_time = clock.monotonic()
                node._on_vision_head_yaw(MockMsg(camera_azimuth_deg))
                node._control_loop()
                clock.advance(0.05)
                targets.append(node._target_yaw)

            return node, targets
        finally:
            H.time = real_time

    def test_one_face_bearing_cannot_walk_the_neck_past_the_field_of_view(self):
        """Measured as distance travelled, not as final angle: with the head free to turn
        a full circle a runaway can wrap right back past its starting point and look
        innocent at the last sample -- which is exactly what the operator saw."""
        _, targets = self._run_with_stale_face(15.8)
        travelled = sum(
            abs(angular_diff_deg(b, a)) for a, b in zip(targets, targets[1:])
        )
        self.assertLessEqual(
            travelled,
            self.CAMERA_HALF_FOV_DEG + 1.0,
            f"Tek bir yuz tespitinden kafa {travelled:.0f} derece yol katetti. Kamera "
            "sadece +-36 deg goruyor, yani yuz en fazla 36 deg otede olabilir; bunun "
            "otesi duzeltmelerin ust uste toplanmasidir (windup), gercek bir hata degil.",
        )

    def test_an_error_that_never_shrinks_stops_driving_the_head(self):
        """A working proportional servo sees its error fall as the head turns. When the
        bearing keeps coming back unchanged the corrections are not landing, so they must
        stop rather than accumulate."""
        _, targets = self._run_with_stale_face(15.8, seconds=12.0)
        tail = targets[-100:]  # last 5 seconds of identical measurements
        drift = max(tail) - min(tail)

        self.assertLessEqual(
            drift,
            0.5,
            f"Son 5 saniyede hedef hala {drift:.1f} derece yurudu (kuyruk: "
            f"{tail[0]:.1f} -> {tail[-1]:.1f}). Ayni hata tekrar tekrar gelirken kafayi "
            "surmeye devam etmek, loglardaki sabit adimli 120 derecelik suzulmenin ta kendisi.",
        )


class TestShortestPathRotation(unittest.TestCase):
    """The neck is a bounded physical joint (safety envelope +-70 to +-85 deg)."""

    def test_ros_travel_limit_never_exceeds_the_firmware_limit(self):
        """If ROS commands past the firmware clamp, the firmware silently truncates and
        the node's dead-reckoned head angle drifts away from reality for good."""
        fw = open(FIRMWARE_FILE, encoding="utf-8").read()
        m = re.search(r"HEAD_MAX_DEG\s*=\s*([0-9.]+)f", fw)
        self.assertIsNotNone(m, "main.cpp icinde HEAD_MAX_DEG bulunamadi")
        firmware_max = float(m.group(1))

        params_src = open(PARAMS_FILE, encoding="utf-8").read()
        m2 = re.search(r"^\s+max_yaw_deg:\s*([0-9.]+)", params_src, re.M)
        self.assertIsNotNone(m2, "astro_params.yaml icinde max_yaw_deg yok")
        ros_max = float(m2.group(1))

        self.assertLessEqual(
            ros_max,
            firmware_max,
            f"ROS {ros_max} deg'e kadar komut veriyor ama firmware {firmware_max} deg'de "
            "kirpiyor; aradaki fark kalici acisal kayma olarak birikir.",
        )

    def test_bounded_physical_joint_safety_envelope(self):
        params_src = open(PARAMS_FILE, encoding="utf-8").read()
        m = re.search(r"^\s+max_yaw_deg:\s*([0-9.]+)", params_src, re.M)
        self.assertIsNotNone(m)
        self.assertLessEqual(
            float(m.group(1)),
            85.0,
            "Boyun continuous 360 joint degildir; mekanik sinirlar +-70 ile +-85 arasinda olmalidir.",
        )


if __name__ == "__main__":
    unittest.main()
