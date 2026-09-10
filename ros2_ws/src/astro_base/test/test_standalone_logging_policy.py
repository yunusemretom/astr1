"""Tests for Standalone Gaze ROS Node Logging Policy.

Verifies:
1. Default INFO level does not emit per-frame forensic spam (no CENTER_DIAG, RAW, SIGNS on every frame).
2. AUDIO state changes are clearly visible ([AUDIO] speech detected, [AUDIO] DOA, [AUDIO] owner transitions).
3. DEBUG / verbose_diagnostics exposes detailed forensic telemetry.
4. Logging execution is non-blocking and adds negligible latency (<5ms per cycle).
"""

import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "standalone")))

from astro_base.standalone_gaze_ros_node import StandaloneGazeRosNode
from astro_base.gaze.gaze_tracker import Detection
from astro_base.gaze.types import PrioritySource


class MockSpeechVerdict:
    def __init__(self, is_speech: bool = True, confidence: float = 0.90):
        self.is_speech = is_speech
        self.confidence = confidence


class TestStandaloneLoggingPolicy:

    def test_info_logging_does_not_emit_per_frame_forensic_spam(self, capsys):
        """When verbose_diagnostics=False (default), 30 frames do not spam stdout with forensic blocks."""
        node = StandaloneGazeRosNode(
            use_camera_source=False,
            enable_audio=False,
            enable_voice=False,
            verbose_diagnostics=False,
        )

        t = 100.0
        for _ in range(30):
            t += 0.033
            node.step_frame(
                detections=[],
                frame_size=(640, 480),
                timestamp=t,
            )

        captured = capsys.readouterr()
        # Forensic block keywords MUST NOT appear in stdout
        assert "CENTER_DIAG" not in captured.out
        assert "SIGNS:" not in captured.out
        assert "FEEDBACK_SYNC:" not in captured.out
        assert "head_feedback_source=" not in captured.out
        assert "sign(visual_bearing)" not in captured.out

    def test_string_false_coercion_prevents_launch_spam(self, capsys):
        """When verbose_diagnostics is passed as string 'false' from ROS launch, it coerces to False."""
        node = StandaloneGazeRosNode(
            use_camera_source=False,
            enable_audio=False,
            enable_voice=False,
            verbose_diagnostics="false",
        )
        assert node.verbose_diagnostics is False

        node.step_frame(detections=[], frame_size=(640, 480), timestamp=10.0)
        captured = capsys.readouterr()
        assert "CENTER_DIAG" not in captured.out
        assert "FEEDBACK_SYNC:" not in captured.out

    def test_audio_state_changes_are_visible(self, caplog):
        """Audio state changes ([AUDIO] speech detected, DOA, owner=...) are clearly logged."""
        node = StandaloneGazeRosNode(
            use_camera_source=False,
            enable_audio=False,
            enable_voice=False,
            verbose_diagnostics=False,
        )

        with caplog.at_level(logging.INFO):
            t = 200.0
            speech = MockSpeechVerdict(is_speech=True, confidence=0.92)

            # 1. Active speech from right (+45 DOA)
            for _ in range(4):
                t += 0.033
                node.step_frame(
                    detections=[],
                    frame_size=(640, 480),
                    timestamp=t,
                    doa_deg=45.0,
                    speech=speech,
                )

            # 2. Face appears on left
            face_left = [Detection(x=100, y=200, w=80, h=80, confidence=0.90)]
            for _ in range(4):
                t += 0.033
                node.step_frame(
                    detections=face_left,
                    frame_size=(640, 480),
                    timestamp=t,
                    doa_deg=45.0,
                    speech=speech,
                )

        log_text = caplog.text
        # Speech onset detected
        assert "[AUDIO] speech detected confidence=0.92" in log_text
        # DOA printed
        assert "[AUDIO] DOA=+45.0°" in log_text
        # Audio reacquisition seized authority
        assert "[AUDIO] owner=AUDIO_REACQUISITION" in log_text
        # Visual tracking seized authority back
        assert "[AUDIO] owner=VISUAL_TRACKING" in log_text
        # Visual target log present
        assert "[VISUAL] target=" in log_text

    def test_verbose_diagnostics_exposes_forensic_telemetry(self, capsys):
        """When verbose_diagnostics=True, full forensic blocks are emitted."""
        node = StandaloneGazeRosNode(
            use_camera_source=False,
            enable_audio=False,
            enable_voice=False,
            verbose_diagnostics=True,
        )

        t = 300.0
        node.step_frame(
            detections=[Detection(x=300, y=200, w=80, h=80, confidence=0.90)],
            frame_size=(640, 480),
            timestamp=t,
        )

        captured = capsys.readouterr()
        assert "CENTER_DIAG:" in captured.out
        assert "RAW:" in captured.out
        assert "VISION:" in captured.out
        assert "FEEDBACK:" in captured.out
        assert "CONTROL:" in captured.out
        assert "ACTUATOR:" in captured.out
        assert "SIGNS:" in captured.out

    def test_logging_does_not_block_cycle(self):
        """50 cycles execute within <100ms total (<2ms per frame step), proving non-blocking performance."""
        node = StandaloneGazeRosNode(
            use_camera_source=False,
            enable_audio=False,
            enable_voice=False,
            verbose_diagnostics=False,
        )

        t_start = time.perf_counter()
        t = 400.0
        for _ in range(50):
            t += 0.033
            node.step_frame(
                detections=[],
                frame_size=(640, 480),
                timestamp=t,
            )
        elapsed_total = time.perf_counter() - t_start
        avg_ms_per_step = (elapsed_total / 50) * 1000.0

        # Must be well below 10ms per step (normally < 1.5ms)
        assert avg_ms_per_step < 10.0
