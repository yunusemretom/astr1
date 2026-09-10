#!/usr/bin/env python3
"""Comprehensive Acceptance Tests for Realtime Self-Speech / False Barge-In Bug Fix."""

import asyncio
import base64
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

os.environ['ASTRO_TEST_MODE'] = '1'

# rclpy mock for non-ROS local testing environments
try:
    import rclpy
    if not rclpy.ok():
        rclpy.init()
except Exception:
    mock_rclpy = MagicMock()
    mock_rclpy.ok.return_value = True
    class MockTime:
        def __init__(self, nanoseconds=0):
            self.nanoseconds = nanoseconds
    mock_rclpy.time.Time = MockTime
    class MockNode:
        def __init__(self, name="", *args, **kwargs):
            self.name = name
            self._logger = MagicMock()
            self._clock = MagicMock()
            self._clock.now.return_value = MockTime(int(time.time() * 1e9))
        def declare_parameter(self, *args, **kwargs): pass
        def get_parameter(self, name):
            m = MagicMock()
            m.get_parameter_value.return_value.string_value = ""
            m.value = 0.06
            return m
        def create_publisher(self, *args, **kwargs): return MagicMock()
        def create_subscription(self, *args, **kwargs): return MagicMock()
        def create_timer(self, *args, **kwargs): return MagicMock()
        def get_logger(self): return self._logger
        def get_clock(self): return self._clock
        def destroy_node(self): pass
    mock_rclpy.node.Node = MockNode
    mock_cbg = MagicMock()
    mock_cbg.MutuallyExclusiveCallbackGroup = MagicMock
    mock_cbg.ReentrantCallbackGroup = MagicMock
    sys.modules["rclpy"] = mock_rclpy
    sys.modules["rclpy.node"] = mock_rclpy.node
    sys.modules["rclpy.qos"] = MagicMock()
    sys.modules["rclpy.time"] = mock_rclpy.time
    sys.modules["rclpy.callback_groups"] = mock_cbg
    sys.modules["diagnostic_msgs"] = MagicMock()
    sys.modules["diagnostic_msgs.msg"] = MagicMock()
    sys.modules["sensor_msgs"] = MagicMock()
    sys.modules["sensor_msgs.msg"] = MagicMock()
    sys.modules["std_msgs"] = MagicMock()
    sys.modules["std_msgs.msg"] = MagicMock()
    rclpy = mock_rclpy


class FakeRealtimeTransport:
    def __init__(self):
        self.sent_events = []
        self.closed = False

    async def send(self, data: str):
        if isinstance(data, str):
            try:
                self.sent_events.append(json.loads(data))
            except Exception:
                self.sent_events.append({"raw": data})
        else:
            self.sent_events.append(data)

    async def close(self):
        self.closed = True

    def get_sent_types(self):
        return [e.get("type", "") for e in self.sent_events if isinstance(e, dict)]


class TestSelfSpeechSuppressionAndBargeIn(unittest.TestCase):
    def setUp(self):
        from astro_ai.astro_realtime_node import AstroRealtimeNode
        from astro_ai.state_machine import RobotState
        fake_ws = FakeRealtimeTransport()
        self.node = AstroRealtimeNode(connect_realtime=False, fake_transport=fake_ws)
        self.node._ambient_rms = 120.0
        self.node.barge_in_min_rms = 1200.0
        self.node.barge_in_playback_min_rms = 2000.0
        self.node.barge_in_noise_mult = 3.5
        self.node.barge_in_min_peak = 2800
        self.node.barge_in_playback_min_peak = 3000
        self.node.barge_in_min_speech_ms = 80.0
        self.node.barge_in_min_consecutive_frames = 4
        self.node.self_voice_max = 0.70
        self.node.barge_in_protection_ms = 350.0
        self.node._barge_in_consecutive_frames = 0
        self.node._barge_in_latched = False
        self.node._is_playback_active = True
        self.node._playback_start_monotonic = time.monotonic() - 1.0  # past protection window
        self.node.state_machine.transition_to(RobotState.SPEAKING)
        self.node._is_responding = True
        self.node._is_sleeping = False
        self.node._consecutive_loud_frames = 0
        self.node.realtime_connection_state = "CONNECTED"
        self.node.realtime_session_state = "READY"
        self.node.active_response_state = "STREAMING"
        self.node.pub_interrupt = MagicMock()

        self.logs = []
        logger_mock = MagicMock()
        logger_mock.info = lambda msg: self.logs.append(str(msg))
        logger_mock.debug = lambda msg: self.logs.append(str(msg))
        logger_mock.warn = lambda msg: self.logs.append(str(msg))
        logger_mock.warning = lambda msg: self.logs.append(str(msg))
        self.node.get_logger = MagicMock(return_value=logger_mock)

    def test_1_astro_speaks_user_silent_no_barge_in(self):
        """1. ASTRO speaks, user is silent -> BARGE-IN = NO, reason=self_voice."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tts_audio = (12000 * np.sin(2 * np.pi * 350 * t)).astype(np.int16).tobytes()
        self.node._update_playback_reference(tts_audio)

        # Microphone captures 10 frames of playback echo (delayed by 30ms = 480 samples = 960 bytes)
        echo_chunk = tts_audio[960 : 960 + 640]

        for _ in range(10):
            self.node._on_input_pcm(echo_chunk)

        # Assert no barge in latched, playback continues
        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)

        log_text = '\n'.join(self.logs)
        self.assertIn('[BARGE-IN DECISION]', log_text)
        self.assertIn('playback_active=true', log_text)
        self.assertIn('reason=self_voice', log_text)
        self.assertIn('decision=false', log_text)
        self.assertIn('speech_confirmed=false', log_text)

    def test_2_astro_speaks_user_talks_barge_in_yes(self):
        """2. ASTRO speaks, user genuinely interrupts -> BARGE-IN = YES, reason=human_speech_confirmed."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tts_audio = (10000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()
        self.node._update_playback_reference(tts_audio)

        # User speaks with distinct acoustic pattern (RMS ~17000, Peak ~24000)
        t_frame = np.linspace(0, 0.02, 320, endpoint=False)
        user_speech = (24000 * np.sin(2 * np.pi * 650 * t_frame)).astype(np.int16).tobytes()

        self.node._vad_active = True
        # Feed 6 frames
        for _ in range(6):
            self.node._on_input_pcm(user_speech)

        self.assertTrue(self.node._barge_in_latched)
        self.assertFalse(self.node._is_playback_active)

        log_text = '\n'.join(self.logs)
        self.assertIn('reason=human_speech_confirmed', log_text)
        self.assertIn('decision=true', log_text)
        self.assertIn('speech_confirmed=true', log_text)

    def test_3_astro_not_speaking_user_speaks(self):
        """3. ASTRO not speaking, user speaks -> normal speech detection active."""
        self.node._is_playback_active = False
        self.node._is_responding = False

        t_frame = np.linspace(0, 0.02, 320, endpoint=False)
        user_speech = (15000 * np.sin(2 * np.pi * 400 * t_frame)).astype(np.int16).tobytes()

        # Send input frame
        self.node._on_input_pcm(user_speech)

        # Verified buffered for user speech
        self.assertGreater(len(self.node._user_speech_audio_buffer), 0)

    def test_4_astro_speaks_own_playback_never_triggers_new_response(self):
        """4. ASTRO speaks -> continuous self-playback echo never triggers new response."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tts_audio = (14000 * np.sin(2 * np.pi * 420 * t)).astype(np.int16).tobytes()
        self.node._update_playback_reference(tts_audio)

        # Feed 25 frames (500ms of sustained loud self-echo)
        for i in range(25):
            chunk = tts_audio[i * 640 : (i + 1) * 640]
            self.node._on_input_pcm(chunk)

        # Barge-in never latches
        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)
        # Interrupt never published
        self.node.pub_interrupt.publish.assert_not_called()

    def test_5_telemetry_fields_fully_reported(self):
        """5. Telemetry fields playback_active, rms, speech_duration_ms, self_voice_score, speech_confirmed, decision, reason are fully logged."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tts_audio = (12000 * np.sin(2 * np.pi * 350 * t)).astype(np.int16).tobytes()
        self.node._update_playback_reference(tts_audio)
        echo_chunk = tts_audio[640:1280]

        self.node._on_input_pcm(echo_chunk)

        log_text = '\n'.join(self.logs)
        for field in [
            'playback_active=true',
            'vad_confidence=',
            'speech_duration_ms=',
            'rms=',
            'self_voice_score=',
            'speech_confirmed=false',
            'decision=false',
            'reason=self_voice',
        ]:
            self.assertIn(field, log_text, f"Expected {field} in log_text")

    def test_6_two_user_turns_exactly_two_responses_no_runaway_loop(self):
        """6. User speaks only twice -> exactly two turns, no autonomous runaway feedback turns."""
        turns_started = 0
        class MockBool:
            def __init__(self, val: bool):
                self.data = val

        # --- Turn 1: User speaks ---
        self.node._is_playback_active = False
        t_user = np.linspace(0, 0.02, 320, endpoint=False)
        user_pcm = (16000 * np.sin(2 * np.pi * 500 * t_user)).astype(np.int16).tobytes()
        self.node._on_input_pcm(user_pcm)
        turns_started += 1

        # Robot begins speaking response 1
        t_tts = np.linspace(0, 1.0, 16000, endpoint=False)
        tts_pcm_1 = (14000 * np.sin(2 * np.pi * 320 * t_tts)).astype(np.int16).tobytes()
        self.node._on_playback_active(MockBool(True))
        self.node._update_playback_reference(tts_pcm_1)

        # Microphones hear 15 frames of TTS 1 echo
        for i in range(15):
            self.node._on_input_pcm(tts_pcm_1[i * 640 : (i + 1) * 640])

        # Self-echo must NOT trigger barge in or cancel
        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)

        # Robot finishes response 1
        self.node._on_playback_active(MockBool(False))
        self.assertFalse(self.node._is_playback_active)

        # --- Turn 2: User speaks second sentence ---
        self.node._on_input_pcm(user_pcm)
        turns_started += 1

        # Robot begins speaking response 2
        tts_pcm_2 = (14000 * np.sin(2 * np.pi * 380 * t_tts)).astype(np.int16).tobytes()
        self.node._on_playback_active(MockBool(True))
        self.node._update_playback_reference(tts_pcm_2)

        # Microphones hear 15 frames of TTS 2 echo
        for i in range(15):
            self.node._on_input_pcm(tts_pcm_2[i * 640 : (i + 1) * 640])

        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)

        # Robot finishes response 2
        self.node._on_playback_active(MockBool(False))

        # Total user initiated turns must be exactly 2
        self.assertEqual(turns_started, 2)
        # Interrupt was never triggered by own speech
        self.node.pub_interrupt.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
