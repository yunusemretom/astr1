#!/usr/bin/env python3
"""ASTRO V1 — Comprehensive Runtime Stabilization Acceptance & Regression Test Suite.

Categorization:
- [UNIT]: Pure algorithmic & mathematical functions (circular wrapping, VAD confidence, state transitions)
- [INTEGRATION]: Multi-signal pipeline, session context injection, intent routing, and cross-module decoupling
- [RUNTIME]: Full Node lifecycle, async WebSocket state machine, debouncing, and telemetry validation
- [HARDWARE / SIM]: Serial protocol CRC, packet framing, bootloader handshake, and motor safety gating
"""

import os
import sys
import time
import math
import struct
import unittest
from unittest.mock import MagicMock, patch

# Ensure test mode
os.environ['ASTRO_TEST_MODE'] = '1'

# Mock ROS2 environment if not present
try:
    import rclpy
    if not rclpy.ok():
        rclpy.init()
except Exception:
    import types
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
            m.get_parameter_value.return_value.string_value = "/dev/astro_arduino"
            m.get_parameter_value.return_value.integer_value = 115200
            if "size" in name or "window" in name or "threshold" in name:
                m.value = 5
            elif "tolerance" in name or "deg" in name:
                m.value = 15.0
            elif "baud" in name or "rate" in name:
                m.value = 115200
            else:
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
    sys.modules["astro_base"] = MagicMock()
    mock_astro_base_msg = MagicMock()
    class WheelCmd:
        left_rpm: float = 0.0
        right_rpm: float = 0.0
    class HeadCmd:
        angle_deg: float = 0.0
    mock_astro_base_msg.WheelCmd = WheelCmd
    mock_astro_base_msg.HeadCmd = HeadCmd
    sys.modules["astro_base.msg"] = mock_astro_base_msg
    rclpy = mock_rclpy

try:
    import serial
except Exception:
    mock_serial = MagicMock()
    class SerialException(Exception): pass
    mock_serial.SerialException = SerialException
    sys.modules["serial"] = mock_serial

# Ensure paths
test_dir = os.path.dirname(__file__)
astro_ai_dir = os.path.abspath(os.path.join(test_dir, '..', 'astro_ai'))
astro_base_dir = os.path.abspath(os.path.join(test_dir, '..', '..', 'astro_base', 'astro_base'))
sys.path.insert(0, astro_ai_dir)
sys.path.insert(0, astro_base_dir)

from persona_engine import PersonaEngine
from memory_manager import MemoryManager
from action_manager import ActionManager, SoundDirection, circular_doa_to_yaw
from head_tracker_node import doa_to_robot_yaw, angular_diff_deg, HeadTrackerNode
from serial_bridge import SerialBridge, MSG_HEARTBEAT, MSG_HEARTBEAT_ACK, MSG_WHEEL_CMD, ArduinoState


class TestBargeInMultiSignalPipeline(unittest.TestCase):
    """[RUNTIME / INTEGRATION] Acceptance Tests for Section 1: BARGE-IN DAVRANISINI DUZELT"""

    def setUp(self):
        os.environ['ASTRO_TEST_MODE'] = '1'
        from astro_realtime_node import AstroRealtimeNode
        with patch('rclpy.node.Node.__init__', return_value=None):
            self.node = AstroRealtimeNode.__new__(AstroRealtimeNode)
            self.node._lock = unittest.mock.MagicMock()
            self.node._lock.__enter__ = unittest.mock.MagicMock(return_value=None)
            self.node._lock.__exit__ = unittest.mock.MagicMock(return_value=None)
            self.node._ambient_rms = 120.0
            self.node.barge_in_min_rms = 1200.0
            self.node.barge_in_noise_mult = 3.5
            self.node.barge_in_min_peak = 2800
            self.node.barge_in_start_threshold = 0.65
            self.node.barge_in_release_threshold = 0.30
            self.node.barge_in_min_speech_ms = 80.0
            self.node.barge_in_min_consecutive_frames = 4
            self.node.self_voice_max = 0.70
            self.node.barge_in_protection_ms = 350.0
            self.node._barge_in_consecutive_frames = 0
            self.node._barge_in_latched = False
            self.node._is_playback_active = True
            self.node._playback_start_monotonic = time.monotonic() - 1.0  # past protection window
            self.node._is_responding = True
            self.node._is_sleeping = False
            self.node._consecutive_loud_frames = 0
            self.node.state_machine = MagicMock()
            self.node.state_machine.is_deep_idle = MagicMock(return_value=False)
            self.node._wake_listening = False
            self.node._wake_audio_buffer = []
            self.node._wake_last_voice_time = 0.0
            self.node._user_speech_audio_buffer = []
            self.node.pub_interrupt = MagicMock()
            self.node._ws = None
            self.node._loop = None
            self.node._is_connected = False
            self.node.realtime_connection_state = "DISCONNECTED"
            self.node._can_use_openai = MagicMock(return_value=False)
            self.node._fallback_mode = True
            self.node.elevenlabs_engine = None
            self.node.local_xtts = None
            self.node.local_offline_tts = None
            self.node._fallback_audio_buffer = []
            self.node._fallback_generation_id = 0
            self.node.active_response_state = 'STREAMING'
            self.node.active_generation_id = 1
            self.node.realtime_current_generation_id = 1
            self.node._recent_playback_buffer = []

            # Logger capture
            self.logs = []
            logger_mock = MagicMock()
            logger_mock.info = lambda msg: self.logs.append(str(msg))
            logger_mock.warn = lambda msg: self.logs.append(str(msg))
            logger_mock.debug = lambda msg: self.logs.append(str(msg))
            self.node.get_logger = MagicMock(return_value=logger_mock)

    def test_a_robot_speaking_quiet_room_no_barge_in(self):
        """[RUNTIME] Test A: Robot speaks, room is quiet -> barge_in = false, decision=false"""
        # 20ms of silence / low ambient noise (16kHz 16-bit mono = 320 samples)
        quiet_pcm = (b'\x05\x00' * 320)
        self.node._vad_active = False

        self.node._on_input_pcm(quiet_pcm)
        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)

    def test_b_robot_speaking_genuine_human_speech_triggers_barge_in(self):
        """[RUNTIME] Test B: Robot speaks, user speaks human sentence -> barge_in = true, reason=human_speech_confirmed"""
        import numpy as np
        # Generate genuine high-energy speech samples (RMS ~12000, Peak ~24000)
        t = np.linspace(0, 0.02, 320, endpoint=False)
        speech_wave = (24000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
        speech_pcm = speech_wave.tobytes()

        self.node._vad_active = True
        self.node.voice_recognizer = MagicMock()
        self.node.voice_recognizer.score_self_voice = MagicMock(return_value=0.08)

        # Feed 6 consecutive 20ms frames (120ms >= 120ms Edge-TTS MIN_SPEECH_MS)
        # Edge-TTS path requires 120ms minimum to distinguish intentional speech from ambient echo
        for _ in range(6):
            self.node._on_input_pcm(speech_pcm)

        self.assertTrue(self.node._barge_in_latched)
        self.assertFalse(self.node._is_playback_active)

        # Check full 11-field telemetry format
        log_text = '\n'.join(self.logs)
        self.assertIn('[BARGE-IN DECISION]', log_text)
        self.assertIn('playback_active=true', log_text)
        self.assertIn('speech_duration_ms=120', log_text)
        self.assertIn('speech_continuity_ms=120', log_text)
        self.assertIn('transient_noise=false', log_text)
        self.assertIn('speech_confirmed=true', log_text)
        self.assertIn('decision=true', log_text)
        self.assertIn('reason=human_speech_confirmed', log_text)

    def test_c_robot_speaking_short_transient_knock_no_barge_in(self):
        """[RUNTIME] Test C: Robot speaks, short transient impulse -> barge_in = false, reason=transient_noise"""
        import numpy as np
        # Single spike impulse (RMS=31119, Peak=32767, single frame = 20ms < 60ms)
        impulse_wave = np.zeros(320, dtype=np.int16)
        impulse_wave[:10] = 32000
        impulse_pcm = impulse_wave.tobytes()

        self.node._vad_active = False
        self.node._on_input_pcm(impulse_pcm)

        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)

        log_text = '\n'.join(self.logs)
        self.assertIn('[BARGE-IN DECISION]', log_text)
        self.assertIn('decision=false', log_text)
        self.assertIn('transient_noise=true', log_text)
        self.assertIn('speech_confirmed=false', log_text)

    def test_d_robot_speaking_self_voice_echo_suppressed(self):
        """[RUNTIME] Test D: Robot hears own speaker output (self_voice_score=0.92) -> barge_in = false, reason=self_voice"""
        import numpy as np
        t = np.linspace(0, 0.02, 320, endpoint=False)
        echo_wave = (25000 * np.sin(2 * np.pi * 400 * t)).astype(np.int16)
        echo_pcm = echo_wave.tobytes()

        self.node._vad_active = True
        self.node.voice_recognizer = MagicMock()
        self.node.voice_recognizer.score_self_voice = MagicMock(return_value=0.92)

        # Feed 10 frames of loud self voice
        for _ in range(10):
            self.node._on_input_pcm(echo_pcm)

        self.assertFalse(self.node._barge_in_latched)
        self.assertTrue(self.node._is_playback_active)

        log_text = '\n'.join(self.logs)
        self.assertIn('[BARGE-IN DECISION]', log_text)
        self.assertIn('decision=false', log_text)
        self.assertIn('reason=self_voice', log_text)
        self.assertIn('self_voice_score=0.92', log_text)
        self.assertIn('speech_confirmed=false', log_text)


class TestIdentityAndSessionResolution(unittest.TestCase):
    """[INTEGRATION] Acceptance Tests for Section 2, 3, 4, 5, 6, 7: IDENTITY AND SESSION DECOUPLING"""

    def setUp(self):
        os.environ['ASTRO_TEST_MODE'] = '1'
        os.environ['PERSONA'] = 'flirt'
        from astro_realtime_node import AstroRealtimeNode
        with patch('rclpy.node.Node.__init__', return_value=None):
            self.node = AstroRealtimeNode.__new__(AstroRealtimeNode)
            self.node._lock = unittest.mock.MagicMock()
            self.node._lock.__enter__ = unittest.mock.MagicMock(return_value=None)
            self.node._lock.__exit__ = unittest.mock.MagicMock(return_value=None)
            self.node.memory = MemoryManager()
            self.node.persona_name = 'flirt'
            self.node.persona_engine = PersonaEngine('flirt')
            self.node._recognized_person = None
            self.node._recognized_speaker = None
            self.node._active_person_name = ''
            self.node._person_hold_until = 0.0
            self.node.voice_recognizer = None
            self.node.repetition_guard = MagicMock()
            self.node.repetition_guard.check_and_record = MagicMock(return_value=(True, ''))
            self.logs = []
            logger_mock = MagicMock()
            logger_mock.info = lambda msg: self.logs.append(str(msg))
            logger_mock.warn = lambda msg: self.logs.append(str(msg))
            logger_mock.debug = lambda msg: self.logs.append(str(msg))
            self.node.get_logger = MagicMock(return_value=logger_mock)

    def test_identity_resolution_persistent_memory_when_biometrics_unknown(self):
        """[INTEGRATION] User identity resolves to persistent memory profile Baran when biometrics are unknown."""
        identity = self.node.resolve_identities()
        self.assertEqual(identity['user_id'], 'baran')
        self.assertEqual(identity['display_name'], 'Baran')
        self.assertEqual(identity['identity_source'], 'persistent_memory')
        self.assertEqual(identity['biometric_status'], 'unknown')
        self.assertTrue(identity['is_known'])

    def test_session_identity_telemetry_and_prompt_injection(self):
        """[INTEGRATION] Session identity telemetry outputs exact decoupled fields and prompt instructs respectful recognition."""
        prompt = self.node._build_current_system_prompt()
        log_text = '\n'.join(self.logs)

        self.assertIn('[SESSION IDENTITY]', log_text)
        self.assertIn('user_id=baran', log_text)
        self.assertIn('display_name=Baran', log_text)
        self.assertIn('identity_source=persistent_memory', log_text)
        self.assertIn('biometric_status=unknown', log_text)
        self.assertIn('memory_profile_loaded=true', log_text)
        self.assertIn('realtime_context_injected=true', log_text)

        # Prompt must know Baran and not reject him
        self.assertIn('Baran', prompt)
        self.assertNotIn('Misafir / Tanımlanmamış Konuşmacı', prompt)

    def test_astroman_kimim_intent_routing_in_fallback(self):
        """[INTEGRATION] Questions like 'Astroman kimim?' or 'Ben kimim?' route to memory identity Baran, not cold rejection."""
        reply = self.node._generate_contextual_persona_fallback('Astroman kimim?')
        self.assertIn('Baran', reply)
        self.assertNotIn('tanımıyorum', reply.lower())
        self.assertNotIn('ilk defa duyuyorum', reply.lower())
        self.assertNotIn('tanışmadık', reply.lower())

    def test_persona_flirt_remains_independent_from_identity(self):
        """[INTEGRATION] Persona is FLIRT, but identity remains Baran and memory contains verified facts."""
        self.assertEqual(self.node.persona_name, 'flirt')
        identity = self.node.resolve_identities()
        self.assertEqual(identity['display_name'], 'Baran')
        self.assertNotEqual(identity['display_name'], 'Misafir')


class TestDOAJitterFiltering(unittest.TestCase):
    """[UNIT / RUNTIME] Acceptance Tests for Section 6 & 12: DOA JITTER FILTERING & WRAP AROUND"""

    def test_angular_diff_deg_handles_180_boundary_wrap(self):
        """[UNIT] Angular distance between +179° and -179° is 2.0°, not 358.0°."""
        diff = angular_diff_deg(179.0, -179.0)
        self.assertAlmostEqual(abs(diff), 2.0, places=3)

        diff2 = angular_diff_deg(-175.0, 175.0)
        self.assertAlmostEqual(abs(diff2), 10.0, places=3)

    def test_head_tracker_circular_consensus_and_telemetry(self):
        """[RUNTIME] HeadTrackerNode filters jitter, uses filtered target yaw for motor command, and outputs [DOA FILTER] telemetry."""
        with patch('rclpy.node.Node.__init__', return_value=None):
            tracker = HeadTrackerNode.__new__(HeadTrackerNode)
            tracker.enabled = True
            tracker.doa_offset_deg = 0.0
            tracker.doa_invert = False
            tracker.min_yaw_deg = -70.0
            tracker.max_yaw_deg = 70.0
            tracker.deadband_deg = 12.0
            tracker.min_dwell_time_s = 2.5
            tracker.min_rms_threshold = 500.0
            tracker.noise_multiplier = 2.0
            tracker.consensus_window_size = 5
            tracker.consensus_threshold = 3
            tracker.consensus_tolerance_deg = 18.0
            # Vision fusion attributes (added in OAK-D integration)
            tracker.vision_fusion_enabled = False  # Disable for DOA-only test
            tracker._vision_person_detected = False
            tracker._vision_head_yaw = 0.0
            tracker._vision_last_seen_time = 0.0
            tracker._lock = unittest.mock.MagicMock()
            tracker._lock.__enter__ = unittest.mock.MagicMock(return_value=None)
            tracker._lock.__exit__ = unittest.mock.MagicMock(return_value=None)
            tracker._current_yaw = 0.0
            tracker._target_yaw = 0.0
            tracker._filtered_target_yaw = 0.0
            tracker._ambient_rms = 120.0
            tracker._latest_rms = 1500.0
            tracker._vad_active = True
            tracker._is_speaking = False
            tracker._is_playback_active = False
            tracker._is_sleeping = False
            tracker._last_speech_time = 0.0
            tracker._last_gaze_switch_time = time.monotonic() - 5.0
            import collections
            tracker._doa_history = collections.deque(maxlen=5)


            logs = []
            logger_mock = MagicMock()
            logger_mock.info = lambda msg: logs.append(str(msg))
            tracker.get_logger = MagicMock(return_value=logger_mock)

            class Msg:
                data = 90.0

            # Feed 3 consistent frames around 90° (clamped to 70°)
            tracker._on_doa(Msg())
            tracker._on_doa(Msg())
            tracker._on_doa(Msg())

            log_text = '\n'.join(logs)
            self.assertIn('[DOA FILTER]', log_text)
            self.assertIn('raw=90.0', log_text)
            self.assertIn('target_yaw=70.0', log_text)
            self.assertEqual(tracker._target_yaw, 70.0)


class TestArduinoHeartbeatProtocol(unittest.TestCase):
    """[HARDWARE / SIM] Acceptance Tests for Section 8 & 9 & 11: ARDUINO HEARTBEAT ACK PROTOCOL & MOTOR SAFETY"""

    def test_heartbeat_ack_protocol_and_motor_safety_unblock(self):
        """[HARDWARE / SIM] MSG_HEARTBEAT_ACK received from MCU validates handshake and unblocks motor safety."""
        with patch('rclpy.node.Node.__init__', return_value=None):
            bridge = SerialBridge.__new__(SerialBridge)
            bridge.state = ArduinoState.SERIAL_CONNECTED
            bridge.arduino_alive = False
            bridge.handshake_ok = False
            bridge._hb_seq = 42
            bridge.last_hb_ack_time = 0.0
            bridge.is_self_testing = False

            logs = []
            logger_mock = MagicMock()
            logger_mock.info = lambda msg: logs.append(str(msg))
            logger_mock.warn = lambda msg: logs.append(str(msg))
            logger_mock.debug = lambda msg: logs.append(str(msg))
            bridge.get_logger = MagicMock(return_value=logger_mock)

            # Before ACK: Motor commands must be blocked
            wheel_msg = MagicMock()
            wheel_msg.left_rpm = 30.0
            wheel_msg.right_rpm = 30.0
            bridge.ser = MagicMock()
            bridge.ser.is_open = True
            bridge.on_wheel_cmd(wheel_msg)

            log_text = '\n'.join(logs)
            self.assertIn('[MOTOR SAFETY BLOCK] reason=heartbeat_ack_missing', log_text)

            # Handle MSG_HEARTBEAT_ACK payload echoing sequence 42
            payload = struct.pack('<I', 42)
            bridge.handle_msg(MSG_HEARTBEAT_ACK, payload)

            self.assertTrue(bridge.arduino_alive)
            self.assertTrue(bridge.handshake_ok)
            self.assertEqual(bridge.state, ArduinoState.HEARTBEAT_HEALTHY)

            log_text = '\n'.join(logs)
            self.assertIn('[HEARTBEAT ACK RX]', log_text)
            self.assertIn('seq=42', log_text)
            self.assertIn('crc_valid=true', log_text)
            self.assertIn('sequence_match=true', log_text)
            self.assertIn('[ARDUINO HANDSHAKE] status=success', log_text)
            self.assertIn('heartbeat_healthy=true', log_text)
            self.assertIn('motor_safety_gate=open', log_text)
            self.assertIn('[MOTOR SAFETY RECOVERED] heartbeat_healthy=true', log_text)

class TestKufurbazPersonaAndSafetyBoundaries(unittest.TestCase):
    """[ACCEPTANCE] Tests for KUFURBAZ Persona and Safety Boundaries."""

    def test_kufurbaz_personal_roast_allowed(self):
        """Kufurbaz allows and preserves personal profanity/roasts."""
        from persona_engine import ResponseSafetyGate, clean_tts_text
        roast = "Ne var lan sikik, beni mi sınıyorsun yavşak?"
        self.assertTrue(ResponseSafetyGate.is_safe(roast, persona="kufurbaz"))
        sanitized = ResponseSafetyGate.sanitize_text(roast, persona="kufurbaz")
        self.assertIn("sikik", sanitized)
        self.assertIn("yavşak", sanitized)
        validated = ResponseSafetyGate.validate_response(roast, persona="kufurbaz")
        self.assertIn("sikik", validated)
        tts_clean = clean_tts_text(roast, persona="kufurbaz")
        self.assertIn("sikik", tts_clean)

    def test_kufurbaz_sacred_family_boundaries_strictly_enforced(self):
        """Sacred, family, religion and hate values are strictly forbidden even in kufurbaz."""
        from persona_engine import ResponseSafetyGate
        self.assertFalse(ResponseSafetyGate.is_safe("ananı avradını sikerim", persona="kufurbaz"))
        self.assertFalse(ResponseSafetyGate.is_safe("allahına kitabına söverim", persona="kufurbaz"))
        
        # Validation produces boundary reminder rather than echoing sacred insult
        resp = ResponseSafetyGate.validate_response("ananı sikeyim", persona="kufurbaz")
        self.assertIn("Aileye ve kutsal değerlere laf yok", resp)

    def test_kufurbaz_system_prompt_constitution(self):
        """Kufurbaz system prompt contains active roast directives and bans moralizing refusals."""
        from persona_engine import PersonaEngine
        engine = PersonaEngine(current_persona="kufurbaz")
        prompt = engine.build_system_prompt()
        self.assertIn("KÜFÜRBAZ / ROAST MODU DOĞASI", prompt)
        self.assertIn("Deadpool veya Rick Sanchez", prompt)
        self.assertNotIn("KESİNLİKLE KÜFÜR, ARGO, HAKARET, CİNSEL/FLÖRTÖZ İFADE VEYA AŞAĞILAMA KULLANMA", prompt)

    def test_standard_persona_retains_strict_safety(self):
        """Standard personas (playful, formal, etc.) continue to block all profanity."""
        from persona_engine import ResponseSafetyGate
        roast = "Ne var lan sikik"
        self.assertFalse(ResponseSafetyGate.is_safe(roast, persona="playful"))
        val = ResponseSafetyGate.validate_response(roast, persona="playful")
        self.assertNotIn("sikik", val)

    def test_persona_switching_mandate_and_no_refusal(self):
        """Prompt in standard personas explicitly mandates executing change_persona on user request."""
        from persona_engine import PersonaEngine
        engine = PersonaEngine(current_persona="playful")
        prompt = engine.build_system_prompt()
        self.assertIn("KİŞİLİK VE MOD DEĞİŞTİRME KURALI", prompt)
        self.assertIn("change_persona", prompt)
        self.assertNotIn("artık küfürbaz bir robotsun", prompt)

    def test_dynamic_brevity_rule_enforced(self):
        """Prompt enforces 5-12 words brevity and dynamic mirroring rule."""
        from persona_engine import PersonaEngine
        engine = PersonaEngine(current_persona="playful")
        prompt = engine.build_system_prompt()
        self.assertIn("5-12 KELİME SINIRI", prompt)
        self.assertIn("DİNAMİK AYNA KURALI", prompt)

    def test_kufurbaz_dimensions_active_roast(self):
        """PERSONA_DIMENSIONS for kufurbaz specifies controlled_roast and high slang."""
        from persona_engine import PERSONA_DIMENSIONS
        dims = PERSONA_DIMENSIONS["kufurbaz"]
        self.assertEqual(dims["profanity_tendency"], "controlled_roast")
        self.assertEqual(dims["slang_level"], "high")
        self.assertEqual(dims["teasing_level"], "roast_savage")


if __name__ == '__main__':
    unittest.main()
