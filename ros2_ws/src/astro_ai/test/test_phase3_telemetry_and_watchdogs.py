"""ASTRO V1 — Phase 3 Acceptance & Verification Test Suite.

Verifies:
  1. System Telemetry JSON payload schema and latency reporting (/astro/telemetry)
  2. Standard ROS 2 DiagnosticArray generation (/diagnostics)
  3. Sensor health watchdogs for RPLiDAR, OAK-D, and Arduino Serial Bridge
  4. Dual-topic diagnostic publishing in SerialBridge
"""

import json
import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch

# Ensure test import paths
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [
    os.path.join(pkg_root, "astro_ai"),
    os.path.join(pkg_root, "astro_ai", "astro_ai"),
    os.path.join(pkg_root, "astro_audio"),
    os.path.join(pkg_root, "astro_audio", "astro_audio"),
    os.path.join(pkg_root, "astro_vision"),
    os.path.join(pkg_root, "astro_vision", "astro_vision"),
    os.path.join(pkg_root, "astro_base"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_ai.astro_realtime_node import AstroRealtimeNode, DiagnosticStatus


class TestPhase3TelemetryAndDiagnostics(unittest.TestCase):
    @patch.dict(os.environ, {"ASTRO_TEST_MODE": "1", "OPENAI_API_KEY": ""})
    def setUp(self):
        self.node = AstroRealtimeNode()
        self.node.pub_telemetry = MagicMock()
        self.node.pub_diagnostics = MagicMock()

    def test_01_telemetry_json_payload_schema(self):
        """Verify /astro/telemetry publishes well-formed JSON with latencies and sensor status."""
        now = time.monotonic()
        self.node._last_laser_scan_time = now - 0.2
        self.node._last_img_time = now - 0.5
        self.node._arduino_heartbeat_healthy = True
        self.node._last_heartbeat_ack_time = now - 0.1
        self.node._active_person_name = "Baran"

        # Record a synthetic turn latency
        self.node.session.latency_tracker.record_turn(gate_ms=45.0, llm_first_token_ms=350.0, total_turn_ms=850.0)

        self.node._publish_system_telemetry()
        self.node.pub_telemetry.publish.assert_called_once()

        call_args = self.node.pub_telemetry.publish.call_args[0][0]
        data = json.loads(call_args.data)

        # Check top-level keys
        self.assertIn("timestamp", data)
        self.assertIn("latency", data)
        self.assertIn("sensors", data)
        self.assertIn("realtime_ws", data)
        self.assertIn("social_state", data)

        # Check latencies
        self.assertEqual(data["latency"]["samples"], 1)
        self.assertAlmostEqual(data["latency"]["p50_total_ms"], 850.0)

        # Check sensors
        self.assertTrue(data["sensors"]["lidar_alive"])
        self.assertTrue(data["sensors"]["camera_alive"])
        self.assertTrue(data["sensors"]["arduino_alive"])

        # Check social state
        self.assertEqual(data["social_state"]["active_person"], "Baran")

    def test_02_ros2_diagnostic_array_structure(self):
        """Verify /diagnostics publishes DiagnosticArray with all core subsystems."""
        now = time.monotonic()
        self.node._last_laser_scan_time = now - 0.1
        self.node._last_img_time = now - 0.2
        self.node._arduino_heartbeat_healthy = True
        self.node._last_heartbeat_ack_time = now - 0.1
        self.node._is_connected = True
        self.node._realtime_state = "CONNECTED"

        self.node._publish_system_telemetry()
        self.node.pub_diagnostics.publish.assert_called_once()

        diag_arr = self.node.pub_diagnostics.publish.call_args[0][0]
        self.assertIsNotNone(diag_arr.status)
        self.assertEqual(len(diag_arr.status), 4)

        names = [st.name for st in diag_arr.status]
        self.assertIn("Astro Realtime / OpenAI WebSocket", names)
        self.assertIn("Astro Base / Serial Controller", names)
        self.assertIn("Astro Safety / RPLiDAR", names)
        self.assertIn("Astro Perception / OAK-D Lite", names)

        # In healthy state, all should be OK
        for st in diag_arr.status:
            self.assertEqual(st.level, DiagnosticStatus.OK)

    def test_03_sensor_watchdog_timeout_levels(self):
        """Verify watchdog raises WARN or ERROR when sensor data is stale or Arduino dead."""
        now = time.monotonic()
        # LiDAR disconnected (> 8.0s) -> ERROR
        self.node._last_laser_scan_time = now - 12.0
        # Camera stale (> 5.0s) -> WARN
        self.node._last_img_time = now - 6.5
        # Arduino heartbeat dead -> ERROR
        self.node._arduino_heartbeat_healthy = False
        self.node._last_heartbeat_ack_time = now - 10.0
        # WS disconnected -> ERROR
        self.node._is_connected = False
        self.node._realtime_state = "DISCONNECTED"

        self.node._publish_system_telemetry()
        diag_arr = self.node.pub_diagnostics.publish.call_args[0][0]

        status_dict = {st.name: st for st in diag_arr.status}
        self.assertEqual(status_dict["Astro Safety / RPLiDAR"].level, DiagnosticStatus.ERROR)
        self.assertEqual(status_dict["Astro Perception / OAK-D Lite"].level, DiagnosticStatus.WARN)
        self.assertEqual(status_dict["Astro Base / Serial Controller"].level, DiagnosticStatus.ERROR)
        self.assertEqual(status_dict["Astro Realtime / OpenAI WebSocket"].level, DiagnosticStatus.ERROR)


class TestPhase3SerialBridgeDiagnostics(unittest.TestCase):
    def test_04_serial_bridge_dual_topic_publishing(self):
        """Verify SerialBridge publishes diagnostics to both /arduino/diagnostics and /diagnostics."""
        from astro_base.serial_bridge import SerialBridge

        with patch.object(SerialBridge, "__init__", return_value=None):
            bridge = SerialBridge()
            bridge.pub_diag = MagicMock()
            bridge.pub_std_diag = MagicMock()
            bridge.arduino_alive = True
            bridge.port = "/dev/ttyACM0"
            bridge.get_clock = MagicMock()
            mock_time = MagicMock()
            mock_time.to_msg.return_value = MagicMock()
            bridge.get_clock.return_value.now.return_value = mock_time

            bridge.publish_diag(vbat_mV=12200, temp_cX100=3450, flags=0)

            bridge.pub_diag.publish.assert_called_once()
            bridge.pub_std_diag.publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
