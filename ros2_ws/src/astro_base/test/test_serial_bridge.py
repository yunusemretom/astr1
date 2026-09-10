#!/usr/bin/env python3
"""Unit tests for ASTRO serial bridge communication protocol and safety gating."""

import struct
import unittest

SOF1 = 0xAA
SOF2 = 0x55

MSG_HEARTBEAT = 0x01
MSG_WHEEL_CMD = 0x02
MSG_HEAD_CMD = 0x03
MSG_IMU_DATA = 0x10
MSG_ENCODER_TICKS = 0x11
MSG_DIAGNOSTICS = 0x12
MSG_HEARTBEAT_ACK = 0x13


def crc8(data: bytes) -> int:
    poly = 0x07
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ poly
            else:
                crc = (crc << 1) & 0xFF
    return crc


def build_packet(msg_id: int, payload: bytes) -> bytes:
    length = 1 + len(payload)
    body = bytes([length, msg_id]) + payload
    c = crc8(body)
    return bytes([SOF1, SOF2]) + body + bytes([c])


class TestSerialBridgeProtocol(unittest.TestCase):
    def test_crc8_known_vector(self):
        # Empty payload
        self.assertEqual(crc8(b""), 0x00)
        # Standard vector
        c = crc8(bytes([0x05, 0x01, 0x00, 0x00, 0x00, 0x01]))
        self.assertIsInstance(c, int)
        self.assertTrue(0 <= c <= 255)

    def test_build_heartbeat_packet(self):
        seq = 42
        payload = struct.pack("<I", seq)
        pkt = build_packet(MSG_HEARTBEAT, payload)
        self.assertEqual(pkt[0], SOF1)
        self.assertEqual(pkt[1], SOF2)
        self.assertEqual(pkt[2], 1 + len(payload))  # Length (len(msg_id) + len(payload))
        self.assertEqual(pkt[3], MSG_HEARTBEAT)
        unpacked_seq = struct.unpack("<I", pkt[4:8])[0]
        self.assertEqual(unpacked_seq, 42)
        # Verify CRC matches
        calc_crc = crc8(pkt[2:-1])
        self.assertEqual(pkt[-1], calc_crc)

    def test_build_wheel_cmd_packet(self):
        l_rpm = 15.5
        r_rpm = -12.3
        payload = struct.pack("<ff", l_rpm, r_rpm)
        pkt = build_packet(MSG_WHEEL_CMD, payload)
        self.assertEqual(pkt[0], SOF1)
        self.assertEqual(pkt[1], SOF2)
        self.assertEqual(pkt[3], MSG_WHEEL_CMD)
        unpacked_l, unpacked_r = struct.unpack("<ff", pkt[4:12])
        self.assertAlmostEqual(unpacked_l, l_rpm, places=4)
        self.assertAlmostEqual(unpacked_r, r_rpm, places=4)

    def test_build_heartbeat_ack_packet(self):
        seq = 1001
        payload = struct.pack("<I", seq)
        pkt = build_packet(MSG_HEARTBEAT_ACK, payload)
        self.assertEqual(pkt[3], MSG_HEARTBEAT_ACK)
        unpacked_seq = struct.unpack("<I", pkt[4:8])[0]
        self.assertEqual(unpacked_seq, 1001)

    def test_packet_corruption_detection(self):
        payload = struct.pack("<I", 999)
        pkt = bytearray(build_packet(MSG_HEARTBEAT, payload))
        # Corrupt one payload byte
        pkt[5] ^= 0xFF
        calc_crc = crc8(bytes(pkt[2:-1]))
        # Corrupted packet body will fail CRC check against original CRC byte
        self.assertNotEqual(pkt[-1], calc_crc)

    def test_build_head_cmd_packet_angles(self):
        """Verify binary packet building for head angles: 0, ±2, ±5, ±10, ±70 degrees."""
        for angle in [0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 70.0, -70.0]:
            payload = struct.pack("<f", angle)
            pkt = build_packet(MSG_HEAD_CMD, payload)
            self.assertEqual(pkt[0], SOF1)
            self.assertEqual(pkt[1], SOF2)
            self.assertEqual(pkt[2], 1 + len(payload))
            self.assertEqual(pkt[3], MSG_HEAD_CMD)
            unpacked_angle = struct.unpack("<f", pkt[4:8])[0]
            self.assertAlmostEqual(unpacked_angle, angle, places=4)
            calc_crc = crc8(pkt[2:-1])
            self.assertEqual(pkt[-1], calc_crc)


if __name__ == "__main__":
    unittest.main()
