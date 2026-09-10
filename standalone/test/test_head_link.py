#!/usr/bin/env python3
"""Tests for the serial link to the head, without a robot attached.

The wire format is the firmware's: [0xAA][0x55][LEN][MSG_ID][PAYLOAD][CRC8], with
LEN counting the id byte and the CRC covering everything from LEN onwards. Framing
and CRC come from the shared gaze core rather than being restated here; what is new
is reading a byte stream back, which the ROS bridge did inside a node.

Two things matter beyond parsing. The firmware disables the motors if it hears
nothing for 500 ms, so a link that stops sending silently stops the robot. And the
encoder angle is the one measurement that keeps every bearing honest — without it
the stack assumes the head is at zero and tracking collapses to centre.
"""

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401  (puts the shared core on the path)
from astro_base.gaze.head_controller import (
    MSG_ENCODER_TICKS,
    MSG_HEAD_CMD,
    MSG_HEARTBEAT,
    build_packet,
)

from head_link import (
    HEARTBEAT_INTERVAL_S,
    HeadLink,
    encode_head_cmd,
    encode_heartbeat,
    head_degrees_from_encoder_payload,
    parse_packets,
)


def _encoder_packet(head_ticks: int) -> bytes:
    payload = struct.pack("<iiiI", 0, 0, head_ticks, 20000)
    return build_packet(MSG_ENCODER_TICKS, payload)


class TestParsingWhatTheFirmwareSends(unittest.TestCase):
    def test_a_whole_packet_is_read_back(self):
        stream = _encoder_packet(259)

        self.assertEqual(
            [(mid, head_degrees_from_encoder_payload(pl)) for mid, pl in parse_packets(stream)],
            [(MSG_ENCODER_TICKS, 100.07)],
        )

    def test_several_packets_in_one_read_are_all_returned(self):
        stream = _encoder_packet(10) + _encoder_packet(20) + _encoder_packet(30)

        self.assertEqual(len(list(parse_packets(stream))), 3)

    def test_leading_noise_is_skipped_rather_than_derailing_the_stream(self):
        """Opening a serial port mid-transmission lands you in the middle of a packet."""
        stream = b"\x12\x34\xaa\x99" + _encoder_packet(42)

        self.assertEqual([mid for mid, _ in parse_packets(stream)], [MSG_ENCODER_TICKS])

    def test_a_corrupted_packet_is_dropped_not_believed(self):
        good = bytearray(_encoder_packet(42))
        good[-1] ^= 0xFF  # break the CRC

        self.assertEqual(list(parse_packets(bytes(good))), [])

    def test_a_truncated_tail_is_left_for_the_next_read(self):
        whole = _encoder_packet(42)
        stream = whole + whole[:5]

        packets, leftover = parse_packets(stream, return_remainder=True)

        self.assertEqual(len(packets), 1)
        self.assertEqual(leftover, whole[:5])


class TestWhatWeSend(unittest.TestCase):
    def test_a_head_command_carries_the_angle_as_a_float(self):
        packet = encode_head_cmd(-42.5)

        msg_id, payload = next(iter(parse_packets(packet)))
        self.assertEqual(msg_id, MSG_HEAD_CMD)
        self.assertAlmostEqual(struct.unpack("<f", payload)[0], -42.5, places=3)

    def test_a_heartbeat_carries_its_sequence_number(self):
        msg_id, payload = next(iter(parse_packets(encode_heartbeat(7))))

        self.assertEqual(msg_id, MSG_HEARTBEAT)
        self.assertEqual(struct.unpack("<I", payload)[0], 7)

    def test_the_heartbeat_interval_leaves_room_under_the_firmware_timeout(self):
        """The firmware cuts the motors after 500 ms of silence."""
        self.assertLess(HEARTBEAT_INTERVAL_S, 0.5)


class _FakePort:
    """Stands in for a serial.Serial: records writes, replays canned bytes."""

    def __init__(self, to_read: bytes = b""):
        self.written = bytearray()
        self._to_read = bytearray(to_read)
        self.closed = False

    @property
    def in_waiting(self):
        return len(self._to_read)

    def write(self, data):
        self.written.extend(data)
        return len(data)

    def read(self, n):
        chunk, self._to_read = self._to_read[:n], self._to_read[n:]
        return bytes(chunk)

    def close(self):
        self.closed = True


class TestHeadLink(unittest.TestCase):
    def test_the_commanded_angle_reaches_the_wire(self):
        link = HeadLink(port=_FakePort())

        link.send_angle(30.0)

        msg_ids = [mid for mid, _ in parse_packets(bytes(link.port.written))]
        self.assertIn(MSG_HEAD_CMD, msg_ids)

    def test_the_encoder_reading_becomes_the_measured_angle(self):
        link = HeadLink(port=_FakePort(_encoder_packet(259)))

        link.poll()

        self.assertAlmostEqual(link.measured_angle_deg, 100.07, places=2)
        self.assertTrue(link.has_feedback)

    def test_a_link_that_has_heard_nothing_admits_it(self):
        """Silence must not read as 'the head is at zero degrees'."""
        link = HeadLink(port=_FakePort())

        link.poll()

        self.assertFalse(link.has_feedback)

    def test_a_heartbeat_goes_out_when_the_interval_has_passed(self):
        link = HeadLink(port=_FakePort())

        link.tick(now=0.0)
        link.tick(now=HEARTBEAT_INTERVAL_S + 0.01)

        beats = [mid for mid, _ in parse_packets(bytes(link.port.written)) if mid == MSG_HEARTBEAT]
        self.assertEqual(len(beats), 2)

    def test_a_heartbeat_is_not_resent_on_every_loop(self):
        link = HeadLink(port=_FakePort())

        for _ in range(50):
            link.tick(now=0.0)

        beats = [mid for mid, _ in parse_packets(bytes(link.port.written)) if mid == MSG_HEARTBEAT]
        self.assertEqual(len(beats), 1)


class TestRunningWithoutTheRobot(unittest.TestCase):
    """The program has to be usable at a desk with no Arduino plugged in."""

    def test_no_port_means_no_crash(self):
        link = HeadLink(port=None)

        link.send_angle(20.0)
        link.tick(now=0.0)
        link.poll()

        self.assertFalse(link.connected)
        self.assertFalse(link.has_feedback)


if __name__ == "__main__":
    unittest.main()
