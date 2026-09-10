#!/usr/bin/env python3
"""Unit tests for deterministic ReSpeaker ALSA device resolution and validation.

Verifies:
1. ReSpeaker resolution strictly binds to ArrayUAC10 (6 channels, 16000 Hz, S16_LE).
2. Jetson Orin Nano onboard APE (tegra-ape, 16 channels) is strictly rejected and never selected.
3. Parsing of /proc/asound/cards identifies the ArrayUAC10 card index accurately.
4. Validation fails fast when required channels (>=6) or format are violated.
5. Forbidden device names/hints are rejected even if they offer multi-channel input.
"""

import os
import sys
import unittest
from unittest.mock import mock_open, patch

# Ensure repository paths
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", ".."))
scripts_dir = os.path.join(root_dir, "scripts")

for p in (root_dir, scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from respeaker_device import (
    FORBIDDEN_NAME_HINTS,
    REQUIRED_CHANNELS,
    REQUIRED_MIC_CHANNELS,
    REQUIRED_SAMPLE_FORMAT,
    REQUIRED_SAMPLE_RATE,
    RESPEAKER_ALSA_DEVICE,
    RESPEAKER_CARD_ID,
    RespeakerDeviceInfo,
    parse_asound_cards,
    resolve_respeaker_capture_device,
    resolve_respeaker_from_devices,
    validate_respeaker_device,
)


MOCK_JETSON_ASOUND_CARDS = """\
 0 [HDA            ]: tegra-hda - NVIDIA Jetson Orin Nano HDA
                      tegra-hda at 0x3518000 irq 77
 1 [APE            ]: tegra-ape - NVIDIA Jetson Orin Nano APE
                      tegra-ape
 2 [ArrayUAC10     ]: USB-Audio - ReSpeaker 4 Mic Array (UAC1.0)
                      Seeed ReSpeaker 4 Mic Array (UAC1.0) at usb-3610000.xhci-2.3, full speed
"""

MOCK_JETSON_NO_RESPEAKER_CARDS = """\
 0 [HDA            ]: tegra-hda - NVIDIA Jetson Orin Nano HDA
                      tegra-hda at 0x3518000 irq 77
 1 [APE            ]: tegra-ape - NVIDIA Jetson Orin Nano APE
                      tegra-ape
"""

MOCK_SOUNDDEVICE_LIST = [
    {
        "name": "NVIDIA Jetson Orin Nano HDA: tegra-hda (hw:0,0)",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "name": "NVIDIA Jetson Orin Nano APE: tegra-ape (hw:1,0)",
        "hostapi": 0,
        "max_input_channels": 16,
        "max_output_channels": 16,
        "default_samplerate": 48000.0,
    },
    {
        "name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)",
        "hostapi": 0,
        "max_input_channels": 6,
        "max_output_channels": 0,
        "default_samplerate": 16000.0,
    },
    {
        "name": "pulse",
        "hostapi": 0,
        "max_input_channels": 32,
        "max_output_channels": 32,
        "default_samplerate": 44100.0,
    },
]


class TestRespeakerDeviceResolution(unittest.TestCase):
    """Test suite for ReSpeaker ALSA device detection and rejection logic."""

    def test_canonical_device_constants(self):
        """Verify the canonical hardware specification requirements."""
        self.assertEqual(RESPEAKER_CARD_ID, "ArrayUAC10")
        self.assertEqual(RESPEAKER_ALSA_DEVICE, "hw:CARD=ArrayUAC10,DEV=0")
        self.assertEqual(REQUIRED_SAMPLE_RATE, 16000)
        self.assertEqual(REQUIRED_CHANNELS, 6)
        self.assertEqual(REQUIRED_SAMPLE_FORMAT, "S16_LE")
        self.assertEqual(REQUIRED_MIC_CHANNELS, (1, 2, 3, 4))

    def test_parse_asound_cards_success(self):
        """Verify parsing of /proc/asound/cards when ReSpeaker is present."""
        card_map = parse_asound_cards(MOCK_JETSON_ASOUND_CARDS)
        self.assertIn(RESPEAKER_CARD_ID, card_map)
        self.assertEqual(card_map[RESPEAKER_CARD_ID], 2)

    def test_parse_asound_cards_missing(self):
        """Verify parsing returns empty/missing when ArrayUAC10 is absent from /proc/asound/cards."""
        card_map = parse_asound_cards(MOCK_JETSON_NO_RESPEAKER_CARDS)
        self.assertNotIn(RESPEAKER_CARD_ID, card_map)
        self.assertIsNone(card_map.get(RESPEAKER_CARD_ID))

    def test_parse_asound_cards_empty(self):
        """Verify parsing handles empty cards string gracefully."""
        card_map = parse_asound_cards("")
        self.assertEqual(card_map, {})

    def test_resolve_respeaker_from_device_list(self):
        """Verify resolution selects ReSpeaker (card 2, 6ch) over Jetson APE (card 1, 16ch)."""
        info = resolve_respeaker_from_devices(MOCK_SOUNDDEVICE_LIST, asound_cards_text=MOCK_JETSON_ASOUND_CARDS)
        self.assertIsNotNone(info)
        self.assertTrue(info.is_valid_respeaker)
        self.assertEqual(info.device_index, 2)
        self.assertEqual(info.card_id, "ArrayUAC10")
        self.assertEqual(info.channels, 6)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.alsa_device_string, "hw:CARD=ArrayUAC10,DEV=0")
        self.assertEqual(info.mic_indices, (1, 2, 3, 4))

    def test_strict_rejection_of_jetson_ape(self):
        """CRITICAL: Jetson APE (16 input channels) must NEVER be selected even if it has >= 6 channels."""
        ape_only_list = [
            {
                "name": "NVIDIA Jetson Orin Nano APE: tegra-ape (hw:1,0)",
                "hostapi": 0,
                "max_input_channels": 16,
                "max_output_channels": 16,
                "default_samplerate": 48000.0,
            },
            {
                "name": "tegra-hda",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]
        info = resolve_respeaker_from_devices(ape_only_list, asound_cards_text=None)
        self.assertFalse(info.is_valid_respeaker, "Jetson APE must be rejected and must not be selected as ReSpeaker!")
        self.assertIn("not detected", info.diagnostic_notes.lower())

    def test_forbidden_hints_rejection(self):
        """Verify any device matching FORBIDDEN_NAME_HINTS is rejected."""
        for hint in FORBIDDEN_NAME_HINTS:
            dev_list = [
                {
                    "name": f"Fake Multi-Channel Device ({hint})",
                    "hostapi": 0,
                    "max_input_channels": 8,
                    "max_output_channels": 0,
                    "default_samplerate": 16000.0,
                }
            ]
            info = resolve_respeaker_from_devices(dev_list, asound_cards_text=None)
            self.assertFalse(info.is_valid_respeaker, f"Device matching forbidden hint '{hint}' was not rejected!")

    def test_insufficient_channels_rejection(self):
        """Verify a ReSpeaker candidate with fewer than REQUIRED_CHANNELS (6) is rejected."""
        two_channel_respeaker = [
            {
                "name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)",
                "hostapi": 0,
                "max_input_channels": 2,  # Downmixed / corrupt
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            }
        ]
        info = resolve_respeaker_from_devices(two_channel_respeaker, asound_cards_text=None)
        self.assertFalse(info.is_valid_respeaker, "ReSpeaker candidate with < 6 channels must be rejected!")

    def test_validate_respeaker_device_success(self):
        """Verify validate_respeaker_device returns True for compliant info."""
        info = RespeakerDeviceInfo(
            alsa_device_string="hw:CARD=ArrayUAC10,DEV=0",
            device_index=2,
            device_name="ReSpeaker 4 Mic Array (UAC1.0)",
            card_id="ArrayUAC10",
            sample_rate=16000,
            sample_format="S16_LE",
            channels=6,
            mic_indices=(1, 2, 3, 4),
            is_valid_respeaker=True,
        )
        self.assertTrue(validate_respeaker_device(info))

    def test_validate_respeaker_device_failure(self):
        """Verify validate_respeaker_device returns False for non-compliant info."""
        invalid_info = RespeakerDeviceInfo(
            alsa_device_string="hw:2,0",
            device_index=2,
            device_name="NVIDIA Jetson Orin Nano APE",
            card_id="APE",
            sample_rate=48000,
            sample_format="S16_LE",
            channels=16,
            mic_indices=(1, 2, 3, 4),
            is_valid_respeaker=False,
            diagnostic_notes="Not a ReSpeaker device: Jetson APE",
        )
        self.assertFalse(validate_respeaker_device(invalid_info))

    def test_simulate_fallback_in_resolve_capture_device(self):
        """Verify allow_simulation=True produces a synthetic compliant device when hardware absent."""
        info = resolve_respeaker_capture_device(allow_simulation=True)
        self.assertIsNotNone(info)
        self.assertTrue(info.is_valid_respeaker)
        self.assertEqual(info.channels, 6)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.sample_format, "S16_LE")
        self.assertEqual(info.mic_indices, (1, 2, 3, 4))
        self.assertEqual(info.card_id, "ArrayUAC10")


    def test_decouple_alsa_card_index_from_portaudio_index(self):
        """CRITICAL: ALSA card index 0 in /proc/asound/cards must NOT force sounddevice index 0.

        If sounddevice index 0 is Tegra HDA (maxChans=2) and sounddevice index 2
        is ReSpeaker ArrayUAC10 (maxChans=6), the resolver must choose PortAudio
        index 2, NEVER index 0 (which would trigger channelCount <= maxChans).
        """
        asound_cards_card0 = """\
 0 [ArrayUAC10     ]: USB-Audio - ReSpeaker 4 Mic Array (UAC1.0)
                      Seeed ReSpeaker 4 Mic Array (UAC1.0) at usb-3610000.xhci-2.3
 1 [HDA            ]: tegra-hda - NVIDIA Jetson Orin Nano HDA
"""
        sd_devices = [
            {
                "name": "tegra-hda: (hw:1,0)",
                "hostapi": 0,
                "max_input_channels": 2,  # Cannot handle 6 channels!
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "tegra-ape: (hw:2,0)",
                "hostapi": 0,
                "max_input_channels": 16,
                "max_output_channels": 16,
                "default_samplerate": 48000.0,
            },
            {
                "name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:0,0)",
                "hostapi": 0,
                "max_input_channels": 6,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            },
        ]
        info = resolve_respeaker_from_devices(sd_devices, asound_cards_text=asound_cards_card0)
        self.assertTrue(info.is_valid_respeaker)
        self.assertEqual(info.device_index, 2, "Must select PortAudio index 2 (6ch ReSpeaker), NOT index 0!")
        self.assertEqual(info.channels, 6)

    def test_portaudio_sysdefault_arrayuac10_accepted(self):
        """Verify PortAudio sysdefault device name for ArrayUAC10 is recognized and not forbidden."""
        sysdefault_devices = [
            {
                "name": "sysdefault:CARD=ArrayUAC10",
                "hostapi": 0,
                "max_input_channels": 6,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            }
        ]
        info = resolve_respeaker_from_devices(sysdefault_devices, asound_cards_text=None)
        self.assertTrue(info.is_valid_respeaker)
        self.assertEqual(info.device_index, 0)
        self.assertEqual(info.channels, 6)


if __name__ == "__main__":
    unittest.main()
