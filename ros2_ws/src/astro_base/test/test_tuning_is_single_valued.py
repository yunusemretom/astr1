#!/usr/bin/env python3
"""Every tuning value must be defined exactly once.

A tuning pass appended new values beside the old ones instead of replacing them:
HEAD_DEADBAND_TICKS and HEAD_MAX_VEL_DEG_S each ended up declared twice in the
firmware, and four planner keys twice in the YAML. C++ refuses to compile a
redefinition, and YAML silently keeps the last of a duplicated key — so the tuning
either could not be flashed or took effect through a rule nobody intended.

The result was a slowdown that was never actually running while being reasoned
about as if it were. These tests make that failure loud.
"""

import os
import re
import unittest

import yaml


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

FIRMWARE = [
    os.path.join(REPO, "arduino", "astro_firmware", "src", "main.cpp"),
    os.path.join(REPO, "arduino", "AstroFirmware", "AstroFirmware.ino"),
]
CONFIGS = [
    os.path.join(REPO, "ros2_ws", "src", "astro_base", "config", "social_gaze_params.yaml"),
    os.path.join(REPO, "ros2_ws", "src", "astro_bringup", "config", "astro_params.yaml"),
    os.path.join(REPO, "ros2_ws", "src", "astro_base", "config", "calibration_params.yaml"),
    os.path.join(REPO, "ros2_ws", "src", "astro_vision", "config", "camera_params.yaml"),
]

_CONSTANT = re.compile(
    r"^\s*static\s+constexpr\s+[\w:]+\s+([A-Z_][A-Z0-9_]*)\s*=", re.MULTILINE
)


class _DuplicateKeyLoader(yaml.SafeLoader):
    """PyYAML keeps the last of a duplicated key; this refuses instead."""


def _no_duplicate_keys(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate key: {key}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


class TestFirmwareConstantsAreDefinedOnce(unittest.TestCase):
    def test_no_constant_is_declared_twice(self):
        for path in FIRMWARE:
            if not os.path.exists(path):
                continue
            with self.subTest(firmware=os.path.basename(path)):
                names = _CONSTANT.findall(open(path, encoding="utf-8").read())
                dupes = sorted({n for n in names if names.count(n) > 1})

                self.assertEqual(dupes, [], f"{path} redefines: {dupes}")


class TestConfigKeysAreDefinedOnce(unittest.TestCase):
    def test_no_yaml_key_is_declared_twice(self):
        for path in CONFIGS:
            if not os.path.exists(path):
                continue
            with self.subTest(config=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    yaml.load(handle, Loader=_DuplicateKeyLoader)


class TestHeadTuningMatchesTheMechanism(unittest.TestCase):
    """The firmware deadband has to clear the gearbox backlash.

    docs/final_validation_report.md measures 0.85 degrees of backlash, and one
    encoder tick is 1/2.5882 = 0.386 degrees. A deadband below the backlash asks the
    controller to close an error the mechanism physically cannot resolve: it drives,
    the output does not follow, the error stays, and it drives again — the hunting
    the head was doing while trying to hold a face.
    """

    BACKLASH_DEG = 0.85

    def _constants(self) -> dict:
        text = open(FIRMWARE[0], encoding="utf-8").read()
        found = {}
        for name in ("HEAD_TICKS_PER_DEG", "HEAD_DEADBAND_TICKS", "HEAD_MAX_VEL_DEG_S"):
            match = re.search(rf"{name}\s*=\s*([0-9.]+)f?", text)
            self.assertIsNotNone(match, f"{name} not found in firmware")
            found[name] = float(match.group(1))
        return found

    def test_the_deadband_covers_the_measured_backlash(self):
        c = self._constants()
        deadband_deg = c["HEAD_DEADBAND_TICKS"] / c["HEAD_TICKS_PER_DEG"]

        self.assertGreaterEqual(deadband_deg, self.BACKLASH_DEG)

    def test_the_deadband_stays_small_enough_to_look_like_eye_contact(self):
        """Parking a degree or two off target is invisible at conversational range;
        much more than that and the robot is plainly looking past the person."""
        c = self._constants()
        deadband_deg = c["HEAD_DEADBAND_TICKS"] / c["HEAD_TICKS_PER_DEG"]

        self.assertLessEqual(deadband_deg, 2.0)

    def test_the_head_speed_is_the_calm_tracking_speed(self):
        self.assertEqual(self._constants()["HEAD_MAX_VEL_DEG_S"], 20.0)


if __name__ == "__main__":
    unittest.main()
