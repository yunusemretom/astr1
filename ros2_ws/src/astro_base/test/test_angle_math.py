"""Unit tests for centralized circular angle mathematics."""

import math
import unittest
import numpy as np

from astro_base.gaze.angle_math import (
    wrap_deg,
    wrap_rad,
    angular_diff_deg,
    circular_distance_deg,
    circular_mean_deg,
    shortest_reachable_arc,
    clamp_deg,
)


class TestAngleMath(unittest.TestCase):
    def test_wrap_deg(self):
        # Basic angles
        self.assertAlmostEqual(wrap_deg(0.0), 0.0)
        self.assertAlmostEqual(wrap_deg(45.0), 45.0)
        self.assertAlmostEqual(wrap_deg(-45.0), -45.0)
        self.assertAlmostEqual(wrap_deg(180.0), 180.0)
        self.assertAlmostEqual(wrap_deg(-180.0), 180.0)

        # Multi-revolution wraps
        self.assertAlmostEqual(wrap_deg(181.0), -179.0)
        self.assertAlmostEqual(wrap_deg(-181.0), 179.0)
        self.assertAlmostEqual(wrap_deg(360.0), 0.0)
        self.assertAlmostEqual(wrap_deg(-360.0), 0.0)
        self.assertAlmostEqual(wrap_deg(540.0), 180.0)
        self.assertAlmostEqual(wrap_deg(-540.0), 180.0)
        self.assertAlmostEqual(wrap_deg(720.0), 0.0)

    def test_angular_diff_deg(self):
        # Direct differences
        self.assertAlmostEqual(angular_diff_deg(10.0, 0.0), 10.0)
        self.assertAlmostEqual(angular_diff_deg(-10.0, 0.0), -10.0)
        self.assertAlmostEqual(angular_diff_deg(0.0, 10.0), -10.0)
        self.assertAlmostEqual(angular_diff_deg(0.0, -10.0), 10.0)

        # Circular wrap difference (seam at ±180°)
        # Moving from +179° to -179° is a +2° turn (counter-clockwise)
        self.assertAlmostEqual(angular_diff_deg(-179.0, 179.0), 2.0)
        # Moving from -179° to +179° is a -2° turn (clockwise)
        self.assertAlmostEqual(angular_diff_deg(179.0, -179.0), -2.0)

        # Opposite extremes
        self.assertAlmostEqual(circular_distance_deg(179.0, -179.0), 2.0)
        self.assertAlmostEqual(circular_distance_deg(0.0, 90.0), 90.0)
        self.assertAlmostEqual(circular_distance_deg(90.0, -90.0), 180.0)

    def test_circular_mean_deg(self):
        # Simple symmetric mean
        self.assertAlmostEqual(circular_mean_deg([10.0, 20.0, 30.0]), 20.0, places=4)
        self.assertAlmostEqual(circular_mean_deg([-10.0, 0.0, 10.0]), 0.0, places=4)

        # Wrap around seam (+170° and -170° should average to 180°, NOT 0°)
        avg = circular_mean_deg([170.0, -170.0])
        self.assertIsNotNone(avg)
        self.assertAlmostEqual(avg, 180.0, places=4)

        # Weighted circular mean
        weighted_avg = circular_mean_deg([0.0, 90.0], weights=[3.0, 1.0])
        self.assertIsNotNone(weighted_avg)
        # Expected: atan2(1*sin(90), 3*cos(0)) = atan2(1, 3) = ~18.43°
        self.assertAlmostEqual(weighted_avg, math.degrees(math.atan2(1.0, 3.0)), places=4)

        # Opposing cancellation returns None
        self.assertIsNone(circular_mean_deg([0.0, 180.0]))
        self.assertIsNone(circular_mean_deg([]))

    def test_shortest_reachable_arc(self):
        # 1. Full circle travel
        delta = shortest_reachable_arc(target_deg=-179.0, current_deg=179.0, min_limit_deg=-180.0, max_limit_deg=180.0)
        self.assertAlmostEqual(delta, 2.0)

        # 2. Constrained limits [-90°, +90°]
        # Direct reachable movement
        delta = shortest_reachable_arc(target_deg=45.0, current_deg=0.0, min_limit_deg=-90.0, max_limit_deg=90.0)
        self.assertAlmostEqual(delta, 45.0)

        # Target out of reach (e.g. 120°): clamped to +90°
        delta = shortest_reachable_arc(target_deg=120.0, current_deg=0.0, min_limit_deg=-90.0, max_limit_deg=90.0)
        self.assertAlmostEqual(delta, 90.0)

        # Target -120°: clamped to -90°
        delta = shortest_reachable_arc(target_deg=-120.0, current_deg=0.0, min_limit_deg=-90.0, max_limit_deg=90.0)
        self.assertAlmostEqual(delta, -90.0)

    def test_clamp_deg(self):
        self.assertEqual(clamp_deg(50.0, -90.0, 90.0), 50.0)
        self.assertEqual(clamp_deg(100.0, -90.0, 90.0), 90.0)
        self.assertEqual(clamp_deg(-100.0, -90.0, 90.0), -90.0)


if __name__ == "__main__":
    unittest.main()
