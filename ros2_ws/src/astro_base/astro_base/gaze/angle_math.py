"""Centralized Circular Angular Mathematics Utilities.

Provides mathematically verified circular operations to eliminate angle-wrapping bugs,
boundary discontinuities at ±180°, and ensure consistent coordinate geometry.
"""

import math
from typing import List, Optional, Sequence, Union


def wrap_deg(angle_deg: float) -> float:
    """Folds any real angle in degrees into the standard interval (-180.0, +180.0].

    Examples:
        wrap_deg(0.0) -> 0.0
        wrap_deg(180.0) -> 180.0
        wrap_deg(-180.0) -> 180.0
        wrap_deg(181.0) -> -179.0
        wrap_deg(-181.0) -> 179.0
        wrap_deg(360.0) -> 0.0
        wrap_deg(540.0) -> 180.0
    """
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    # Floating point precision correction for -180.0 edge case
    if wrapped == -180.0:
        return 180.0
    return float(wrapped)


def wrap_rad(angle_rad: float) -> float:
    """Folds any real angle in radians into the standard interval (-pi, +pi]."""
    wrapped = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return float(wrapped)


def angular_diff_deg(target_deg: float, current_deg: float) -> float:
    """Computes the minimal signed circular rotation from current_deg to target_deg.

    Result is strictly in (-180.0, +180.0]:
      - Positive: counter-clockwise (turn left in robot frame)
      - Negative: clockwise (turn right in robot frame)

    Examples:
        angular_diff_deg(10.0, 0.0) -> +10.0
        angular_diff_deg(-10.0, 0.0) -> -10.0
        angular_diff_deg(-179.0, 179.0) -> +2.0
        angular_diff_deg(179.0, -179.0) -> -2.0
    """
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    if diff == -180.0:
        return 180.0
    return float(diff)


def circular_distance_deg(a_deg: float, b_deg: float) -> float:
    """Computes the absolute shortest angular distance between two angles in [0.0, 180.0]."""
    return float(abs(angular_diff_deg(a_deg, b_deg)))


def circular_mean_deg(
    angles_deg: Sequence[float],
    weights: Optional[Sequence[float]] = None
) -> Optional[float]:
    """Computes the weighted trigonometric circular mean of a collection of angles.

    Uses vector addition:
      sin_sum = sum(w_i * sin(theta_i))
      cos_sum = sum(w_i * cos(theta_i))
      mean = atan2(sin_sum, cos_sum)

    Returns:
      Mean angle in (-180.0, +180.0], or None if sequence is empty or vectors cancel out.
    """
    if not angles_deg:
        return None

    if weights is None:
        weights = [1.0] * len(angles_deg)
    elif len(weights) != len(angles_deg):
        raise ValueError("angles_deg and weights must have identical length")

    sin_sum = 0.0
    cos_sum = 0.0
    total_w = 0.0

    for a, w in zip(angles_deg, weights):
        if w < 0.0:
            continue
        rad = math.radians(a)
        sin_sum += w * math.sin(rad)
        cos_sum += w * math.cos(rad)
        total_w += w

    if total_w <= 0.0:
        return None

    # Check for near-zero vector magnitude (perfect opposing cancellation)
    mag = math.hypot(sin_sum, cos_sum)
    if mag < 1e-6:
        return None

    avg_rad = math.atan2(sin_sum, cos_sum)
    return wrap_deg(math.degrees(avg_rad))


def shortest_reachable_arc(
    target_deg: float,
    current_deg: float,
    min_limit_deg: float = -90.0,
    max_limit_deg: float = 90.0
) -> float:
    """Calculates the signed rotation to reach target_deg respecting mechanical joint limits.

    If travel is limited (e.g. [-90°, +90°]), a direct wrap around the rear might breach
    physical stops. This function finds the shortest rotation whose trajectory stays
    completely within [min_limit_deg, max_limit_deg].

    Returns:
      Signed delta in degrees such that (current_deg + delta) is reachable within limits.
    """
    # Full circle continuous travel check
    is_full_circle = (max_limit_deg - min_limit_deg) >= 359.9
    short = angular_diff_deg(target_deg, current_deg)

    if is_full_circle:
        return short

    # 1. Test standard short arc
    candidate_1 = current_deg + short
    if min_limit_deg <= candidate_1 <= max_limit_deg:
        return short

    # 2. Test complementary long arc
    long_way = short - math.copysign(360.0, short) if short != 0.0 else 0.0
    candidate_2 = current_deg + long_way
    if min_limit_deg <= candidate_2 <= max_limit_deg:
        return long_way

    # 3. Neither arc fits: clamp target to closest mechanical limit
    clamped_target = max(min_limit_deg, min(max_limit_deg, candidate_1))
    return float(clamped_target - current_deg)


def clamp_deg(val: float, min_val: float, max_val: float) -> float:
    """Clamps a floating point angle value between min_val and max_val."""
    return float(max(min_val, min(max_val, val)))
