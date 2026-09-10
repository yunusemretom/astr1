#!/usr/bin/env python3
"""Canonical GazeTracker re-exported directly from standalone.tracker (Golden Reference 2e0b70c).

This module provides the shared gaze tracker by importing directly from
standalone/tracker.py, ensuring that ROS and standalone share the EXACT
same single source of truth without code duplication.
"""

import os
import sys

from pathlib import Path


def _resolve_standalone_dir() -> str:
    cur = Path(__file__).resolve().parent
    while cur.parent != cur:
        cand = cur / "standalone"
        if (cand / "tracker.py").exists():
            return str(cand)
        cur = cur.parent
    return str(Path(__file__).resolve().parents[5] / "standalone")


_STANDALONE_DIR = _resolve_standalone_dir()
if _STANDALONE_DIR not in sys.path:
    sys.path.insert(0, _STANDALONE_DIR)

from tracker import (
    DEFAULT_CALIBRATION_PATH,
    UNSCORED_CONFIDENCE,
    Detection,
    GazeResult,
    GazeTracker,
    _load_calibration,
)

__all__ = [
    "Detection",
    "GazeResult",
    "GazeTracker",
    "UNSCORED_CONFIDENCE",
    "DEFAULT_CALIBRATION_PATH",
    "_load_calibration",
]
