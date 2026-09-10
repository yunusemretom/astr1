#!/usr/bin/env python3
"""ASTRO — Standalone Gaze Node (Alias to StandaloneGazeRosNode).

Directly wraps the golden 2e0b70c standalone GazeTracker.
"""

from astro_base.standalone_gaze_ros_node import (
    StandaloneGazeRosNode,
    main,
)

# Backwards-compatible class name
StandaloneGazeNode = StandaloneGazeRosNode

if __name__ == "__main__":
    main()
