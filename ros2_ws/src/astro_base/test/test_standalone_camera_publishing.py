#!/usr/bin/env python3
"""Test Suite for Camera Image Publishing in StandaloneGazeRosNode.

Verifies:
1. Publisher initialization on /oak/rgb/image_raw with qos_profile_sensor_data.
2. Publishing of valid Image messages during step_camera_frame.
3. Rate limiting according to camera_publish_fps (15 FPS default).
4. Full round-trip compatibility with astro_realtime_node's imgmsg_to_bgr decoder.
5. Clean disable behavior when publish_camera_image is False.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest

# Resolve paths
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[3]
STANDALONE_DIR = REPO_ROOT / "standalone"
ASTRO_BASE_DIR = CUR_DIR.parents[1]
ASTRO_AI_DIR = REPO_ROOT / "ros2_ws" / "src" / "astro_ai"

for path_str in (str(REPO_ROOT), str(STANDALONE_DIR), str(ASTRO_BASE_DIR), str(ASTRO_AI_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from astro_base.standalone_gaze_ros_node import StandaloneGazeRosNode, bgr_to_imgmsg
from astro_ai.astro_realtime_node import imgmsg_to_bgr


class TestStandaloneCameraPublishing:
    """Validates camera frame publishing from StandaloneGazeRosNode to ROS topics."""

    def test_camera_publisher_initialized_by_default(self):
        """StandaloneGazeRosNode should create a publisher on /oak/rgb/image_raw."""
        node = StandaloneGazeRosNode(use_camera_source=False, enable_audio=False)
        assert node.pub_camera_image is not None
        assert node.camera_image_topic == "/oak/rgb/image_raw"
        assert node.publish_camera_image is True
        assert node.camera_publish_fps == 15.0

    def test_camera_publisher_disabled_flag(self):
        """When publish_camera_image parameter is False, no publisher should be created."""
        node = StandaloneGazeRosNode(use_camera_source=False, enable_audio=False)
        node.declare_parameter("publish_camera_image", False)
        node_disabled = StandaloneGazeRosNode(use_camera_source=False, enable_audio=False)
        node_disabled.publish_camera_image = False
        node_disabled.pub_camera_image = None
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        node_disabled.step_camera_frame(dummy_frame)
        assert node_disabled.pub_camera_image is None

    def test_step_camera_frame_publishes_valid_image(self):
        """Stepping camera frame should publish an Image with bgr8 encoding and valid shape."""
        node = StandaloneGazeRosNode(use_camera_source=False, enable_audio=False)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        node.step_camera_frame(test_frame)

        assert node.pub_camera_image.last_msg is not None
        pub_msg = node.pub_camera_image.last_msg
        assert pub_msg.height == 480
        assert pub_msg.width == 640
        assert pub_msg.encoding == "bgr8"
        assert pub_msg.step == 640 * 3
        assert len(pub_msg.data) == 480 * 640 * 3

    def test_imgmsg_to_bgr_decoder_roundtrip(self):
        """astro_realtime_node.imgmsg_to_bgr must perfectly reconstruct the original frame."""
        node = StandaloneGazeRosNode(use_camera_source=False, enable_audio=False)
        original_frame = np.full((120, 160, 3), fill_value=42, dtype=np.uint8)

        node.step_camera_frame(original_frame)
        pub_msg = node.pub_camera_image.last_msg

        reconstructed_frame = imgmsg_to_bgr(pub_msg)
        assert reconstructed_frame is not None
        assert reconstructed_frame.shape == (120, 160, 3)
        assert np.array_equal(original_frame, reconstructed_frame)

    def test_camera_publishing_rate_limiting(self):
        """Frames provided faster than 1/camera_publish_fps should be throttled."""
        node = StandaloneGazeRosNode(use_camera_source=False, enable_audio=False)
        node.camera_publish_fps = 10.0
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        node.step_camera_frame(dummy_frame)
        initial_count = node.pub_camera_image.count
        assert initial_count == 1

        node.step_camera_frame(dummy_frame)
        assert node.pub_camera_image.count == initial_count

        node._last_camera_pub_time = time.monotonic() - 0.15
        node.step_camera_frame(dummy_frame)
        assert node.pub_camera_image.count == initial_count + 1
