"""Image conversion without cv_bridge (avoids NumPy 2.x ABI issues on Jetson)."""
import array

import numpy as np
try:
    from sensor_msgs.msg import Image
except ImportError:
    class Image:
        pass


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding == "bgr8":
        channels = 3
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        return data.copy()
    if msg.encoding == "rgb8":
        import cv2

        channels = 3
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if msg.encoding in ("mono8", "8UC1"):
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        import cv2

        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def bgr_to_imgmsg(frame: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(frame.shape[1] * 3)
    # array.array('B') is the one type rclpy's uint8[] setter accepts without
    # inspecting it. Anything else — bytes included — is walked twice in pure Python
    # (`all(isinstance(v, int) ...)` then `all(0 <= v < 256 ...)`) before being
    # converted to exactly this, which cost 45 ms per 640x480 frame and throttled
    # every vision node to a few Hz. The stored payload is identical either way.
    msg.data = array.array("B", frame.tobytes())
    return msg
