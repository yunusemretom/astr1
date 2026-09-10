"""Makes the shared gaze core importable from this standalone program.

The whole point of this folder is to run the *same* brain without ROS, not a second
copy of it: two hand-maintained implementations of the same behaviour drift, and
this repository has already paid for that once — the Arduino IDE firmware copy
diverged in 147 of 535 lines and would apply every angle 1.73x too large.

So astro_base/gaze, the astro_vision helpers and the astro_ai conversation
pieces are imported from where they live.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGES = (
    os.path.join(REPO, "ros2_ws", "src", "astro_base"),
    os.path.join(REPO, "ros2_ws", "src", "astro_vision"),
    os.path.join(REPO, "ros2_ws", "src", "astro_audio"),
    os.path.join(REPO, "ros2_ws", "src", "astro_ai"),
)

for _path in _PACKAGES:
    if _path not in sys.path:
        sys.path.insert(0, _path)
