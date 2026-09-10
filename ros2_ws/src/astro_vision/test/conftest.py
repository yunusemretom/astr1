"""Test fixtures for astro_vision.

The vision nodes carry a mock Node shim that activates when rclpy cannot be
imported, so their logic can be unit tested without a ROS graph. On a machine with
/opt/ros/humble sourced rclpy *is* importable, so the shim never activates and node
construction fails with NotInitializedException.

Blocking the import selects the shim path deterministically, and the block is lifted
as soon as collection ends so the other packages' tests are unaffected. This mirrors
ros2_ws/src/astro_base/test/conftest.py.
"""

import sys

_BLOCKED = ("rclpy", "rclpy.node", "rclpy.qos")
_saved = {name: sys.modules.get(name) for name in _BLOCKED}

for _name in _BLOCKED:
    sys.modules[_name] = None  # type: ignore[assignment]


def pytest_collection_finish(session):
    """Restore rclpy once every astro_vision test module has been imported."""
    for name, module in _saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
