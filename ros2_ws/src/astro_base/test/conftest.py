"""Test fixtures for astro_base.

head_tracker_node / serial_bridge each carry a mock Node shim that activates when
rclpy cannot be imported, so their logic can be unit tested without a ROS graph.
On a machine with /opt/ros/humble sourced rclpy *is* importable, so the shim never
activates and every node construction fails — first with NotInitializedException,
then (after rclpy.init) with _TYPE_SUPPORT errors, because the custom astro_base
messages (HeadCmd, WheelCmd) only exist in a built workspace.

Blocking the import selects the shim path deterministically. The block is lifted as
soon as collection ends: astro_ai and astro_audio tests import rclpy lazily inside
setUp, and leaving it blocked for the whole session breaks them.
"""

import sys

_BLOCKED = ("rclpy", "rclpy.node", "rclpy.qos")
_saved = {name: sys.modules.get(name) for name in _BLOCKED}

for _name in _BLOCKED:
    sys.modules[_name] = None  # type: ignore[assignment]


def pytest_collection_finish(session):
    """Restore rclpy once every astro_base test module has been imported."""
    for name, module in _saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
