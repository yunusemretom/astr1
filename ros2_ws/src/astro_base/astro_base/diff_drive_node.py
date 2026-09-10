#!/usr/bin/env python3
"""ASTRO V1 — Standalone Differential Drive Kinematics Node.

Subscribes to:
  - /cmd_vel (geometry_msgs/msg/Twist)

Publishes to:
  - /wheel_cmds (astro_base/msg/WheelCmd)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from astro_base.msg import WheelCmd


class DiffDriveNode(Node):
    def __init__(self):
        super().__init__("diff_drive_node")

        self.declare_parameter("wheel_radius", 0.06)
        self.declare_parameter("wheel_separation", 0.26)
        self.declare_parameter("max_rpm", 150.0)

        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.wheel_separation = float(self.get_parameter("wheel_separation").value)
        self.max_rpm = float(self.get_parameter("max_rpm").value)

        self.sub_cmd_vel = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )
        self.pub_wheel_cmds = self.create_publisher(
            WheelCmd, "/wheel_cmds", 10
        )

        self.get_logger().info(
            f"🚗 [DiffDriveNode] Ready. wheel_radius={self.wheel_radius}m, "
            f"wheel_separation={self.wheel_separation}m, max_rpm={self.max_rpm}"
        )

    def cmd_vel_callback(self, msg: Twist):
        v = float(msg.linear.x)
        w = float(msg.angular.z)

        # Differential drive inverse kinematics
        v_left = v - (w * self.wheel_separation / 2.0)
        v_right = v + (w * self.wheel_separation / 2.0)

        # Convert linear wheel velocities (m/s) to RPM
        left_rpm = (v_left / self.wheel_radius) * (60.0 / (2.0 * math.pi))
        right_rpm = (v_right / self.wheel_radius) * (60.0 / (2.0 * math.pi))

        # Clamp to max_rpm
        if self.max_rpm > 0.0:
            left_rpm = max(-self.max_rpm, min(self.max_rpm, left_rpm))
            right_rpm = max(-self.max_rpm, min(self.max_rpm, right_rpm))

        cmd = WheelCmd()
        cmd.left_rpm = float(left_rpm)
        cmd.right_rpm = float(right_rpm)
        self.pub_wheel_cmds.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DiffDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
