import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("astro_bringup")
    params_file = os.path.join(pkg_dir, "config", "astro_params.yaml")
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_head_tracker = LaunchConfiguration("enable_head_tracker")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock",
            ),
            DeclareLaunchArgument(
                "enable_head_tracker",
                default_value="true",
                description="Start social sound-localization head tracking node",
            ),
            Node(
                package="astro_base",
                executable="serial_bridge",
                name="serial_bridge",
                output="screen",
                parameters=[params_file, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="astro_base",
                executable="social_gaze",
                name="social_gaze_node",
                output="screen",
                condition=IfCondition(enable_head_tracker),
                parameters=[params_file, {"use_sim_time": use_sim_time}],
            ),
        ]
    )
