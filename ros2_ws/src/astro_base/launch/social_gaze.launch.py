"""ROS 2 Launch file for ASTRO Social Gaze & Hardware System."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('astro_base')
    default_social_config = os.path.join(pkg_share, 'config', 'social_gaze_params.yaml')
    default_calib_config = os.path.join(pkg_share, 'config', 'calibration_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_audio_stream',
            default_value='true',
            description='ReSpeaker DOA/VAD yayıncısını başlat; zaten çalışıyorsa false',
        ),
        DeclareLaunchArgument(
            'audio_doa_profile',
            default_value='respeaker_eye_20260908',
            description='8 Eylül göz montajı sektörleri veya geometric (0=ön)',
        ),
        DeclareLaunchArgument(
            'launch_serial_bridge',
            default_value='true',
            description='Launch serial bridge driver to Arduino Mega hardware',
        ),
        DeclareLaunchArgument(
            'social_config_file',
            default_value=default_social_config,
            description='Path to social gaze parameter yaml file',
        ),
        DeclareLaunchArgument(
            'calib_config_file',
            default_value=default_calib_config,
            description='Path to unified calibration yaml file',
        ),

        # Serial Hardware Bridge Node
        Node(
            package='astro_base',
            executable='serial_bridge',
            name='serial_bridge',
            output='screen',
            condition=IfCondition(LaunchConfiguration('launch_serial_bridge')),
            parameters=[{
                'port': '/dev/ttyCH341USB0',
                'baud': 115200,
            }],
        ),

        # Authoritative Standalone Gaze & Hardware Pipeline Node
        Node(
            package='astro_audio',
            executable='audio_stream_node',
            name='audio_stream_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('launch_audio_stream')),
        ),
        Node(
            package='astro_base',
            executable='standalone_gaze_ros',
            name='standalone_gaze_ros_node',
            output='screen',
            parameters=[
                LaunchConfiguration('social_config_file'),
                {'calibration_path': LaunchConfiguration('calib_config_file')},
                {'audio_source_mode': 'topics',
                 'audio_doa_profile': LaunchConfiguration('audio_doa_profile'),
                 'enable_audio': True, 'enable_voice': False},
            ],
        ),
    ])
