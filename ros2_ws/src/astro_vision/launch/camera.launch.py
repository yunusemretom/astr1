"""ASTRO V1 — kamera + görü düğümleri launch.

    ros2 launch astro_vision camera.launch.py                        # OAK-D sürücüsü
    ros2 launch astro_vision camera.launch.py source:=webcam         # USB webcam
    ros2 launch astro_vision camera.launch.py source:=none           # yalnızca görü düğümü
    ros2 launch astro_vision camera.launch.py use_native_spatial:=true   # OAK-D VPU üzerinde

Tüm kaynaklar aynı konuya (/oak/rgb/image_raw) yayınlar; görü düğümleri hangisinin
yayınladığını bilmek zorunda değildir.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap


def _source_is(value: str):
    return IfCondition(PythonExpression(["'", LaunchConfiguration("source"), f"' == '{value}'"]))


def _depthai_camera(params_file, use_sim_time):
    """OAK-D ROS sürücüsü — paket kurulu değilse launch'u komple düşürmemek için ayrı."""
    try:
        depthai_pkg = get_package_share_directory("depthai_ros_driver")
    except Exception:
        return LogInfo(
            msg=(
                "⚠️  depthai_ros_driver bulunamadı — OAK-D başlatılamıyor.\n"
                "    Kurmak için: sudo apt install ros-humble-depthai-ros\n"
                "    USB kamerayla denemek için: "
                "ros2 launch astro_vision camera.launch.py source:=webcam"
            )
        )

    return GroupAction(
        actions=[
            SetRemap("/camera/color/image_raw", "/oak/rgb/image_raw"),
            SetRemap("/camera/color/camera_info", "/oak/rgb/camera_info"),
            SetRemap("/camera/depth/image_raw", "/oak/depth/image_raw"),
            SetRemap("/camera/points", "/oak/points"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(depthai_pkg, "launch", "camera.launch.py")
                ),
                launch_arguments={
                    "params_file": params_file,
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
        ],
        # Yerleşik (on-chip) boru hattı seçiliyse ROS sürücüsü çalıştırılmaz.
        condition=UnlessCondition(LaunchConfiguration("use_native_spatial")),
    )


def generate_launch_description():
    pkg_dir = get_package_share_directory("astro_vision")
    params_file = os.path.join(pkg_dir, "config", "camera_params.yaml")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_native_spatial = LaunchConfiguration("use_native_spatial")

    return LaunchDescription([
        # Bir 640x480 bgr8 kare 900 KB; Linux'un varsayılan 208 KB'lık UDP alım
        # tamponu tek kareyi bile tutamıyor ve FastDDS parçaladığı mesajın çoğunu
        # düşürüyordu (yayıncı 30 Hz, dedektör 5.8 Hz — %70 kayıp). Paylaşımlı bellek
        # taşıması bunu aynı makinede tamamen ortadan kaldırır; UDP dışarısı için açık kalır.
        SetEnvironmentVariable(
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            os.path.join(pkg_dir, "config", "fastdds_shm.xml"),
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="Use simulation clock"
        ),
        DeclareLaunchArgument(
            "source",
            default_value="oakd",
            description="Görüntü kaynağı: oakd | webcam | none",
        ),
        DeclareLaunchArgument(
            "use_native_spatial",
            default_value="false",
            description="OAK-D VPU üzerinde çalışan yerleşik uzamsal algı boru hattı",
        ),
        DeclareLaunchArgument(
            "show_debug",
            default_value="false",
            description="Canlı bounding box penceresi göster (cv2.imshow)",
        ),

        # --- Görüntü kaynakları ---
        GroupAction(actions=[_depthai_camera(params_file, use_sim_time)], condition=_source_is("oakd")),
        Node(
            package="astro_vision",
            executable="webcam_publisher_node",
            name="webcam_publisher_node",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            condition=_source_is("webcam"),
        ),

        # --- Görü düğümleri ---
        Node(
            package="astro_vision",
            executable="face_detector_node",
            name="face_detector_node",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time, "show_debug": LaunchConfiguration("show_debug")}],
            condition=UnlessCondition(use_native_spatial),
        ),
        Node(
            package="astro_vision",
            executable="oak_spatial_native_node",
            name="oak_spatial_native_node",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(use_native_spatial),
        ),
    ])
