cmake_minimum_required(VERSION 3.8)
project(astro_base)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Ubuntu 22.04'te CMake, Debian'ın distutils şemasını kullanıp üretilen python
# mesaj modülünü install/astro_base/LOCAL/lib/python3.10/dist-packages altına
# koyuyordu. colcon oraya PYTHONPATH hook'u üretmediği için paket kurulu
# görünmesine rağmen `from astro_base.msg import WheelCmd` ModuleNotFoundError
# veriyordu — hem serial_bridge.py çalışma anında, hem de testlerinde.
# Hedefi açıkça vererek hook'un üretilmesini garantiliyoruz.
if(NOT DEFINED PYTHON_INSTALL_DIR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(PYTHON_INSTALL_DIR
      "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
      CACHE STRING "Üretilen python mesaj modülünün kurulacağı yer")
endif()

# Generate custom messages (ROS 2 otomatik olarak python mesaj paketini de üretecek)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/WheelCmd.msg"
  "msg/HeadCmd.msg"
  "msg/HeadState.msg"
  "msg/GazeStatus.msg"
  DEPENDENCIES std_msgs
)


# Install Python modules
install(DIRECTORY
  astro_base
  DESTINATION "${PYTHON_INSTALL_DIR}"
)

# Install Python executables - Çalıştırılabilir scriptimizi hedefe kopyalıyoruz
install(PROGRAMS
  astro_base/serial_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME serial_bridge
)

install(PROGRAMS
  astro_base/social_gaze_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME social_gaze
)

install(PROGRAMS
  astro_base/head_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME head_tracker
)

install(PROGRAMS
  astro_base/standalone_gaze_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME standalone_gaze
)

install(PROGRAMS
  astro_base/standalone_gaze_ros_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME standalone_gaze_ros
)

# Install launch and config directories
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Linting
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()


