import os
from glob import glob

from setuptools import setup

package_name = "astro_vision"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml") + glob("config/*.xml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Baran Eren",
    maintainer_email="baran@example.com",
    description="ASTRO V1 OAK-D Lite camera wrapper and face detection",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "face_detector_node = astro_vision.face_detector_node:main",
            "webcam_publisher_node = astro_vision.webcam_publisher_node:main",
            "oak_perception_node = astro_vision.oak_perception_node:main",
            "oak_spatial_native_node = astro_vision.oak_spatial_native_node:main",
            "session_recorder = astro_vision.session_recorder:main",
        ],
    },
)
