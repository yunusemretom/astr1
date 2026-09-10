import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'astro_base'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f'{package_name}.gaze'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Baran Eren',
    maintainer_email='baran@example.com',
    description='ASTRO V1 base hardware interface and social gaze system',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'serial_bridge = astro_base.serial_bridge:main',
            'social_gaze = astro_base.social_gaze_node:main',
            'standalone_gaze = astro_base.standalone_gaze_node:main',
            'standalone_gaze_ros = astro_base.standalone_gaze_ros_node:main',
            'head_tracker = astro_base.head_tracker_node:main',
            'diff_drive = astro_base.diff_drive_node:main',
        ],
    },
)

