import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'swarm_gazebo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro') + glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SwarmX Team',
    maintainer_email='swarmx@graduation.project',
    description='Gazebo simulation and ROS 2 bridge for SwarmX Autonomous Swarm Car Search',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_bridge_node = swarm_gazebo.slam_bridge:main',
            'coordinator_node = swarm_gazebo.coordinator_node:main',
            'robot_controller_node = swarm_gazebo.robot_controller:main',
            'perception_bridge_node = swarm_gazebo.perception_bridge:main',
            'mission_monitor_node = swarm_gazebo.mission_monitor:main',
        ],
    },
)
