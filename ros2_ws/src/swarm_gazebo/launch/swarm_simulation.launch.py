#!/usr/bin/env python3
"""
Master ROS 2 Launch File for SwarmX Gazebo Simulation.
Starts Gazebo World, spawns 4 robots, and launches the complete SwarmX coordination stack.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('swarm_gazebo')

    robots = ['robot_1', 'robot_2', 'robot_3', 'robot_4']

    # Include Gazebo World & Robot Spawner Launch
    gazebo_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'gazebo_world.launch.py')
        ),
        launch_arguments={'headless': 'True'}.items(),
    )

    # 1. Gazebo SLAM Bridge Node (LiDAR -> OccupancyGrid -> ISLAMProvider)
    slam_bridge_node = Node(
        package='swarm_gazebo',
        executable='slam_bridge_node',
        name='gazebo_slam_bridge',
        output='screen',
    )

    # 2. Gazebo Perception Bridge Node (Camera -> OCR -> Watchlist Matching)
    perception_bridge_node = Node(
        package='swarm_gazebo',
        executable='perception_bridge_node',
        name='gazebo_perception_bridge',
        output='screen',
    )

    # 3. Robot Controller Nodes (Waypoints -> Obstacle Avoidance -> /robot_*/cmd_vel)
    controller_nodes = [
        Node(
            package='swarm_gazebo',
            executable='robot_controller_node',
            name=f'{rid}_controller',
            arguments=['--robot-id', rid],
            output='screen',
        )
        for rid in robots
    ]

    # 4. SwarmX ACO Coordinator Node
    coordinator_node = Node(
        package='swarm_gazebo',
        executable='coordinator_node',
        name='swarm_coordinator_node',
        output='screen',
    )

    # 5. Mission Monitor & HUD Node
    monitor_node = Node(
        package='swarm_gazebo',
        executable='mission_monitor_node',
        name='mission_monitor_node',
        output='screen',
    )

    return LaunchDescription([
        gazebo_world_launch,
        slam_bridge_node,
        perception_bridge_node,
        coordinator_node,
        monitor_node,
        *controller_nodes,
    ])
