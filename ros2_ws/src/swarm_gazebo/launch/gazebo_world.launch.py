#!/usr/bin/env python3
"""
ROS 2 Launch file for Gazebo Urban World and 4 Swarm Robots.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('swarm_gazebo')
    world_file = os.path.join(pkg_dir, 'worlds', 'urban_search_world.world')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'swarm_robot.urdf.xacro')

    headless = LaunchConfiguration('headless')
    declare_headless_cmd = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Whether to run Gazebo in headless mode',
    )

    # Robot Colors & Start Positions
    robot_configs = [
        ('robot_1', 'Blue', '0.2 0.5 0.9 1.0', 6.0, 6.0, 0.0),
        ('robot_2', 'Orange', '1.0 0.5 0.0 1.0', 6.0, 44.0, -1.5708),
        ('robot_3', 'Green', '0.1 0.8 0.2 1.0', 44.0, 6.0, 1.5708),
        ('robot_4', 'Purple', '0.7 0.2 0.8 1.0', 44.0, 44.0, 3.14159),
    ]

    # Gazebo server & client execution commands
    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen',
    )
    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=launch.conditions.UnlessCondition(headless) if 'launch.conditions' in globals() else None,
    )

    nodes = [
        declare_headless_cmd,
        gzserver_cmd,
    ]

    for rid, color, color_rgba, x, y, yaw in robot_configs:
        # Generate URDF via xacro
        robot_description = Command([
            'xacro ', xacro_file,
            ' robot_name:=', rid,
            ' robot_color:=', color,
            ' robot_color_rgba:="', color_rgba, '"',
        ])

        # State publisher
        rsp_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=rid,
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
                'frame_prefix': f'{rid}/',
            }],
        )

        # Gazebo spawn entity node
        spawn_node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name=f'spawn_{rid}',
            arguments=[
                '-topic', f'/{rid}/robot_description',
                '-entity', rid,
                '-x', str(x),
                '-y', str(y),
                '-z', '0.1',
                '-Y', str(yaw),
                '-robot_namespace', rid,
            ],
            output='screen',
        )

        nodes.extend([rsp_node, spawn_node])

    return LaunchDescription(nodes)
