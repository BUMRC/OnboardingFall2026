"""Starts the simulator and the autonomy node.

ros2 launch rover_autonomy autonomy.launch.py [course:=course1] [gui:=true] [rviz:=true]
                                              [localization:=mcl|oracle]
localization:=oracle uses the simulator's perfect map -> odom instead of your particle filter.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rover_sim.course import load_course


def launch_setup(context):
    share = get_package_share_directory('rover_sim')
    course_arg = LaunchConfiguration('course').perform(context)
    course_path = course_arg if course_arg.endswith('.yaml') else os.path.join(share, 'courses', f'{course_arg}.yaml')
    use_mcl = LaunchConfiguration('localization').perform(context) != 'oracle'

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'sim.launch.py')),
            launch_arguments={'course': course_path, 'gui': LaunchConfiguration('gui'),
                              'rviz': LaunchConfiguration('rviz'),
                              'oracle': 'false' if use_mcl else 'true'}.items()),
        Node(package='rover_autonomy', executable='autonomy', output='screen',
             parameters=[{'use_mcl': use_mcl, 'start': [float(v) for v in load_course(course_path).start],
                          'use_sim_time': True}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='course1',
                              description='Course name in rover_sim/courses, or a path to a course YAML'),
        DeclareLaunchArgument('gui', default_value='true', description='Open the Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('localization', default_value='mcl',
                              description='mcl: your particle filter; oracle: the simulator publishes map -> odom'),
        OpaqueFunction(function=launch_setup),
    ])
