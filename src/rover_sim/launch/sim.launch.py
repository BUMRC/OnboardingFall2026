"""Starts the simulator, the rover, the ROS bridge, the support nodes, the referee and RViz.

ros2 launch rover_sim sim.launch.py [course:=course1] [gui:=true] [rviz:=true]
                                    [oracle:=true] [noise:=true]
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from rover_sim.course import load_course, world_sdf


def flag(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ('true', '1', 'yes')


def launch_setup(context):
    share = get_package_share_directory('rover_sim')

    course_arg = LaunchConfiguration('course').perform(context)
    course_path = course_arg if course_arg.endswith('.yaml') else os.path.join(share, 'courses', f'{course_arg}.yaml')
    course = load_course(course_path)
    world_path = os.path.join(tempfile.gettempdir(), f'rover_sim_{course.name}.sdf')
    with open(world_path, 'w') as f:
        f.write(world_sdf(course))

    gz_cmd = ['ign', 'gazebo', '-r', world_path] if flag(context, 'gui') else ['ign', 'gazebo', '-r', '-s', world_path]
    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(share, 'urdf', 'mini_rover.urdf.xacro')]), value_type=str)
    x, y, yaw = course.start
    sim_time = {'use_sim_time': True}

    actions = [
        ExecuteProcess(cmd=gz_cmd, output='screen', on_exit=Shutdown()),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}, sim_time]),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-topic', 'robot_description', '-name', 'mini_rover',
                        '-x', str(x), '-y', str(y), '-z', '0.01', '-Y', str(yaw)]),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': os.path.join(share, 'config', 'bridge.yaml')}]),
        Node(package='rover_sim', executable='odometry',
             parameters=[{'noise': flag(context, 'noise'),
                          'publish_map_to_odom': flag(context, 'oracle')}, sim_time]),
        Node(package='rover_sim', executable='course_publisher',
             parameters=[{'course': course_path}, sim_time]),
        Node(package='rover_sim', executable='cmd_vel_relay', parameters=[sim_time]),
        Node(package='rover_sim', executable='referee', output='screen',
             parameters=[{'course': course_path}, sim_time]),
    ]
    if flag(context, 'rviz'):
        actions.append(Node(package='rviz2', executable='rviz2',
                            arguments=['-d', os.path.join(share, 'rviz', 'sim.rviz')],
                            parameters=[sim_time]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='course1',
                              description='Course name in rover_sim/courses, or a path to a course YAML'),
        DeclareLaunchArgument('gui', default_value='true', description='Open the Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('oracle', default_value='true',
                              description='Publish map->odom from ground truth (turn off to use your localizer)'),
        DeclareLaunchArgument('noise', default_value='true', description='Add drift to wheel odometry'),
        OpaqueFunction(function=launch_setup),
    ])
