from glob import glob

from setuptools import setup

package_name = 'rover_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/courses', glob('courses/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aakash Kumar',
    maintainer_email='maintainer@example.com',
    description='Gazebo obstacle course, mini rover and support nodes for the autonomy onboarding.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'odometry = rover_sim.odometry:main',
            'course_publisher = rover_sim.course_publisher:main',
            'cmd_vel_relay = rover_sim.cmd_vel_relay:main',
            'referee = rover_sim.referee:main',
        ],
    },
)
