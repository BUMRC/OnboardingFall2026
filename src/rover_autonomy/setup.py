from glob import glob

from setuptools import setup

package_name = 'rover_autonomy'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aakash Kumar',
    maintainer_email='maintainer@example.com',
    description='Localization (MCL), planning (A*) and control (MPPI) for the mini rover.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'autonomy = rover_autonomy.autonomy_node:main',
        ],
    },
)
