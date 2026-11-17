from setuptools import find_packages, setup

package_name = 'drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/drone_system.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'drone_control = drone_control.drone_control:main',
            'rc_monitor_node = drone_control.rc_monitor_node:main',
            'web_map_node = drone_control.web_map_node:main',
            'camera_node = drone_control.camera_node:main',
            'AL_node = drone_control.AL_node:main',
            'aruco_node = drone_control.aruco_node:main',
            'aruco_debug_saver = drone_control.aruco_debug_saver:main',
            'landing_node = drone_control.landing_node:main',
        ],
    },
)
