from setuptools import setup

package_name = 'deepreach_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/deepreach_inference.launch.py']),
        ('share/' + package_name + '/launch', ['launch/deepreach_goal2.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/deepreach_goal2.rviz']),
        ('share/' + package_name + '/rviz', ['rviz/deepreach_goal2_topdown.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deepreach user',
    maintainer_email='user@example.com',
    description='ROS 2 inference package for DeepReach Dubins3D safety overrides.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deepreach_inference_node = deepreach_ros.inference_node:main',
            'safety_bubble_visualizer = deepreach_ros.safety_bubble_visualizer:main',
            'deepreach_stress_test = deepreach_ros.stress_test_node:main',
        ],
    },
)
