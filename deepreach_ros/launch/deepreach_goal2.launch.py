from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import LaunchConfigurationEquals
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share = get_package_share_directory('deepreach_ros')
    rviz_orbit_config_path = package_share + '/rviz/deepreach_goal2.rviz'
    rviz_topdown_config_path = package_share + '/rviz/deepreach_goal2_topdown.rviz'

    model_path = '/workspaces/ros2_ws/src/project/deepreach_CMPT419/runs/dubins3d_run/training/checkpoints/model_final.pth'

    common_model_params = {
        'model_path': model_path,
        'device': 'cpu',
        'x_min': -5.0,
        'x_max': 5.0,
        'y_min': -5.0,
        'y_max': 5.0,
        'theta_min': -3.141592653589793,
        'theta_max': 3.141592653589793,
        'time_query': 1.0,
        'hidden_features': 512,
        'num_hidden_layers': 3,
    }

    inference_params = dict(common_model_params)
    inference_params.update(
        {
            'omega_max': 1.1,
            'set_mode': 'avoid',
            'safety_value_threshold': 0.0,
        }
    )

    viz_params = dict(common_model_params)
    viz_params.update(
        {
            'frame_id': 'map',
            'publish_rate_hz': 2.0,
            'sample_dx': 1.6,
            'sample_dy': 1.6,
            'sample_dtheta': 1.2,
            'sample_nx': 19,
            'sample_ny': 19,
            'sample_ntheta': 17,
            'level_epsilon': 0.02,
            'theta_to_z_scale': 0.35,
        }
    )

    stress_params = {
        'publish_rate_hz': 20.0,
        'duration_sec': 18.0,
        'initial_x': -4.0,
        'initial_y': 0.0,
        'initial_theta': 0.0,
        'v_cmd': 1.0,
        'nominal_omega_cmd': 0.0,
        'obstacle_x': 0.0,
        'obstacle_y': 0.0,
        'obstacle_radius': 0.6,
        'robot_radius': 0.22,
        'auto_shutdown_on_complete': False,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'rviz_preset',
                default_value='orbit',
                description='RViz preset to launch: orbit or topdown',
            ),
            Node(
                package='deepreach_ros',
                executable='deepreach_inference_node',
                name='deepreach_inference_node',
                output='screen',
                parameters=[inference_params],
            ),
            Node(
                package='deepreach_ros',
                executable='safety_bubble_visualizer',
                name='safety_bubble_visualizer',
                output='screen',
                parameters=[viz_params],
            ),
            TimerAction(
                period=6.0,
                actions=[
                    Node(
                        package='deepreach_ros',
                        executable='deepreach_stress_test',
                        name='deepreach_stress_test',
                        output='screen',
                        parameters=[stress_params],
                    )
                ],
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='deepreach_goal2_rviz',
                output='screen',
                arguments=['-d', rviz_orbit_config_path],
                remappings=[('/visualization_marker_array', '/safety_bubble_markers')],
                condition=LaunchConfigurationEquals('rviz_preset', 'orbit'),
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='deepreach_goal2_rviz_topdown',
                output='screen',
                arguments=['-d', rviz_topdown_config_path],
                remappings=[('/visualization_marker_array', '/safety_bubble_markers')],
                condition=LaunchConfigurationEquals('rviz_preset', 'topdown'),
            ),
        ]
    )