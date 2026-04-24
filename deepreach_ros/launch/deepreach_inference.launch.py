from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='deepreach_ros',
            executable='deepreach_inference_node',
            name='deepreach_inference_node',
            output='screen',
            parameters=[
                {
                    'model_path': '/workspaces/ros2_ws/src/project/deepreach_CMPT419/runs/dubins3d_run/training/checkpoints/model_final.pth',
                    'device': 'cpu',
                    'x_min': -5.0,
                    'x_max': 5.0,
                    'y_min': -5.0,
                    'y_max': 5.0,
                    'theta_min': -3.141592653589793,
                    'theta_max': 3.141592653589793,
                    'time_query': 1.0,
                    'omega_max': 1.1,
                    'set_mode': 'avoid',
                    'safety_value_threshold': 0.0,
                    'hidden_features': 512,
                    'num_hidden_layers': 3,
                }
            ],
        )
    ])
