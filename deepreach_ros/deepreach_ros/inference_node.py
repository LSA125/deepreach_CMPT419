import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from .dubins3d_control import dubins3d_optimal_control_from_grad_theta
from .normalization import CoordinateNormalizer, NormalizationConfig
from .siren_model import SingleBVPNet


class DeepReachInferenceNode(Node):
    def __init__(self):
        super().__init__('deepreach_inference_node')

        self.declare_parameter('model_path', '/workspaces/ros2_ws/src/project/deepreach_CMPT419/runs/dubins3d_run/training/checkpoints/model_final.pth')
        self.declare_parameter('device', 'cpu')

        self.declare_parameter('x_min', -5.0)
        self.declare_parameter('x_max', 5.0)
        self.declare_parameter('y_min', -5.0)
        self.declare_parameter('y_max', 5.0)
        self.declare_parameter('theta_min', -3.141592653589793)
        self.declare_parameter('theta_max', 3.141592653589793)
        self.declare_parameter('time_query', 1.0)

        self.declare_parameter('omega_max', 1.1)
        self.declare_parameter('set_mode', 'avoid')
        self.declare_parameter('safety_value_threshold', 0.0)

        self.declare_parameter('hidden_features', 512)
        self.declare_parameter('num_hidden_layers', 3)

        model_path = self.get_parameter('model_path').value
        device_param = self.get_parameter('device').value
        self.device = torch.device(device_param)

        config = NormalizationConfig(
            x_range=(float(self.get_parameter('x_min').value), float(self.get_parameter('x_max').value)),
            y_range=(float(self.get_parameter('y_min').value), float(self.get_parameter('y_max').value)),
            theta_range=(float(self.get_parameter('theta_min').value), float(self.get_parameter('theta_max').value)),
            time_max=1.0,
        )
        self.normalizer = CoordinateNormalizer(config)

        self.omega_max = float(self.get_parameter('omega_max').value)
        self.set_mode = str(self.get_parameter('set_mode').value)
        self.safety_value_threshold = float(self.get_parameter('safety_value_threshold').value)
        self.time_query = float(self.get_parameter('time_query').value)

        hidden_features = int(self.get_parameter('hidden_features').value)
        num_hidden_layers = int(self.get_parameter('num_hidden_layers').value)

        self.model = SingleBVPNet(
            in_features=4,
            out_features=1,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.latest_nominal_omega = 0.0

        self.state_sub = self.create_subscription(Pose2D, 'state', self.state_callback, 10)
        self.nominal_sub = self.create_subscription(Float32, 'nominal_omega', self.nominal_callback, 10)

        self.safe_omega_pub = self.create_publisher(Float32, 'safe_omega', 10)
        self.safety_value_pub = self.create_publisher(Float32, 'safety_value', 10)
        self.override_active_pub = self.create_publisher(Bool, 'override_active', 10)

        self.get_logger().info(f'Loaded model from: {model_path}')
        self.get_logger().info(f'Running on device: {self.device}')

    def nominal_callback(self, msg: Float32):
        self.latest_nominal_omega = float(msg.data)

    def state_callback(self, msg: Pose2D):
        state_real = np.array([msg.x, msg.y, msg.theta], dtype=np.float32)
        state_norm = self.normalizer.normalize_state(state_real, time=self.time_query)

        coords = torch.tensor(state_norm, dtype=torch.float32, device=self.device).view(1, 1, 4)

        with torch.enable_grad():
            out = self.model({'coords': coords})
            value = out['model_out']
            grad_all = torch.autograd.grad(value.sum(), out['model_in'], create_graph=False)[0]

        value_scalar = float(value.item())
        grad_norm_state = grad_all[0, 0, 1:].detach().cpu().numpy()
        grad_real_state = self.normalizer.denormalize_gradient(grad_norm_state)

        omega_override = dubins3d_optimal_control_from_grad_theta(
            dV_dtheta=float(grad_real_state[2]),
            omega_max=self.omega_max,
            set_mode=self.set_mode,
        )

        override_active = value_scalar <= self.safety_value_threshold
        omega_safe = omega_override if override_active else self.latest_nominal_omega

        safe_msg = Float32()
        safe_msg.data = float(omega_safe)
        self.safe_omega_pub.publish(safe_msg)

        value_msg = Float32()
        value_msg.data = value_scalar
        self.safety_value_pub.publish(value_msg)

        active_msg = Bool()
        active_msg.data = bool(override_active)
        self.override_active_pub.publish(active_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DeepReachInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
