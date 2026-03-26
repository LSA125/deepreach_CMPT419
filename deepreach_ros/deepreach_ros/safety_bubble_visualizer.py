import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, Pose2D
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from .normalization import CoordinateNormalizer, NormalizationConfig
from .siren_model import SingleBVPNet


class SafetyBubbleVisualizer(Node):
    def __init__(self):
        super().__init__('safety_bubble_visualizer')

        self.declare_parameter('model_path', '/workspaces/ros2_ws/src/project/deepreach_CMPT419/runs/dubins3d_run/training/checkpoints/model_final.pth')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate_hz', 2.0)

        self.declare_parameter('x_min', -5.0)
        self.declare_parameter('x_max', 5.0)
        self.declare_parameter('y_min', -5.0)
        self.declare_parameter('y_max', 5.0)
        self.declare_parameter('theta_min', -3.141592653589793)
        self.declare_parameter('theta_max', 3.141592653589793)
        self.declare_parameter('time_query', 1.0)
        self.declare_parameter('hidden_features', 512)
        self.declare_parameter('num_hidden_layers', 3)

        self.declare_parameter('sample_dx', 1.6)
        self.declare_parameter('sample_dy', 1.6)
        self.declare_parameter('sample_dtheta', 1.2)
        self.declare_parameter('sample_nx', 19)
        self.declare_parameter('sample_ny', 19)
        self.declare_parameter('sample_ntheta', 17)
        self.declare_parameter('level_epsilon', 0.02)
        self.declare_parameter('theta_to_z_scale', 0.35)
        self.declare_parameter('fixed_obstacle_x', 0.0)
        self.declare_parameter('fixed_obstacle_y', 0.0)
        self.declare_parameter('fixed_obstacle_radius', 0.6)
        self.declare_parameter('fixed_obstacle_on_path', True)
        self.declare_parameter('fixed_obstacle_distance_ahead', 1.8)
        self.declare_parameter('moving_obstacle_start_x', 2.0)
        self.declare_parameter('moving_obstacle_start_y', 1.5)
        self.declare_parameter('moving_obstacle_radius', 0.45)
        self.declare_parameter('moving_obstacle_speed', 0.5)
        self.declare_parameter('moving_obstacle_start_distance_ahead', 3.0)
        self.declare_parameter('moving_obstacle_start_lateral_offset', 1.0)
        self.declare_parameter('moving_obstacle_reset_distance', 0.25)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.time_query = float(self.get_parameter('time_query').value)
        self.level_epsilon = float(self.get_parameter('level_epsilon').value)
        self.theta_to_z_scale = float(self.get_parameter('theta_to_z_scale').value)
        self.fixed_obstacle_x = float(self.get_parameter('fixed_obstacle_x').value)
        self.fixed_obstacle_y = float(self.get_parameter('fixed_obstacle_y').value)
        self.fixed_obstacle_radius = float(self.get_parameter('fixed_obstacle_radius').value)
        self.fixed_obstacle_on_path = bool(self.get_parameter('fixed_obstacle_on_path').value)
        self.fixed_obstacle_distance_ahead = float(self.get_parameter('fixed_obstacle_distance_ahead').value)
        self.moving_obstacle_radius = float(self.get_parameter('moving_obstacle_radius').value)
        self.moving_obstacle_speed = float(self.get_parameter('moving_obstacle_speed').value)
        self.moving_obstacle_start_distance_ahead = float(self.get_parameter('moving_obstacle_start_distance_ahead').value)
        self.moving_obstacle_start_lateral_offset = float(self.get_parameter('moving_obstacle_start_lateral_offset').value)
        self.moving_obstacle_reset_distance = float(self.get_parameter('moving_obstacle_reset_distance').value)
        self.moving_obstacle_pos = np.array(
            [
                float(self.get_parameter('moving_obstacle_start_x').value),
                float(self.get_parameter('moving_obstacle_start_y').value),
            ],
            dtype=np.float32,
        )
        self.moving_obstacle_start_pos = self.moving_obstacle_pos.copy()
        self.obstacles_initialized = False
        self.last_obstacle_update_time = self.get_clock().now()

        self.device = torch.device(str(self.get_parameter('device').value))
        hidden_features = int(self.get_parameter('hidden_features').value)
        num_hidden_layers = int(self.get_parameter('num_hidden_layers').value)

        model_path = str(self.get_parameter('model_path').value)

        config = NormalizationConfig(
            x_range=(float(self.get_parameter('x_min').value), float(self.get_parameter('x_max').value)),
            y_range=(float(self.get_parameter('y_min').value), float(self.get_parameter('y_max').value)),
            theta_range=(float(self.get_parameter('theta_min').value), float(self.get_parameter('theta_max').value)),
            time_max=1.0,
        )
        self.normalizer = CoordinateNormalizer(config)

        self.model = SingleBVPNet(
            in_features=4,
            out_features=1,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.dx_values = np.linspace(-float(self.get_parameter('sample_dx').value), float(self.get_parameter('sample_dx').value), int(self.get_parameter('sample_nx').value), dtype=np.float32)
        self.dy_values = np.linspace(-float(self.get_parameter('sample_dy').value), float(self.get_parameter('sample_dy').value), int(self.get_parameter('sample_ny').value), dtype=np.float32)
        self.dtheta_values = np.linspace(-float(self.get_parameter('sample_dtheta').value), float(self.get_parameter('sample_dtheta').value), int(self.get_parameter('sample_ntheta').value), dtype=np.float32)

        self.latest_state = None
        self.create_subscription(Pose2D, 'state', self.state_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'safety_bubble_markers', 10)

        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(0.1, publish_rate_hz), self.publish_markers)

        self.get_logger().info(f'Safety bubble visualizer loaded model from: {model_path}')

    def state_callback(self, msg: Pose2D):
        self.latest_state = np.array([msg.x, msg.y, self.wrap_angle(msg.theta)], dtype=np.float32)

    @staticmethod
    def wrap_angle(theta: float) -> float:
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    def publish_markers(self):
        if self.latest_state is None:
            return

        state = self.latest_state
        x0, y0, theta0 = float(state[0]), float(state[1]), float(state[2])

        if not self.obstacles_initialized:
            heading_x = float(np.cos(theta0))
            heading_y = float(np.sin(theta0))
            lateral_x = -heading_y
            lateral_y = heading_x
            if self.fixed_obstacle_on_path:
                self.fixed_obstacle_x = x0 + heading_x * self.fixed_obstacle_distance_ahead
                self.fixed_obstacle_y = y0 + heading_y * self.fixed_obstacle_distance_ahead
            self.moving_obstacle_start_pos = np.array(
                [
                    x0 + heading_x * self.moving_obstacle_start_distance_ahead + lateral_x * self.moving_obstacle_start_lateral_offset,
                    y0 + heading_y * self.moving_obstacle_start_distance_ahead + lateral_y * self.moving_obstacle_start_lateral_offset,
                ],
                dtype=np.float32,
            )
            self.moving_obstacle_pos = self.moving_obstacle_start_pos.copy()
            self.obstacles_initialized = True

        grid = np.stack(np.meshgrid(self.dx_values, self.dy_values, self.dtheta_values, indexing='ij'), axis=-1).reshape(-1, 3)
        sample_states = np.zeros((grid.shape[0], 3), dtype=np.float32)
        sample_states[:, 0] = x0 + grid[:, 0]
        sample_states[:, 1] = y0 + grid[:, 1]
        sample_states[:, 2] = np.arctan2(np.sin(theta0 + grid[:, 2]), np.cos(theta0 + grid[:, 2]))

        norm_states = self.normalizer.normalize_state(sample_states, time=self.time_query)
        coords = torch.tensor(norm_states, dtype=torch.float32, device=self.device).view(-1, 1, 4)

        with torch.no_grad():
            out = self.model({'coords': coords})
            values = out['model_out'].view(-1).detach().cpu().numpy()

        near_boundary = np.abs(values) <= self.level_epsilon
        boundary_states = sample_states[near_boundary]

        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.header.frame_id = self.frame_id
        clear_marker.header.stamp = now_msg
        clear_marker.ns = 'safety_bubble'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        robot_marker = Marker()
        robot_marker.header.frame_id = self.frame_id
        robot_marker.header.stamp = now_msg
        robot_marker.ns = 'safety_bubble'
        robot_marker.id = 1
        robot_marker.type = Marker.SPHERE
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = x0
        robot_marker.pose.position.y = y0
        robot_marker.pose.position.z = 0.0
        robot_marker.pose.orientation.w = 1.0
        robot_marker.scale.x = 0.25
        robot_marker.scale.y = 0.25
        robot_marker.scale.z = 0.25
        robot_marker.color.r = 0.0
        robot_marker.color.g = 0.8
        robot_marker.color.b = 0.2
        robot_marker.color.a = 0.95
        marker_array.markers.append(robot_marker)

        fixed_obstacle_marker = Marker()
        fixed_obstacle_marker.header.frame_id = self.frame_id
        fixed_obstacle_marker.header.stamp = now_msg
        fixed_obstacle_marker.ns = 'safety_bubble'
        fixed_obstacle_marker.id = 3
        fixed_obstacle_marker.type = Marker.CYLINDER
        fixed_obstacle_marker.action = Marker.ADD
        fixed_obstacle_marker.pose.position.x = self.fixed_obstacle_x
        fixed_obstacle_marker.pose.position.y = self.fixed_obstacle_y
        fixed_obstacle_marker.pose.position.z = 0.0
        fixed_obstacle_marker.pose.orientation.w = 1.0
        fixed_obstacle_marker.scale.x = self.fixed_obstacle_radius * 2.0
        fixed_obstacle_marker.scale.y = self.fixed_obstacle_radius * 2.0
        fixed_obstacle_marker.scale.z = 0.25
        fixed_obstacle_marker.color.r = 0.2
        fixed_obstacle_marker.color.g = 0.2
        fixed_obstacle_marker.color.b = 1.0
        fixed_obstacle_marker.color.a = 0.85
        marker_array.markers.append(fixed_obstacle_marker)

        now_time = self.get_clock().now()
        dt = (now_time - self.last_obstacle_update_time).nanoseconds * 1e-9
        self.last_obstacle_update_time = now_time
        if dt > 0.0:
            to_robot = np.array([x0, y0], dtype=np.float32) - self.moving_obstacle_pos
            dist = float(np.linalg.norm(to_robot))
            if dist <= self.moving_obstacle_reset_distance:
                self.moving_obstacle_pos = self.moving_obstacle_start_pos.copy()
            elif dist > 1e-6:
                step = min(self.moving_obstacle_speed * dt, dist)
                self.moving_obstacle_pos = self.moving_obstacle_pos + (to_robot / dist) * step

        moving_obstacle_marker = Marker()
        moving_obstacle_marker.header.frame_id = self.frame_id
        moving_obstacle_marker.header.stamp = now_msg
        moving_obstacle_marker.ns = 'safety_bubble'
        moving_obstacle_marker.id = 4
        moving_obstacle_marker.type = Marker.CYLINDER
        moving_obstacle_marker.action = Marker.ADD
        moving_obstacle_marker.pose.position.x = float(self.moving_obstacle_pos[0])
        moving_obstacle_marker.pose.position.y = float(self.moving_obstacle_pos[1])
        moving_obstacle_marker.pose.position.z = 0.0
        moving_obstacle_marker.pose.orientation.w = 1.0
        moving_obstacle_marker.scale.x = self.moving_obstacle_radius * 2.0
        moving_obstacle_marker.scale.y = self.moving_obstacle_radius * 2.0
        moving_obstacle_marker.scale.z = 0.25
        moving_obstacle_marker.color.r = 1.0
        moving_obstacle_marker.color.g = 0.1
        moving_obstacle_marker.color.b = 0.1
        moving_obstacle_marker.color.a = 0.9
        marker_array.markers.append(moving_obstacle_marker)

        bubble_marker = Marker()
        bubble_marker.header.frame_id = self.frame_id
        bubble_marker.header.stamp = now_msg
        bubble_marker.ns = 'safety_bubble'
        bubble_marker.id = 2
        bubble_marker.type = Marker.SPHERE_LIST
        bubble_marker.action = Marker.ADD
        bubble_marker.pose.orientation.w = 1.0
        bubble_marker.scale.x = 0.06
        bubble_marker.scale.y = 0.06
        bubble_marker.scale.z = 0.06
        bubble_marker.color.r = 1.0
        bubble_marker.color.g = 0.25
        bubble_marker.color.b = 0.1
        bubble_marker.color.a = 0.55

        for state_i in boundary_states:
            p = Point()
            p.x = float(state_i[0])
            p.y = float(state_i[1])
            p.z = float((self.wrap_angle(float(state_i[2]) - theta0)) * self.theta_to_z_scale)
            bubble_marker.points.append(p)

        marker_array.markers.append(bubble_marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyBubbleVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()