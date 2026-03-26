import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class DeepReachStressTestNode(Node):
    def __init__(self):
        super().__init__('deepreach_stress_test')

        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('duration_sec', 18.0)

        self.declare_parameter('initial_x', -4.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)

        self.declare_parameter('v_cmd', 1.0)
        self.declare_parameter('nominal_omega_cmd', 0.0)

        self.declare_parameter('obstacle_x', 0.0)
        self.declare_parameter('obstacle_y', 0.0)
        self.declare_parameter('obstacle_radius', 0.6)
        self.declare_parameter('robot_radius', 0.22)

        self.declare_parameter('auto_shutdown_on_complete', False)

        self.rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.dt = 1.0 / max(1.0, self.rate_hz)
        self.duration_sec = float(self.get_parameter('duration_sec').value)

        self.v_cmd = float(self.get_parameter('v_cmd').value)
        self.nominal_omega_cmd = float(self.get_parameter('nominal_omega_cmd').value)

        self.obstacle_x = float(self.get_parameter('obstacle_x').value)
        self.obstacle_y = float(self.get_parameter('obstacle_y').value)
        self.obstacle_radius = float(self.get_parameter('obstacle_radius').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.collision_distance = self.obstacle_radius + self.robot_radius

        initial_state = np.array(
            [
                float(self.get_parameter('initial_x').value),
                float(self.get_parameter('initial_y').value),
                float(self.get_parameter('initial_theta').value),
            ],
            dtype=np.float32,
        )

        self.filtered_state = initial_state.copy()
        self.baseline_state = initial_state.copy()

        self.latest_safe_omega = self.nominal_omega_cmd
        self.override_active = False

        self.override_count = 0
        self.step_count = 0
        self.max_steps = int(self.duration_sec * self.rate_hz)

        self.min_dist_filtered = np.inf
        self.min_dist_baseline = np.inf

        self.state_pub = self.create_publisher(Pose2D, 'state', 10)
        self.nominal_pub = self.create_publisher(Float32, 'nominal_omega', 10)
        self.test_pass_pub = self.create_publisher(Bool, 'stress_test_passed', 10)

        self.create_subscription(Float32, 'safe_omega', self.safe_omega_callback, 10)
        self.create_subscription(Bool, 'override_active', self.override_callback, 10)

        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info('Stress test started: aggressive nominal command toward obstacle center')

    @staticmethod
    def wrap_angle(theta: float) -> float:
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    def safe_omega_callback(self, msg: Float32):
        self.latest_safe_omega = float(msg.data)

    def override_callback(self, msg: Bool):
        self.override_active = bool(msg.data)

    def propagate(self, state: np.ndarray, omega_cmd: float) -> np.ndarray:
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        x_next = x + self.v_cmd * np.cos(theta) * self.dt
        y_next = y + self.v_cmd * np.sin(theta) * self.dt
        theta_next = self.wrap_angle(theta + omega_cmd * self.dt)
        return np.array([x_next, y_next, theta_next], dtype=np.float32)

    def distance_to_obstacle(self, state: np.ndarray) -> float:
        dx = float(state[0]) - self.obstacle_x
        dy = float(state[1]) - self.obstacle_y
        return float(np.sqrt(dx * dx + dy * dy))

    def step(self):
        nominal_msg = Float32()
        nominal_msg.data = float(self.nominal_omega_cmd)
        self.nominal_pub.publish(nominal_msg)

        state_msg = Pose2D()
        state_msg.x = float(self.filtered_state[0])
        state_msg.y = float(self.filtered_state[1])
        state_msg.theta = float(self.filtered_state[2])
        self.state_pub.publish(state_msg)

        self.filtered_state = self.propagate(self.filtered_state, self.latest_safe_omega)
        self.baseline_state = self.propagate(self.baseline_state, self.nominal_omega_cmd)

        if self.override_active:
            self.override_count += 1

        self.min_dist_filtered = min(self.min_dist_filtered, self.distance_to_obstacle(self.filtered_state))
        self.min_dist_baseline = min(self.min_dist_baseline, self.distance_to_obstacle(self.baseline_state))

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.finish_test()

    def finish_test(self):
        self.timer.cancel()

        baseline_collision = self.min_dist_baseline <= self.collision_distance
        filtered_collision = self.min_dist_filtered <= self.collision_distance
        passed = baseline_collision and (not filtered_collision) and (self.override_count > 0)

        pass_msg = Bool()
        pass_msg.data = bool(passed)
        self.test_pass_pub.publish(pass_msg)

        self.get_logger().info(
            'Stress test summary: '
            f'baseline_collision={baseline_collision}, '
            f'filtered_collision={filtered_collision}, '
            f'override_count={self.override_count}, '
            f'min_dist_baseline={self.min_dist_baseline:.3f}, '
            f'min_dist_filtered={self.min_dist_filtered:.3f}, '
            f'pass={passed}'
        )

        if bool(self.get_parameter('auto_shutdown_on_complete').value):
            self.get_logger().info('Auto shutdown enabled; exiting stress test node')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DeepReachStressTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()