"""Decoupled ROS 2 node wrapper for Visual Positioning System (VPS).

Subscribes to standard sensor topics:
- /camera/image_raw (sensor_msgs/Image)
- /mavros/imu/data (sensor_msgs/Imu)
- /mavros/local_position/velocity_body (geometry_msgs/TwistStamped)
- /mavros/altitude (std_msgs/Float64 or custom baro)

Publishes downward AGL height:
- /vps/height_agl (sensor_msgs/Range, min=0.1m, max=300.0m)

Note: Architectural isolation ensures core library has zero ROS dependencies.
This wrapper is entirely optional and degrades gracefully if rclpy is not installed.
"""

from pathlib import Path
from typing import Optional
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vps.config import VPSConfig, load_config
from vps.mavlink.telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from vps.fusion.fusion_engine import VPSFusionEngine

# Optional ROS 2 imports with graceful fallback
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, Imu, Range
    from geometry_msgs.msg import TwistStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Minimal mock Node class for typing / non-ROS testing
    class Node:  # type: ignore
        def __init__(self, node_name: str, **kwargs):
            self.node_name = node_name


class VPSRos2Node(Node):
    """ROS 2 wrapper node for VPS Height Estimator."""

    def __init__(self, node_name: str = "vps_height_estimator", **kwargs) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS 2 (rclpy) is not installed in the current Python environment. "
                "Use the standalone runner 'python vps/main.py' or activate a ROS 2 workspace."
            )
        super().__init__(node_name, **kwargs)

        # Declare ROS 2 parameters
        self.declare_parameter("config_file", "config/default_config.yaml")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("imu_topic", "/mavros/imu/data")
        self.declare_parameter("velocity_topic", "/mavros/local_position/velocity_body")
        self.declare_parameter("output_range_topic", "/vps/height_agl")
        self.declare_parameter("baseline_offset_m", 0.15)

        cfg_path = Path(self.get_parameter("config_file").get_parameter_value().string_value)
        cam_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        vel_topic = self.get_parameter("velocity_topic").get_parameter_value().string_value
        range_topic = self.get_parameter("output_range_topic").get_parameter_value().string_value
        baseline_m = float(self.get_parameter("baseline_offset_m").get_parameter_value().double_value)

        # Load VPS Config
        if cfg_path.is_file():
            self.cfg = load_config(cfg_path)
        else:
            self.cfg = VPSConfig()
        self.cfg.camera.baseline_ground_offset_m = baseline_m

        # Initialize Core Fusion Engine
        self.ring_buffer = TelemetryRingBuffer(max_duration_s=10.0)
        self.fusion_engine = VPSFusionEngine(config=self.cfg, initial_height_m=baseline_m)

        # State cache
        self._latest_vel_ned = np.zeros(3, dtype=np.float64)
        self._latest_att_rpy = np.zeros(3, dtype=np.float64)
        self._latest_gyro = np.zeros(3, dtype=np.float64)
        self._latest_baro = baseline_m
        self._frame_count = 0

        # ROS 2 Subscriptions
        self.sub_img = self.create_subscription(
            Image, cam_topic, self._image_callback, qos_profile_sensor_data
        )
        self.sub_imu = self.create_subscription(
            Imu, imu_topic, self._imu_callback, qos_profile_sensor_data
        )
        self.sub_vel = self.create_subscription(
            TwistStamped, vel_topic, self._velocity_callback, qos_profile_sensor_data
        )

        # ROS 2 Publisher
        self.pub_range = self.create_publisher(Range, range_topic, 10)

        self.get_logger().info(
            f"VPS Height Estimator ROS 2 node initialized.\n"
            f"  Subscribed: Image '{cam_topic}', IMU '{imu_topic}', Velocity '{vel_topic}'\n"
            f"  Publishing: Range '{range_topic}'"
        )

    def _imu_callback(self, msg: "Imu") -> None:
        """Cache latest IMU angular velocity and attitude quaternion."""
        t_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        gx = float(msg.angular_velocity.x)
        gy = float(msg.angular_velocity.y)
        gz = float(msg.angular_velocity.z)
        self._latest_gyro = np.array([gx, gy, gz], dtype=np.float64)

        # Quaternion to Roll, Pitch, Yaw
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self._latest_att_rpy = np.array([roll, pitch, yaw], dtype=np.float64)

        # Ingest into telemetry buffer
        sample = TelemetrySample(
            timestamp_s=t_s,
            gyro_rad_s=self._latest_gyro.copy(),
            attitude_rpy_rad=self._latest_att_rpy.copy(),
            vel_ned_mps=self._latest_vel_ned.copy(),
            baro_alt_m=self._latest_baro,
        )
        self.ring_buffer.add_sample(sample)

    def _velocity_callback(self, msg: "TwistStamped") -> None:
        """Cache latest body/NED velocity."""
        vx = float(msg.twist.linear.x)
        vy = float(msg.twist.linear.y)
        vz = float(msg.twist.linear.z)
        self._latest_vel_ned = np.array([vx, vy, vz], dtype=np.float64)

    def _image_callback(self, msg: "Image") -> None:
        """Process incoming camera frame and publish rangefinder height."""
        t_capture = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._frame_count += 1

        # Convert ROS Image buffer directly to numpy array
        # IMX296 mono8 or BGR8
        if msg.encoding in ("mono8", "8UC1"):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        elif msg.encoding in ("bgr8", "8UC3"):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        elif msg.encoding in ("rgb8",):
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img = rgb[:, :, ::-1]  # RGB to BGR
        else:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
            return

        # Query synchronized telemetry
        telem = self.ring_buffer.get_interpolated(t_capture)
        if telem is None:
            telem = TelemetrySample(
                timestamp_s=t_capture,
                gyro_rad_s=self._latest_gyro,
                attitude_rpy_rad=self._latest_att_rpy,
                vel_ned_mps=self._latest_vel_ned,
                baro_alt_m=self._latest_baro,
            )

        # Process through master fusion engine
        fusion_out = self.fusion_engine.process_telemetry_and_frame(
            frame=img,
            timestamp_s=t_capture,
            pos_ned_m=telem.pos_ned_m,
            vel_ned_mps=telem.vel_ned_mps,
            attitude_rpy_rad=telem.attitude_rpy_rad,
            gyro_rates_rad_s=telem.gyro_rad_s,
            baro_alt_m=telem.baro_alt_m,
            frame_idx=self._frame_count,
        )

        # Construct sensor_msgs/Range message
        range_msg = Range()
        range_msg.header = msg.header
        range_msg.radiation_type = Range.INFRARED
        range_msg.field_of_view = float(np.radians(62.2))  # Sony IMX296 HFOV
        range_msg.min_range = float(self.cfg.mavlink.min_distance_cm * 0.01)
        range_msg.max_range = float(self.cfg.mavlink.max_distance_cm * 0.01)
        range_msg.range = float(fusion_out.height_agl_m)

        self.pub_range.publish(range_msg)


def main(args=None):
    if not ROS2_AVAILABLE:
        print("[ERROR] ROS 2 (rclpy) is not installed in current environment.")
        sys.exit(1)

    rclpy.init(args=args)
    node = VPSRos2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
