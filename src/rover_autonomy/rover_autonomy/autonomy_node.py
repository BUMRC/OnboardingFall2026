"""The autonomy node: localization, planning and control, once per lidar scan.

Provided. The algorithms live in mcl.py, planner.py and controller.py. On every scan:
1. Localize: look up the odometry pose at the scan's time, run the particle filter, and publish
   map -> odom so that map -> odom -> base_link lands on the estimate.
2. Plan, the first time only: A* through every gate on the prior map.
3. Control: MPPI from the latest pose, avoiding mapped obstacles and anything in the scan.
With use_mcl false, the simulator's oracle publishes map -> odom instead, so you can work on planning
and control before localization works.
"""
import bisect
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rover_autonomy import controller, mcl, planner
from rover_autonomy.geometry import Grid, compose, inverse, wrap

SCAN_OBSTACLE_RANGE = 4.0  # m: scan hits closer than this count as obstacles for MPPI
TF_TOLERANCE = 0.1         # s: map -> odom is stamped this far ahead so "latest" lookups succeed
PARTICLES_SHOWN = 300
ROLLOUTS_SHOWN = 40


def yaw_of(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def make_pose(x, y, yaw):
    pose = Pose()
    pose.position.x, pose.position.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
    return pose


class Autonomy(Node):
    def __init__(self):
        super().__init__('autonomy')
        self.use_mcl = self.declare_parameter('use_mcl', True).value
        self.start = list(self.declare_parameter('start', [2.0, 3.0, 0.0]).value)

        self.grid = None
        self.gates = None
        self.laser_offset = None
        self.odom = deque(maxlen=100)  # (time, x, y, yaw, v, w) from /odom, about 2 s at 50 Hz
        self.pf = None
        self.tracker = None
        self.mppi = controller.MPPI()
        self.disabled = set()
        self.finished = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        newest_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.particles_pub = self.create_publisher(PoseArray, 'particles', 1)
        self.plan_pub = self.create_publisher(Path, 'plan', latched)
        self.local_plan_pub = self.create_publisher(Path, 'local_plan', 1)
        self.rollouts_pub = self.create_publisher(MarkerArray, 'mppi/rollouts', 1)
        self.create_subscription(OccupancyGrid, 'map', self.on_map, latched)
        self.create_subscription(PoseArray, 'course/waypoints', self.on_waypoints, latched)
        self.create_subscription(Odometry, 'odom', self.on_odom, 50)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.on_initial_pose, 1)
        self.create_subscription(LaserScan, 'scan', self.on_scan, newest_only)

    def on_map(self, msg):
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.grid = Grid(data >= 50, msg.info.resolution, origin)

    def on_waypoints(self, msg):
        self.gates = [(p.position.x, p.position.y, yaw_of(p.orientation)) for p in msg.poses]

    def on_odom(self, msg):
        p, v = msg.pose.pose, msg.twist.twist
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.odom.append((t, p.position.x, p.position.y, yaw_of(p.orientation), v.linear.x, v.angular.z))

    def on_initial_pose(self, msg):
        """RViz's 2D Pose Estimate tool: restart the particle filter around the clicked pose."""
        if self.pf is not None:
            p = msg.pose.pose
            self.pf.initialize((p.position.x, p.position.y, yaw_of(p.orientation)))
            self.get_logger().info(f'Particles reset around ({p.position.x:.2f}, {p.position.y:.2f})')

    def on_scan(self, msg):
        if self.grid is None or self.gates is None or len(self.odom) < 2:
            return
        if self.laser_offset is None:
            self.laser_offset = self.lookup('base_link', msg.header.frame_id)
            if self.laser_offset is None:
                return
        started = time.perf_counter()
        stamp = Time.from_msg(msg.header.stamp)
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        odom_at_scan = self.odom_at(stamp.nanoseconds * 1e-9)

        # 1. Localize.
        map_to_odom = self.localize(odom_at_scan, ranges, angles, stamp)
        if map_to_odom is None or self.finished:
            return
        latest = self.odom[-1]
        pose = compose(map_to_odom, latest[1:4])  # the estimate, brought up to date with odometry
        state = (pose[0], pose[1], pose[2], latest[4], latest[5])

        # 2. Plan once, through every gate.
        if self.tracker is None:
            path = self.run('planner', planner.plan_course, self.grid, pose[:2], self.gates)
            if path is None:
                if 'planner' not in self.disabled:
                    self.get_logger().warn('No path through the gates', throttle_duration_sec=5.0)
                return
            self.tracker = controller.PathTracker(path)
            self.plan_pub.publish(self.path_msg(path))
            self.get_logger().info(f'Planned {self.tracker.s[-1]:.1f} m through {len(self.gates)} gates')

        # 3. Control.
        path_lookup = self.tracker.lookup_ahead(pose[:2])
        if self.tracker.remaining() < 0.3:
            self.cmd_pub.publish(Twist())
            self.finished = True
            self.get_logger().info('Reached the end of the path')
            return
        laser = compose(compose(map_to_odom, odom_at_scan), self.laser_offset)
        close = np.isfinite(ranges) & (ranges < SCAN_OBSTACLE_RANGE)
        beams = laser[2] + angles[close]
        hits = np.column_stack([laser[0] + ranges[close] * np.cos(beams), laser[1] + ranges[close] * np.sin(beams)])
        result = self.run('controller', self.mppi.step, state, path_lookup,
                          controller.obstacle_distance_function(self.grid, hits))
        if result is None:
            return
        command, planned, samples = result
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(command[0]), float(command[1])
        self.cmd_pub.publish(cmd)
        self.local_plan_pub.publish(self.path_msg(planned[:, :2]))
        self.rollouts_pub.publish(self.rollout_markers(samples))

        elapsed = time.perf_counter() - started
        if elapsed > 0.1:
            self.get_logger().warn(f'Handling a scan took {elapsed * 1000:.0f} ms; scans come every 100 ms',
                                   throttle_duration_sec=5.0)

    def localize(self, odom_at_scan, ranges, angles, stamp):
        """map -> odom, from the particle filter or, with use_mcl off, from the simulator."""
        if not self.use_mcl:
            return self.lookup('map', 'odom')
        if self.pf is None:
            self.pf = mcl.ParticleFilter(self.grid, self.laser_offset)
            self.pf.initialize(self.start)
        result = self.run('mcl', self.pf.step, odom_at_scan, ranges, angles)
        if result is None:
            return None
        estimate, odom_used = result
        # The estimate goes with the odometry pose of the filter's last update.
        map_to_odom = compose(estimate, inverse(odom_used))
        self.publish_transform(map_to_odom, stamp + Duration(seconds=TF_TOLERANCE))
        self.publish_particles(stamp)
        return map_to_odom

    def run(self, module, function, *args):
        """Call into an algorithm. A TODO that isn't written yet switches that part off, with one message."""
        if module in self.disabled:
            return None
        try:
            return function(*args)
        except NotImplementedError as e:
            self.disabled.add(module)
            hint = ' Run with localization:=oracle to work on planning and control first.' if module == 'mcl' else ''
            self.get_logger().warn(f'{module}: {e} is not written yet, so {module} is off.{hint}')
            return None

    def odom_at(self, t):
        """The odometry pose at time t, interpolated between the /odom messages around it."""
        times = [entry[0] for entry in self.odom]
        i = bisect.bisect_right(times, t)
        if i == 0 or i == len(times):
            return np.array(self.odom[min(i, len(times) - 1)][1:4])
        (t0, x0, y0, a0, _, _), (t1, x1, y1, a1, _, _) = self.odom[i - 1], self.odom[i]
        f = (t - t0) / (t1 - t0)
        return np.array([x0 + f * (x1 - x0), y0 + f * (y1 - y0), a0 + f * float(wrap(a1 - a0))])

    def lookup(self, parent, child):
        """Latest pose (x, y, yaw) of child in parent from TF, or None if it isn't available yet."""
        try:
            t = self.tf_buffer.lookup_transform(parent, child, Time()).transform
        except TransformException:
            self.get_logger().info(f'Waiting for TF {parent} -> {child}', throttle_duration_sec=2.0)
            return None
        return np.array([t.translation.x, t.translation.y, yaw_of(t.rotation)])

    def publish_transform(self, pose, stamp):
        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id, t.child_frame_id = 'map', 'odom'
        t.transform.translation.x, t.transform.translation.y = float(pose[0]), float(pose[1])
        t.transform.rotation.z, t.transform.rotation.w = float(np.sin(pose[2] / 2)), float(np.cos(pose[2] / 2))
        self.tf_broadcaster.sendTransform(t)

    def publish_particles(self, stamp):
        particles = self.pf.particles
        shown = particles[np.linspace(0, len(particles) - 1, min(PARTICLES_SHOWN, len(particles))).astype(int)]
        msg = PoseArray()
        msg.header.stamp, msg.header.frame_id = stamp.to_msg(), 'map'
        msg.poses = [make_pose(*p) for p in shown]
        self.particles_pub.publish(msg)

    def path_msg(self, points):
        msg = Path()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), 'map'
        yaws = np.append(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])), 0.0)
        for (x, y), yaw in zip(points, yaws):
            pose = PoseStamped(header=msg.header)
            pose.pose = make_pose(x, y, yaw)
            msg.poses.append(pose)
        return msg

    def rollout_markers(self, samples):
        marker = Marker(type=Marker.LINE_LIST, ns='rollouts', id=0)
        marker.header.stamp, marker.header.frame_id = self.get_clock().now().to_msg(), 'map'
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.color.r = marker.color.g = marker.color.b = 0.6
        marker.color.a = 0.5
        for xy in samples[np.random.choice(len(samples), min(ROLLOUTS_SHOWN, len(samples)), replace=False), :, :2]:
            segments = np.stack([xy[:-1], xy[1:]], axis=1).reshape(-1, 2)
            marker.points += [Point(x=float(x), y=float(y)) for x, y in segments]
        return MarkerArray(markers=[marker])


def main():
    # Handle Ctrl-C here instead of in rclpy, so the node can still send a last stop command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Autonomy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
