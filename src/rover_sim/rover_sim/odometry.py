"""Wheel odometry with drift, computed from the simulator's ground truth.

Publishes /odom and the odom -> base_link transform. With publish_map_to_odom on, it also
publishes map -> odom from ground truth: a perfect localizer to use until yours works.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_from_quaternion(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def make_transform(stamp, parent, child, x, y, yaw):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.z = math.sin(yaw / 2)
    t.transform.rotation.w = math.cos(yaw / 2)
    return t


class SimOdometry(Node):
    def __init__(self):
        super().__init__('odometry')
        self.noise = self.declare_parameter('noise', True).value
        # Random drift. Variances grow with distance driven and angle turned, independent of update rate.
        self.trans_noise = self.declare_parameter('trans_noise', 0.0005).value  # m^2 per m driven
        self.rot_noise = self.declare_parameter('rot_noise', 0.002).value  # rad^2 per rad turned
        self.rot_per_trans_noise = self.declare_parameter('rot_per_trans_noise', 0.0004).value  # rad^2 per m driven
        # Systematic drift, like a slightly wrong wheel radius or wheel separation.
        self.trans_scale = self.declare_parameter('trans_scale', 1.02).value
        self.rot_scale = self.declare_parameter('rot_scale', 0.97).value
        seed = self.declare_parameter('seed', -1).value
        self.publish_map_to_odom = self.declare_parameter('publish_map_to_odom', True).value

        self.rng = np.random.default_rng(None if seed < 0 else seed)
        self.prev = None
        self.pose = [0.0, 0.0, 0.0]
        self.pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, 'ground_truth', self.on_ground_truth, 10)

    def on_ground_truth(self, msg):
        p = msg.pose.pose
        gt = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev is None:
            self.prev = (gt, t)
            self.publish(msg.header.stamp, gt, 0.0, 0.0)
            return

        (px, py, pyaw), pt = self.prev
        dt = t - pt
        if dt <= 0.0:
            return
        self.prev = (gt, t)

        # What the wheels see: distance along the heading and change in heading.
        # Wheels can't measure sideways slip, so that component is dropped.
        dx, dy = gt[0] - px, gt[1] - py
        d_trans = math.cos(pyaw) * dx + math.sin(pyaw) * dy
        d_rot = wrap(gt[2] - pyaw)

        if self.noise:
            trans_std = math.sqrt(self.trans_noise * abs(d_trans))
            rot_std = math.sqrt(self.rot_noise * abs(d_rot) + self.rot_per_trans_noise * abs(d_trans))
            d_trans = d_trans * self.trans_scale + self.rng.normal(0.0, trans_std)
            d_rot = d_rot * self.rot_scale + self.rng.normal(0.0, rot_std)

        x, y, yaw = self.pose
        mid = yaw + d_rot / 2
        self.pose = [x + d_trans * math.cos(mid), y + d_trans * math.sin(mid), wrap(yaw + d_rot)]
        self.publish(msg.header.stamp, gt, d_trans / dt, d_rot / dt)

    def publish(self, stamp, gt, v, w):
        x, y, yaw = self.pose
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw / 2)
        odom.pose.pose.orientation.w = math.cos(yaw / 2)
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.pub.publish(odom)

        transforms = [make_transform(stamp, 'odom', 'base_link', x, y, yaw)]
        if self.publish_map_to_odom:
            # map->odom = (map->base_link from ground truth) * inverse(odom->base_link)
            yaw_mo = wrap(gt[2] - yaw)
            c, s = math.cos(yaw_mo), math.sin(yaw_mo)
            transforms.append(make_transform(stamp, 'map', 'odom',
                                             gt[0] - (c * x - s * y), gt[1] - (s * x + c * y), yaw_mo))
        self.tf.sendTransform(transforms)


def main():
    rclpy.init()
    node = SimOdometry()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()