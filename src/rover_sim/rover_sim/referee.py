"""Referee: times a run from ground truth, checks the gates are passed in order and counts contacts.

The clock starts when the rover first moves. A gate counts when the rover crosses it in the direction
of travel. A contact is the rover's center coming within contact_distance of any obstacle, including
the ones missing from /map. The result is logged and published on /referee/result.
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from rover_sim.course import load_course


def distance_to(obstacle, x, y):
    """Distance from (x, y) to the obstacle's edge, negative inside."""
    dx, dy = x - obstacle.center[0], y - obstacle.center[1]
    if obstacle.shape == 'cylinder':
        return math.hypot(dx, dy) - obstacle.radius
    c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
    ox = abs(c * dx + s * dy) - obstacle.size[0] / 2
    oy = abs(-s * dx + c * dy) - obstacle.size[1] / 2
    return math.hypot(max(ox, 0.0), max(oy, 0.0)) + min(max(ox, oy), 0.0)


def crosses(p, q, a, b):
    """Whether segment p-q crosses segment a-b."""
    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    return side(a, b, p) * side(a, b, q) <= 0.0 and side(p, q, a) * side(p, q, b) <= 0.0


class Referee(Node):
    def __init__(self):
        super().__init__('referee')
        course = load_course(self.declare_parameter('course', '').value)
        self.contact_distance = self.declare_parameter('contact_distance', 0.2).value
        self.obstacles, self.gates = course.obstacles, course.gates
        self.first = self.prev = self.start_time = None
        self.next_gate = self.contacts = 0
        self.touching = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.result_pub = self.create_publisher(String, 'referee/result', latched)
        self.create_subscription(Odometry, 'ground_truth', self.on_ground_truth, 10)

    def on_ground_truth(self, msg):
        if self.next_gate == len(self.gates):
            return
        p = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.first is None:
            self.first = self.prev = p
            return
        if self.start_time is None:
            if math.dist(p, self.first) < 0.05:
                return
            self.start_time = t
            self.get_logger().info('Clock started')
        elapsed = t - self.start_time

        touching = min(distance_to(o, *p) for o in self.obstacles) < self.contact_distance
        if touching and not self.touching:
            self.contacts += 1
            self.get_logger().warn(f'Contact {self.contacts} at ({p[0]:.2f}, {p[1]:.2f})')
        self.touching = touching

        gate = self.gates[self.next_gate]
        forward = (p[0] - self.prev[0]) * math.cos(gate.heading) + (p[1] - self.prev[1]) * math.sin(gate.heading) > 0.0
        if forward and crosses(self.prev, p, gate.start, gate.end):
            self.next_gate += 1
            self.get_logger().info(f'Gate {self.next_gate}/{len(self.gates)} {gate.name}: {elapsed:.2f} s')
            if self.next_gate == len(self.gates):
                result = f'Finished in {elapsed:.2f} s with {self.contacts} contacts'
                self.get_logger().info(result)
                self.result_pub.publish(String(data=result))
        self.prev = p


def main():
    rclpy.init()
    node = Referee()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
