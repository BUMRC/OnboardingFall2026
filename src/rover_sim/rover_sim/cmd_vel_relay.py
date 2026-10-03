"""Forwards /cmd_vel to the simulator, and stops the rover when commands stop arriving.

Gazebo's drive plugin keeps executing the last command it got, so without this a crashed
controller leaves the rover driving into a wall.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class CmdVelRelay(Node):
    def __init__(self):
        super().__init__('cmd_vel_relay')
        self.timeout = self.declare_parameter('timeout', 1.0).value
        self.pub = self.create_publisher(Twist, 'sim/cmd_vel', 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.last = None
        self.create_timer(0.05, self.check_timeout)

    def on_cmd_vel(self, msg):
        self.pub.publish(msg)
        self.last = self.get_clock().now()

    def check_timeout(self):
        if self.last is None:
            return
        if (self.get_clock().now() - self.last).nanoseconds * 1e-9 > self.timeout:
            self.pub.publish(Twist())
            self.last = None


def main():
    rclpy.init()
    node = CmdVelRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()