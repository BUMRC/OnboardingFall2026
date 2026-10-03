"""Publishes the course for planners and RViz.

/map            prior map: walls and known obstacles (what the rover is given)
/map_truth      every obstacle, including ones missing from the prior map (for debugging)
/course/gates   gate markers for RViz
/course/waypoints  gate centers in order, oriented along the direction of travel
All are latched (transient local), so nodes that start later still receive them.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from rover_sim.course import load_course, rasterize

GATE_COLOR = (1.0, 0.85, 0.1)
FINISH_COLOR = (0.1, 0.8, 0.2)


class CoursePublisher(Node):
    def __init__(self):
        super().__init__('course_publisher')
        course = load_course(self.declare_parameter('course', '').value)
        resolution = self.declare_parameter('resolution', 0.05).value
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.map_pub = self.create_publisher(OccupancyGrid, 'map', latched)
        self.truth_pub = self.create_publisher(OccupancyGrid, 'map_truth', latched)
        self.gates_pub = self.create_publisher(MarkerArray, 'course/gates', latched)
        self.waypoints_pub = self.create_publisher(PoseArray, 'course/waypoints', latched)

        self.map_pub.publish(self.grid_msg(*rasterize(course, resolution, False), resolution))
        self.truth_pub.publish(self.grid_msg(*rasterize(course, resolution, True), resolution))
        self.gates_pub.publish(self.gate_markers(course))
        self.waypoints_pub.publish(self.waypoints(course))
        self.get_logger().info(f'Published course "{course.name}" with {len(course.gates)} gates')

    @staticmethod
    def grid_msg(grid, origin, resolution):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = resolution
        msg.info.height, msg.info.width = grid.shape
        msg.info.origin.position.x, msg.info.origin.position.y = origin
        msg.info.origin.orientation.w = 1.0
        msg.data = (grid.astype(np.int8) * 100).ravel().tolist()
        return msg

    @staticmethod
    def gate_markers(course):
        markers = MarkerArray()
        for i, gate in enumerate(course.gates):
            color = FINISH_COLOR if i == len(course.gates) - 1 else GATE_COLOR

            line = Marker(type=Marker.LINE_STRIP, ns='gates', id=i)
            line.header.frame_id = 'map'
            line.pose.orientation.w = 1.0
            line.scale.x = 0.08
            line.color.r, line.color.g, line.color.b, line.color.a = *color, 1.0
            line.points = [Point(x=gate.start[0], y=gate.start[1]), Point(x=gate.end[0], y=gate.end[1])]

            label = Marker(type=Marker.TEXT_VIEW_FACING, ns='gate_names', id=i)
            label.header.frame_id = 'map'
            label.pose.position.x, label.pose.position.y, label.pose.position.z = *gate.center, 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.4
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f'{i + 1}. {gate.name}'

            markers.markers += [line, label]
        return markers

    @staticmethod
    def waypoints(course):
        msg = PoseArray()
        msg.header.frame_id = 'map'
        for gate in course.gates:
            pose = Pose()
            pose.position.x, pose.position.y = gate.center
            pose.orientation.z = math.sin(gate.heading / 2)
            pose.orientation.w = math.cos(gate.heading / 2)
            msg.poses.append(pose)
        return msg


def main():
    rclpy.init()
    node = CoursePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()