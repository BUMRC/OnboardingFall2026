"""Helpers shared by the algorithms: angles, 2D poses and occupancy grids.

Provided. Poses are (x, y, yaw) in meters and radians.
"""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def wrap(angle):
    """Wrap an angle, or an array of angles, to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """Pose b, given relative to pose a, expressed in a's parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], float(wrap(a[2] + b[2]))])


def inverse(a):
    """The pose that undoes a: compose(a, inverse(a)) is (0, 0, 0)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


class Grid:
    """An occupancy grid as numpy arrays.

    occupied[row, col] is True for obstacle cells. Rows go up in y and columns go up in x, the same
    layout as nav_msgs/OccupancyGrid. origin is the world (x, y) of the grid's lower-left corner.
    distance[row, col] is the distance in meters from the cell to the nearest obstacle.
    """

    def __init__(self, occupied, resolution, origin):
        self.occupied = np.asarray(occupied, dtype=bool)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.height, self.width = self.occupied.shape
        self.distance = distance_transform_edt(~self.occupied) * self.resolution

    def to_cell(self, x, y):
        """World coordinates to (row, col) indices. Works on arrays."""
        col = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return row, col

    def to_world(self, row, col):
        """Cell indices to the world coordinates of the cell centers. Works on arrays."""
        x = self.origin[0] + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(row) + 0.5) * self.resolution
        return x, y

    def distance_at(self, x, y, outside=0.0):
        """Distance to the nearest obstacle at world points (any array shape); `outside` off the grid."""
        row, col = self.to_cell(x, y)
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        out = np.full(np.shape(row), outside, dtype=float)
        out[inside] = self.distance[row[inside], col[inside]]
        return out

    def downsample(self, factor):
        """A coarser grid where a cell is occupied if any of the cells it covers is."""
        h, w = self.height // factor, self.width // factor
        occupied = self.occupied[:h * factor, :w * factor].reshape(h, factor, w, factor).any(axis=(1, 3))
        return Grid(occupied, self.resolution * factor, self.origin)
