"""Course definitions: load a course YAML, build its Gazebo world, rasterize its maps."""
import math
from dataclasses import dataclass

import numpy as np
import yaml

WALL_THICKNESS = 0.2
WALL_COLOR = (0.75, 0.72, 0.68)
ROCK_COLOR = (0.45, 0.30, 0.25)
GROUND_COLOR = (0.80, 0.55, 0.40)


@dataclass
class Obstacle:
    shape: str
    center: tuple
    size: tuple = (0.0, 0.0)
    radius: float = 0.0
    height: float = 1.0
    yaw: float = 0.0
    known: bool = True
    color: tuple = None


@dataclass
class Gate:
    name: str
    start: tuple
    end: tuple
    heading: float

    @property
    def center(self):
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)


@dataclass
class Course:
    name: str
    width: float
    height: float
    wall_height: float
    start: tuple
    obstacles: list
    gates: list


def load_course(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    arena = data['arena']
    width, height = float(arena['width']), float(arena['height'])
    wall_height = float(arena.get('wall_height', 1.0))

    t = WALL_THICKNESS
    perimeter = [
        Obstacle('box', (width / 2, -t / 2), (width + 2 * t, t), height=wall_height),
        Obstacle('box', (width / 2, height + t / 2), (width + 2 * t, t), height=wall_height),
        Obstacle('box', (-t / 2, height / 2), (t, height), height=wall_height),
        Obstacle('box', (width + t / 2, height / 2), (t, height), height=wall_height),
    ]

    obstacles = []
    for o in data.get('obstacles', []):
        obstacles.append(Obstacle(
            shape=o['shape'],
            center=tuple(o['center']),
            size=tuple(o.get('size', (0.0, 0.0))),
            radius=float(o.get('radius', 0.0)),
            height=float(o.get('height', wall_height)),
            yaw=float(o.get('yaw', 0.0)),
            known=bool(o.get('known', True)),
            color=tuple(o['color']) if 'color' in o else None,
        ))

    gates = [Gate(g['name'], tuple(g['from']), tuple(g['to']), float(g['heading']))
             for g in data.get('gates', [])]

    return Course(data['name'], width, height, wall_height,
                  tuple(data['start']), perimeter + obstacles, gates)


def rasterize(course, resolution, include_unknown):
    """Occupancy grid covering the arena and its walls: True where a cell center is inside an obstacle.

    Returns (grid, origin). grid[row, col] has row = y index and col = x index, the layout
    nav_msgs/OccupancyGrid uses. origin is the (x, y) of the grid's lower-left corner.
    """
    t = WALL_THICKNESS
    origin = (-t, -t)
    nx = int(round((course.width + 2 * t) / resolution))
    ny = int(round((course.height + 2 * t) / resolution))
    xs = origin[0] + (np.arange(nx) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)

    grid = np.zeros((ny, nx), dtype=bool)
    for o in course.obstacles:
        if not o.known and not include_unknown:
            continue
        dx, dy = X - o.center[0], Y - o.center[1]
        if o.shape == 'cylinder':
            grid |= dx ** 2 + dy ** 2 <= o.radius ** 2
        else:
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            local_x, local_y = c * dx + s * dy, -s * dx + c * dy
            grid |= (np.abs(local_x) <= o.size[0] / 2) & (np.abs(local_y) <= o.size[1] / 2)
    return grid, origin


def _material(rgb):
    r, g, b = rgb
    return (f'<material><ambient>{r} {g} {b} 1</ambient>'
            f'<diffuse>{r} {g} {b} 1</diffuse></material>')


def _obstacle_model(index, o):
    if o.shape == 'cylinder':
        geometry = f'<cylinder><radius>{o.radius}</radius><length>{o.height}</length></cylinder>'
        color = o.color or ROCK_COLOR
    else:
        geometry = f'<box><size>{o.size[0]} {o.size[1]} {o.height}</size></box>'
        color = o.color or WALL_COLOR
    return f"""
    <model name="obstacle_{index}">
      <static>true</static>
      <pose>{o.center[0]} {o.center[1]} {o.height / 2} 0 0 {o.yaw}</pose>
      <link name="link">
        <collision name="collision"><geometry>{geometry}</geometry></collision>
        <visual name="visual"><geometry>{geometry}</geometry>{_material(color)}</visual>
      </link>
    </model>"""


def _gate_model(index, g, is_finish):
    length = math.dist(g.start, g.end)
    yaw = math.atan2(g.end[1] - g.start[1], g.end[0] - g.start[0])
    color = (0.1, 0.8, 0.2) if is_finish else (1.0, 0.85, 0.1)
    return f"""
    <model name="gate_{index}">
      <static>true</static>
      <pose>{g.center[0]} {g.center[1]} 0.005 0 0 {yaw}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{length} 0.1 0.01</size></box></geometry>
          {_material(color)}
        </visual>
      </link>
    </model>"""


def world_sdf(course):
    obstacles = ''.join(_obstacle_model(i, o) for i, o in enumerate(course.obstacles))
    gates = ''.join(_gate_model(i, g, i == len(course.gates) - 1) for i, g in enumerate(course.gates))
    return f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="{course.name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.85 0.75 0.65 1</background>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.85 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>

    <model name="ground">
      <static>true</static>
      <pose>{course.width / 2} {course.height / 2} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{course.width + 20} {course.height + 20}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{course.width + 20} {course.height + 20}</size></plane></geometry>
          {_material(GROUND_COLOR)}
        </visual>
      </link>
    </model>
{obstacles}
{gates}
  </world>
</sdf>
"""