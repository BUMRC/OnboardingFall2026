"""Tests for mcl.py. From src/rover_autonomy: python3 -m pytest test/test_mcl.py"""
import math

import numpy as np
import pytest

from rover_autonomy import mcl
from rover_autonomy.geometry import Grid

ROOM = 6.0   # a square room, walls 0.2 m thick, open space from 0.2 to 5.8
WALL = 0.2


def room_grid(resolution=0.05):
    n = int(round(ROOM / resolution))
    wall = int(round(WALL / resolution))
    occupied = np.zeros((n, n), dtype=bool)
    occupied[:wall, :] = occupied[-wall:, :] = occupied[:, :wall] = occupied[:, -wall:] = True
    return Grid(occupied, resolution, (0.0, 0.0))


def room_scan(pose, angles):
    """Exact ranges from a laser at pose to the room's inner walls."""
    lo, hi = WALL, ROOM - WALL
    ranges = []
    for a in angles:
        dx, dy = math.cos(pose[2] + a), math.sin(pose[2] + a)
        hits = [(w - pose[0]) / dx for w in (lo, hi) if abs(dx) > 1e-9] + \
               [(w - pose[1]) / dy for w in (lo, hi) if abs(dy) > 1e-9]
        ranges.append(min(t for t in hits if t > 0))
    return np.array(ranges)


@pytest.fixture
def no_noise(monkeypatch):
    monkeypatch.setattr(mcl, 'ALPHAS', (0.0, 0.0, 0.0, 0.0))


def test_motion_is_replayed_from_each_particles_heading(no_noise):
    particles = np.array([[5.0, 5.0, 0.0], [5.0, 5.0, math.pi / 2]])
    moved = mcl.sample_motion_model(particles, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.random.default_rng(0))
    np.testing.assert_allclose(moved[0], [6.0, 5.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(moved[1], [5.0, 6.0, math.pi / 2], atol=1e-9)


def test_motion_ignores_where_the_odom_frame_is(no_noise):
    # The same 1 m forward drive, measured in an odom frame that's shifted and rotated.
    particles = np.array([[5.0, 5.0, 0.0]])
    moved = mcl.sample_motion_model(particles, (3.0, -2.0, math.pi / 2), (3.0, -1.0, math.pi / 2),
                                    np.random.default_rng(0))
    np.testing.assert_allclose(moved[0], [6.0, 5.0, 0.0], atol=1e-9)


def test_turning_in_place_does_not_move_the_particles(no_noise):
    particles = np.array([[1.0, 1.0, 0.0]])
    moved = mcl.sample_motion_model(particles, (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2), np.random.default_rng(0))
    np.testing.assert_allclose(moved[0], [1.0, 1.0, math.pi / 2], atol=1e-9)


def test_driving_backward(no_noise):
    particles = np.array([[5.0, 5.0, 0.0]])
    moved = mcl.sample_motion_model(particles, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), np.random.default_rng(0))
    np.testing.assert_allclose(moved[0, :2], [4.0, 5.0], atol=1e-9)
    assert abs(math.remainder(moved[0, 2], 2 * math.pi)) < 1e-9


def test_motion_noise_spreads_the_particles():
    particles = np.zeros((5000, 3))
    moved = mcl.sample_motion_model(particles, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.random.default_rng(0))
    assert moved.shape == (5000, 3)
    assert abs(moved[:, 0].mean() - 1.0) < 0.05
    assert 0.02 < moved[:, 0].std() < 0.5 and 0.02 < moved[:, 1].std() < 0.5
    assert np.all((moved[:, 2] >= -math.pi) & (moved[:, 2] < math.pi))


def test_true_pose_scores_highest():
    grid = room_grid()
    true_pose = (2.0, 3.0, 0.3)
    angles = np.linspace(-math.pi, math.pi, 60, endpoint=False)
    ranges = room_scan(true_pose, angles)
    particles = np.array([true_pose, (2.3, 3.0, 0.3), (2.0, 2.7, 0.3), (2.0, 3.0, 0.5)])
    scores = mcl.scan_log_likelihood(particles, ranges, angles, grid, (0.0, 0.0, 0.0))
    assert scores.shape == (4,)
    assert np.argmax(scores) == 0


def test_laser_offset_is_used():
    grid = room_grid()
    rover = (2.0, 3.0, math.pi / 2)
    laser = (2.0, 3.1, math.pi / 2)  # the laser 0.1 m ahead of the rover, which faces +y
    angles = np.linspace(-math.pi, math.pi, 60, endpoint=False)
    ranges = room_scan(laser, angles)
    particles = np.array([rover, laser])
    scores = mcl.scan_log_likelihood(particles, ranges, angles, grid, (0.1, 0.0, 0.0))
    assert scores[0] > scores[1]


def test_beams_ending_off_the_map_get_the_lowest_score():
    grid = room_grid()
    angles = np.linspace(-math.pi, math.pi, 20, endpoint=False)
    ranges = np.full(20, 1.0)
    scores = mcl.scan_log_likelihood(np.array([[-50.0, -50.0, 0.0]]), ranges, angles, grid, (0.0, 0.0, 0.0))
    assert scores[0] == pytest.approx(20 * math.log(mcl.Z_RAND / mcl.MAX_RANGE), rel=1e-3)


def test_resample_copies_heavy_particles():
    particles = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    for seed in range(20):
        out = mcl.low_variance_resample(particles, np.array([0.5, 0.25, 0.25, 0.0]), np.random.default_rng(seed))
        assert out.shape == (4, 3)
        assert sorted(out[:, 0]) == [0.0, 0.0, 1.0, 2.0]


def test_resample_with_equal_weights_keeps_every_particle():
    particles = np.random.default_rng(1).normal(size=(100, 3))
    out = mcl.low_variance_resample(particles, np.full(100, 0.01), np.random.default_rng(2))
    np.testing.assert_allclose(np.sort(out[:, 0]), np.sort(particles[:, 0]))


def test_estimate_is_the_weighted_mean():
    particles = np.array([[0.0, 0.0, 0.0], [4.0, 8.0, 0.0]])
    np.testing.assert_allclose(mcl.estimate_pose(particles, np.array([0.75, 0.25])), [1.0, 2.0, 0.0], atol=1e-9)


def test_estimate_averages_yaw_as_an_angle():
    particles = np.array([[0.0, 0.0, math.radians(179)], [0.0, 0.0, math.radians(-179)]])
    yaw = mcl.estimate_pose(particles, np.array([0.5, 0.5]))[2]
    assert abs(abs(yaw) - math.pi) < 1e-6
