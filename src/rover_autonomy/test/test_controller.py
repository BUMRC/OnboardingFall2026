"""Tests for controller.py. From src/rover_autonomy: python3 -m pytest test/test_controller.py"""
import math

import numpy as np
import pytest

from rover_autonomy import controller


def constant(v, w, k=1):
    controls = np.zeros((k, controller.HORIZON, 2))
    controls[:, :, 0], controls[:, :, 1] = v, w
    return controls


def test_rollout_shape_and_standing_still():
    poses = controller.rollout((1.0, 2.0, 0.5, 0.0, 0.0), constant(0.0, 0.0, k=5))
    assert poses.shape == (5, controller.HORIZON, 3)
    np.testing.assert_allclose(poses[:, -1], [[1.0, 2.0, 0.5]] * 5)


def test_rollout_respects_acceleration():
    # From rest, v climbs 0.1 m/s per step to 1.0, then holds: 0.55 m in the first 10 steps, 0.1 m per step after.
    poses = controller.rollout((0.0, 0.0, 0.0, 0.0, 0.0), constant(1.0, 0.0))
    assert poses[0, 9, 0] == pytest.approx(0.55)
    assert poses[0, -1, 0] == pytest.approx(0.55 + 0.1 * (controller.HORIZON - 10))
    assert np.allclose(poses[0, :, 1], 0.0)


def test_rollout_turns_in_place():
    # w climbs 0.4 rad/s per step: 0.4, 0.8, then 1.0 from the third step on.
    poses = controller.rollout((0.0, 0.0, 0.0, 0.0, 0.0), constant(0.0, 1.0))
    assert poses[0, -1, 2] == pytest.approx(0.1 * (0.4 + 0.8 + 1.0 * (controller.HORIZON - 2)))
    assert np.allclose(poses[0, :, :2], 0.0)


def test_rollout_starts_from_the_current_speed():
    poses = controller.rollout((0.0, 0.0, math.pi / 2, 1.0, 0.0), constant(1.0, 0.0))
    assert poses[0, 0, 1] == pytest.approx(0.1)
    assert abs(poses[0, 0, 0]) < 1e-9


# A straight path along the x axis from the rover, and one obstacle point at (2, 0.6).
def path_lookup(points):
    return np.abs(points[:, 1]), np.clip(points[:, 0], 0.0, None)


def obstacle_distance(points):
    return np.hypot(points[:, 0] - 2.0, points[:, 1] - 0.6)


def line(x0, x1, y0=0.0, y1=0.0):
    t = np.linspace(0.0, 1.0, controller.HORIZON)
    return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, np.zeros_like(t)], axis=1)


def test_cost_rewards_progress():
    costs = controller.trajectory_cost(np.stack([line(0, 3, -0.3, -0.3), line(0, 0, -0.3, -0.3)]),
                                       path_lookup, obstacle_distance)
    assert costs.shape == (2,)
    assert costs[0] < costs[1]


def test_cost_penalizes_leaving_the_path():
    costs = controller.trajectory_cost(np.stack([line(0, 1, -0.2, -0.2), line(0, 1, -0.7, -0.7)]),
                                       path_lookup, obstacle_distance)
    assert costs[0] < costs[1]


def test_cost_penalizes_collisions():
    # The second trajectory drives through the obstacle point.
    costs = controller.trajectory_cost(np.stack([line(0, 3, -0.3, -0.3), line(0, 3, 0.6, 0.6)]),
                                       path_lookup, obstacle_distance)
    assert costs[0] < costs[1]


def test_cutting_far_from_the_path_earns_no_extra_progress():
    # Both end 3 m along; the second gets there by going 2 m off the path.
    detour = np.concatenate([line(0, 1.5, -0.3, -2.0)[:15], line(1.5, 3, -2.0, -0.3)[15:]])
    costs = controller.trajectory_cost(np.stack([line(0, 3, -0.3, -0.3), detour]), path_lookup, obstacle_distance)
    assert costs[0] < costs[1]


def test_weights_sum_to_one_and_favor_low_cost():
    w = controller.mppi_weights(np.array([3.0, 1.0, 2.0]))
    assert w.sum() == pytest.approx(1.0)
    assert np.argmax(w) == 1 and w[1] > w[2] > w[0]


def test_weights_handle_large_costs():
    w = controller.mppi_weights(np.array([5000.0, 5001.0]))
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0) and w[0] > w[1]


def test_equal_costs_get_equal_weights():
    np.testing.assert_allclose(controller.mppi_weights(np.full(4, 7.0)), 0.25)
