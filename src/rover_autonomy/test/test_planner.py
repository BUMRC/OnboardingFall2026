"""Tests for planner.py. From src/rover_autonomy: python3 -m pytest test/test_planner.py"""
import math

import numpy as np

from rover_autonomy import planner


def path_cost(path, cost):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) * cost[b] for a, b in zip(path, path[1:]))


def check_valid(path, blocked, start, goal):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1, f'{a} -> {b} is not a single 8-connected step'
        assert not blocked[b], f'{b} is blocked'


def test_open_grid_goes_diagonal():
    blocked = np.zeros((10, 10), dtype=bool)
    cost = np.ones((10, 10))
    path = planner.astar(blocked, cost, (0, 0), (9, 9))
    check_valid(path, blocked, (0, 0), (9, 9))
    assert path_cost(path, cost) == pytest_approx(9 * math.sqrt(2))


def test_goes_through_the_gap_in_a_wall():
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[:, 5] = True
    blocked[8, 5] = False
    cost = np.ones((10, 10))
    path = planner.astar(blocked, cost, (0, 0), (0, 9))
    check_valid(path, blocked, (0, 0), (0, 9))
    assert (8, 5) in path


def test_unreachable_goal_returns_none():
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[:, 5] = True
    assert planner.astar(blocked, np.ones((10, 10)), (0, 0), (0, 9)) is None


def test_start_equals_goal():
    assert planner.astar(np.zeros((3, 3), dtype=bool), np.ones((3, 3)), (1, 1), (1, 1)) == [(1, 1)]


def test_detours_around_expensive_cells():
    # The middle row is expensive, so the cheapest path leaves it and steps back in only at the goal.
    blocked = np.zeros((3, 5), dtype=bool)
    cost = np.ones((3, 5))
    cost[1, :] = 10.0
    path = planner.astar(blocked, cost, (1, 0), (1, 4))
    check_valid(path, blocked, (1, 0), (1, 4))
    assert path_cost(path, cost) == pytest_approx(13 + math.sqrt(2))


def test_finds_the_cheapest_path_not_just_a_path():
    rng = np.random.default_rng(0)
    for _ in range(20):
        blocked = rng.random((15, 15)) < 0.25
        blocked[0, 0] = blocked[14, 14] = False
        cost = 1.0 + 4.0 * rng.random((15, 15))
        path = planner.astar(blocked, cost, (0, 0), (14, 14))
        best = dijkstra(blocked, cost, (0, 0), (14, 14))
        if best is None:
            assert path is None
        else:
            check_valid(path, blocked, (0, 0), (14, 14))
            assert path_cost(path, cost) == pytest_approx(best)


def dijkstra(blocked, cost, start, goal):
    import heapq
    dist = {start: 0.0}
    frontier = [(0.0, start)]
    while frontier:
        d, cell = heapq.heappop(frontier)
        if cell == goal:
            return d
        if d > dist[cell]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (cell[0] + dr, cell[1] + dc)
                if n == cell or not (0 <= n[0] < blocked.shape[0] and 0 <= n[1] < blocked.shape[1]) or blocked[n]:
                    continue
                nd = d + math.hypot(dr, dc) * cost[n]
                if nd < dist.get(n, math.inf):
                    dist[n] = nd
                    heapq.heappush(frontier, (nd, n))
    return None


def pytest_approx(x):
    import pytest
    return pytest.approx(x, rel=1e-9, abs=1e-9)
