import numpy as np
from driving_gridworld.rewards import DeterministicReward
from driving_gridworld.rewards import StochasticReward
from driving_gridworld.rewards import sample_reward_parameters
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import NO_OP, ACTIONS
import pytest
from itertools import product


def headlight_range():
    return 4


def new_road(*obstacles, speed=headlight_range(), car_col=1):
    return Road(headlight_range(), Car(car_col, speed=speed), obstacles)


def test_sample_reward_parameters():
    u, C, d, H = sample_reward_parameters(headlight_range() + 1)

    assert len(u) == headlight_range() + 2 == len(d)
    assert C.shape == (headlight_range() + 2, headlight_range() + 1) == H.shape

    rows = range(0, headlight_range() + 2)
    colums = range(0, headlight_range() + 1)
    for i, j in product(rows, colums):
        assert u[i] > d[i] > H[i, j]
        assert u[i] > C[i, j] > H[i, j]
        if j > 0:
            assert C[i, j - 1] > C[i, j]
            assert H[i, j - 1] > H[i, j]


def determinstic_reward_function():
    return DeterministicReward(
        *sample_reward_parameters(headlight_range() + 1))


@pytest.mark.parametrize("new_reward_function",
                         [determinstic_reward_function, StochasticReward])
def test_collision_is_worse_than_no_collision(new_reward_function):
    np.random.seed(42)
    _patient = new_reward_function()

    def patient(s, sp):
        np.random.seed(42)
        return _patient(s, NO_OP, sp)

    s1 = new_road(Bump(row=-1, col=1, speed=0))
    s1_p = new_road(Bump(row=headlight_range(), col=1, speed=0))
    s2 = new_road()
    s2_p = new_road()
    assert patient(s1, s1_p) < patient(s2, s2_p)


@pytest.mark.parametrize("new_reward_function",
                         [determinstic_reward_function, StochasticReward])
def test_collision_with_faster_obstacle_is_worse_than_one_with_a_slower_obstacle(
        new_reward_function):
    np.random.seed(42)
    _patient = new_reward_function()

    def patient(s, sp):
        np.random.seed(42)
        return _patient(s, NO_OP, sp)

    s1 = new_road(Bump(row=-1, col=1, speed=4))
    s1_p = new_road(Bump(row=headlight_range(), col=1, speed=4))
    s2 = new_road(Bump(row=-1, col=1, speed=3))
    s2_p = new_road(Bump(row=headlight_range(), col=1, speed=3))
    assert patient(s1, s1_p) < patient(s2, s2_p)


@pytest.mark.parametrize("new_reward_function",
                         [determinstic_reward_function, StochasticReward])
def test_more_collisions_are_worse(new_reward_function):
    np.random.seed(42)
    _patient = new_reward_function()

    def patient(s, sp):
        np.random.seed(42)
        return _patient(s, NO_OP, sp)

    s1 = new_road(*[Bump(row=-1, col=1, speed=0)] * 2)
    s1_p = new_road(*[Bump(row=headlight_range(), col=1, speed=0)] * 2)
    s2 = new_road(Bump(row=-1, col=1, speed=0))
    s2_p = new_road(Bump(row=headlight_range(), col=1, speed=0))
    assert patient(s1, s1_p) < patient(s2, s2_p)


@pytest.mark.parametrize("new_reward_function",
                         [determinstic_reward_function, StochasticReward])
def test_driving_faster_with_no_obstacles_gives_larger_reward(
        new_reward_function):
    np.random.seed(42)
    _patient = new_reward_function()

    def patient(s, sp):
        np.random.seed(42)
        return _patient(s, NO_OP, sp)

    for speed in range(headlight_range()):
        s1 = new_road(speed=speed)
        s2 = new_road(speed=speed + 1)
        assert patient(s1, s1) < patient(s2, s2)


@pytest.mark.parametrize("new_reward_function",
                         [determinstic_reward_function, StochasticReward])
def test_reward_is_minimal_when_car_hits_moving_pedestrian(
        new_reward_function):
    np.random.seed(42)
    patient = new_reward_function()

    s = new_road(Pedestrian(-1, 1, speed=0))
    sp = new_road(Pedestrian(headlight_range(), 1, speed=0))

    assert patient(s, NO_OP, sp) == patient.reward_for_critical_error


@pytest.mark.parametrize("columns", [(0, -1), (3, 4)])
@pytest.mark.parametrize("new_reward_function",
                         [determinstic_reward_function, StochasticReward])
@pytest.mark.parametrize("action", ACTIONS)
def test_crashing_into_a_wall(columns, new_reward_function, action):
    np.random.seed(42)
    patient = new_reward_function()

    s = new_road(car_col=columns[0])
    sp = new_road(car_col=columns[1])

    assert not s.has_crashed()
    assert sp.has_crashed()

    assert patient(s, action, sp) == patient.reward_for_critical_error
    assert patient(sp, action, sp) == patient.reward_for_critical_error
