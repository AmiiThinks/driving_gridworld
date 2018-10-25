import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from driving_gridworld.rewards import \
    SituationalReward, \
    TfSituationalReward, \
    WcSituationalReward, \
    BcSituationalReward
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


def wc_non_critical_error_reward():
    return -1.0


def stopping_reward():
    return 0.0


def new_road(*obstacles, speed=headlight_range(), car_col=1):
    return Road(headlight_range(), Car(car_col, speed=speed), obstacles)


def situational_reward_function(cls, critical_error_reward=-1):
    return cls(
        wc_non_critical_error_reward(),
        stopping_reward(),
        critical_error_reward=critical_error_reward)


def unshifted_worst_case_determinstic_reward_function(
        critical_error_reward=-1.0):
    return DeterministicReward.worst_case_reward_unshifted(
        headlight_range() + 1, critical_error_reward=critical_error_reward)


def unshifted_best_case_determinstic_reward_function(
        critical_error_reward=-1.0):
    return DeterministicReward.best_case_reward_unshifted(
        headlight_range() + 1, critical_error_reward=critical_error_reward)


def unshifted_average_case_determinstic_reward_function(
        critical_error_reward=-1.0):
    return DeterministicReward.average_reward_unshifted(
        headlight_range() + 1, critical_error_reward=critical_error_reward)


all_situational_reward_function_constructors = list(map(
    lambda cls: lambda *args, **kwargs: situational_reward_function(cls, *args, **kwargs), [
        SituationalReward, TfSituationalReward,
        WcSituationalReward, BcSituationalReward
    ]
))


@pytest.mark.parametrize("new_reward_function",
                         all_situational_reward_function_constructors)
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
                         all_situational_reward_function_constructors)
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
                         all_situational_reward_function_constructors)
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
                         all_situational_reward_function_constructors)
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
                         all_situational_reward_function_constructors)
def test_reward_is_minimal_when_car_hits_moving_pedestrian(
        new_reward_function):
    np.random.seed(42)
    patient = new_reward_function()

    s = new_road(Pedestrian(-1, 1, speed=0))
    sp = new_road(Pedestrian(headlight_range(), 1, speed=0))

    assert patient(s, NO_OP, sp) == patient.critical_error_reward


@pytest.mark.parametrize("columns", [(0, -1), (3, 4)])
@pytest.mark.parametrize("new_reward_function",
                         all_situational_reward_function_constructors)
@pytest.mark.parametrize("action", ACTIONS)
def test_crashing_into_a_wall(columns, new_reward_function, action):
    np.random.seed(42)
    patient = new_reward_function()

    s = new_road(car_col=columns[0])
    sp = new_road(car_col=columns[1])

    assert not s.has_crashed()
    assert sp.has_crashed()

    assert patient(s, action, sp) == patient.critical_error_reward
    assert patient(sp, action, sp) == patient.critical_error_reward


@pytest.mark.parametrize("columns", [(0, -1), (3, 4)])
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("critical_reward",
                         [float(i) for i in range(-1, -11, -1)])
def test_min_reward_function(columns, action, critical_reward):
    np.random.seed(42)
    patient = unshifted_worst_case_determinstic_reward_function(
        critical_error_reward=critical_reward)
    bias = patient.critical_error_reward - critical_reward
    assert bias == 0.0

    s = new_road(car_col=columns[0])
    sp = new_road(car_col=columns[1])

    assert not s.has_crashed()
    assert sp.has_crashed()

    assert patient(s, action, sp) == patient.critical_error_reward
    assert patient(sp, action, sp) == patient.critical_error_reward


@pytest.mark.parametrize("new_reward_function", [
    unshifted_worst_case_determinstic_reward_function,
    unshifted_best_case_determinstic_reward_function,
    unshifted_average_case_determinstic_reward_function
])
def test_best_worst_and_average_case_reward_parameters(new_reward_function):
    patient = new_reward_function()
    u = patient.u
    C = patient.c
    d = patient.d
    H = patient.h
    assert len(u) == headlight_range() + 2 == len(d)
    assert C.shape == (headlight_range() + 2, headlight_range() + 1) == H.shape

    rows = range(0, headlight_range() + 2)
    columns = range(0, headlight_range() + 1)

    for i, j in product(rows, columns):
        assert u[i] > d[i]
        assert d[i] > H[i, j]
        assert u[i] > C[i, j]
        assert C[i, j] > H[i, j]
        if j > 0:
            assert C[i, j - 1] > C[i, j]
            assert H[i, j - 1] > H[i, j]


def test_two_samples_with_tf():
    tf.set_random_seed(42)

    patient = TfSituationalReward(
        wc_non_critical_error_reward=-np.ones([2]),
        stopping_reward=np.zeros([2]),
        critical_error_reward=np.full([2], -1000.0),
        bc_unobstructed_progress_reward=np.ones([2]),
        num_samples=2)
    s = new_road(car_col=0)
    rewards = patient(s, NO_OP, s).numpy()
    assert rewards[0] == pytest.approx(1.1475897)
    assert rewards[1] == pytest.approx(0.9217568)
