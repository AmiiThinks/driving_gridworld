import numpy as np
from driving_gridworld.rewards import DeterministicReward
from driving_gridworld.rewards import StochasticReward
from driving_gridworld.rewards import \
    sample_reward_parameters, \
    r
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS, RIGHT, NO_OP, LEFT, UP, DOWN
import pytest


def driving_conditions():
    hlr = 4
    action = NO_OP
    car = Car(1, 1)
    return hlr, action, car


def test_sample_reward_parameters():
    hlr = driving_conditions()[0]
    u, C, d, H = sample_reward_parameters(hlr)
    for i in range(0, hlr + 2):
        assert u[i] > d[i]
        assert u[i] > C[i, 0] > H[i, 0]
        assert d[i] > H[i, 0]
        for j in range(1, hlr + 1):
            assert u[i] > C[i, j] > H[i, j]
            assert d[i] > H[i, j]
            assert C[i, j - 1] > C[i, j]
            assert H[i, j - 1] > H[i, j]


def test_collision_gives_lower_reward_than_no_collision():
    np.random.seed(42)
    hlr, action, car = driving_conditions()
    u, C, d, H = sample_reward_parameters(hlr)
    det_rew_func = DeterministicReward(u, C, d, H)
    sto_rew_func = StochasticReward()

    obst_list = [Bump(-1, 1, speed=4), Bump(-1, 1)]
    det_rew_list = list()
    sto_rew_list = list()
    for obst in obst_list:
        np.random.seed(42)
        s = Road(hlr, car, [obst])
        s_p = list(s.successors(action))[0][0]
        det_reward = det_rew_func(s, action, s_p)
        det_rew_list.append(det_reward)
        sto_reward = sto_rew_func(s, action, s_p)
        sto_rew_list.append(sto_reward)
    assert len(det_rew_list) == len(sto_rew_list) == 2
    assert det_rew_list[0] < det_rew_list[1]
    assert sto_rew_list[0] < sto_rew_list[1]


def test_hitting_more_obstacles_gives_lower_reward():
    np.random.seed(42)
    hlr, action, car = driving_conditions()
    obst_list = [Bump(-1, 1, speed=4)]*3
    u, C, d, H = sample_reward_parameters(hlr)
    det_rew_func = DeterministicReward(u, C, d, H)
    sto_rew_func = StochasticReward()

    det_reward_for_less_collisions = 0
    sto_reward_for_less_collisions = 0
    for i in range(1, len(obst_list)+1):
        np.random.seed(42)
        s = Road(hlr, car, obst_list[0:i])
        s_p = list(s.successors(action))[0][0]

        reward_for_more_collisions = det_rew_func(s, action, s_p)
        assert det_reward_for_less_collisions > reward_for_more_collisions
        det_reward_for_less_collisions = reward_for_more_collisions

        reward_for_more_collisions = sto_rew_func(s, action, s_p)
        assert sto_reward_for_less_collisions > reward_for_more_collisions
        sto_reward_for_less_collisions = reward_for_more_collisions


def test_driving_faster_with_no_obstacles_gives_larger_reward():
    hlr, action, car = driving_conditions()
    u, C, d, H = sample_reward_parameters(hlr)
    det_rew_list = list()
    sto_rew_list = list()
    det_rew_func = DeterministicReward(u, C, d, H)
    sto_rew_func = StochasticReward()

    for current_speed in range(0, hlr + 2):
        np.random.seed(42)
        s = Road(hlr, Car(1, current_speed))
        s_p = list(s.successors(action))[0][0]
        det_r = det_rew_func(s, action, s_p)
        det_rew_list.append(det_r)
        sto_r = sto_rew_func(s, action, s_p)
        sto_rew_list.append(sto_r)

    for i in range(1, hlr + 2):
        assert det_rew_list[i - 1] < det_rew_list[i]
        assert sto_rew_list[i - 1] < sto_rew_list[i]


def test_reward_is_negative_one_when_car_hits_moving_pedestrian():
    np.random.seed(42)
    hlr, action, car = driving_conditions()
    obst_list = [Pedestrian(-1, 1, speed=4)]
    s = Road(hlr, car, obst_list)
    s_p = list(s.successors(action))[0][0]

    u, c, d, h = sample_reward_parameters(hlr)
    det_rew_func = DeterministicReward(u, c, d, h)
    sto_rew_func = StochasticReward()
    det_reward = det_rew_func(s, action, s_p)
    sto_reward = sto_rew_func(s, action, s_p)
    bias_det = det_rew_func.bias
    bias_sto = sto_rew_func.bias

    assert det_reward == -1.0 + bias_det
    assert sto_reward == -1.0 + bias_sto
