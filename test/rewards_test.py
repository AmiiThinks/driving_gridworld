import numpy as np
from driving_gridworld.rewards import Deterministic_Reward
from driving_gridworld.rewards import Stochastic_Reward
from driving_gridworld.rewards import \
    sample_reward_parameters, \
    r
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS, RIGHT, NO_OP, LEFT, UP, DOWN
import pytest


def test_all_entries_in_u_vector_are_greater_than_d_vector():
    hlr = 4
    u, C, d, H = sample_reward_parameters(hlr)
    for i in range(0, hlr+2):
        assert u[i] > d[i]


def test_all_entries_in_u_vector_are_greater_than_C_matrix_which_are_greater_than_H_matrix():
    hlr = 4
    u, C, d, H = sample_reward_parameters(hlr)
    for i in range(0, hlr+2):
        for j in range(0, hlr+1):
            assert u[i] > C[i,j] > H[i,j]


def test_all_entries_in_d_vector_are_greater_than_H_matrix():
    hlr = 4
    u, C, d, H = sample_reward_parameters(hlr)
    for i in range(0, hlr+2):
        for j in range(0, hlr+1):
            assert d[i] > H[i,j]


def test_hit_a_moving_obstacle_with_higher_speed_gives_lower_reward():
    hlr = 4
    u, C, d, H = sample_reward_parameters(hlr)
    for i in range(0, hlr+2):
        for j in range(1, hlr+1):
            assert C[i, j-1] > C[i, j]
            assert H[i, j-1] > H[i, j]


def test_collision_gives_lower_deterministic_reward_than_no_collision():
    np.random.seed(42)
    hlr = 4
    action = NO_OP
    car = Car(1, 1)
    u, C, d, H = sample_reward_parameters(hlr)
    det_rew_func = Deterministic_Reward(u, C, d, H)

    obst_list = [Bump(-1, 1, speed=4)]
    s = Road(hlr, car, obst_list)
    successors_list = list(s.successors(action))
    s_p = successors_list[0][0]
    reward_collision = det_rew_func(s, action, s_p)

    obst_list2 = [Bump(-1, 1, speed=1)]
    s2 = Road(hlr, car, obst_list2)
    successors_list2 = list(s2.successors(action))
    s2_p = successors_list2[0][0]
    reward_no_collision = det_rew_func(s2, action, s2_p)
    assert reward_collision < reward_no_collision


def test_collision_gives_lower_stochastic_reward_than_no_collision():
    np.random.seed(42)
    hlr = 4
    action = NO_OP
    car = Car(1, 1)
    sto_rew_func = Stochastic_Reward()

    obst_list = [Bump(-1, 1, speed=4)]
    s = Road(hlr, car, obst_list)
    successors_list = list(s.successors(action))
    s_p = successors_list[0][0]
    reward_collision = sto_rew_func(s, action, s_p)

    obst_list2 = [Bump(-1, 1)]
    s2 = Road(hlr, car, obst_list2)
    successors_list2 = list(s2.successors(action))
    s2_p = successors_list2[0][0]
    np.random.seed(42)
    reward_no_collision = sto_rew_func(s2, action, s2_p)
    assert reward_collision < reward_no_collision


def test_hit_more_obstacles_gives_lower_deterministic_reward():
    np.random.seed(42)
    hlr = 4
    obst_list = [Bump(-1, 1, speed=4), Bump(-1, 1, speed=4), Bump(-1, 1, speed=4)]
    action = NO_OP
    car = Car(1, 1)
    u, C, d, H = sample_reward_parameters(hlr)
    det_rew_func = Deterministic_Reward(u, C, d, H)

    reward_less_collisions = 0
    for i in range(1, len(obst_list)):
        np.random.seed(42)
        s = Road(hlr, car, obst_list[0:i])
        successors_list = list(s.successors(action))
        s_p = successors_list[0][0]
        reward_more_collisions = det_rew_func(s, action, s_p)
        assert reward_less_collisions > reward_more_collisions
        reward_less_collisions = reward_more_collisions


def test_hit_more_obstacles_gives_lower_stochastic_reward():
    np.random.seed(42)
    hlr = 4
    obst_list = [Bump(-1, 1, speed=4), Bump(-1, 1, speed=4), Bump(-1, 1, speed=4)]
    action = NO_OP
    car = Car(1, 1)
    sto_rew_func = Stochastic_Reward()

    reward_less_collisions = 0
    for i in range(1, len(obst_list)):
        np.random.seed(42)
        s = Road(hlr, car, obst_list[0:i])
        successors_list = list(s.successors(action))
        s_p = successors_list[0][0]
        reward_more_collisions = sto_rew_func(s, action, s_p)
        assert reward_less_collisions > reward_more_collisions
        reward_less_collisions = reward_more_collisions


def test_driving_faster_no_obstacles_gives_larger_reward():
    np.random.seed(42)
    headlight_range = 4
    action =  NO_OP
    u, c, d, h = sample_reward_parameters(headlight_range)
    det_rew_list = list()
    sto_rew_list = list()
    det_rew_func = Deterministic_Reward(u, c, d, h)
    sto_rew_func = Stochastic_Reward()

    for current_speed in range(0, headlight_range + 2):
        np.random.seed(42)
        s = Road(headlight_range, Car(1, current_speed))
        successors_list = list(s.successors(action))
        s_p = successors_list[0][0]
        det_r = det_rew_func(s, action, s_p)
        det_rew_list.append(det_r)
        sto_r = sto_rew_func(s, action, s_p)
        sto_rew_list.append(sto_r)

    for i in range(1, headlight_range + 2):
        assert det_rew_list[i-1] < det_rew_list[i]
        assert sto_rew_list[i-1] < sto_rew_list[i]


def test_deterministic_reward_is_negative_one_when_car_hits_moving_pedestrian():
    np.random.seed(42)
    hlr = 4
    obst_list = [Pedestrian(-1, 1, speed=4)]
    action = NO_OP
    car = Car(1, 1)
    s = Road(hlr, car, obst_list)
    successors_list = list(s.successors(action))
    s_p = successors_list[0][0]

    u, c, d, h = sample_reward_parameters(hlr)
    det_rew_func = Deterministic_Reward(u, c, d, h)
    bias = det_rew_func.bias
    reward = det_rew_func(s, action, s_p)
    assert reward == -1.0 + bias


def test_stochastic_reward_is_negative_one_when_car_hits_moving_pedestrian():
    np.random.seed(42)
    hlr = 4
    obst_list = [Pedestrian(-1, 1, speed=4)]
    action = NO_OP
    car = Car(1, 1)
    s = Road(hlr, car, obst_list)
    successors_list = list(s.successors(action))
    s_p = successors_list[0][0]

    sto_rew_func = Stochastic_Reward()
    bias = sto_rew_func.bias
    reward = sto_rew_func(s, action, s_p)
    assert reward == -1.0 + bias
