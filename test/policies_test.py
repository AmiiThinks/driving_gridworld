import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.policies import hand_coded_data_gathering_policy
from driving_gridworld.actions import UP, DOWN, RIGHT, LEFT, NO_OP
import pytest


def test_pedestrian_in_front_of_car():
    obstacles = [Pedestrian(3, 2)]
    headlight_range = 4
    road = Road(headlight_range, Car(2, 1), obstacles)
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == LEFT


def test_policy_if_overdrive_headlights():
    headlight_range = 4
    road = Road(headlight_range, Car(2, headlight_range + 1), [])
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == DOWN


def test_single_bump_in_front_of_car():
    obstacles = [Bump(3, 2)]
    headlight_range = 4
    road = Road(headlight_range, Car(2, 1), obstacles)
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == LEFT


def test_car_speeds_up_if_obstacle_not_visible():
    headlight_range = 4
    obstacles = [Bump(headlight_range + 1, 2)]
    road = Road(headlight_range, Car(2, 2), obstacles)
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == UP


def test_two_adjacent_bumps():
    headlight_range = 4
    obstacles = [Bump(0, 1), Bump(0, 2)]
    road = Road(headlight_range, Car(2, 2), obstacles)
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == DOWN


def test_car_offroad():
    headlight_range = 4
    road1 = Road(headlight_range, Car(0, 1), [])
    action_taken1 = hand_coded_data_gathering_policy(road1)
    assert action_taken1 == RIGHT

    road2 = Road(headlight_range, Car(3, 1), [])
    action_taken2 = hand_coded_data_gathering_policy(road2)
    assert action_taken2 == LEFT


def test_no_obstacle_in_front_of_car():
    headlight_range = 4
    road = Road(headlight_range, Car(1, 1), [])
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == UP
