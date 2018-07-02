import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.policies import \
    hand_coded_data_gathering_policy, \
    hand_coded_score_for_columns_adjacent_to_car, \
    hand_coded_obstacle_score
from driving_gridworld.actions import UP, DOWN, RIGHT, LEFT, NO_OP
import pytest


@pytest.mark.parametrize("obst_class", [Bump, Pedestrian])
@pytest.mark.parametrize("car_col", [0, 1, 2, 3])
def test_scores_obst_more_than_1_col_away(obst_class, car_col):
    headlight_range = 4

    for i in [2, 3]:
        obst_col = (car_col + i) % 5
        obst = obst_class(0, obst_col)
        road = Road(headlight_range, Car(car_col, 1), [obst])
        scores = hand_coded_score_for_columns_adjacent_to_car(road)

        if car_col == 0:
            assert scores == [-np.inf, -2, 0]
        elif car_col == 1:
            assert scores == [-2, 0, 0]
        elif car_col == 2:
            assert scores == [0, 0, -2]
        else:
            assert scores == [0, -2, -np.inf]


@pytest.mark.parametrize("obst_class", [Bump, Pedestrian])
@pytest.mark.parametrize("car_col", [0, 1, 2, 3])
def test_scores_obst_1_col_away_left(obst_class, car_col):
    headlight_range = 4
    obst = obst_class(0, car_col - 1)
    road = Road(headlight_range, Car(car_col, 1), [obst])
    score = hand_coded_obstacle_score(obst, road)
    scores = hand_coded_score_for_columns_adjacent_to_car(road)
    if car_col == 0:
        assert scores == [-np.inf, -2, 0]
    elif car_col == 1:
        assert scores == [-2 + score, 0, 0]
    elif car_col == 2:
        assert scores == [score, 0, -2]
    else:
        assert scores == [score, -2, -np.inf]


@pytest.mark.parametrize("obst_class", [Bump, Pedestrian])
@pytest.mark.parametrize("car_col", [0, 1, 2, 3])
def test_scores_obst_1_col_away_right(obst_class, car_col):
    headlight_range = 4
    obst = obst_class(0, car_col + 1)
    road = Road(headlight_range, Car(car_col, 1), [obst])
    score = hand_coded_obstacle_score(obst, road)
    scores = hand_coded_score_for_columns_adjacent_to_car(road)
    if car_col == 0:
        assert scores == [-np.inf, -2, score]
    elif car_col == 1:
        assert scores == [-2, 0, score]
    elif car_col == 2:
        assert scores == [0, 0, -2 + score]
    else:
        assert scores == [0, -2, -np.inf]


@pytest.mark.parametrize("obst_class", [Bump, Pedestrian])
@pytest.mark.parametrize("car_col", [0, 1, 2, 3])
def test_scores_obst_same_col_as_car(obst_class, car_col):
    headlight_range = 4
    obst = obst_class(0, car_col)
    road = Road(headlight_range, Car(car_col, 1), [obst])
    score = hand_coded_obstacle_score(obst, road)
    scores = hand_coded_score_for_columns_adjacent_to_car(road)
    if car_col == 0:
        assert scores == [-np.inf, -2 + score, 0]
    elif car_col == 1:
        assert scores == [-2, score, 0]
    elif car_col == 2:
        assert scores == [0, score, -2]
    else:
        assert scores == [0, -2 + score, -np.inf]


def test_pedestrian_in_front_of_car():
    obstacles = [Pedestrian(3, 2)]
    headlight_range = 4
    road = Road(headlight_range, Car(2, 1), obstacles)
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == LEFT


def test_policy_speeds_up_but_does_not_overdrive_headlights():
    headlight_range = 4
    road = Road(headlight_range, Car(2, headlight_range + 1), [])
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == DOWN

    road = Road(headlight_range, Car(2, headlight_range), [])
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == DOWN

    road = Road(headlight_range, Car(2, headlight_range - 1), [])
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == NO_OP

    road = Road(headlight_range, Car(2, headlight_range - 2), [])
    action_taken = hand_coded_data_gathering_policy(road)
    assert action_taken == UP


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
