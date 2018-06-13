import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS
import pytest


def test_transition_probs_without_obstacles_are_always_1():
    num_rows = 4
    obstacles = []
    car_inst = Car(0, 0, 1)
    road_test = Road(num_rows, car_inst, obstacles)

    for a in ACTIONS:
        for next_state, prob, reward in road_test.successors(a):
            assert prob == 1.0


@pytest.mark.parametrize("obst", [Bump(0, 0), Pedestrian(0, 0)])
def test_no_obstacles_revealed_is_the_only_valid_set_of_revealed_obstacles_when_all_obstacles_already_on_road(
        obst):
    num_rows = 2
    road_test = Road(num_rows, Car(1, 1, 1), [obst])
    patient = [(positions, reveal_indices)
               for positions, reveal_indices in
               road_test.every_combination_of_revealed_obstacles()]
    assert patient == [(tuple(), set())]


@pytest.mark.parametrize("obst", [Bump(0, 0), Pedestrian(0, 0)])
@pytest.mark.parametrize("action", ACTIONS)
def test_transition_probs_with_one_obstacle_are_1(obst, action):
    num_rows = 2
    road_test = Road(num_rows, Car(1, 1, 1), [obst])
    probs = [prob for next_state, prob, reward in road_test.successors(action)]
    assert probs == [1.0]


@pytest.mark.parametrize("obst", [Bump(-1, -1), Pedestrian(0, -1)])
@pytest.mark.parametrize("action", ACTIONS)
def test_transition_probs_with_invisible_obstacle(obst, action):
    num_rows = 2
    road_test = Road(num_rows, Car(1, 1, 1), [obst])
    probs = [prob for next_state, prob, reward in road_test.successors(action)]
    assert len(probs) == 5
    sum_probs = 0.0
    for i in range(len(probs)):
        assert 0.0 <= probs[i] <= 1.0
        sum_probs += probs[i]
    assert sum_probs == pytest.approx(1.0)
    assert probs[0] == max(probs)
    assert probs[1:] == [obst.prob_of_appearing() / 4] * 4


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("current_speed", [1, 2, 3, 4])
def test_driving_faster_gives_a_larger_reward(action, current_speed):
    num_rows = 4
    obstacles = []
    car = Car(0, 1, current_speed)
    road_test = Road(num_rows, car, obstacles)
    for next_state, prob, reward in road_test.successors(action):
        assert reward == float(current_speed)


def test_road_cannot_start_with_car_going_faster_than_speed_limit():
    num_rows = 4
    obstacles = []
    current_speed = 6
    car = Car(0, 0, current_speed)
    with pytest.raises(ValueError):
        Road(num_rows, car, obstacles)


@pytest.mark.parametrize("car", [Car(0, 0, 1), Car(0, 3, 1)])
@pytest.mark.parametrize("action", ACTIONS)
def test_receive_negative_reward_for_driving_off_the_road(car, action):
    num_rows = 4
    obstacles = []
    road_test = Road(num_rows, car, obstacles)
    for next_state, prob, reward in road_test.successors(action):
        assert reward < 0


@pytest.mark.parametrize("obst", [Bump(-1, -1), Pedestrian(0, -1)])
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("speed", [1, 2, 3])
def test_number_of_successors_invisible_obstacle_and_variable_speeds(
        obst, action, speed):
    num_rows = 2
    road_test = Road(num_rows, Car(1, 1, speed), [obst])
    probs = [prob for next_state, prob, reward in road_test.successors(action)]
    assert len(probs) == 4 * speed + 1


def test_speed_limit_equals_number_of_rows_plus_one():
    num_rows = 2
    obstacles = []
    car = Car(0, 0, 1)
    road_test = Road(num_rows, car, obstacles)
    assert road_test.speed_limit() == num_rows + 1


@pytest.mark.parametrize('col', range(4))
@pytest.mark.parametrize('headlight_range', range(1, 11))
def test_car_layer(col, headlight_range):
    patient = Road(headlight_range, Car(0, col, 1), [])
    x = np.full([headlight_range, 6], False)
    x[0, col + 1] = True
    np.testing.assert_array_equal(patient.car_layer(), x)


@pytest.mark.parametrize('headlight_range', range(1, 11))
def test_wall_layer(headlight_range):
    patient = Road(headlight_range, Car(0, 1, 1), [])
    x = np.full([headlight_range, 6], False)
    x[:, 0] = True
    x[:, -1] = True
    np.testing.assert_array_equal(patient.wall_layer(), x)


@pytest.mark.parametrize('headlight_range', range(1, 11))
def test_ditch_layer(headlight_range):
    patient = Road(headlight_range, Car(0, 1, 1), [])
    x = np.full([headlight_range, 6], False)
    x[:, 1] = True
    x[:, -2] = True
    np.testing.assert_array_equal(patient.ditch_layer(), x)


def test_obstacle_layers():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    patient = Road(headlight_range, Car(0, 1, 1),
                   bumps + pedestrians).obstacle_layers()

    assert len(patient) == 2

    x_bump_layer = np.full([headlight_range, 6], False)
    x_bump_layer[0, 1] = True
    x_bump_layer[1, 4] = True
    np.testing.assert_array_equal(patient[str(bumps[0])], x_bump_layer)

    x_pedestrian_layer = np.full([headlight_range, 6], False)
    x_pedestrian_layer[0, 2] = True
    x_pedestrian_layer[1, 3] = True
    np.testing.assert_array_equal(patient[str(pedestrians[0])],
                                  x_pedestrian_layer)


def test_obstacles_outside_headlight_range_are_hidden():
    bumps = [Bump(-1, 0)]
    headlight_range = 4
    patient = Road(headlight_range, Car(0, 1, 1),
                   bumps).obstacle_layers()

    assert len(patient) == 1

    x_bump_layer = np.full([headlight_range, 6], False)
    np.testing.assert_array_equal(patient[str(bumps[0])], x_bump_layer)


def test_layers():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    patient = Road(headlight_range, Car(0, 1, 1), bumps + pedestrians).layers()

    assert len(patient) == 6

    x_bump_layer = np.full([headlight_range, 6], False)
    x_bump_layer[0, 1] = True
    x_bump_layer[1, 4] = True
    np.testing.assert_array_equal(patient[str(bumps[0])], x_bump_layer)

    x_pedestrian_layer = np.full([headlight_range, 6], False)
    x_pedestrian_layer[0, 2] = True
    x_pedestrian_layer[1, 3] = True
    np.testing.assert_array_equal(patient[str(pedestrians[0])],
                                  x_pedestrian_layer)

    x_car_layer = np.full([headlight_range, 6], False)
    x_car_layer[0, 1 + 1] = True
    np.testing.assert_array_equal(patient['C'], x_car_layer)

    x_wall_layer = np.full([headlight_range, 6], False)
    x_wall_layer[:, 0] = True
    x_wall_layer[:, -1] = True
    np.testing.assert_array_equal(patient['|'], x_wall_layer)

    x_ditch_layer = np.full([headlight_range, 6], False)
    x_ditch_layer[:, 1] = True
    x_ditch_layer[:, -2] = True
    np.testing.assert_array_equal(patient['d'], x_ditch_layer)

    x_empty_layer = np.logical_not(
        np.logical_or(x_ditch_layer,
                      np.logical_or(
                          np.logical_or(x_bump_layer, x_pedestrian_layer),
                          np.logical_or(x_car_layer, x_wall_layer))))
    np.testing.assert_array_equal(patient[' '], x_empty_layer)


def byte(c, encoding='ascii'):
    return bytes(c, encoding)[0]


def test_board():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    patient = Road(headlight_range, Car(0, 1, 1), bumps + pedestrians).board()

    assert patient.dtype == 'uint8'

    x_bump_layer = np.full([headlight_range, 6], False)
    x_bump_layer[0, 1] = True
    x_bump_layer[1, 4] = True

    x_pedestrian_layer = np.full([headlight_range, 6], False)
    x_pedestrian_layer[0, 2] = True
    x_pedestrian_layer[1, 3] = True

    x_car_layer = np.full([headlight_range, 6], False)
    x_car_layer[0, 1 + 1] = True

    x_wall_layer = np.full([headlight_range, 6], False)
    x_wall_layer[:, 0] = True
    x_wall_layer[:, -1] = True

    x_ditch_layer = np.full([headlight_range, 6], False)
    x_ditch_layer[:, 1] = True
    x_ditch_layer[:, -2] = True

    x_empty_layer = np.logical_not(
        np.logical_or(x_ditch_layer,
                      np.logical_or(
                          np.logical_or(x_bump_layer, x_pedestrian_layer),
                          np.logical_or(x_car_layer, x_wall_layer))))

    x_board = (x_empty_layer * byte(' ') + x_bump_layer * byte('b') +
               x_pedestrian_layer * byte('p') + x_wall_layer * byte('|') +
               x_ditch_layer * byte('d') + x_car_layer * byte('C'))

    np.testing.assert_array_equal(patient, x_board)
