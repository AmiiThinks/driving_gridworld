import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS, RIGHT, LEFT, UP, DOWN, NO_OP
from driving_gridworld.road import product_combination_pairs
import pytest


@pytest.mark.parametrize("action", ACTIONS)
def test_transition_probs_without_obstacles_are_always_1(action):
    headlight_range = 4
    patient = Road(headlight_range, Car(0, 1))
    for next_state, prob in patient.successors(action):
        assert prob == 1.0


@pytest.mark.parametrize("obst", [Bump(0, 0), Pedestrian(0, 0)])
def test_no_obstacles_revealed_is_the_only_valid_set_of_revealed_obstacles_when_all_obstacles_already_on_road(
        obst):
    headlight_range = 2
    road_test = Road(headlight_range, Car(1, 1), [obst])
    patient = list(road_test.every_combination_of_revealed_obstacles(1))
    assert patient == [{}]


@pytest.mark.parametrize("obst", [Bump(0, 0), Pedestrian(0, 0)])
@pytest.mark.parametrize("action", ACTIONS)
def test_transition_probs_with_one_obstacle_are_1(obst, action):
    headlight_range = 2
    road_test = Road(headlight_range, Car(1, 1), [obst])
    probs = [prob for next_state, prob in road_test.successors(action)]
    assert probs == [1.0]


@pytest.mark.parametrize("obst", [Bump(-1, -1), Pedestrian(0, -1)])
@pytest.mark.parametrize("action", ACTIONS)
def test_transition_probs_with_invisible_obstacle(obst, action):
    headlight_range = 2
    road_test = Road(headlight_range, Car(1, 1), [obst])
    probs = [prob for next_state, prob in road_test.successors(action)]
    if action == LEFT or action == RIGHT:
        assert probs == [1.0]
    else:
        assert len(probs) == 5
        sum_probs = 0.0
        for i in range(len(probs)):
            assert 0.0 <= probs[i] <= 1.0
            sum_probs += probs[i]
        assert sum_probs == pytest.approx(1.0)
        assert probs[1:] == [obst.prob_of_appearing / 4] * 4
        assert probs[0] == 1 - sum([obst.prob_of_appearing / 4] * 4)


@pytest.mark.parametrize("obst", [Bump(-1, -1), Pedestrian(0, -1)])
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("speed", [1, 2, 3])
def test_number_of_successors_invisible_obstacle_and_variable_speeds(
        obst, action, speed):
    headlight_range = 2
    road_test = Road(headlight_range, Car(1, speed), [obst])
    probs = [prob for next_state, prob in road_test.successors(action)]
    assert (len(probs) ==
            4 * max(0, speed - int(action == LEFT or action == RIGHT)) + 1)


def test_car_can_only_overdrive_headlights_by_one_unit():
    headlight_range = 2
    with pytest.raises(ValueError):
        patient = Road(headlight_range, Car(0, speed=headlight_range + 2))

    patient = Road(headlight_range, Car(0, headlight_range + 1))
    assert patient.speed_limit() == headlight_range + 1

    patient = next(patient.successors(UP))[0]
    assert patient.speed_limit() == headlight_range + 1
    patient._car.speed == patient.speed_limit()

    patient = next(patient.successors(DOWN))[0]
    patient._car.speed == headlight_range


@pytest.mark.parametrize('col', range(4))
@pytest.mark.parametrize('headlight_range', range(1, 11))
def test_car_layer(col, headlight_range):
    patient = Road(headlight_range, Car(col, 1), [])
    x = np.full([headlight_range + 1, 7], False)
    x[headlight_range, col + 1] = True
    np.testing.assert_array_equal(patient.car_layer(), x)


@pytest.mark.parametrize('headlight_range', range(1, 11))
def test_wall_layer(headlight_range):
    patient = Road(headlight_range, Car(1, 1), [])
    x = np.full([headlight_range + 1, 7], False)
    x[:, 0] = True
    x[:, -2] = True
    np.testing.assert_array_equal(patient.wall_layer(), x)


@pytest.mark.parametrize('headlight_range', range(1, 11))
def test_ditch_layer(headlight_range):
    patient = Road(headlight_range, Car(1, 1), [])
    x = np.full([headlight_range + 1, 7], False)
    x[:, 1] = True
    x[:, -3] = True
    np.testing.assert_array_equal(patient.ditch_layer(), x)


@pytest.mark.parametrize('speed', range(4))
def test_speedometer_layer(speed):
    headlight_range = 4
    patient = Road(headlight_range, Car(1, speed=speed), [])
    x = np.full([headlight_range + 1, 7], False)
    x[headlight_range + 1 - speed:, -1] = True
    np.testing.assert_array_equal(patient.speedometer_layer(), x)


def test_obstacle_layers():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    patient = Road(headlight_range, Car(1, 1),
                   bumps + pedestrians).obstacle_layers()

    assert len(patient) == 2

    x_bump_layer = np.full([headlight_range + 1, 7], False)
    x_bump_layer[0, 1] = True
    x_bump_layer[1, 4] = True
    np.testing.assert_array_equal(patient[str(bumps[0])], x_bump_layer)

    x_pedestrian_layer = np.full([headlight_range + 1, 7], False)
    x_pedestrian_layer[0, 2] = True
    x_pedestrian_layer[1, 3] = True
    np.testing.assert_array_equal(patient[str(pedestrians[0])],
                                  x_pedestrian_layer)


def test_obstacles_outside_headlight_range_are_hidden():
    bumps = [Bump(-1, 0)]
    headlight_range = 4
    patient = Road(headlight_range, Car(1, 1), bumps).obstacle_layers()

    assert len(patient) == 1

    x_bump_layer = np.full([headlight_range + 1, 7], False)
    np.testing.assert_array_equal(patient[str(bumps[0])], x_bump_layer)


def test_layers():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    speed = 1
    patient = Road(headlight_range, Car(1, speed=speed),
                   bumps + pedestrians).layers()

    assert len(patient) == 7

    x_bump_layer = np.full([headlight_range + 1, 7], False)
    x_bump_layer[0, 1] = True
    x_bump_layer[1, 4] = True
    np.testing.assert_array_equal(patient[str(bumps[0])], x_bump_layer)

    x_pedestrian_layer = np.full([headlight_range + 1, 7], False)
    x_pedestrian_layer[0, 2] = True
    x_pedestrian_layer[1, 3] = True
    np.testing.assert_array_equal(patient[str(pedestrians[0])],
                                  x_pedestrian_layer)

    x_car_layer = np.full([headlight_range + 1, 7], False)
    x_car_layer[headlight_range, 1 + 1] = True
    np.testing.assert_array_equal(patient['C'], x_car_layer)

    x_wall_layer = np.full([headlight_range + 1, 7], False)
    x_wall_layer[:, 0] = True
    x_wall_layer[:, -2] = True
    np.testing.assert_array_equal(patient['|'], x_wall_layer)

    x_ditch_layer = np.full([headlight_range + 1, 7], False)
    x_ditch_layer[:, 1] = True
    x_ditch_layer[:, -3] = True
    np.testing.assert_array_equal(patient['d'], x_ditch_layer)

    x_speedometer_layer = np.full([headlight_range + 1, 7], False)
    x_speedometer_layer[-speed:, -1] = True
    np.testing.assert_array_equal(patient['^'], x_speedometer_layer)

    x_empty_layer = np.logical_not(
        np.logical_or(x_speedometer_layer,
                      np.logical_or(
                          x_ditch_layer,
                          np.logical_or(
                              np.logical_or(x_bump_layer, x_pedestrian_layer),
                              np.logical_or(x_car_layer, x_wall_layer)))))
    np.testing.assert_array_equal(patient[' '], x_empty_layer)


def _byte(c, encoding='ascii'):
    return bytes(c, encoding)[0]


def test_board():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    speed = 1
    patient = Road(headlight_range, Car(1, speed=speed),
                   bumps + pedestrians).board()

    assert patient.dtype == 'uint8'

    x_board = np.full(
        [headlight_range + 1, Road._stage_width], _byte(' '), dtype='uint8')

    x_board[:, 0] = _byte('|')
    x_board[:, -2] = _byte('|')

    x_board[:, 1] = _byte('d')
    x_board[:, -3] = _byte('d')

    x_board[-speed:, -1] = _byte('^')

    x_board[0, 1] = bumps[0].to_byte()
    x_board[1, 4] = bumps[0].to_byte()

    x_board[0, 2] = pedestrians[0].to_byte()
    x_board[1, 3] = pedestrians[0].to_byte()

    x_board[headlight_range, 1 + 1] = _byte('C')

    np.testing.assert_array_equal(patient, x_board)


@pytest.mark.parametrize('col', [0, 1])
def test_obstacles_appear_over_dirt_and_pavement(col):
    bumps = [Bump(0, col)]
    headlight_range = 4
    patient = Road(headlight_range, Car(2, 1), bumps).board()
    assert patient[0, col + 1] == bumps[0].to_byte()


@pytest.mark.parametrize('col', [0, 1, 2])
def test_car_appears_over_dirt_and_pavement(col):
    bumps = [Bump(0, 2)]
    headlight_range = 4
    c = Car(col, 1)
    patient = Road(headlight_range, c, bumps).board()
    assert patient[headlight_range, col + 1] == c.to_byte()


def test_to_key():
    bumps = [Bump(0, 2)]
    headlight_range = 1
    car = Car(2, 1)
    patient = Road(headlight_range, car, bumps).to_key()
    assert patient == (2, 1, frozenset([('b', 0, 2, 0, 0)]))

    obstacles = [
        Bump(-1, 0),
        Bump(-1, -1),
        Bump(0, 2),
        Pedestrian(1, 1),
        Pedestrian(1, 2),
        Pedestrian(2, 3)
    ]
    headlight_range = 1
    car = Car(2, 1)
    patient = Road(headlight_range, car, obstacles).to_key()
    assert patient == (
        2,
        1,
        frozenset([('b', 0, 2, 0, 0), ('p', 1, 1, 0, 0), ('p', 1, 2, 0, 0)]))  # yapf:disable


def test_to_s():
    bumps = [Bump(-1, -1), Bump(0, 0), Bump(1, 3)]
    pedestrians = [Pedestrian(-1, -1), Pedestrian(0, 1), Pedestrian(1, 2)]
    headlight_range = 4
    speed = 1
    patient = Road(headlight_range, Car(1, speed=speed),
                   bumps + pedestrians).to_s()
    assert patient == '|bp d| \n|d pb| \n|d  d| \n|d  d| \n|dC d|^'


@pytest.mark.parametrize('col', [-1, 4])
def test_car_has_crashed(col):
    patient = Road(4, Car(col, 1), [])
    assert patient.has_crashed()
    assert patient.has_crashed(Car(col, 1))
    assert not patient.has_crashed(Car(0, 1))


def test_crashing_into_left_wall():
    patient = Road(1, Car(0, 1), [])
    successors = list(patient.successors(LEFT))
    assert len(successors) == 1
    s, p = successors[0]
    assert p == 1.0
    assert s.to_key() != patient.to_key()
    assert s.to_key() == (-1, 0, frozenset())
    assert s.to_s() == '|d  d| \nCd  d| '


def test_crashing_into_right_wall():
    patient = Road(1, Car(3, 1), [])
    successors = list(patient.successors(RIGHT))
    assert len(successors) == 1
    s, p = successors[0]
    assert p == 1.0
    assert s.to_key() != patient.to_key()
    assert s.to_key() == (4, 0, frozenset())
    assert s.to_s() == '|d  d| \n|d  dC '


@pytest.mark.parametrize("speed", range(7))
def test_successor_probabilities(speed):
    state = Road(headlight_range=6, car=Car(1, speed), obstacles=[Bump(-1, -1)])
    probs = [p for (s,p) in state.successors(UP)]
    assert sum(probs) == pytest.approx(1.0)


def test_probabilities_error():
    state = Road(headlight_range=3, car=Car(1, 1), obstacles=[Bump(-1, -1), Bump(-1, -1)])
    probs = [p for (s,p) in state.successors(NO_OP)]
    assert sum(probs) == pytest.approx(1.0)


def test_permutations():
    patient = product_combination_pairs(
        [(0, i) for i in range(4)],
        [0, 1],
        2
    )
    assert len(list(patient)) == 16


def test_successor_function_with_identical_states():
    expected_probs = [0.64, 0.08, 0.08, 0.08, 0.08, 0.0025, 0.005, 0.005, 0.005,
        0.0025, 0.005, 0.005, 0.0025, 0.005, 0.0025]
    state = Road(headlight_range = 3, car=Car(1, 1), obstacles=[Bump(-1, -1), Bump(-1, -1)])
    probs = [p for (s,p) in state.successors(NO_OP)]
    assert sum(probs) == pytest.approx(1.0)
    assert len(probs) == len(expected_probs)
    for i in range(len(probs)):
        assert probs[i] == pytest.approx(expected_probs[i])
