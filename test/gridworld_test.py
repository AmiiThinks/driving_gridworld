import pytest
import numpy as np
from driving_gridworld.gridworld import DrivingGridworld, RecordingDrivingGridworld
from driving_gridworld.actions import ACTIONS, QUIT, LIST_CONTROLS, NO_OP


@pytest.mark.parametrize("action", ACTIONS + [QUIT, LIST_CONTROLS])
def test_game_over(action):
    patient = DrivingGridworld(4, 0, 0, 1, discount=0.123)
    o, r, d = patient.its_showtime()
    assert d == 0.123
    assert not patient.game_over
    o, r, d = patient.play(action)
    assert patient.game_over == (action == QUIT)
    assert d == 0.123


def test_initial_observation():
    patient = DrivingGridworld(3, 0, 0, 1, discount=0.8)
    assert patient.road.to_s() == '|d  d| \n|d  d| \n|d  d| \n|d Cd|^'

    o, r, d = patient.its_showtime()
    assert r == 0
    assert d == 0.8

    np.testing.assert_array_equal(
        o.board,
        np.array(
            [[124, 100,  32,  32, 100, 124, 32],
             [124, 100,  32,  32, 100, 124, 32],
             [124, 100,  32,  32, 100, 124, 32],
             [124, 100,  32,  67, 100, 124, 94]]).astype('uint8')
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['C'],
        np.array(
            [[False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False],
             [False, False, False,  True, False, False, False]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['|'],
        np.array(
            [[ True, False, False, False, False,  True, False],
             [ True, False, False, False, False,  True, False],
             [ True, False, False, False, False,  True, False],
             [ True, False, False, False, False,  True, False]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['d'],
        np.array(
            [[False,  True, False, False,  True, False, False],
             [False,  True, False, False,  True, False, False],
             [False,  True, False, False,  True, False, False],
             [False,  True, False, False,  True, False, False]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers[' '],
        np.array(
            [[False, False,  True,  True, False, False, True],
             [False, False,  True,  True, False, False, True],
             [False, False,  True,  True, False, False, True],
             [False, False,  True, False, False, False, False]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['^'],
        np.array(
            [[False, False,  False,  False, False, False, False],
             [False, False,  False,  False, False, False, False],
             [False, False,  False,  False, False, False, False],
             [False, False,  False, False, False, False, True]])
    )  # yapf:disable


def test_recording_gridworld_creation():
    patient = RecordingDrivingGridworld()
    assert patient.recorded() == []
    observation, reward, discount = patient.its_showtime()
    expected_recorded = [(patient.road.copy(), reward, discount)]
    assert patient.recorded() == expected_recorded
    observation, reward, discount = patient.play(0)
    expected_recorded.append((patient.road.copy(), reward, discount, 0))
    assert patient.recorded() == expected_recorded


def test_obstacles_always_appear_with_the_same_probability():
    headlight_range = 4
    patient = DrivingGridworld(
        headlight_range, num_bumps=0, num_pedestrians=1, speed=1, discount=0.99, pedestrian_appearance_prob=0.01)

    all_x = [
        (frozenset(), 0.99, 0.0),
        (frozenset([('p', 0, 1)]), 0.0025, 0.0),
        (frozenset([('p', 0, 3)]), 0.0025, 0.0),
        (frozenset([('p', 0, 0)]), 0.0025, 0.0),
        (frozenset([('p', 0, 2)]), 0.0025, 0.0)
    ]
    all_x.reverse()
    successors = tuple(patient.road.successors(NO_OP))
    for s, prob, r in successors:
        x_obstacles, x_prob, x_r = all_x.pop()
        assert s.to_key()[-1] == x_obstacles
        assert prob == x_prob
        assert r == x_r

    patient.road = successors[1][0]
    for _ in range(headlight_range):
        patient.play(NO_OP)

    assert patient.road.to_s() == '|d  d| \n|d  d| \n|d  d| \n|d  d| \n|dpCd|^'
    patient.play(NO_OP)
    assert patient.road.to_s() == '|d  d| \n|d  d| \n|d  d| \n|d  d| \n|d Cd|^'

    all_x = [
        (frozenset(), 0.99, 0.0),
        (frozenset([('p', 0, 1)]), 0.0025, 0.0),
        (frozenset([('p', 0, 3)]), 0.0025, 0.0),
        (frozenset([('p', 0, 0)]), 0.0025, 0.0),
        (frozenset([('p', 0, 2)]), 0.0025, 0.0)
    ]
    all_x.reverse()
    successors = tuple(patient.road.successors(NO_OP))
    for s, prob, r in successors:
        x_obstacles, x_prob, x_r = all_x.pop()
        assert s.to_key()[-1] == x_obstacles
        assert prob == x_prob
        assert r == x_r
