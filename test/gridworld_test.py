import pytest
import numpy as np
from driving_gridworld.gridworld import DrivingGridworld, RecordingDrivingGridworld
from driving_gridworld.actions import ACTIONS, QUIT, LIST_CONTROLS


@pytest.mark.parametrize("action", ACTIONS + [QUIT, LIST_CONTROLS])
def test_game_over(action):
    patient = DrivingGridworld(4, 0, 0, 1, discount=0.123)
    o, r, d = patient.its_showtime()
    assert d == 0
    assert not patient.game_over
    o, r, d = patient.play(action)
    assert patient.game_over == (action == QUIT)
    assert d == 0.123


def test_initial_observation():
    patient = DrivingGridworld(3, 0, 0, 1, discount=0.8)
    assert patient.road.to_s(show_walls=False) == 'd  d\nd  d\nd  d\nd Cd'

    o, r, d = patient.its_showtime()
    assert r == 0
    assert d == 0

    np.testing.assert_array_equal(
        o.board,
        np.array(
            [[124, 100,  32,  32, 100, 124],
             [124, 100,  32,  32, 100, 124],
             [124, 100,  32,  32, 100, 124],
             [124, 100,  32,  67, 100, 124]]).astype('uint8')
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['C'],
        np.array(
            [[False, False, False, False, False, False],
             [False, False, False, False, False, False],
             [False, False, False, False, False, False],
             [False, False, False,  True, False, False]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['|'],
        np.array(
            [[ True, False, False, False, False,  True],
             [ True, False, False, False, False,  True],
             [ True, False, False, False, False,  True],
             [ True, False, False, False, False,  True]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers['d'],
        np.array(
            [[False,  True, False, False,  True, False],
             [False,  True, False, False,  True, False],
             [False,  True, False, False,  True, False],
             [False,  True, False, False,  True, False]])
    )  # yapf:disable

    np.testing.assert_array_equal(
        o.layers[' '],
        np.array(
            [[False, False,  True,  True, False, False],
             [False, False,  True,  True, False, False],
             [False, False,  True,  True, False, False],
             [False, False,  True, False, False, False]])
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
