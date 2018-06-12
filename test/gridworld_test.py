import pytest
import numpy as np
from driving_gridworld.gridworld import DrivingGridworld
from driving_gridworld.actions import ACTIONS, QUIT, LIST_CONTROLS


@pytest.mark.skip(reason='Ignoring until pycolab is removed.')
@pytest.mark.parametrize("speed", range(1, 4))
def test_to_road_without_observation_argument_always_returns_the_root_state(
        speed):
    patient = DrivingGridworld(2, 0, 0, speed)
    road = patient.to_road()
    assert road.to_s(show_walls=False) == '    \n   c '


@pytest.mark.parametrize("action", ACTIONS + [QUIT, LIST_CONTROLS])
def test_game_over(action):
    patient = DrivingGridworld(4, 0, 0, 1)
    patient.its_showtime()
    assert not patient.game_over
    patient.play(action)
    assert patient.game_over == (action == QUIT)


@pytest.mark.xfail()
def test_initial_observation():
    patient = DrivingGridworld(4, 0, 0, 1)
    road = patient.to_road()
    assert road.to_s(show_walls=False) == 'd  d\nd  d\nd  d\nd Cd'

    o, r, d = patient.its_showtime()
    assert r == 1
    assert d == 1

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
