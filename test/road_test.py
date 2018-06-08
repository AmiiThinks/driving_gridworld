from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS
import pytest


def test_transition_probs_without_obstacles_are_always_1():
    num_rows = 4
    obstacles = []
    speed_limit = 1
    car_inst = Car(0, 0, 1)
    road_test = Road(num_rows, car_inst, obstacles, speed_limit)

    for a in ACTIONS:
        for next_state, prob, reward in road_test.successors(a):
            assert prob == 1.0


@pytest.mark.parametrize("obst", [Bump(0, 0), Pedestrian(0, 0)])
def test_no_obstacles_revealed_is_the_only_valid_set_of_revealed_obstacles_when_all_obstacles_already_on_road(obst):
    num_rows = 2
    speed_limit = 1

    road_test = Road(num_rows, Car(1, 1, 1), [obst], speed_limit)
    patient = [
        (positions, reveal_indices)
        for positions, reveal_indices in
        road_test.every_combination_of_revealed_obstacles()
    ]
    assert patient == [(tuple(), set())]


@pytest.mark.parametrize("obst", [Bump(0, 0), Pedestrian(0, 0)])
@pytest.mark.parametrize("action", ACTIONS)
def test_transition_probs_with_one_obstacle_are_1(obst, action):
    num_rows = 2
    speed_limit = 1

    road_test = Road(num_rows, Car(1, 1, 1), [obst], speed_limit)
    probs = [
        prob
        for next_state, prob, reward in road_test.successors(action)
    ]
    assert probs == [1.0]
