import pytest
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Obstacle
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS


def test_transition_probs_without_obstacles_are_always_1():
    num_rows = 4
    obstacles = []
    speed_limit = 1
    car_inst = Car(0, 0, 1)
    road_test = Road(num_rows, car_inst, obstacles, speed_limit)
    for a in ACTIONS:
        for next_state, prob, reward in road_test.successors(a):
            assert prob == 1.0


@pytest.mark.parametrize("obst", [Bump(2, 2), Pedestrian(2, 2)])
@pytest.mark.parametrize("car_inst", [Car(3, 2, 1)])
def test_transition_probs_with_one_obstacle_are_1(obst, car_inst):
    num_rows = 2
    obstacles = [obst]
    speed_limit = 1
    road_test = Road(num_rows, car_inst, obstacles, speed_limit)
    for a in ACTIONS:
        for next_state, prob, reward in road_test.successors(a):
            assert prob == 1.0
