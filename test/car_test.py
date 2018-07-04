import numpy as np
from driving_gridworld.car import Car
from driving_gridworld.actions import ACTIONS, RIGHT, NO_OP, LEFT
import pytest

@pytest.mark.parametrize("col_action", [(0, LEFT), (3, RIGHT)])

def test_car_does_not_bounce_off_wall(col_action):
    col = col_action[0]
    action = col_action[1]
    car = Car(col, 1)
    speed_limit = 5
    next_car = car.next(action, speed_limit)
    assert next_car.col < 0 or next_car.col > 3
