from driving_gridworld.obstacles import Bump, CarObstacle
import pytest
import numpy as np

def test_creation():
    patient = Bump(2, 2)
    assert patient.row == 2
    assert patient.col == 2


@pytest.mark.parametrize("distance", [0, 1, 2, 3])
def test_car_obstacle(distance):
    row = 1
    speed = 1
    patient = CarObstacle(row, 1)
    next_patient = patient.next(distance)
    assert next_patient.row == (row + speed + distance)
