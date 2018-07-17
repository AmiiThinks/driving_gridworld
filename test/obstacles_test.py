from driving_gridworld.obstacles import Obstacle, Pedestrian
import pytest
import numpy as np

def test_creation():
    patient = Obstacle(2, 2)
    assert patient.row == 2
    assert patient.col == 2


@pytest.mark.parametrize("car_speed", [0, 1, 2, 3])
def test_moving_pedestrian_at_speed_1(car_speed):
    patient = Pedestrian(1, 1, 1)
    next_patient = patient.next(car_speed)
    assert next_patient.row == (patient.row + car_speed + patient.speed)
