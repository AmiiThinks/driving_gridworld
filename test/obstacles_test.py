from driving_gridworld.obstacles import Obstacle, Pedestrian
import pytest


def test_creation():
    patient = Obstacle(2, 2)
    assert patient.row == 2
    assert patient.col == 2


@pytest.mark.parametrize("car_speed", range(10))
@pytest.mark.parametrize("ped_speed", range(10))
def test_moving_pedestrian(car_speed, ped_speed):
    patient = Pedestrian(1, 1, speed=ped_speed)
    next_patient = patient.next(car_speed)
    assert next_patient.row == (patient.row + car_speed + patient.speed)
