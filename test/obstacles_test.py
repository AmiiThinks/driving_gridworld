from driving_gridworld.obstacles import Obstacle
from driving_gridworld.car import Car


def test_creation():
    patient = Obstacle(2, 2)
    assert patient.row == 2
    assert patient.col == 2


def test_no_collision_mismatched_columns():
    patient = Obstacle(2, 2)
    car = Car(3, 1, 1)
    assert not patient.next(car).has_collided(car)


def test_collision_matched_columns():
    patient = Obstacle(2, 2)
    car = Car(3, 2, 1)
    next_patient = patient.next(car)
    assert next_patient.row == 3
    assert next_patient.has_collided(car)
