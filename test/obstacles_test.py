from driving_gridworld.obstacles import Obstacle


def test_creation():
    patient = Obstacle(2, 2)
    assert patient.row == 2
    assert patient.col == 2
