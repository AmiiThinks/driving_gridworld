import numpy as np


class Obstacle(object):
    def __init__(self, row, col, prob_of_appearing=0.2, obst_speed=0.0):
        self.row = row
        self.col = col
        self.prob_of_appearing = prob_of_appearing
        self.obst_speed = obst_speed

    def position(self):
        return (self.row, self.col)

    def expected_reward_for_collision(self, speed):
        raise NotImplementedError()

    def copy_at_position(self, row, col):
        return self.__class__(
            row, col, prob_of_appearing=self.prob_of_appearing)

    def next(self, distance):
        return self.__class__(
            self.row + distance + self.obst_speed,
            self.col,
            prob_of_appearing=self.prob_of_appearing)

    def __str__(self):
        raise NotImplementedError()

    def to_byte(self, encoding='ascii'):
        return bytes(str(self), encoding)[0]

    def reward_for_collision(self, speed, stddev=0.0):
        proportional_stddev = stddev * speed
        return (self.expected_reward_for_collision(speed) +
                np.random.normal(0, proportional_stddev))


class Bump(Obstacle):
    def expected_reward_for_collision(self, speed):
        return -2 * speed

    def __str__(self):
        return 'b'


class Pedestrian(Obstacle):
    def expected_reward_for_collision(self, speed):
        return -8e2**speed

    def __str__(self):
        return 'p'


class CarObstacle(Obstacle):
    def __init__(self, row, col, prob_of_appearing=0.2, obst_speed=1.0):
        self.row = row
        self.col = col
        self.prob_of_appearing = prob_of_appearing
        self.obst_speed = obst_speed

    def expected_reward_for_collision(self, speed):
        return -8e2**(speed + self.obst_speed) 

    def __str__(self):
        return 'c'
