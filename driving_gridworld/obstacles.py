import numpy as np


class Obstacle(object):
    def __init__(self, row, col, prob_of_appearing=0.2, speed=0):
        self.row = row
        self.col = col
        self.prob_of_appearing = prob_of_appearing
        self.speed = speed

    def position(self):
        return (self.row, self.col)

    def expected_reward_for_collision(self, speed):
        raise NotImplementedError()

    def copy_at_position(self, row, col):
        return self.__class__(
            row, col, prob_of_appearing=self.prob_of_appearing)

    def next(self, distance):
        return self.__class__(
            self.row + distance + self.speed,
            self.col,
            prob_of_appearing=self.prob_of_appearing,
            speed=self.speed)

    def __str__(self):
        raise NotImplementedError()

    def to_byte(self, encoding='ascii'):
        return bytes(str(self), encoding)[0]

    def reward_for_collision(self, speed, stddev=0.0):
        stddev_car = stddev * speed
        stddev_car_obst = stddev * self.speed
        return (self.expected_reward_for_collision(speed) +
                np.random.normal(0, np.sqrt(np.square(stddev_car) +
                np.square(stddev_car_obst))))


class Bump(Obstacle):
    def expected_reward_for_collision(self, speed):
        return -2 * speed

    def __str__(self):
        return 'b'


class Pedestrian(Obstacle):
    def expected_reward_for_collision(self, speed):
        return -8e2**(speed + self.speed)

    def __str__(self):
        return 'p'
