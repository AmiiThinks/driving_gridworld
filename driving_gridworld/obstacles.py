class Obstacle(object):
    def __init__(self, row, col, prob_of_appearing=0.2, speed=0):
        self.row = row
        self.col = col
        self.prob_of_appearing = prob_of_appearing
        self.speed = speed

    def position(self):
        return (self.row, self.col)

    def copy_at_position(self, row, col):
        return self.__class__(
            row,
            col,
            prob_of_appearing=self.prob_of_appearing,
            speed=self.speed)

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


class Bump(Obstacle):
    def __str__(self):
        return 'b'


class Pedestrian(Obstacle):
    def __str__(self):
        return 'p'
