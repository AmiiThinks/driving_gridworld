class Obstacle(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def position(self):
        return (self.row, self.col)

    def prob_of_appearing(self):
        raise NotImplementedError()

    def reward_for_collision(self, speed):
        raise NotImplementedError()

    def next(self, distance):
        return self.__class__(self.row + distance, self.col)

    def __str__(self):
        raise NotImplementedError()

    def to_byte(self, encoding='ascii'):
        return bytes(str(self), encoding)[0]


class Bump(Obstacle):
    def prob_of_appearing(self):
        return 0.1

    def reward_for_collision(self, speed):
        return -2 * speed

    def __str__(self):
        return 'b'


class Pedestrian(Obstacle):
    def prob_of_appearing(self):
        return 0.05

    def reward_for_collision(self, speed):
        return -8e2**speed

    def __str__(self):
        return 'p'
