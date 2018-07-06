from .actions import UP, DOWN, LEFT, RIGHT, NO_OP


class Car(object):
    def __init__(self, col, speed):
        self.col = col
        self.speed = speed

    def next(self, action, speed_limit):
        assert speed_limit > 0

        col = self.col
        speed = self.speed
        if action == UP:
            speed = min(self.speed + 1, speed_limit)
        elif action == DOWN:
            speed = max(self.speed - 1, 0)
        elif action == LEFT:
            if self.speed > 0:
                col = self.col - 1
        elif action == RIGHT:
            if self.speed > 0:
                col = self.col + 1
        elif action != NO_OP:
            raise ValueError('Unrecognized action, "{}".'.format(action))
        return Car(col, speed)

    def progress_toward_destination(self, action):
        return max(self.speed - int(action == LEFT or action == RIGHT), 0)

    def reward(self, action):
        return float(self.progress_toward_destination(action))

    def __str__(self):
        return 'C'

    def to_byte(self, encoding='ascii'):
        return bytes(str(self), encoding)[0]
