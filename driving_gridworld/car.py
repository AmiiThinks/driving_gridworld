from .actions import UP, DOWN, LEFT, RIGHT, NO_OP


def car_row_array(position=2, show_walls=True):
    if show_walls:
        row = [' ', 'd', ' ', ' ', 'd', ' ']
    else:
        row = ['d', ' ', ' ', 'd']
    assert position < len(row) - int(show_walls)
    row[position + int(show_walls)] = 'C'
    return row


def car_row(position=2):
    return ''.join(car_row_array(position))


class Car(object):
    def __init__(self, row, col, speed):
        self.row = row
        self.col = col
        self.speed = speed

    def position(self): return (self.row, self.col)

    def next(self, action, speed_limit):
        assert speed_limit > 0
        if action == UP:
            return Car(self.row, self.col, min(self.speed + 1, speed_limit))
        elif action == DOWN:
            return Car(self.row, self.col, max(self.speed - 1, 1))
        elif action == LEFT:
            return Car(self.row, max(self.col - 1, 0), self.speed)
        elif action == RIGHT:
            return Car(self.row, min(self.col + 1, 3), self.speed)
        elif action == NO_OP:
            return Car(self.row, self.col, self.speed)
        else:
            raise 'Unrecognized action, "{}".'.format(action)

    def reward(self):
        return float(self.speed)

    def __str__(self):
        return 'C'

    def to_byte(self, encoding='ascii'):
        return bytes(str(self), encoding)[0]
