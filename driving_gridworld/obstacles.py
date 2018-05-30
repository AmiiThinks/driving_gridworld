class Obstacle(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def position(self):
        return (self.row, self.col)

    def prob_of_appearing(self):
        raise 'Abstract.'

    def reward_for_collision(self):
        raise 'Abstract'

    def reward(self, car):
        return self.reward_for_collision() if self.has_collided(car) else 0

    def has_collided(self, car):
        old_row = self.row - car.speed
        was_in_front_of_car = old_row < car.row
        is_under_or_behind_car = self.row >= car.row
        is_in_car_lane = self.col == car.col
        return (was_in_front_of_car and is_under_or_behind_car
                and is_in_car_lane)

    def next(self, car):
        if self.has_collided(car):
            next_row = car.row + 1
        else:
            next_row = self.row + car.speed
        return self.__class__(next_row, self.col)


class Bump(Obstacle):
    def prob_of_appearing(self):
        return 0.1

    def reward_for_collision(self, car):
        return -2 * car.speed


class Pedestrian(Obstacle):
    def prob_of_appearing(self):
        return 0.05

    def reward_for_collision(self, car):
        return -1e2**car.speed
