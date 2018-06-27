import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import UP, DOWN, RIGHT, LEFT, NO_OP

def hand_coded_data_gathering_policy(road):
    scores = road.score_for_columns_adjacent_to_car()
    max_idx = scores.index(max(scores))
    if max_idx == 0 and scores[0] == scores[1]:
        max_idx = 1

    if max_idx == 0:
        return LEFT

    elif max_idx == 2:
        return RIGHT

    elif max_idx == 1 and scores[1] < 0:
        if road._car.speed == 1:
            return NO_OP
        else:
            return DOWN

    elif max_idx == 1 and scores[1] >= 0:
        if road._car.speed >= road._headlight_range:
            return DOWN
        elif road._car.speed < road._headlight_range - 1:
            return UP
        else:
            return NO_OP
