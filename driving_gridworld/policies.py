import numpy as np
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.actions import UP, DOWN, RIGHT, LEFT, NO_OP


def hand_coded_score_for_columns_adjacent_to_car(road):
    scores = [-np.inf, -2, 0, 0, -2, -np.inf]
    for obst in road._obstacles:
        if road.obstacle_is_visible(obst):
            scores[obst.col + 1] += hand_coded_obstacle_score(obst, road)
    return scores[road._car.col:road._car.col + 3]


def hand_coded_obstacle_score(obst, road):
    if isinstance(obst, Pedestrian):
        return -2 * road._headlight_range
    elif isinstance(obst, Bump):
        return -1
    else:
        return 0


def hand_coded_data_gathering_policy(road):
    scores = hand_coded_score_for_columns_adjacent_to_car(road)
    max_idx = np.argmax(scores)
    if scores[max_idx] == scores[1]: max_idx = 1
    if max_idx == 0: return LEFT
    elif max_idx == 2: return RIGHT
    else:
        if scores[1] < 0 and road._car.speed > 1: return DOWN
        elif road._car.speed >= road._headlight_range: return DOWN
        elif road._car.speed < road._headlight_range - 1: return UP
    return NO_OP
