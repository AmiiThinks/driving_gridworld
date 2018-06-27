import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.actions import UP, DOWN, RIGHT, LEFT, NO_OP


def hand_coded_score_for_columns_adjacent_to_car(road):
    adj_cols_idx = list(
        range(max(0, road._car.col - 1), min(road._car.col + 2, 4)))
    scores = [0] * len(adj_cols_idx)

    if road._car.col == 0:
        scores.insert(0, -np.inf)
        scores[1] = -2

    elif road._car.col == 1:
        scores[0] = -2

    elif road._car.col == 2:
        scores[2] = -2

    else:
        scores.append(-np.inf)
        scores[1] = -2

    for obst in road._obstacles:
        if not road.obstacle_outside_car_path(obst):
            delta = abs(obst.col - road._car.col)
            if delta < 2:
                i = obst.col - road._car.col + 1
                scores[i] += hand_coded_obstacle_score(obst, road)
    return scores


def hand_coded_obstacle_score(obst, road):
    if isinstance(obst, Pedestrian):
        return -2 * road._headlight_range
    elif isinstance(obst, Bump):
        return -1
    else:
        return 0


def hand_coded_data_gathering_policy(road):
    scores = hand_coded_score_for_columns_adjacent_to_car(road)
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
