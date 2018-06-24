import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.obstacles import Bump
from driving_gridworld.obstacles import Pedestrian
from driving_gridworld.car import Car
from driving_gridworld.actions import UP, DOWN, RIGHT, LEFT, NO_OP


def policy_if_pedestrian_in_front(obstacle, car_col):
    if car_col == 1:
        return RIGHT
    elif car_col == 2:
        return LEFT

def policy_if_overdrive_headlights():
    return DOWN

def policy_if_bump_in_front(obstacle, car_col):
    return DOWN

def policy_if_offroad(car_col):
    if car_col == 0:
        return RIGHT
    elif car_col == 3:
        return LEFT

def policy_if_no_obstacle_in_front():
    return UP

def hand_coded_data_gathering_policy(road):
    car_col = road._car.col
    if road._obstacles != []:
        for obst in road._obstacles:
            # Priority 1: if there is a pedestrian in front of the car, avoid pedestrian
            if isinstance(obst, Pedestrian) and obst.col == car_col:
                action = policy_if_pedestrian_in_front(obst, car_col)
                #TODO: need an else statement if we're off-road???
                #TODO: what happens if there are two pedestrians side by side? We need to check whether we have [pedestrian(r, c), ped(r, c+1)]...

            # Priority 3: Never hit bumps at full speed. If bump is ahead, reduce speed.
            elif isinstance(obst, Bump) and obst.col == car_col:
                action = policy_if_bump_in_front(obst, car_col)

            # Priority 5: if no obstacles in front of the car, speed up to the speed_limit
            elif obst.col != car_col:
                action = policy_if_no_obstacle_in_front()

    # Priority 2: Don't over-drive headlights. The speed-limit + current position <= headlight position
    elif road._car.speed > road._headlight_range:
        action = policy_if_overdrive_headlights()

    # Priority 4: avoid driving off-road, unless a pedestrian is on the road and in front of you.
    elif car_col == 0 or car_col == 3:
        action = policy_if_offroad(car_col)

    #TODO: Is the constraint for car.speed <= speed_limit already implemented ?

    else:
        action = NO_OP

    return action
