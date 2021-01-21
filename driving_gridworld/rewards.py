import warnings
import numpy as np
from itertools import product
from driving_gridworld.obstacles import Pedestrian, Bump


class PedestrianHit(RuntimeError):
    pass


def check_for_pedestrian_collision(obs):
    if isinstance(obs, Pedestrian): raise PedestrianHit()


def reward(s, a, s_prime):
    '''
    Simple default reward function.

    +1 for every space travelled forward, -2 for every space travelled
    off-road, and -2 times the speed of the car relative to a Bump.
    Crashing into the barriers or a Pedestrian is a catastrophic condition
    that receives -4 times the speed limit as if the car hit a bump in the
    ditch at full speed but didn't make any forward progress.
    '''
    if s.has_crashed() or s_prime.has_crashed():
        return -4 * s.speed_limit()

    def check_for_bump_collision(obs=None):
        return (2 * (obs.speed + 1) *
                s.car.speed) if isinstance(obs, Bump) else 0

    try:
        r = s.car.progress_toward_destination(a)
        if s_prime.is_in_a_ditch() or s.is_in_a_ditch():
            r -= 2 * s.car.speed
        r -= s.count_obstacle_collisions(s_prime, check_for_bump_collision,
                                         check_for_pedestrian_collision)[0]
        return r
    except PedestrianHit:
        return -4 * s.speed_limit()
