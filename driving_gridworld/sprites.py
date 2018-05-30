from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab.protocols import logging as plab_logging
import numpy as np

from .obstacles import Bump, Pedestrian
from .drapes import DitchDrape
from .actions import UP, DOWN, LEFT, RIGHT, NO_OP


class ObstacleSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character, virtual_position):
        super(ObstacleSprite, self).__init__(
            corner,
            position,
            character,
            egocentric_scroller=False,
            impassable='|')
        self._teleport(virtual_position)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if self.on_the_board:
            # Check if the player's car ran over yourself.
            # If so, give the player a
            # negative reward proportional to their speed.
            new_row = self.virtual_position.row + things['C'].speed
            if (new_row >= things['C'].virtual_position.row
                    and self.virtual_position.col == things['C']
                    .virtual_position.col):
                the_plot.add_reward(self.reward_for_collision(things['C']))
            self._teleport((new_row, self.virtual_position.col))
        else:
            # Check how many legal spaces there are to teleport to.
            # This just depends on how fast the player is going and the other
            # obstacles on the board.
            possibly_allowed_positions = {}
            for row in range(0, things['C'].speed):
                for col in range(1, 5):
                    possibly_allowed_positions[(row, col)] = True
            for thing in things.values():
                if isinstance(thing, (DitchDrape, CarSprite)): continue
                disallowed_position = (thing.virtual_position.row,
                                       thing.virtual_position.col)
                if disallowed_position in possibly_allowed_positions:
                    del possibly_allowed_positions[disallowed_position]

            allowed_positions = possibly_allowed_positions
            reveal = (len(allowed_positions) > 0
                      and np.random.uniform() < self.prob_of_appearing())
            if reveal:
                allowed_positions = tuple(allowed_positions.keys())
                i = np.random.randint(0, len(allowed_positions))
                self._teleport(allowed_positions[i])


class BumpSprite(ObstacleSprite, Bump):
    pass


class PedestrianSprite(ObstacleSprite, Pedestrian):
    pass


class CarSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for our player, the car."""

    def __init__(self, corner, position, character, virtual_position):
        """Constructor: player is egocentric and can't walk through walls."""
        super(CarSprite, self).__init__(
            corner,
            position,
            character,
            egocentric_scroller=True,
            impassable='|')
        self._teleport(virtual_position)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused

        the_plot.add_reward(self.speed)

        if actions == UP:
            self._speed = min(self.speed + 1, self._speed_limit)
        elif actions == DOWN:
            self._speed = max(self.speed - 1, 1)
        elif actions == LEFT:
            self._west(board, the_plot)
        elif actions == RIGHT:
            self._east(board, the_plot)
        elif actions == NO_OP:
            self._stay(board, the_plot)
        elif actions == 5:
            the_plot.terminate_episode()
        elif actions == 6:
            print(plab_logging.consume(the_plot))


def car_sprite_class(speed=1, speed_limit=3):
    class MyCarSprite(CarSprite):
        def __init__(self, *args, **kwargs):
            super(MyCarSprite, self).__init__(*args, **kwargs)
            self._speed = speed
            self._speed_limit = speed_limit

        @property
        def speed(self):
            return self._speed

    return MyCarSprite
