from pycolab.prefab_parts import drapes as prefab_drapes


class DitchDrape(prefab_drapes.Scrolly):
    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_pattern_position = self.pattern_position_prescroll(
            things['C'].position,
            the_plot
        )

        for i in range(1, things['C'].speed + 1):
            if self.whole_pattern[
                (
                    player_pattern_position.row - i,
                    player_pattern_position.col
                )
            ]:
                the_plot.add_reward(-4)
