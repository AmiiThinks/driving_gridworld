class Rollout(object):
    def __init__(self, policy, game, policy_on_game=False):
        self.policy = policy
        self.game = game
        self.policy_on_game = policy_on_game

    def __iter__(self):
        self.observation, self.reward, self.discount = self.game.its_showtime()
        self.t = 0
        self.discounted_return = 0.0
        return self

    def __next__(self):
        if self.discount > 0:
            a = self.policy(self.game
                            if self.policy_on_game else self.observation)
            observation, r, d = self.game.play(a)

            self.discounted_return += (d**self.t) * r
            yield (self.t, self.observation, a, r, self.discount, observation,
                   self.discounted_return)

            self.observation = observation
            self.reward = r
            self.discount = d
            self.t += 1
        else:
            raise StopIteration()
