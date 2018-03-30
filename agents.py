# THIS IS SINGLE CPU VERSION
import gym
from nes import NES


class Pong(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.es = NES(weights=self.weights,
                      reward_func=self.reward_func,
                      alpha=0.01,
                      sigma=0.01,
                      population_size=15)
        self._weights = []

    def play(self):
        pass

    def train(self):
        pass

    def predict_action(self, input_state):
        pass

    def reward_func(self):
        pass

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
