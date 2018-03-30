# THIS IS SINGLE CPU VERSION
import numpy as np


class NES(object):
    def __init__(self, weights, reward_func, alpha, sigma, population_size):
        self._weights = weights
        self._reward_func = reward_func
        self._alpha = alpha
        self._sigma = sigma
        self._population_size = population_size

    def w_try(self, i_popu):
        return [self.weights[i] + self.sigma * noise for i, noise in enumerate(i_popu)]

    def train(self, n_iters=300, p_steps=20):
        for iter in range(n_iters):
            if iter % p_steps == 0:
                print(' iter: %s, reward: %s' % (iter, self.reward_func(self.weights)))

            # randomly initialize weights of each population
            populations = [[np.random.randn(*w.shape) for w in self.weights] for _ in range(self.population_size)]
            # initialize rewards
            rewards = np.zeros(self.population_size)
            # jitter weights
            for i, i_popu in enumerate(populations):
                w_try = self.w_try(i_popu)
                rewards[i] = self.reward_func(w_try)

            A = (rewards - np.mean(rewards)) / np.std(rewards)
            for i, _ in enumerate(self.weights):
                N = np.array([pop[i] for pop in populations])
                self.weights[i] += self.alpha/(self.population_size*self.sigma) * np.dot(N.T, A).T

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def reward_func(self):
        return self._reward_func

    @reward_func.setter
    def reward_func(self, reward_func):
        self._reward_func = reward_func

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, SIGMA):
        self._sigma = SIGMA

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, POPULATION_SIZE):
        self._population_size = POPULATION_SIZE
