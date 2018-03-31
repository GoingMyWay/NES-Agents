# THIS IS SINGLE CPU VERSION
import numpy as np


class NES(object):
    def __init__(self, sess, weights, reward_func, alpha, sigma, population_size):
        self._sess = sess
        self._weights = weights
        self._real_weights = self._sess.run(self._weights)
        self._reward_func = reward_func
        self._alpha = alpha
        self._sigma = sigma
        self._population_size = population_size

    def w_try(self, i_popu):
        return [self._real_weights[i] + self.sigma * noise for i, noise in enumerate(i_popu)]

    def train(self, n_iters=300, p_steps=20):
        for iter in range(n_iters):
            if iter % p_steps == 0:
                print(' iter: %s, reward: %s' % (iter, self._reward_func(self._real_weights)))

            # randomly initialize weights of each population
            populations = [[np.random.randn(*w.shape) for w in self._real_weights] for _ in range(self.population_size)]

            # jitter weights and get the rewards
            rewards = [
                self._reward_func(self.w_try(i_popu)) for i, i_popu in enumerate(populations)
            ]
            A = (rewards - np.mean(rewards)) / np.std(rewards)

            # update the value of weights
            for i, _ in enumerate(self._real_weights):
                N = np.array([pop[i] for pop in populations])
                self._real_weights[i] += self.alpha/(self.population_size*self.sigma) * np.dot(N.T, A).T

            # assign value to tensorflow's graph
            self._sess.run([tf_var.assign(real_var) for tf_var, real_var in zip(self.weights, self._real_weights)])

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

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
