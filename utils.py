import numpy as np
import copy
import random
from collections import namedtuple, deque


class OUNoise:

    def __init__(self, size, mu, theta, sigma):
        """

        :param size: ?
        :param mu: This is the gaussian noise mu or mean parameter
        :param theta: This is the gaussian noise theta parametre
        :param sigma:  This is the gaussian noise standard deviation
        """

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """ Reset internal state that is the noise to the mean """
        self.state = copy.copy(self.mu)


    def sample(self):
        """ Update internal state and return it as a noise sample. """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """ Buffer to store experience tuples up to a size """

    def __init__(self, buffer_size, batch_size):
        """

        :param buffer_size: maximum size of the buffer
        :param batch_size: size of each training batch
        """

        self.internal_memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(typename="Experience",
                                     field_names=["state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"])
    def add(self, state, action, reward, next_state, done):
        """ Add experience to the internal memory"""
        e = self.experience(state,
                            action,
                            reward,
                            next_state,
                            done)
        self.internal_memory.append(e)

    def sample(self):
        """ Randomly return  a batch of experiences from the internal memory """
        return random.sample(self.internal_memory, k=self.batch_size)


    def __len__(self):
        """ Return the size of the current internal memory"""
        return len(self.internal_memory)