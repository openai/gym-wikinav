import gym
from gym import error, spaces, utils
from gym.utils import seeding


class WikiNavEnv(gym.Env):

    def __init__(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _render(self, mode="human", close=False):
        raise NotImplementedError
