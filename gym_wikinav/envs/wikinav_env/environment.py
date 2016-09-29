from io import StringIO
import sys

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_wikinav.envs.wikinav_env import web_graph


class WikiNavEnv(gym.Env):

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, beam_size=32, graph=None, goal_reward=10.0):
        """
        Args:
            beam_size: Number of candidates to present as actions at each
                timestep
            graph:
        """
        super(WikiNavEnv, self).__init__()

        if graph is None:
            graph = web_graph.EmbeddedWikispeediaGraph.get_default_graph()
        self.graph = graph

        # TODO verify beam size

        self.beam_size = beam_size
        self.goal_reward = goal_reward

        self.path_length = self.graph.path_length

        self.navigator = web_graph.Navigator(self.graph, self.beam_size,
                                             self.path_length)

        self._action_space = spaces.Discrete(self.beam_size)

        self._just_reset = False

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        # abstract
        raise NotImplementedError

    @property
    def cur_article_id(self):
        return self.navigator.cur_article_id

    @property
    def gold_path_length(self):
        return self.navigator.gold_path_length

    def get_article_for_action(self, action):
        return self.navigator.get_article_for_action(action)

    def _step(self, action):
        reward = self._reward(action)
        self.navigator.step(action)

        obs = self._observe()
        done = self.navigator.done
        info = {}

        return obs, reward, done, info

    def _reset(self):
        self.navigator.reset()
        self._just_reset = True
        obs = self._observe()
        self._just_reset = False
        return obs

    def _observe(self):
        # abstract
        raise NotImplementedError

    def _reward(self, action):
        """
        Compute single-timestep reward after having taken the action specified
        by `action`.
        """
        # abstract
        raise NotImplementedError

    def _render(self, mode="human", close=False):
        if close: return

        outfile = StringIO() if mode == "ansi" else sys.stdout

        cur_page = self.graph.get_article_title(self.cur_article_id)
        outfile.write("%s\n" % cur_page)
        return outfile


class EmbeddingWikiNavEnv(WikiNavEnv):

    """
    WikiNavEnv which represents articles with embeddings.
    """

    def __init__(self, *args, **kwargs):
        super(EmbeddingWikiNavEnv, self).__init__(*args, **kwargs)

        self.embedding_dim = self.graph.embedding_dim

        self._query_embedding = None

    @property
    def observation_space(self):
        # 2 embeddings (query and current page) plus the embeddings of
        # articles on the beam
        return spaces.Box(low=-np.inf, high=np.inf,
                          shape=(2 + self.beam_size, self.embedding_dim))

    def _observe(self):
        if self._just_reset:
            self._query_embedding = \
                    self.graph.get_query_embeddings([self.navigator._path])[0]

        current_page_embedding = \
                self.graph.get_article_embeddings([self.cur_article_id])[0]
        beam_embeddings = self.graph.get_article_embeddings(self.navigator.beam)

        return self._query_embedding, current_page_embedding, beam_embeddings

    def _reward(self, idx):
        if idx == self.graph.stop_sentinel:
            if self.navigator.on_target or self.navigator.done:
                # Return goal reward when first stopping on target and also at
                # every subsequent timestep.
                return self.goal_reward
            else:
                # Penalize for stopping on wrong page.
                return -self.goal_reward

        next_page = self.navigator.get_article_for_action(idx)
        overlap = self.graph.get_relative_word_overlap(next_page,
                                                       self.navigator.target_id)
        return overlap * self.goal_reward
