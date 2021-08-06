import gym

from .entities import Cluster


class SimulatedCluster(gym.Env):
    cluster: Cluster

    def reset(self):
        self.cluster = Cluster()

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass
