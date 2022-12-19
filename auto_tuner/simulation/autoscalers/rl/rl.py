import os

import simpy
from ray.rllib.agents.ppo import PPOTrainer

from auto_tuner.simulation.config import get_environment_config
from auto_tuner.simulation.kube_cluster import SimulatedCluster
from auto_tuner.simulation.monitoring import Monitoring
from auto_tuner.simulation.autoscalers.base import RecommenderBase


class RLRecommender(RecommenderBase):

    def __init__(self, env: simpy.Environment, cluster: SimulatedCluster, monitoring: Monitoring):
        super().__init__(env, cluster, monitoring)
        self.current_action = None
        config = {
            "env": SimulatedCluster,
            "framework": "tf",
            "env_config": get_environment_config()
        }
        self.agent = PPOTrainer(config=config, env=SimulatedCluster)

        with open(os.path.dirname(__file__) + "checkpoint.txt", "r") as f:
            checkpoint = f.read()
        self.agent.restore(checkpoint)

    def get_state(self, cluster_metrics, application_metrics):
        pass

    def recommend(self, state: list) -> int:
        self.current_action = self.agent.compute_action(state)
        return self.current_action

    def receive_action_response(self, next_state: list, reward: float):
        pass

    def plot_results(self):
        pass
