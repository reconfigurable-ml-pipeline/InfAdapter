import math
import matplotlib.pyplot as plt
import time

from autoscaler.envs.simulation.entities import LOAD_HISTORY_LAST_N
from autoscaler.envs.simulation.kube_cluster import SimulatedCluster
from autoscaler.recommenders.base import RecommenderBase


class HPARecommender(RecommenderBase):
    collect_metrics_per = 15
    do_scaling_per = 15

    def __init__(self, cluster: SimulatedCluster):
        super().__init__(cluster)
        self.downscale_stabilization = 5 * 60
        self.tolerance = 0.1

        self.utilization_history = [0]
        self.replica_history = [self.cluster.min_replicas]
        self.prev_recommendations = []

    def recommend(self, state: list):
        # time.sleep(1)
        self.remove_old_recommendations()
        last_utilization = state[LOAD_HISTORY_LAST_N-1]
        self.utilization_history.append(last_utilization)
        current_replicas = self.get_replicas(state)
        c = last_utilization / self.cluster.utilization_target
        max_replicas = min(self.cluster.max_replicas, max(4, current_replicas * 2))
        self.render(state)
        desired_replicas = current_replicas
        if abs(c - 1) > self.tolerance:
            desired_replicas = math.ceil(current_replicas * c)

        new_replicas = desired_replicas

        if current_replicas - new_replicas > 4:
            new_replicas = current_replicas - 4

        if new_replicas < current_replicas:
            print(f"scaling in recommended: {new_replicas}. going to stabilize...")
            for timestamp, recommendation in self.prev_recommendations:
                new_replicas = max(new_replicas, recommendation)
            print("new replicas after stabilization is:", new_replicas)
        elif new_replicas > current_replicas:
            print(f"scaling out recommended: {new_replicas}")
        else:
            print(f"no change: {new_replicas}")
        print()

        if new_replicas > max_replicas:
            new_replicas = max_replicas
        if new_replicas < self.cluster.min_replicas:
            new_replicas = self.cluster.min_replicas
        # Record the unstable recommendation
        self.prev_recommendations.append((self.time_step, desired_replicas))

        self.replica_history.append(new_replicas)
        return new_replicas

    @staticmethod
    def get_replicas(state):
        return state[-1]

    def render(self, state):
        last_utilization = state[LOAD_HISTORY_LAST_N-1]
        print(f"calculating new recommendation at {self.time_step} ...")
        print("last utilization:", last_utilization)
        print("target utilization:", self.cluster.utilization_target)
        print("current replicas:", self.get_replicas(state))

    def remove_old_recommendations(self):
        recommendations = self.prev_recommendations[:]
        self.prev_recommendations = []
        for timestamp, recommendation in recommendations:
            if timestamp >= self.time_step - self.downscale_stabilization:
                self.prev_recommendations.append((timestamp, recommendation))

    def plot_results(self):

        x = [i * 15 for i in range(len(self.utilization_history))]
        plt.xlabel("Simulation Time")
        plt.plot(x, self.utilization_history, label="utilization", marker="o")
        plt.plot(x, self.replica_history, label="replicas", marker="o")
        plt.title(
            f"min_replicas: {self.cluster.min_replicas}     max_replicas: {self.cluster.max_replicas}     "
            f"utilization_target: {self.cluster.utilization_target}"
        )
        plt.legend()
        plt.show()
