import math
import matplotlib.pyplot as plt

import simpy

from auto_tuner.envs.simulation.kube_cluster import SimulatedCluster
from auto_tuner.envs.simulation.monitoring import Monitoring
from auto_tuner.autoscalers.base import RecommenderBase


class HPARecommender(RecommenderBase):
    collect_metrics_per = 15
    do_scaling_per = 15

    def __init__(self, env: simpy.Environment, cluster: SimulatedCluster, monitoring: Monitoring):
        super().__init__(env=env, cluster=cluster, monitoring=monitoring)
        self.downscale_stabilization = 5 * 60
        self.tolerance = 0.1

        self.utilization_history = []
        self.replica_history = []
        self.prev_recommendations = []

    def get_state(self, cluster_metrics, application_metrics):
        return cluster_metrics

    def recommend(self, state: list):
        # time.sleep(1)
        self.remove_old_recommendations()
        last_utilization = state[self.cluster.cluster.LOAD_HISTORY_LAST_N-1]
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
            # print(f"scaling in recommended: {new_replicas}. going to stabilize...")
            for timestamp, recommendation in self.prev_recommendations:
                new_replicas = max(new_replicas, recommendation)
            # print("new replicas after stabilization is:", new_replicas)

        if new_replicas > max_replicas:
            new_replicas = max_replicas
            # print(f"using max replicas: {max_replicas}")
        if new_replicas < self.cluster.min_replicas:
            new_replicas = self.cluster.min_replicas
            # print(f"using min replicas: {self.cluster.min_replicas}")
        # Record the unstable recommendation
        self.prev_recommendations.append((self.env.now, desired_replicas))

        self.replica_history.append(new_replicas)
        # print(f"{new_replicas=}")
        return new_replicas

    @staticmethod
    def get_replicas(state):
        return state[-1]

    def render(self, state):
        return
        last_utilization = state[self.cluster.cluster.LOAD_HISTORY_LAST_N-1]
        print(f"calculating new recommendation at {self.env.now} ...")
        print("last utilization:", last_utilization)
        print("target utilization:", self.cluster.utilization_target)
        print("current replicas:", self.get_replicas(state))

    def remove_old_recommendations(self):
        recommendations = self.prev_recommendations[:]
        self.prev_recommendations = []
        for timestamp, recommendation in recommendations:
            if timestamp >= self.env.now - self.downscale_stabilization:
                self.prev_recommendations.append((timestamp, recommendation))

    def plot_results(self):
        x = [i * self.collect_metrics_per for i in range(len(self.utilization_history))]
        plt.xlabel("Simulation Time")
        plt.plot(x, self.monitoring.records, label="request count", marker="o")
        plt.title("HPA")
        plt.legend()
        plt.show()

        plt.plot(x, self.utilization_history, label="utilization", marker="o")
        plt.plot(x, self.replica_history, label="replicas", marker="o")
        plt.title(
            f"min_replicas: {self.cluster.min_replicas}   max_replicas: {self.cluster.max_replicas}   HPA   "
            f"utilization_target: {self.cluster.utilization_target}"
        )
        plt.legend()
        plt.show()

        plt.plot(x, self.monitoring.request_statistics, label="mean response time per 15 seconds", marker="o")
        plt.legend()
        plt.show()


