import math
import time

from autoscaler.envs.simulation.entities import LOAD_HISTORY_LAST_N
from autoscaler.envs.simulation.kube_cluster import SimulatedCluster


class Recommender:
    def __init__(self, cluster: SimulatedCluster):
        self.cluster = cluster
        self.sync_period = 15
        self.downscale_stabilization = 5 * 60
        self.tolerance = 0.1

        self.prev_recommendations = []
        self.time_step = 0

    def recommend(self, state: list) -> int:
        time.sleep(1)
        last_utilization = state[LOAD_HISTORY_LAST_N-1]
        current_replicas = self.get_replicas(state)
        c = last_utilization / self.cluster.utilization_target

        self.render(state)
        desired_replicas = current_replicas
        if abs(c - 1) > self.tolerance:
            desired_replicas = min(
                max(
                    math.ceil(current_replicas * c), self.cluster.min_replicas
                ), self.cluster.max_replicas
            )

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

        # Record the unstable recommendation
        self.prev_recommendations.append((self.time_step, desired_replicas))

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

    def run(self):
        self.cluster.reset()
        while True:
            self.time_step += 1
            if self.time_step % self.sync_period != 0:
                continue
            self.remove_old_recommendations()
            self.cluster.collect_metrics()
            state = self.cluster.get_state()
            action = self.recommend(state)
            self.cluster.step([action])
