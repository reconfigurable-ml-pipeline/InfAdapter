import math
import numpy as np
import simpy
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.simulation.kube_cluster import SimulatedCluster
from auto_tuner.simulation.monitoring import Monitoring
from auto_tuner.simulation.autoscalers.base import RecommenderBase


class BiLSTMRecommender(RecommenderBase):
    collect_metrics_per = 60
    do_scaling_per = 60
    query_n_requests_from_monitoring = 10

    MAX_REQUESTS_A_POD_RESPONDS_IN_MINUTE = 500
    RRS = 0.6

    def __init__(self, env: simpy.Environment, cluster: SimulatedCluster, monitoring: Monitoring):
        super().__init__(env=env, cluster=cluster, monitoring=monitoring)
        self.replica_history = []
        self.prev_recommendations = []
        self.model = load_model(f"{AUTO_TUNER_DIRECTORY}/autoscalers/bi_lstm/saved")

    def get_state(self, cluster_metrics, application_metrics):
        return [application_metrics, cluster_metrics[-1]]

    def recommend(self, state: list):
        requests, prev_replicas = state
        requests = tf.convert_to_tensor(np.array(requests).reshape((-1, 10, 1)), dtype=tf.float32)
        next_minutes_load = self.model.predict(requests)
        new_replicas = int(math.ceil(next_minutes_load / self.MAX_REQUESTS_A_POD_RESPONDS_IN_MINUTE))
        new_replicas = min(max(new_replicas, self.cluster.min_replicas), self.cluster.max_replicas)
        if new_replicas < prev_replicas:
            pods_surplus = (prev_replicas - new_replicas) * self.RRS
            new_replicas = int(prev_replicas - pods_surplus)
        self.replica_history.append(new_replicas)
        return new_replicas

    def render(self, state):
        return
        last_utilization = state[self.cluster.cluster.LOAD_HISTORY_LAST_N-1]
        print(f"calculating new recommendation at {self.env.now} ...")
        print("last utilization:", last_utilization)
        print("target utilization:", self.cluster.utilization_target)
        print("current replicas:", self.get_replicas(state))

    def plot_results(self):
        x = [i * self.collect_metrics_per for i in range(len(self.monitoring.records))]
        plt.xlabel("Simulation Time")
        plt.plot(x, self.monitoring.records, label="request count", marker="o")
        plt.title("BiLSTM")
        plt.legend()
        plt.show()

        x2 = [i * self.collect_metrics_per for i in range(len(self.replica_history))]
        plt.plot(x2, self.replica_history, label="replicas", marker="o")
        plt.title(
            f"min_replicas: {self.cluster.min_replicas}   max_replicas: {self.cluster.max_replicas}   BiLSTM"
        )
        plt.legend()
        plt.show()

        plt.plot(x, self.monitoring.request_statistics, label="mean response time per 60 seconds", marker="o")
        plt.title("BiLSTM")
        plt.legend()
        plt.show()


