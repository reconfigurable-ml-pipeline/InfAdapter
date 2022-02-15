import sys

import simpy

from autoscaler.envs.simulation.load_balancer import LoadBalancer
from autoscaler.envs.simulation.monitoring import Monitoring
from autoscaler.envs.simulation.requests import RequestGenerator
from autoscaler.recommenders import recommenders
from autoscaler.envs.simulation.kube_cluster import SimulatedCluster
from autoscaler.config import get_environment_config


class Runner:
    def __init__(self):
        recommender_dict = {
            '1': recommenders.HPARecommender,
            '2': recommenders.BiLSTMRecommender
        }
        print("select a recommender:\n1: HPARecommender\n2: BiLSTMRecommender")
        selected_recommender = input()
        recommender = recommender_dict.get(selected_recommender)
        if recommender is None:
            print("Wrong input!")
            sys.exit(1)
        config = get_environment_config()
        self.env = config["env"]
        self.pod_name = config["pod_name"]
        self.cluster = SimulatedCluster(
            config
        )
        self.cluster.reset()
        self.monitoring = Monitoring(cluster=self.cluster)
        self.cluster.get_response_time = lambda: self.monitoring.request_statistics[-1]
        self.load_balancer = LoadBalancer(cluster=self.cluster, monitoring=self.monitoring)
        self.recommender = recommender(env=self.env, cluster=self.cluster, monitoring=self.monitoring)
        self.request_generator = RequestGenerator(self.load_balancer, pod_name=self.pod_name)

    def run(self):
        self.env.process(self.monitoring.run())
        self.env.process(self.load_balancer.run())
        self.env.process(self.recommender.run())
        load_generator = self.env.process(self.request_generator.run())
        self.env.run(until=load_generator)
        self.recommender.plot_results()


if __name__ == "__main__":
    Runner().run()
