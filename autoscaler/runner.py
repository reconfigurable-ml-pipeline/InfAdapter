import simpy

from autoscaler.envs.simulation.load_balancer import LoadBalancer
from autoscaler.envs.simulation.monitoring import Monitoring
from autoscaler.envs.simulation.requests import RequestGenerator
from autoscaler.recommenders import recommenders
from autoscaler.envs.simulation.kube_cluster import SimulatedCluster


class Runner:
    def __init__(self):
        self.env = simpy.Environment()
        self.pod_name = "nginx"
        self.cluster = SimulatedCluster(
            env=self.env, pod_name=self.pod_name, min_replicas=5, max_replicas=100, utilization_target=50, seed=31
        )
        self.cluster.reset()
        self.monitoring = Monitoring(cluster=self.cluster)
        self.recommender = recommenders.HPARecommender(env=self.env, cluster=self.cluster, monitoring=self.monitoring)
        self.load_balancer = LoadBalancer(cluster=self.cluster, monitoring=self.monitoring)
        self.request_generator = RequestGenerator(self.load_balancer, pod_name=self.pod_name)

    def run(self):
        self.env.process(self.monitoring.run())
        self.env.process(self.recommender.run())
        self.env.process(self.request_generator.run())
        self.env.process(self.load_balancer.run())
        self.env.run(until=1000)
        self.recommender.plot_results()


if __name__ == "__main__":
    Runner().run()
