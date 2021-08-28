from autoscaler.envs.simulation.recommender import Recommender
from autoscaler.envs.simulation.kube_cluster import SimulatedCluster


class Runner:
    def __init__(self):
        self.pod_name = "nginx"
        self.cluster = SimulatedCluster(
            pod_name=self.pod_name, min_replicas=1, max_replicas=20, utilization_target=50, seed=31
        )
        self.recommender = Recommender(self.cluster)

    def run(self):
        self.recommender.run()


if __name__ == "__main__":
    Runner().run()
