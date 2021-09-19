from autoscaler.recommenders import recommenders
from autoscaler.envs.simulation.kube_cluster import SimulatedCluster


class Runner:
    def __init__(self):
        self.pod_name = "nginx"
        self.cluster = SimulatedCluster(
            pod_name=self.pod_name, min_replicas=10, max_replicas=100, utilization_target=70, seed=31
        )
        self.recommender = recommenders.HPARecommender(self.cluster)

    def run(self):
        self.recommender.run(run_for=1750)


if __name__ == "__main__":
    Runner().run()
