import simpy

from autoscaler.envs.simulation.kube_cluster import SimulatedCluster
from autoscaler.envs.simulation.monitoring import Monitoring


class RecommenderBase:
    collect_metrics_per = None  # in seconds
    do_scaling_per = None
    query_n_requests_from_monitoring = None

    def __init__(self, env: simpy.Environment, cluster: SimulatedCluster, monitoring: Monitoring):
        self.env = env
        self.cluster = cluster
        self.monitoring = monitoring
        self.monitoring.set_cluster_metrics_collection_period(self.collect_metrics_per)

    def recommend(self, state: list) -> int:
        raise NotImplementedError

    def plot_results(self):
        raise NotImplementedError

    def get_state(self, cluster_metrics, application_metrics):
        raise NotImplementedError

    def run(self):
        if self.collect_metrics_per is None or self.do_scaling_per is None:
            print("Set collect_metrics_per and do_scaling_per attributes in your recommender class (in minutes)")
            return
        while True:
            yield self.env.timeout(self.do_scaling_per + 0.00001)
            application_metrics = self.monitoring.get_application_metrics(self.query_n_requests_from_monitoring)
            if (
                    self.query_n_requests_from_monitoring and
                    len(application_metrics) < self.query_n_requests_from_monitoring
            ):
                continue
            state = self.get_state(
                self.monitoring.get_cluster_metrics(),
                application_metrics
            )
            action = self.recommend(state)
            self.cluster.step([action])
