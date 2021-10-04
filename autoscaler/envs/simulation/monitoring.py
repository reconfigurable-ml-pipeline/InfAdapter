from autoscaler.envs.simulation.kube_cluster import SimulatedCluster


class Monitoring:
    APPLICATION_METRICS_WINDOW_SIZE = 100

    def __init__(self, cluster: SimulatedCluster):
        self._cluster = cluster
        self._cluster_metrics_collection_period = None
        self.records = []
        self.request_statistics = []

    def set_cluster_metrics_collection_period(self, period: int):
        self._cluster_metrics_collection_period = period

    def add_record(self, record):
        self.records.append(record)

    def add_request_statistics(self, statistics):
        self.request_statistics.extend(statistics)

    def collect_cluster_metrics(self):
        self._cluster.cluster.collect_cluster_load(self._cluster.pod_name)

    def get_cluster_metrics(self):
        return self._cluster.get_state()

    def get_application_metrics(self):
        return self.records[-self.APPLICATION_METRICS_WINDOW_SIZE:]

    def run(self):
        while True:
            yield self._cluster.env.timeout(self._cluster_metrics_collection_period)
            self.collect_cluster_metrics()
