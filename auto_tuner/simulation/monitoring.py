from auto_tuner.simulation.kube_cluster import SimulatedCluster


class Monitoring:
    def __init__(self, cluster: SimulatedCluster):
        self._cluster = cluster
        self.cluster_metrics_collection_period = None
        self.records = []  # requests count
        self.request_statistics = []  # mean response times

    def set_cluster_metrics_collection_period(self, period: int):
        assert period > 0, "metric collection period must be a positive integer value"
        self.cluster_metrics_collection_period = period

    def add_record(self, record):
        self.records.append(record)

    def add_request_statistics(self, statistics):
        self.request_statistics.append(statistics)

    def collect_cluster_metrics(self):
        self._cluster.cluster.collect_cluster_load(self._cluster.pod_name)

    def get_cluster_metrics(self):
        return self._cluster.get_state()

    def get_application_metrics(self, last_n):
        if not last_n:
            return []
        return self.records[-last_n:]

    def run(self):
        while True:
            yield self._cluster.env.timeout(self.cluster_metrics_collection_period)
            self.collect_cluster_metrics()
