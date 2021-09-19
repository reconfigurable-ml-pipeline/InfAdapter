from autoscaler.envs.simulation.kube_cluster import SimulatedCluster


class RecommenderBase:
    collect_metrics_per = None  # in seconds
    do_scaling_per = None

    def __init__(self, cluster: SimulatedCluster):
        self.cluster = cluster
        self.time_step = 0

    def recommend(self, state: list) -> int:
        raise NotImplementedError

    def plot_results(self):
        raise NotImplementedError

    def run(self, run_for=1000):
        if self.collect_metrics_per is None or self.do_scaling_per is None:
            print("Set collect_metrics_per and do_scaling_per attributes in your recommender class (in minutes)")
            return
        self.cluster.reset()
        while True:
            self.time_step += 1
            if self.time_step % self.collect_metrics_per == 0:
                self.cluster.collect_metrics()
            if self.time_step % self.do_scaling_per == 0:
                state = self.cluster.get_state()
                action = self.recommend(state)
                self.cluster.step([action])
            if self.time_step == run_for:
                print("Finished simulation. going to plot some figures...")
                self.plot_results()
                return
