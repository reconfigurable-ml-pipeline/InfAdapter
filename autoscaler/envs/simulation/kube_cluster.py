import numpy as np
import gym
from gym.spaces import Box
from gym.utils import seeding

from autoscaler.envs.simulation.entities import Cluster, Pod, HPA, LOAD_HISTORY_LAST_N


class SimulatedCluster(gym.Env):

    def __init__(
            self,
            pod_name: str,
            min_replicas: int,
            max_replicas: int,
            utilization_target: int,
            seed: int = None
    ):
        self.pod_name = pod_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.utilization_target = utilization_target
        self.seed = seed
        self.time_step = 0
        self.global_time_step = 0
        self.cluster = Cluster()
        self.action_space = Box(
            low=np.array([self.min_replicas]),
            high=np.array([self.max_replicas])
        )
        self.observation_space = Box(
            low=np.array([*[0.0 for _ in range(LOAD_HISTORY_LAST_N)], min_replicas]),
            high=np.array([*[1.0 for _ in range(LOAD_HISTORY_LAST_N)], max_replicas])
        )
        self.state = None
        np.random.seed(self.seed)
        self.np_random, seed = seeding.np_random(self.seed)

    def reset(self):
        self.time_step = 0
        self.cluster = Cluster()
        hpa = HPA(self.pod_name, self.utilization_target, self.min_replicas, self.max_replicas)
        pod = Pod(
            ip=Pod.IP_FORMAT.format(Pod.COUNTER),
            cpu_limit=150,
            memory_limit=int(150e6),
            cpu_request=100,
            memory_request=int(100e6),
            name=self.pod_name,
            image="nginx"
        )
        hpa.pods.append(pod)
        for i in range(self.min_replicas - 1):
            hpa.pods.append(pod.replicate())
        self.cluster.deploy_hpa(hpa)
        self.state = self.cluster.get_state(self.pod_name)
        return self.state

    def step(self, action: list):
        assert self.action_space.contains(action), f"invalid action: {action}, {type(action)}"
        self.time_step += 1
        self.global_time_step += 1
        self.cluster.master.rescale_pod(self.pod_name, action[0])
        self.state = self.cluster.get_state(self.pod_name)
        reward = 1  # Fixme: how to calculate reward
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        state = self.cluster.get_state(self.pod_name)
        load_history = state[:LOAD_HISTORY_LAST_N]
        replicas = state[LOAD_HISTORY_LAST_N]
        print(f"time step: {self.time_step}, {self.global_time_step}")
        print(f"load history in the last {LOAD_HISTORY_LAST_N} steps:")
        print(load_history)
        print(f"current replicas: {replicas}")
        print(f"utilization target: {self.utilization_target}")

    def collect_metrics(self):
        self.cluster.collect_cluster_load(self.pod_name)

    def get_state(self):
        return self.cluster.get_state(self.pod_name)

    @property
    def done(self) -> bool:
        return False
