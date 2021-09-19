import typing
import numpy as np
from typing import List
from kubernetes.client.models import V1ResourceRequirements


LOAD_HISTORY_LAST_N = 5


def load_generator(pod: "Pod"):
    step = np.pi / 50
    x = 0
    while True:
        yield ((pod.resources.requests["cpu"] + (pod.resources.limits["cpu"])) / 2) * (1 + np.sin(x)) / 2
        x += step


class Pod:
    COUNTER = 1
    IP_FORMAT = "192.168.1.{}"

    def __init__(
            self,
            ip: str,
            name: str,
            image: str,
            memory_request: int,  # bytes
            cpu_request: int,  # milli-cores
            memory_limit: int = None,
            cpu_limit: int = None,
            labels: dict = None,
            env_vars: dict = None,
            namespace: str = "autoscaling_sim",
    ):
        self.ip = ip
        self.name = name
        self.image = image
        self.namespace = namespace
        self.labels = labels
        self.resources = V1ResourceRequirements(
            limits={"cpu": cpu_limit, "memory": memory_limit},
            requests={"cpu": cpu_request, "memory": memory_request}
        )
        self.env_vars = env_vars
        self.load_generator = load_generator(self)

    def calculate_load(self):
        return next(self.load_generator)

    def replicate(self) -> "Pod":
        Pod.COUNTER += 1
        return Pod(
            ip=self.IP_FORMAT.format(Pod.COUNTER),
            name=self.name,
            image=self.image,
            namespace=self.namespace,
            labels=self.labels.copy() if self.labels else None,
            cpu_request=self.resources.requests["cpu"],
            memory_request=self.resources.requests["memory"],
            cpu_limit=self.resources.limits["cpu"],
            memory_limit=self.resources.limits["memory"],
            env_vars=self.env_vars.copy() if self.env_vars else None
        )


# class Service:
#     def __init__(self, ip: str):
#         self.pods: List[Pod] = []
#         self.ip = ip
#         self.idx = 0
#
#     def add_pod(self, pod: Pod):
#         self.pods.append(pod)
#
#     def balance(self, request):
#         self.pods[self.idx].respond(request)
#         self.idx = (self.idx + 1) % len(self.pods)


class HPA:
    def __init__(self, pod_name, utilization_target: float, min_replicas: int, max_replicas: int):
        self.pods: List[Pod] = []
        self.utilization_target = utilization_target
        self.pod_name = pod_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    def rescale(self, new_replicas: int):
        scaled_out = False
        pods = []
        new_replicas = min(max(new_replicas, self.min_replicas), self.max_replicas)
        if len(self.pods) > new_replicas:
            for _ in range(len(self.pods) - new_replicas):
                pods.append(self.pods.pop())
        elif len(self.pods) < new_replicas:
            scaled_out = True
            for _ in range(new_replicas - len(self.pods)):
                new_replica = self.pods[0].replicate()
                self.pods.append(new_replica)
                pods.append(new_replica)
        return pods, scaled_out


class Worker:
    def __init__(self, seq_no: int, cpu_capacity: int, mem_capacity: int):
        self.id = seq_no
        self.cpu_capacity = cpu_capacity  # 8000 means 8000 milli-cores
        self.mem_capacity = mem_capacity  # bytes
        self.free_cpu = cpu_capacity
        self.free_mem = mem_capacity
        self.pods: List[Pod] = []

    def add_job(self, job: Pod):
        self.pods.append(job)

    def has_job(self, job: Pod):
        return job in self.pods

    def delete_job(self, job: Pod):
        self.pods.remove(job)

    def register(self, master: "Master"):
        master.register(self)


class Master:
    def __init__(self):
        self.workers: List[Worker] = []
        # self.services: List[Service] = []
        self.hpa_list: List[HPA] = []

    def register(self, worker: Worker):
        self.workers.append(worker)

    def schedule(self, job: Pod):
        cpu_request = job.resources.requests["cpu"]
        mem_request = job.resources.requests["memory"]

        for worker in self.workers:
            if worker.free_mem >= mem_request and worker.free_cpu >= cpu_request:
                worker.add_job(job)

    def get_hpa_object(self, pod_name) -> typing.Optional[HPA]:
        hpa = None
        for hpa_obj in self.hpa_list:
            if hpa_obj.pod_name == pod_name:
                hpa = hpa_obj
                break
        return hpa

    def get_pod_replicas(self, pod_name):
        return self.get_hpa_object(pod_name).pods

    def rescale_pod(self, pod_name, new_replicas):

        pods, scaled_out = self.get_hpa_object(pod_name).rescale(new_replicas)
        if scaled_out:
            for new_replica in pods:
                self.schedule(new_replica)
        else:
            for pod_to_delete in pods:
                for worker in self.workers:
                    if worker.has_job(pod_to_delete):
                        worker.delete_job(pod_to_delete)


class Cluster:
    def __init__(self):
        self.master = Master()
        self.workers = [Worker(i, 1000, int(1e9)) for i in range(2)]
        for worker in self.workers:
            worker.register(self.master)
        self.load_history = [0.0 for _ in range(LOAD_HISTORY_LAST_N)]  # history of average resource (CPU) utilization

    def schedule(self, job: Pod):
        self.master.schedule(job)

    def deploy_hpa(self, hpa: HPA):
        self.master.hpa_list.append(hpa)

    def collect_cluster_load(self, pod_name):
        load = 0
        requests = 0
        num_replicas = 0
        for pod in self.master.get_pod_replicas(pod_name):
            load += pod.calculate_load()
            requests += pod.resources.requests["cpu"]
            num_replicas += 1
        self.load_history.append(100 * load / requests)

    def get_state(self, pod_name) -> list:
        hpa = self.master.get_hpa_object(pod_name)
        load_history = self.load_history[-LOAD_HISTORY_LAST_N:]
        replicas = len(hpa.pods)
        return [*load_history, replicas]
