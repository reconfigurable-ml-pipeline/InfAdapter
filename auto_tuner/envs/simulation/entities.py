import typing

import numpy as np
from typing import List

import simpy
from kubernetes.client.models import V1ResourceRequirements

from auto_tuner.envs.simulation.requests import Request


class Pod:
    COUNTER = 1
    IP_FORMAT = "192.168.1.{}"
    MAX_REQUESTS = 8

    def __init__(
            self,
            ip: str,
            name: str,
            image: str,
            memory_request: int,  # bytes
            cpu_request: int,  # milli-cores
            memory_limit: int = None,
            cpu_limit: int = None,
            namespace: str = "autoscaling_sim",
    ):
        self.ip = ip
        self.name = name
        self.image = image
        self.namespace = namespace
        self.resources = V1ResourceRequirements(
            limits={"cpu": cpu_limit, "memory": memory_limit},
            requests={"cpu": cpu_request, "memory": memory_request}
        )
        self.request_count = 0

    def process_request(self, env: simpy.Environment, request: Request) -> dict:
        self.request_count += 1
        yield env.timeout(max(.1, np.random.random() / 2))
        # calculate response time, missing deadline (maybe through load balancer)
        self.request_count -= 1
        request.done(env.now)
        return {"success": True}

    def calculate_cpu_load(self):
        return (self.request_count / self.MAX_REQUESTS) * self.resources.requests["cpu"]

    def replicate(self) -> "Pod":
        Pod.COUNTER += 1
        return Pod(
            ip=self.IP_FORMAT.format(Pod.COUNTER),
            name=self.name,
            image=self.image,
            namespace=self.namespace,
            cpu_request=self.resources.requests["cpu"],
            memory_request=self.resources.requests["memory"],
            cpu_limit=self.resources.limits["cpu"],
            memory_limit=self.resources.limits["memory"],
        )


class Service:
    def __init__(self, pod_name: str, ip: str, port: int):
        self.pod_name = pod_name
        self.pods: List[Pod] = []
        self.ip = ip
        self.port = port
        self.idx = -1

    def add_pod(self, pod: Pod):
        self.pods.append(pod)

    def balance(self, env: simpy.Environment, request: Request) -> dict:
        self.idx = (self.idx + 1) % len(self.pods)
        response = self.pods[self.idx].process_request(env, request)
        return response


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
        new_replicas = int(min(max(new_replicas, self.min_replicas), self.max_replicas))
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
        cpu_request = job.resources.requests["cpu"]
        mem_request = job.resources.requests["memory"]
        self.free_mem -= mem_request
        self.free_cpu -= cpu_request
        self.pods.append(job)

    def has_job(self, job: Pod):
        return job in self.pods

    def delete_job(self, job: Pod):
        self.pods.remove(job)

    def register(self, master: "Master"):
        self.master = master
        self.env = self.master.env
        self.master.register(self)

    def process_request(self, request: Request):
        service = self.master.get_service_object(request.pod_name)
        response = service.balance(self.env, request)
        return response


class Master:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.workers: List[Worker] = []
        self.services: List[Service] = []
        self.hpa_list: List[HPA] = []

    def register(self, worker: Worker):
        self.workers.append(worker)

    def schedule(self, job: Pod):
        cpu_request = job.resources.requests["cpu"]
        mem_request = job.resources.requests["memory"]

        for worker in self.workers:
            if worker.free_mem >= mem_request and worker.free_cpu >= cpu_request:
                worker.add_job(job)
                break

    def get_hpa_object(self, pod_name) -> typing.Optional[HPA]:
        hpa = None
        for hpa_obj in self.hpa_list:
            if hpa_obj.pod_name == pod_name:
                hpa = hpa_obj
                break
        return hpa

    def get_service_object(self, pod_name) -> typing.Optional[Service]:
        service = None
        for svc in self.services:
            if svc.pod_name == pod_name:
                service = svc
                break
        return service

    def get_pod_replicas(self, pod_name):
        return self.get_hpa_object(pod_name).pods

    def rescale_pod(self, pod_name, new_replicas):
        service = self.get_service_object(pod_name)
        pods, scaled_out = self.get_hpa_object(pod_name).rescale(new_replicas)
        if scaled_out:
            for new_replica in pods:
                self.schedule(new_replica)
                service.pods.append(new_replica)
        else:
            for pod_to_delete in pods:
                service.pods.remove(pod_to_delete)
                for worker in self.workers:
                    if worker.has_job(pod_to_delete):
                        worker.delete_job(pod_to_delete)


class Cluster:
    LOAD_HISTORY_LAST_N = 100

    def __init__(self, env: simpy.Environment):
        self.env = env
        self.master = Master(self.env)
        self.workers = [Worker(i, 16000, int(16e9)) for i in range(2)]
        for worker in self.workers:
            worker.register(self.master)
        self._cpu_load_history = [0.0 for _ in range(self.LOAD_HISTORY_LAST_N)]  # history of average CPU utilization

    def schedule(self, job: Pod):
        self.master.schedule(job)

    def deploy_hpa(self, hpa: HPA):
        self.master.hpa_list.append(hpa)

    def deploy_service(self, service: Service):
        self.master.services.append(service)

    def collect_cluster_load(self, pod_name):
        load = 0
        requests = 0
        for pod in self.master.get_pod_replicas(pod_name):
            load += pod.calculate_cpu_load()
            requests += pod.resources.requests["cpu"]
        self._cpu_load_history.append(100 * load / requests)

    def get_state(self, pod_name) -> list:
        hpa = self.master.get_hpa_object(pod_name)
        load_history = self._cpu_load_history[-self.LOAD_HISTORY_LAST_N:]
        replicas = len(hpa.pods)
        return [*load_history, replicas]
