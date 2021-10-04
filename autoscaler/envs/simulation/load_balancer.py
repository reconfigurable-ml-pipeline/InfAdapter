from typing import List
import simpy

from autoscaler.envs.simulation.kube_cluster import SimulatedCluster
from autoscaler.envs.simulation.monitoring import Monitoring
from autoscaler.envs.simulation.requests import Request


class LoadBalancer:
    def __init__(self, cluster: SimulatedCluster, monitoring: Monitoring):
        self.env = cluster.env
        self._workers = cluster.cluster.workers
        self.min_replicas = cluster.min_replicas
        self._monitoring = monitoring
        self._idx = 0
        self._counter = 1
        self._new_requests: List[Request] = []
        self._request_queue: List[Request] = []
        self._all_requests: List[Request] = []
        self._request_count_history = []
        self._missing_deadlines = {}

    def process_request(self, request: Request):
        worker = self._workers[self._idx]
        self._idx = (self._idx + 1) % len(self._workers)
        response = worker.process_request(request)
        return response

    def receive_requests(self, requests: List[Request]):
        self._new_requests = requests
        self._request_count_history.append(len(self._new_requests))
        i = 0
        for req in self._new_requests:
            req.ID = self._counter + i
            i += 1
        self._counter += len(self._new_requests)
        self._all_requests.extend(self._new_requests[:])

    @property
    def replica_count(self):
        count = 0
        for worker in self._workers:
            count += len(worker.pods)
        return count

    @property
    def servers_capacity(self):
        count = 0
        for worker in self._workers:
            for pod in worker.pods:
                count += (pod.MAX_REQUESTS - pod.request_count)
        return count

    def process_queue(self):
        while True:
            try:
                yield self.env.timeout(5)
            except simpy.Interrupt as i:
                request = i.cause  # type: Request
                if request.deadline < request.finished_at:
                    request.miss()
                    self._missing_deadlines[int(self.env.now) + 1] = (
                            self._missing_deadlines.get(int(self.env.now) + 1, 0) + 1
                    )
                if self._request_queue:
                    request = self._request_queue.pop(0)
                    self.env.process(self.process_request(request))

    def run(self):
        self.queue_process = self.env.process(self.process_queue())
        while True:
            servers_capacity = self.servers_capacity
            new_requests = self._new_requests[:]
            for req in new_requests:
                req.done_process = self.queue_process
            requests = self._request_queue[:servers_capacity]
            self._request_queue = self._request_queue[servers_capacity:]
            if len(requests) < servers_capacity:
                p = servers_capacity - len(requests)
                requests = requests + new_requests[:p]
                self._request_queue.extend(new_requests[p:])
            else:
                self._request_queue.extend(new_requests)
            for request in requests:
                self.env.process(self.process_request(request))

            if self.env.now % 15 == 0 and self.env.now != 0:
                total_response_time = 0
                count = 0
                for req in self._all_requests:
                    if req.arrived_at > self.env.now - 15 and req.finished_at:
                        total_response_time += req.finished_at - req.arrived_at
                        count += 1
                self._monitoring.add_record(total_response_time / count)
                self._monitoring.add_request_statistics(self._request_count_history[-15:])

            yield self.env.timeout(1)
