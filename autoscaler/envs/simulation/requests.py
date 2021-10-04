import numpy as np


class Request:
    def __init__(self, now, pod_name, ID=None, done_process=None):
        self.arrived_at = now
        self.pod_name = pod_name
        self.ID = ID
        self.deadline = self.arrived_at + 2
        self.finished_at = None
        self.missed = False
        self.done_process = done_process

    def __str__(self):
        return f"Request {self.ID}"

    def miss(self):
        self.missed = True

    def done(self, now):
        self.finished_at = now
        self.done_process.interrupt(self)


class RequestGenerator:

    def __init__(self, load_balancer, pod_name: str):
        self.env = load_balancer.env
        self.load_balancer = load_balancer
        self._pod_name = pod_name

    def request_generator(self):
        while True:
            yield int(
                30 * self.load_balancer.min_replicas * (
                        max(0.45, np.sin(self.env.now * np.pi / 100) + (np.sin(self.env.now * np.pi / 175)))
                )
            )

    def generate(self):
        new_requests_count = next(self.request_generator())
        new_requests = [
            Request(
                self.env.now, self._pod_name,
            ) for _ in range(new_requests_count)
        ]
        return new_requests

    def run(self):
        while True:
            self.load_balancer.receive_requests(self.generate())
            yield self.env.timeout(1)
