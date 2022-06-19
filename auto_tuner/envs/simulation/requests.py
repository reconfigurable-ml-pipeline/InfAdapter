from auto_tuner import AUTO_TUNER_DIRECTORY


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

        self._counter = 0

        with open(f"{AUTO_TUNER_DIRECTORY}/dataset/worldcup/workload.txt", "r") as f:
            reqs = f.readlines()
            reqs = reqs[0].split()
            self.workload = list(map(int, reqs))

        self.request_generator = self.request_generator_()

    def request_generator_(self):
        c = 5857043
        while True:
            if c == 5891043:
                break
            if c % 10000 == 0:
                print(f"day {c/60/60/24}")
            yield self.workload[c]
            c += 1

    def generate(self):
        try:
            new_requests_count = next(self.request_generator)
        except StopIteration:
            return None
        new_requests = [
            Request(
                self.env.now, self._pod_name, ID=self._counter + i
            ) for i in range(new_requests_count)
        ]
        self._counter += new_requests_count
        return new_requests

    def run(self):
        while True:
            requests = self.generate()
            if requests is None:
                return
            self.load_balancer.receive_requests(requests)
            yield self.env.timeout(1)
