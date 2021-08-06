from kubernetes.client.models import V1ResourceRequirements


class Pod:
    def __init__(
            self,
            name: str,
            image: str,
            labels: dict = None,
            memory_request: str = None,
            cpu_request: str = None,
            memory_limit: str = None,
            cpu_limit: str = None,
            env_vars: dict = None,
            namespace: str = "autoscaling_sim",
    ):
        self.ip = None  # generate unique ip in namespace
        self.name = name
        self.image = image
        self.namespace = namespace
        self.labels = labels
        self.resources = V1ResourceRequirements(
            limits={"cpu": cpu_limit, "memory": memory_limit},
            requests={"cpu": cpu_request, "memory": memory_request}
        )
        self.env_vars = env_vars


class Node:
    def __init__(self, cpu_capacity, mem_capacity):
        self.cpu_capacity = cpu_capacity  # 8000 means 8000 milli-cores
        self.mem_capacity = mem_capacity  # bytes
        self.free_cpu = cpu_capacity
        self.free_mem = mem_capacity
        self.pods = []
        self.services = []
        self.hpas = []

    def add_job(self, job):
        self.pods.append(job)


class Cluster:
    def __init__(self):
        self.nodes = []

    def schedule(self, job):
        #  choose a suitable node to schedule job
        pass
