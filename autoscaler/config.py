import simpy


def get_environment_config():
    return {
        "env": simpy.Environment(),
        "pod_name": "nginx",
        "min_replicas": 10,
        "max_replicas": 500,
        "utilization_target": 70,
        "seed": 31
    }