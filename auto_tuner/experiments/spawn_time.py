import time
import os
import csv
import numpy as np
from datetime import datetime

from kube_resources.deployments import get_deployment
from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.parameters import ParamTypes
from auto_tuner.experiments.utils import (
    apply_config,
    delete_previous_deployment,
)



namespace = "mehran"
service_name = "tfserving-resnet"


def start_service(hardware, cpu, arch):
    config = {
        ParamTypes.REPLICA: 1,
        ParamTypes.HARDWARE: hardware,
        ParamTypes.CPU: cpu,
        ParamTypes.MEMORY: f"{cpu}G",
        ParamTypes.MODEL_ARCHITECTURE: arch,
        ParamTypes.BATCH: 1,
        ParamTypes.INTRA_OP_PARALLELISM: 1,
        ParamTypes.INTER_OP_PARALLELISM: cpu,
    }
    apply_config(
        service_name,
        namespace,
        config
    )
    return config


def delete_service():
    delete_previous_deployment(service_name, namespace)


def start_experiment(hardware, cpu, arch):
    start_service(hardware, cpu, arch)
    t = time.perf_counter()
    ready_replicas = None
    while ready_replicas is None or ready_replicas < 1:
        time.sleep(0.1)
        try:
            ready_replicas = get_deployment(service_name, namespace)["status"]["ready_replicas"]
        except Exception:
            continue
    output = time.perf_counter() - t
    time.sleep(2)
    delete_service()
    time.sleep(8)
    return output


if __name__ == "__main__":
    ip = "192.5.86.160"

    for cpu in [2, 8]:
        for arch in [18, 34, 50, 101, 152]:
            lst = []
            for repeat in range(50):
                hardware = "cpu"
                spawn_time = start_experiment(hardware, cpu, arch)
                lst.append(spawn_time)
            filepath = f'{AUTO_TUNER_DIRECTORY}/../results/spawn_time_experiment.csv'
            file_exists = os.path.exists(filepath)
            with open(
                filepath, 'a', newline=''
            ) as csvfile:
                field_names = [
                    "ARCH",
                    "CPU",
                    "min",
                    "p95",
                    "max",
                    "timestamp"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "ARCH": arch,
                        "CPU": cpu,
                        "min": round(min(lst), 2),
                        "p95": round(np.percentile(lst, 95), 2),
                        "max": round(max(lst), 2),
                        "timestamp": datetime.now().isoformat()
                    }
                )
