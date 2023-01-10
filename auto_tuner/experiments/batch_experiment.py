import requests
from kube_resources.services import get_service
from auto_tuner import AUTO_TUNER_DIRECTORY
import requests
import time
import numpy as np
import csv
import json
import os
from datetime import datetime
from auto_tuner.parameters import ParamTypes
from auto_tuner.experiments.utils import apply_config, delete_previous_deployment, wait_till_pods_are_ready


images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)
)


def start(cpu, mem, arch, intra_op, inter_op):
    namespace = "mehran"
    service_name = "tfserving-resnet"
    apply_config(
        service_name,
        namespace,
        {
            "CPU": cpu,
            "MEM": mem,
            "ARCH": arch,
            "BATCH": 1,
            "INTRA_OP_PARALLELISM":intra_op,
            "INTER_OP_PARALLELISM": inter_op,
            "REPLICA": 1
        }
    )
    time.sleep(5)
    wait_till_pods_are_ready(f"{service_name}-predictor-default", namespace)
    ip = "192.5.86.160"
    port = get_service(f"{service_name}-rest", namespace)["node_port"]
    url = f"http://{ip}:{port}/v1/models/resnet:predict"
    for warmup in range(10):
        requests.post(url, data=images[np.random.randint(0, 200)]["data"])
    

    latencies = {}
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        latencies[batch_size] = []
        batch_images = images[:batch_size]
        d = {"inputs": [json.loads(di["data"])["inputs"][0] for di in batch_images]}
        for repeat in range(40):
            t = time.perf_counter()
            response = requests.post(url, json=d)
            if response.status_code == 200:
                latencies[batch_size].append(time.perf_counter() - t)
            else:
                print(response.json())
    delete_previous_deployment(service_name, namespace)
    return {k: 
        {
            "p99": round(np.percentile(v, 99), 2),
            "min": round(min(v), 2),
            "max": round(max(v), 2),
            "avg": round(sum(v) / len(v), 2)
        }
        for k, v in latencies.items()
    }


if __name__ == "__main__":
    memory = "4G"
    for cpu in [2, 4, 16]:
        for model_version in [18, 152]:
            for inter_op in [1, cpu]:
                for intra_op in [1, cpu]:
                    if inter_op == 1 and intra_op == 1:
                        continue
                    result = start(cpu, memory, model_version, intra_op=intra_op, inter_op=inter_op)
                    filepath = f'{AUTO_TUNER_DIRECTORY}/../results/batch_result.csv'
                    file_exists = os.path.exists(filepath)
                    with open(
                        filepath, 'a', newline=''
                    ) as csvfile:
                        field_names = [
                            ParamTypes.CPU,
                            ParamTypes.MEMORY,
                            ParamTypes.MODEL_ARCHITECTURE,
                            ParamTypes.INTER_OP_PARALLELISM,
                            ParamTypes.INTRA_OP_PARALLELISM,
                            ParamTypes.BATCH,
                            "p99",
                            "avg",
                            "min",
                            "max",
                            "timestamp"
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        if not file_exists:
                            writer.writeheader()
                        for k, v in result.items():
                            writer.writerow(
                                {
                                    **v,
                                    ParamTypes.CPU: cpu,
                                    ParamTypes.MEMORY: memory,
                                    ParamTypes.MODEL_ARCHITECTURE: model_version,
                                    ParamTypes.INTER_OP_PARALLELISM: inter_op,
                                    ParamTypes.INTRA_OP_PARALLELISM: intra_op,
                                    ParamTypes.BATCH: k,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                    time.sleep(5)
