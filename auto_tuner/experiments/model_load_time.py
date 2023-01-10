import time
import os
import csv
from datetime import datetime

from kube_resources.services import get_service
from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.parameters import ParamTypes
from auto_tuner.experiments.utils import (
    apply_config,
    wait_till_pods_are_ready,
    delete_previous_deployment,
)
from auto_tuner.utils.prometheus import PrometheusClient

from auto_tuner.utils.tfserving.model_switching import request_pod_to_switch_model_version



namespace = "mehran"
service_name = "tfserving-resnet"


def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v


def start_service(hardware, cpu):
    config = {
        ParamTypes.REPLICA: 1,
        ParamTypes.HARDWARE: hardware,
        ParamTypes.CPU: cpu,
        ParamTypes.MEMORY: "2G",
        ParamTypes.MODEL_ARCHITECTURE: 18,
        ParamTypes.BATCH: 1,
        ParamTypes.INTRA_OP_PARALLELISM: 1,
        ParamTypes.INTER_OP_PARALLELISM: cpu,
    }
    apply_config(
        service_name,
        namespace,
        config
    )
    time.sleep(5)
    wait_till_pods_are_ready(service_name, namespace)
    time.sleep(5)
    return config


def delete_service():
    delete_previous_deployment(service_name, namespace)


def start_experiment(hardware, cpu, ip, prom):
    start_service(hardware, cpu)
    
    port = get_service(f"tfserving-resnet-grpc", "mehran")["node_port"]
    repeat = 5
    output = {}
    for _ in range(repeat):
        for arch in [18, 34, 50, 101, 152]:
            r = request_pod_to_switch_model_version(f"{ip}:{port}", arch)
            
            time.sleep(10)
            load_latency = prom.get_instant(
                f':tensorflow:cc:saved_model:load_latency{{container="{service_name}", model_path="/models/resnet/{arch}"}}'
            )
            load_latency = _get_value(load_latency, divide_by=1000)
            load_latency = load_latency - sum(output.get(arch, []))
            # print(f"load_latency for model {arch} with cpu={cpu} is: {load_latency}")
            output[arch] = output.get(arch, []) + [load_latency]
    
    print(f"output for cpu={cpu}", output)
    for k in output:
        output[k] = max(output[k])
    
    delete_service()
    time.sleep(15)
    
    return output

    
    
if __name__ == "__main__":
    ip = "192.5.86.160"
    prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
    prom = PrometheusClient(ip, prom_port)
    for cpu in range(1, 5):
        hardware = "cpu"
        models_load_latency = start_experiment(hardware, cpu, ip, prom)
        
        filepath = f'{AUTO_TUNER_DIRECTORY}/../results/model_load_time_experiment.csv'
        file_exists = os.path.exists(filepath)
        with open(
            filepath, 'a', newline=''
        ) as csvfile:
            field_names = [
                "ARCH",
                "CPU",
                "load_latency",
                "timestamp"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()
            for model, load_latency in models_load_latency.items():
                writer.writerow(
                    {
                        "ARCH": model,
                        "CPU": cpu,
                        "load_latency": load_latency,
                        "timestamp": datetime.now().isoformat()
                    }
                )