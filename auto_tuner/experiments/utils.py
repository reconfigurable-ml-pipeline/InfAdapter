import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import csv

from kube_resources.deployments import get_deployment
from kube_resources.configmaps import delete_configmap
from kube_resources.kserve import delete_inference_service
from kube_resources.services import delete_service as delete_kubernetes_service

from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.experiments.parameters import ParamTypes
from auto_tuner.utils.kserve_ml_inference.tfserving import deploy_ml_service
from auto_tuner.utils.prometheus import PrometheusClient


def is_config_valid(c: dict) -> bool:
    cpu = c.get(ParamTypes.CPU)
    memory = c.get(ParamTypes.MEMORY)
    replicas = c.get(ParamTypes.REPLICA)

    return True


def apply_config(service_name: str, namespace: str, config: dict):
    deploy_ml_service(
        service_name=service_name,
        active_model_version=config.get(ParamTypes.MODEL_ARCHITECTURE),
        namespace=namespace,
        selector={"inference_framework": "kserve", "ML_framework": "tensorflow", "model_server": "tfserving"},
        predictor_container_ports=[8501, 8500],
        # predictor_container_ports=[8501, 8500, 9081],
        predictor_image="mehransi/main:tfserving_resnet_b64",
        predictor_request_mem=config.get(ParamTypes.MEMORY),
        predictor_request_cpu=config.get(ParamTypes.CPU),
        predictor_limit_mem=config.get(ParamTypes.MEMORY),
        predictor_limit_cpu=config.get(ParamTypes.CPU),
        predictor_min_replicas=config.get(ParamTypes.REPLICA),
        max_batch_size=config.get(ParamTypes.BATCH),
        max_batch_latency=config.get(ParamTypes.BATCH_TIMEOUT),
        num_batch_threads=config.get(ParamTypes.NUM_BATCH_THREADS),
        max_enqueued_batches=config.get(ParamTypes.MAX_ENQUEUED_BATCHES),
        predictor_args=[
            f"--tensorflow_intra_op_parallelism={config.get(ParamTypes.INTRA_OP_PARALLELISM)}",
            f"--tensorflow_inter_op_parallelism={config.get(ParamTypes.INTER_OP_PARALLELISM)}"
        ]
    )


def wait_till_pods_are_ready(deploy_name: str, namespace: str):
    deployment = get_deployment(deploy_name, namespace=namespace)
    replicas = deployment["replicas"]
    ready_replicas = deployment["status"]["ready_replicas"]
    while ready_replicas is None or ready_replicas < replicas:
        time.sleep(1)
        ready_replicas = get_deployment(deploy_name, namespace)["status"]["ready_replicas"]


def delete_previous_deployment(service_name: str, namespace: str):
    delete_inference_service(service_name, namespace=namespace)
    delete_configmap(f"{service_name}-cm", namespace)
    delete_kubernetes_service(f"{service_name}-rest", namespace=namespace)
    delete_kubernetes_service(f"{service_name}-grpc", namespace=namespace)
    # delete_kubernetes_service(f"{service_name}-batch", namespace=namespace)


def _get_value(prom_res):
    for tup in prom_res:
        if tup[1] != "NaN":
            v = float(tup[1])
            if v > 1:
                v = v / 1000
            return round(v, 2)


def save_results(config: dict, prom: PrometheusClient, total: int, failed: int, start_time: int):
    # Todo: Add CPU and memory usage
    percent_708ms = prom.get_instant(
        'sum(rate(:tensorflow:serving:runtime_latency_bucket{instance=~".*:8501", le = "708235"}[5m]))'
        ' / sum(rate(:tensorflow:serving:request_latency_count{instance=~".*:8501"}[5m]))'
    )
    percentile_50 = prom.get_instant(
        'histogram_quantile(0.50, rate(:tensorflow:serving:request_latency_bucket{instance=~".*:8501"}[5m]))'
    )
    percentile_95 = prom.get_instant(
        'histogram_quantile(0.95, rate(:tensorflow:serving:request_latency_bucket{instance=~".*:8501"}[5m]))'
    )
    percentile_99 = prom.get_instant(
        'histogram_quantile(0.99, rate(:tensorflow:serving:request_latency_bucket{instance=~".*:8501"}[5m]))'
    )
    filepath = f'{AUTO_TUNER_DIRECTORY}/../results/experiment_result.csv'
    file_exists = os.path.exists(filepath)
    with open(
        filepath, 'a', newline=''
    ) as csvfile:
        params = ParamTypes.get_all()
        field_names = [*params, "percent_708ms", "p50", "p95", "p99", "failed", "total", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                **config,
                "percent_708ms": _get_value(percent_708ms),
                "p50": _get_value(percentile_50),
                "p95": _get_value(percentile_95),
                "p99": _get_value(percentile_99),
                "failed": failed,
                "total": total,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # print("------------------------------")
    # end_time = (datetime.now() + timedelta(seconds=20)).timestamp()
    # request_rates = prom.get_range(
    #     'sum(:tensorflow:serving:request_latency_count{instance=~".*:8501"})',
    #     start_time=start_time,
    #     end_time=end_time,
    #     step=1
    # )
    # if not request_rates:
    #     return
    
    # values = list(map(lambda x: int(x[1]), request_rates))
    # print("last value", values[-1])
    # print("monitoring workload", len(values), values)
    # for i in range(len(values)-1, 0, -1):
    #     values[i] = values[i] - values[i-1]
    # while True:
    #     if values[-1] == 0:
    #         values.pop(-1)
    #     else:
    #         break
    # if values[:2] in [[0,3], [3,0]]:
    #     values = values[2:]
    # elif values[0] == 3:
    #     values = values[1:]
    # print("monitoring workload after", len(values), values)
    
    # print("sum values", sum(values))

    # plt.xlabel("time (seconds)")
    # plt.plot(range(1, len(values) + 1), values, label="request count")
    # plt.legend()
    # plt.savefig("load_monitoring.png", format="png")
    # plt.close()


def save_load_results(config: dict, total: int, result: dict):
    filepath = f'{AUTO_TUNER_DIRECTORY}/../results/load_result.csv'
    file_exists = os.path.exists(filepath)
    with open(
        filepath, 'a', newline=''
    ) as csvfile:
        params = ParamTypes.get_all()
        field_names = [*params, "total", *result.keys(), "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                **config,
                "total": total,
                **result,
                "timestamp": datetime.now().isoformat()
            }
        )
