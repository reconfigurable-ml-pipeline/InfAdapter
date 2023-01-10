import time
import os
from datetime import datetime
import csv

from kube_resources.deployments import get_deployment
from kube_resources.configmaps import delete_configmap
from kube_resources.deployments import delete_deployment
from kube_resources.services import delete_service as delete_kubernetes_service

from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.parameters import ParamTypes
from auto_tuner.utils.tfserving import deploy_ml_service
from auto_tuner.utils.prometheus import PrometheusClient


def is_config_valid(c: dict) -> bool:
    cpu = c.get(ParamTypes.CPU)
    memory = c.get(ParamTypes.MEMORY)
    replicas = c.get(ParamTypes.REPLICA)

    return True


def apply_config(service_name: str, namespace: str, config: dict, hardware = ParamTypes.HARDWARE_CPU):
    if hardware == ParamTypes.HARDWARE_CPU:
        image = "tensorflow/serving:2.8.0"
    else:
        image = "tensorflow/serving:2.8.0-gpu"
    deploy_ml_service(
        service_name=service_name,
        image=image,
        replicas=config.get(ParamTypes.REPLICA),
        active_model_version=config.get(ParamTypes.MODEL_ARCHITECTURE),
        namespace=namespace,
        selector={"ML_framework": "tensorflow", "model_server": "tfserving"},
        container_ports=[8501, 8500],
        request_mem=config.get(ParamTypes.MEMORY),
        request_cpu=config.get(ParamTypes.CPU),
        limit_mem=config.get(ParamTypes.MEMORY),
        limit_cpu=config.get(ParamTypes.CPU),
        max_batch_size=config.get(ParamTypes.BATCH),
        max_batch_latency=config.get(ParamTypes.BATCH_TIMEOUT),
        num_batch_threads=config.get(ParamTypes.NUM_BATCH_THREADS),
        max_enqueued_batches=config.get(ParamTypes.MAX_ENQUEUED_BATCHES),
        args=[
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
    delete_deployment(service_name, namespace=namespace)
    delete_configmap(f"{service_name}-cm", namespace)
    delete_kubernetes_service(f"{service_name}-rest", namespace=namespace)
    delete_kubernetes_service(f"{service_name}-grpc", namespace=namespace)
    # delete_kubernetes_service(f"{service_name}-batch", namespace=namespace)


def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v


def save_results(config: dict, prom: PrometheusClient, total: int, failed: int, start_time: float):
    # Todo: Add CPU and memory usage
    duration_int = round(datetime.now().timestamp() - start_time)
    percent_708ms = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501", le = "708235"}}[{duration_int}s]))'
        f' / sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    average_latency = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_sum{{instance=~".*:8501"}}[{duration_int}s]))'
        f' / sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    average_latency = _get_value(average_latency, divide_by=1000, should_round=False)
    percentile_99 = prom.get_instant(
        f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501"}}[{duration_int}s])) by (le))'
    )
    rate = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    queueing_delay_p99 = prom.get_instant(
        f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:batching_session:queuing_latency_bucket{{instance=~".*:8501"}}[{duration_int}s])) by(le))'
    )
    filepath = f'{AUTO_TUNER_DIRECTORY}/../results/experiment_result.csv'
    file_exists = os.path.exists(filepath)
    row = {
        **config,
        "percent_708ms": _get_value(percent_708ms),
        "rate": _get_value(rate),
        "1/avg_latency": round(1000 / average_latency, 2) if average_latency else None,
        "p99": _get_value(percentile_99, divide_by=1000),
        "p99_queueing": _get_value(queueing_delay_p99, 1000),
        "failed": failed,
        "total": total,
        "duration": duration_int,
        "timestamp": datetime.now().isoformat()
    }
    with open(
        filepath, 'a', newline=''
    ) as csvfile:
        field_names = list(row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
