import time
from datetime import datetime
import matplotlib.pyplot as plt

from kube_resources.deployments import get_deployment
from kube_resources.configmaps import delete_configmap
from kube_resources.kserve import delete_inference_service
from kube_resources.services import delete_service as delete_kubernetes_service

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
        predictor_container_ports=[8501, 8500, 9081],
        predictor_image="mehransi/main:tfserving_resnet",
        predictor_request_mem=config.get(ParamTypes.MEMORY),
        predictor_request_cpu=config.get(ParamTypes.CPU),
        predictor_limit_mem=config.get(ParamTypes.MEMORY),
        predictor_limit_cpu=config.get(ParamTypes.CPU),
        predictor_min_replicas=config.get(ParamTypes.REPLICA),
        max_batch_size=config.get(ParamTypes.BATCH),
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


def save_results(prom: PrometheusClient, start_time: int):
    """SLO
    sum(rate(: tensorflow:serving: runtime_latency_bucket{instance = ~".*:8501", le = "708235"}[1m])) /
     sum(rate(: tensorflow:serving: request_latency_count{instance = ~".*:8501"}[1m]))

    """
    request_rates = prom.get_range(
        'sum(:tensorflow:serving:request_latency_count{instance=~".*:8501"})',
        start_time=start_time,
        end_time=int(datetime.now().timestamp()),
        step=1
    )
    print("request rates", request_rates)
    if not request_rates:
        return
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(request_rates) + 1), list(map(lambda x: x[1], request_rates)), label="request count")
    plt.legend()
    plt.savefig("load_monitoring.png", format="png")
    plt.close()
