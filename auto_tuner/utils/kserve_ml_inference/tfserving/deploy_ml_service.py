from auto_tuner.utils.kserve_ml_inference.tfserving.serving_configuration import (
    get_serving_configuration, get_batch_configuration
)
from kube_resources.configmaps import create_configmap
from kube_resources.kserve import create_inference_service
from kube_resources.services import create_service as create_kubernetes_service


def deploy_ml_service(
        service_name: str, active_model_version: int, selector: dict, namespace="default", **kwargs
):
    volume_mount_path = "/etc/tfserving"
    configmap_name = f"{service_name}-cm"
    volume_name = f"{service_name}-vol"

    max_batch_size = kwargs.pop("max_batch_size", None)
    max_batch_latency = kwargs.pop("max_batch_latency", 10000)
    num_batch_threads = kwargs.pop("num_batch_threads", kwargs.get("predictor_request_cpu"))
    max_enqueued_batches = kwargs.pop("max_enqueued_batches", 1000000)
    enable_batching = False
    if max_batch_size is not None and max_batch_size > 1:
        enable_batching = True
    
    if not kwargs.get("predictor_env_vars"):
        kwargs["predictor_env_vars"] = {}
    kwargs["predictor_env_vars"]["TF_CPP_VMODULE"] = "http_server=1"

    if not kwargs.get("predictor_args"):
        kwargs["predictor_args"] = []
    kwargs["predictor_args"].extend([
        f"--model_config_file={volume_mount_path}/models.config",
        f"--monitoring_config_file={volume_mount_path}/monitoring.config",
        f"--enable_batching={'true' if enable_batching else 'false'}",   
    ])
    if enable_batching:
        kwargs["predictor_args"].append(f"--batching_parameters_file={volume_mount_path}/batch.config")

    if not kwargs.get("labels"):
        kwargs["labels"] = {}
    kwargs["labels"].update(selector)

    if not kwargs.get("volumes"):
        kwargs["volumes"] = []
    kwargs["volumes"].append({"name": volume_name, "config_map": {"name": configmap_name}})

    if not kwargs.get("predictor_volume_mounts"):
        kwargs["predictor_volume_mounts"] = []
    kwargs["predictor_volume_mounts"].append({"name": volume_name, "mount_path": "/etc/tfserving"})

    create_configmap(
        configmap_name,
        namespace=namespace,
        data={
            "models.config": get_serving_configuration("resnet", "/models/resnet/", "tensorflow", active_model_version),
            "monitoring.config": """
                prometheus_config {
                  enable: true,
                  path: "/monitoring/prometheus/metrics"
                }
            """,
            "batch.config": get_batch_configuration(max_batch_size, max_batch_latency, num_batch_threads, max_enqueued_batches)
        }
    )
    create_inference_service(service_name, namespace, **kwargs)
    create_kubernetes_service(
        f"{service_name}-grpc",
        target_port=8500,
        port=8500,
        namespace=namespace,
        expose_type="NodePort",
        selector=selector
    )
    create_kubernetes_service(
        f"{service_name}-rest",
        target_port=8501,
        port=8501,
        namespace=namespace,
        expose_type="NodePort",
        selector=selector
    )
    # create_kubernetes_service(
    #     f"{service_name}-batch",
    #     target_port=9081,
    #     port=9081,
    #     namespace=namespace,
    #     expose_type="NodePort",
    #     selector=selector
    # )
