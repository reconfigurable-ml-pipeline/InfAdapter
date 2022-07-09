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
    max_batch_latency = kwargs.pop("max_batch_latency", 10)  # Fixme: add to config space
    

    if not kwargs.get("predictor_args"):
        kwargs["predictor_args"] = []
    kwargs["predictor_args"].extend([
        f"--model_config_file={volume_mount_path}/models.config",
        f"--monitoring_config_file={volume_mount_path}/monitoring.config",
        "--enable_batching=true",
        f"--batching_parameters_file={volume_mount_path}/batch.config"
    ])

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
            # Fixme: Add num_batch_size to configuration space
            "batch.config": get_batch_configuration(max_batch_size, max_batch_latency, kwargs["predictor_request_cpu"])
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
