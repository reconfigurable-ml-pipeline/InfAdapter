from auto_tuner.utils.tfserving.serving_configuration import (
    get_serving_configuration, get_batch_configuration
)
from kube_resources.configmaps import create_configmap
from kube_resources.deployments import create_deployment
from kube_resources.services import create_service as create_kubernetes_service


def deploy_ml_service(
        service_name: str,
        image:str, 
        replicas: int, 
        active_model_version: int, 
        selector: dict, 
        namespace="default", 
        **kwargs
):
    volume_mount_path = "/etc/tfserving"
    configmap_name = f"{service_name}-cm"
    config_volume_name = f"{service_name}-config-vol"
    models_volume_name = f"{service_name}-models-vol"

    max_batch_size = kwargs.pop("max_batch_size", None)
    max_batch_latency = kwargs.pop("max_batch_latency", None)
    if max_batch_latency is None:
        max_batch_latency = 10000
    num_batch_threads = kwargs.pop("num_batch_threads", None)
    if num_batch_threads is None:
        num_batch_threads = kwargs.get("request_cpu")
    max_enqueued_batches = kwargs.pop("max_enqueued_batches", None)
    if max_enqueued_batches is None:
        max_enqueued_batches = 1000000
    enable_batching = False
    if max_batch_size is not None and max_batch_size > 1:
        enable_batching = True
    
    # if not kwargs.get("env_vars"):
    #     kwargs["env_vars"] = {}
    # kwargs["env_vars"]["TF_CPP_VMODULE"] = "http_server=1"

    if not kwargs.get("args"):
        kwargs["args"] = []
    kwargs["args"].extend([
        f"--model_config_file={volume_mount_path}/models.config",
        f"--monitoring_config_file={volume_mount_path}/monitoring.config",
        f"--enable_batching={'true' if enable_batching else 'false'}",   
    ])
    if enable_batching:
        kwargs["args"].append(f"--batching_parameters_file={volume_mount_path}/batch.config")

    if not kwargs.get("labels"):
        kwargs["labels"] = {}
    kwargs["labels"].update(selector)

    if not kwargs.get("volumes"):
        kwargs["volumes"] = []
    kwargs["volumes"].append(
        {"name": config_volume_name, "config_map": {"name": configmap_name}}
    )
    kwargs["volumes"].append(
        {"name": models_volume_name, "nfs": {"server": "192.5.86.160", "path": "/fileshare/tensorflow_resnet_b64"}}
    )

    if not kwargs.get("volume_mounts"):
        kwargs["volume_mounts"] = []
    kwargs["volume_mounts"].append({"name": config_volume_name, "mount_path": "/etc/tfserving"})
    kwargs["volume_mounts"].append({"name": models_volume_name, "mount_path": "/models/resnet"})

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
            "batch.config": get_batch_configuration(
                max_batch_size, 
                max_batch_latency, 
                num_batch_threads, 
                max_enqueued_batches
            )
        }
    )
    create_deployment(service_name, image, replicas, namespace=namespace, **kwargs)
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
