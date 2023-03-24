import os

from auto_tuner.utils.tfserving.serving_configuration import (
    get_serving_configuration, get_batch_configuration
)
from kube_resources.configmaps import create_configmap
from kube_resources.deployments import create_deployment
from kube_resources.services import create_service as create_kubernetes_service


def deploy_ml_service(
        service_name: str,
        replicas: int, 
        active_model_version: int, 
        selector: dict, 
        namespace="default",
        mount_all_models=False,
        **kwargs
):
    model_name = "resnet"
    volume_mount_path = "/etc/tfserving"
    configmap_name = f"{service_name}-cm"
    config_volume_name = f"{service_name}-config-vol"
    models_volume_name = f"{service_name}-models-vol"
    
    kwargs.setdefault("name", f"{service_name}-container")
    kwargs.setdefault("image", "mehransi/main:tfserving_with_warmup")

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
    if max_batch_size is not None:
        enable_batching = True
    
    if not kwargs.get("env_vars"):
        kwargs["env_vars"] = {}
    kwargs["env_vars"]["MODEL_NAME"] = model_name
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
    
    labels = kwargs.pop("labels")

    if not kwargs.get("volumes"):
        kwargs["volumes"] = []
    kwargs["volumes"].append(
        {"name": config_volume_name, "config_map": {"name": configmap_name}}
    )
    warmup_volume_name = "warmup-vol"
    kwargs["volumes"].append(
        {"name": warmup_volume_name, "empty_dir": "{}"}
    )
    
    model_nfs_path = "/fileshare/tensorflow_resnet_b64"
    if mount_all_models is False:
        model_nfs_path += f"/{active_model_version}"
    kwargs["volumes"].append(
        {
            "name": models_volume_name, 
            "nfs": {
                "server": os.getenv("NFS_SERVER"),
                "path": model_nfs_path
            }
        }
    )
    
    volumes = kwargs.pop("volumes")

    if not kwargs.get("volume_mounts"):
        kwargs["volume_mounts"] = []
    mount_model_to = "/models/resnet"
    if mount_all_models is False:
        mount_model_to += f"/{active_model_version}"
    kwargs["volume_mounts"].append({"name": config_volume_name, "mount_path": volume_mount_path})
    kwargs["volume_mounts"].append({"name": models_volume_name, "mount_path": mount_model_to})
    kwargs["volume_mounts"].append({"name": warmup_volume_name, "mount_path": "/warmup"})

    create_configmap(
        configmap_name,
        namespace=namespace,
        data={
            "models.config": get_serving_configuration(
                model_name, f"/models/{model_name}/", "tensorflow", active_model_version
            ),
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
    kwargs.update(readiness_probe={"exec": ["cat", "/warmup/done"], "period_seconds": 1})
    create_deployment(
        service_name, 
        containers=[
            kwargs,
            {
                "name": "warmup",
                "image": "mehransi/main:warmup_for_tfserving",
                "env_vars": {
                    "WARMUP_COUNT": "3", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python", "PYTHONUNBUFFERED": "1"
                },
                "request_cpu": "128m",
                "request_mem": "128Mi",
                "volume_mounts": [{"name": warmup_volume_name, "mount_path": "/warmup"}]
            }
        ], 
        replicas=replicas, 
        namespace=namespace, 
        labels=labels, 
        volumes=volumes
    )
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
