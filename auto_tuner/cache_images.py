import os
import time
import json
from kube_resources.deployments import create_deployment, delete_deployment, get_deployments
from kube_resources.services import delete_service, get_services
from kube_resources.configmaps import delete_configmap, create_configmap
from auto_tuner.experiments.utils import wait_till_pods_are_ready
from auto_tuner.utils.tfserving.serving_configuration import get_batch_configuration, get_serving_configuration


baseline_accuracies = {
    18: .69758,
    34: .73314,
    50: .7613,
    101: .77374,
    152: .78312,
}

load_times = {
    18: 6.44,
    34: 6.79,
    50: 6.82,
    101: 7.77,
    152: 8.52,
}

memory_sizes = {
    18: "1.25G",
    34: "1.5G",
    50: "1.75G",
    101: "2.25G",
    152: "2.5G"
}

model_versions = [18, 34, 50, 101, 152]
base_model_name = "resnet"
base_service_name = f"tfserving-{base_model_name}"
namespace = "mehran"
node_ip = os.getenv("CLUSTER_NODE_IP")
namespace = "mehran"
adapter_svc = "adapter-test"
adapter_labels = {"project": "infadapter", "module": "adapter"}
labels = {
    "ML_framework": "tensorflow",
    "model_server": "tfserving",
}
env_vars = {
    "BASE_MODEL_NAME": base_model_name,
    "BASE_SERVICE_NAME": base_service_name,
    "MODEL_VERSIONS": json.dumps(model_versions),
    "BASELINE_ACCURACIES": json.dumps(baseline_accuracies),
    "MEMORY_SIZES": json.dumps(memory_sizes),
    "LOAD_TIMES": json.dumps(load_times),
    "PROMETHEUS_ENDPOINT": f"{node_ip}:1000",
    "NFS_SERVER_IP": os.getenv("NFS_SERVER_IP", node_ip),
    "CLUSTER_NODE_IP": node_ip,
    "CONTAINER_PORTS": json.dumps([8500, 8501]),
    "CONTAINER_LABELS": json.dumps(labels),
    "K8S_IN_CLUSTER_CLIENT": "true",
    "K8S_NAMESPACE": namespace,
    "ALPHA": str(0.2),
    "BETA": str(0.001),
    "MAX_CPU": str(20),
    "LATENCY_SLO_MS": 750,
    "LSTM_PREDICTION_ERROR_PERCENTAGE": 10,
    "EMW_PREDICTION_ERROR_PERCENTAGE": 10,
    "WARMUP_COUNT": 3,
    "FIRST_DECIDE_DELAY_MINUTES": 2,
    "SOLVER_TYPE": "i",
    "STABILIZATION_INTERVAL": 4,
    "DECISION_INTERVAL": 30,
    "VPA_TYPE": "P",
    "CAPACITY_COEF": 0.8,
}


def create_ml_service(model_version: int, size: int, deploy_version: int = None):

    service_name = base_service_name + f"-{model_version}"
    kwargs = {"name": f"{service_name}-container", "env_vars": {}}
    kwargs["env_vars"]["MODEL_NAME"] = base_model_name
    volume_mount_path = "/etc/tfserving"
    configmap_name = f"{service_name}-cm"
    config_volume_name = f"{service_name}-config-vol"
    models_volume_name = f"{service_name}-models-vol"
    warmup_volume_name = f"{service_name}-warmup-vol"
    
    
    kwargs.update({"limit_cpu": size, "limit_mem": "4G"})
    kwargs["container_ports"] = [8500, 8501]
    kwargs["image"] = "tensorflow/serving:2.8.0"

    kwargs["args"] = [
        f"--tensorflow_intra_op_parallelism=1",
        f"--tensorflow_inter_op_parallelism={size}",
        f"--rest_api_num_threads={15 * size}",
        f"--model_config_file={volume_mount_path}/models.config",
        f"--monitoring_config_file={volume_mount_path}/monitoring.config",
        f"--batching_parameters_file={volume_mount_path}/batch.config",
        "--enable_batching=true",
    ]

    labels = {"model": "resnet" + f"-{model_version}"}

    volumes = [
        {
            "name": config_volume_name,
            "config_map": {"name": configmap_name}
        },
        {
            "name": models_volume_name,
            "nfs": {"server": node_ip, "path": f"/fileshare/tensorflow_resnet_b64/{model_version}"}
        },
        {
            "name": warmup_volume_name,
            "empty_dir": "{}"
        }
    ]
    
    kwargs["volume_mounts"] = [
        {"name": config_volume_name, "mount_path": "/etc/tfserving"},
        {"name": models_volume_name, "mount_path": f"/models/{base_model_name}/{model_version}"},
        {"name": warmup_volume_name, "mount_path": "/warmup"}
    ]
    kwargs.update(readiness_probe={"exec": ["cat", "/warmup/done"], "period_seconds": 1})
    
    deployment_name = service_name
    if deploy_version is not None:
        deployment_name = deployment_name + f"-{deploy_version}"
    
    create_configmap(
        base_service_name + f"-{model_version}-cm",
        namespace=namespace,
        data={
            "models.config": get_serving_configuration(
                base_model_name, f"/models/{base_model_name}/", "tensorflow", model_version
            ),
            "monitoring.config": """
                prometheus_config {
                enable: true,
                path: "/monitoring/prometheus/metrics"
                }
            """,
            "batch.config": get_batch_configuration(
                1, 0, size, 100
            )
        }
    )
        
    create_deployment(
        deployment_name, 
        containers=[
            kwargs,
            {
                "name": "warmup",
                "image": "mehransi/main:warmup_for_tfserving",
                "env_vars": {
                    "WARMUP_COUNT": 3,
                    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                    "PYTHONUNBUFFERED": "1"
                },
                "request_cpu": "128m",
                "request_mem": "128Mi",
                "volume_mounts": [{"name": warmup_volume_name, "mount_path": "/warmup"}]
            }
        ], 
        replicas=3, 
        namespace=namespace, 
        labels=labels, 
        volumes=volumes
    )


if __name__ == "__main__":
    create_deployment(
        adapter_svc,
        replicas=3,
        namespace=namespace,
        labels=adapter_labels,
        containers=[
            {
                "name": adapter_svc + "-container",
                "image": "mehransi/main:adapter",
                "limit_cpu": 2, 
                "limit_mem": "4G",
                "request_cpu": 1, 
                "container_ports": [8000],
                "env_vars": env_vars
            }
        ]
    )

    time.sleep(2)
    wait_till_pods_are_ready(adapter_svc, namespace)
    
    create_deployment(
        "dispatcher-svc",
        containers=[
            {
                "name": "dispatcher-svc-container",
                "image": "mehransi/main:dispatcher",
                "request_cpu": 1,
                "limit_cpu": 2,
                "container_ports": [8000],
                "env_vars": {
                    "URL_PATH": "/v1/models/resnet:predict"
                }
            }
        ],
        replicas=3,
        namespace=namespace,
        labels={"project": "infadapter", "module": "dispatcher"}
    )
    time.sleep(2)
    wait_till_pods_are_ready(adapter_svc, namespace)
    
    create_ml_service(50, 4)
    time.sleep(2)
    wait_till_pods_are_ready(base_service_name + f"-50", namespace)
    
    time.sleep(30)
    # Clear the namespace
    services = get_services(namespace)
    for svc in services:
        delete_service(svc["name"], namespace)
    deployments = get_deployments(namespace)
    for deploy in deployments:
        delete_deployment(deploy["name"], namespace)
    for m in model_versions:
        try:
            delete_configmap(f"{base_service_name}-{m}-cm", namespace)
        except Exception:
            pass
    
    print("OK")
