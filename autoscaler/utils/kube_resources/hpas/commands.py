import time
import yaml

from pykube.core import autoscaling_api as api


def _get_hpa_info(hpa):
    return {
        "kind": "HorizontalPodAutoscaler",
        "namespace": hpa.metadata.namespace,
        "name": hpa.metadata.name,
        "max_replicas": hpa.spec.max_replicas,
        "min_replicas": hpa.spec.min_replicas,
        "target": {
            "api_version": hpa.spec.scale_target_ref.api_version,
            "kind": hpa.spec.scale_target_ref.kind,
            "name": hpa.spec.scale_target_ref.name,
        },
        "target_cpu_utilization_percentage": hpa.spec.target_cpu_utilization_percentage,
        "status": {
            "current_cpu_utilization": hpa.status.current_cpu_utilization_percentage,
            "current_replicas": hpa.status.current_replicas,
            "desired_replicas": hpa.status.desired_replicas,
        }
    }

def create_hpa(file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=yaml_content)
        time.sleep(1)
        return get_hpa(response.metadata.name, namespace)


def get_hpas(namespace="default"):
    if namespace == "all":
        response = api.list_horizontal_pod_autoscaler_for_all_namespaces(watch=False)
    else:
        response = api.list_namespaced_horizontal_pod_autoscaler(namespace, watch=False)
    return list(
        map(
            lambda hpa: _get_hpa_info(hpa),
            response.items
        )
    )


def get_hpa(autoscaler_name, namespace="default"):
    response = api.read_namespaced_horizontal_pod_autoscaler(name=autoscaler_name, namespace=namespace)
    return _get_hpa_info(response)


def update_hpa(autoscaler_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.replace_namespaced_horizontal_pod_autoscaler(
            name=autoscaler_name, namespace=namespace, body=yaml_content
        )
        time.sleep(1)
        return get_hpa(response.metadata.name, namespace)


def partially_update_hpa(autoscaler_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.patch_namespaced_horizontal_pod_autoscaler(
            name=autoscaler_name, namespace=namespace, body=yaml_content
        )
        time.sleep(1)
        return get_hpa(response.metadata.name, namespace)


def delete_hpa(autoscaler_name, namespace="default"):
    response = api.delete_namespaced_horizontal_pod_autoscaler(name=autoscaler_name, namespace=namespace)
    return {"status": response.status}
