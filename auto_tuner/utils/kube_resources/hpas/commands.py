import time

from kubernetes.client.models import V1HorizontalPodAutoscaler

from auto_tuner.utils.kube_resources.utils import construct_hpa
from auto_tuner.utils.kube_resources import autoscaling_api as api


def _get_hpa_info(hpa: V1HorizontalPodAutoscaler):
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


def create_hpa(
        name: str,
        target_cpu_utilization: int,
        min_replicas: int,
        max_replicas: int,
        target_api_version: str,
        target_kind: str,
        target_name: str,
        namespace="default"
):
    hpa = construct_hpa(
        name=name,
        namespace=namespace,
        target_cpu_utilization=target_cpu_utilization,
        max_replicas=max_replicas,
        min_replicas=min_replicas,
        target_api_version=target_api_version,
        target_kind=target_kind,
        target_name=target_name
    )
    response = api.create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=hpa)
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


def update_hpa(
        name,
        target_cpu_utilization: int = None,
        min_replicas: int = None,
        max_replicas: int = None,
        target_api_version: str = None,
        target_kind: str = None,
        target_name: str = None,
        partial=False,
        namespace="default"
):
    hpa = api.read_namespaced_horizontal_pod_autoscaler(name=name, namespace=namespace)

    hpa = construct_hpa(
        name=name,
        namespace=hpa.metadata.namespace,
        target_cpu_utilization=target_cpu_utilization or hpa.spec.target_cpu_utilization_percentage,
        max_replicas=max_replicas or hpa.spec.max_replicas,
        min_replicas=min_replicas or hpa.spec.min_replicas,
        target_api_version=target_api_version or hpa.spec.scale_target_ref.api_version,
        target_kind=target_kind or hpa.spec.scale_target_ref.kind,
        target_name=target_name or hpa.spec.scale_target_ref.name
    )
    if partial:
        response = api.patch_namespaced_horizontal_pod_autoscaler(
            name=name, namespace=namespace, body=hpa
        )
    else:
        response = api.replace_namespaced_horizontal_pod_autoscaler(
            name=name, namespace=namespace, body=hpa
        )
    time.sleep(1)
    return get_hpa(response.metadata.name, namespace)


def delete_hpa(name, namespace="default"):
    response = api.delete_namespaced_horizontal_pod_autoscaler(name=name, namespace=namespace)
    return {"status": response.status}
