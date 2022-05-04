import time

from kubernetes.client.models import V1ConfigMap

from auto_tuner.utils.kube_resources import core_api as api
from auto_tuner.utils.kube_resources.utils import construct_configmap


def _get_configmap_info(cm: V1ConfigMap) -> dict:
    return {
        "Kind": "ConfigMap",
        "namespace": cm.metadata.namespace,
        "name": cm.metadata.name,
        "data": cm.data
    }


def get_configmap(configmap_name, namespace="default") -> dict:
    response = api.read_namespaced_config_map(configmap_name, namespace)
    return _get_configmap_info(response)


def create_configmap(configmap_name: str, data: dict, namespace="default") -> dict:
    cm = construct_configmap(name=configmap_name, namespace=namespace, data=data)
    response = api.create_namespaced_config_map(namespace, cm)
    return _get_configmap_info(response)


def update_configmap(configmap_name: str, data: dict, namespace="default", partial=False) -> dict:
    old_cm = get_configmap(configmap_name, namespace)
    if partial:
        data = {**old_cm["data"], **data}

    cm = construct_configmap(name=configmap_name, namespace=namespace, data=data)
    response = api.replace_namespaced_config_map(namespace=namespace, name=configmap_name, body=cm)
    return _get_configmap_info(response)


def delete_configmap(configmap_name: str, namespace="default"):
    response = api.delete_namespaced_config_map(name=configmap_name, namespace=namespace)
    return {"status": response.status}
