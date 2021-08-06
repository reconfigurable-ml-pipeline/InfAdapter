from kubernetes.client.models import V1Service

from autoscaler.utils.kube_resources.utils import construct_service
from autoscaler.utils.kube_resources import core_api as api


def _get_service_info(service: V1Service):
    return {
        "namespace": service.metadata.namespace,
        "name": service.metadata.name,
        "port": service.spec.ports[0].port,
        "target_port": service.spec.ports[0].target_port,
        "port_name": service.spec.ports[0].name,
        "protocol": service.spec.ports[0].protocol,
        "selector": service.spec.selector
    }


def create_service(
        name: str,
        port: int,
        target_port: int,
        port_name: str,
        selector: dict,
        protocol: str = None,
        namespace="autoscaling"
):
    service = construct_service(
        name=name,
        namespace=namespace,
        port=port,
        target_port=target_port,
        port_name=port_name,
        selector=selector,
        protocol=protocol
    )
    response = api.create_namespaced_service(namespace=namespace, body=service)
    return get_service(response.metadata.name, namespace)


def get_services(namespace="autoscaling"):
    if namespace == "all":
        response = api.list_service_for_all_namespaces(watch=False)
    else:
        response = api.list_namespaced_service(namespace, watch=False)
    return list(
        map(
            lambda s: _get_service_info(s),
            response.items
        )
    )


def get_service(name, namespace="autoscaling"):
    response = api.read_namespaced_service(name=name, namespace=namespace)
    return _get_service_info(response)


def update_service(
        name,
        port: int = None,
        target_port: int = None,
        port_name: str = None,
        selector: dict = None,
        protocol: str = None,
        partial=False,
        namespace="autoscaling"
):
    service = api.read_namespaced_service(name=name, namespace=namespace)  # type: V1Service
    service_port = service.spec.ports[0]
    service = construct_service(
        name=name,
        namespace=service.metadata.namespace,
        port=port or service_port.port,
        target_port=target_port or service_port.target_port,
        port_name=port_name or service_port.name,
        protocol=protocol or service_port.protocol,
        selector=selector or service.spec.selector
    )
    if partial:
        response = api.patch_namespaced_service(name=name, namespace=namespace, body=service)
    else:
        response = api.replace_namespaced_service(name=name, namespace=namespace, body=service)
    return get_service(response.metadata.name, namespace)


def delete_service(name, namespace="autoscaling"):
    response = api.delete_namespaced_service(name=name, namespace=namespace)
    return {"status": response.status}
