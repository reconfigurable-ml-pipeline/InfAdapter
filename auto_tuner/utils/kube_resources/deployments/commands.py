import time

from kubernetes.client.models import V1Deployment

from auto_tuner.utils.kube_resources.utils import construct_deployment
from auto_tuner.utils.kube_resources import apps_api as api


def _get_deployment_info(deployment: V1Deployment):
    return {
        "kind": "Deployment",
        "namespace": deployment.metadata.namespace,
        "name": deployment.metadata.name,
        "replicas": deployment.spec.replicas,
        "selector": {
            "match_labels": deployment.spec.selector.match_labels,
            "match_expressions": deployment.spec.selector.match_expressions
        },
        "rolling_update_strategy": {
            "max_surge": deployment.spec.strategy.rolling_update.max_surge,
            "max_unavailable": deployment.spec.strategy.rolling_update.max_unavailable,
        },
        "containers": list(map(
            lambda c: {
                "name": c.name,
                "image": c.image,
                "ports": c.ports,
                "resources": {
                    "limits": c.resources.limits,
                    "requests": c.resources.requests
                }
            },
            deployment.spec.template.spec.containers
        )),
        "status": {
            "available_replicas": deployment.status.available_replicas,
            "replicas": deployment.status.replicas,
            "ready_replicas": deployment.status.ready_replicas,
        }
    }


def create_deployment(
        name: str,
        image: str,
        replicas: int,
        namespace="default",
        **kwargs
):
    deployment = construct_deployment(
        name=name,
        namespace=namespace,
        image=image,
        replicas=replicas,
        **kwargs
    )
    response = api.create_namespaced_deployment(namespace=namespace, body=deployment)
    time.sleep(1)
    return get_deployment(response.metadata.name, namespace)


def get_deployments(namespace="default"):
    if namespace == "all":
        response = api.list_deployment_for_all_namespaces(watch=False)
    else:
        response = api.list_namespaced_deployment(namespace, watch=False)
    return list(
        map(
            lambda d: _get_deployment_info(d),
            response.items
        )
    )


def get_deployment(name, namespace="default"):
    response = api.read_namespaced_deployment(name=name, namespace=namespace)
    return _get_deployment_info(response)


def update_deployment(
        name: str,
        replicas: int = None,
        labels: dict = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
        env_vars: dict = None,
        partial=False,
        namespace="default"
):
    deployment = api.read_namespaced_deployment(name=name, namespace=namespace)  # type: V1Deployment
    container = deployment.spec.template.spec.containers[0]
    deployment = construct_deployment(
        name=name,
        namespace=deployment.metadata.namespace,
        image=container.image,
        replicas=replicas or deployment.spec.replicas,
        labels=labels or deployment.spec.template.metadata.labels,
        request_mem=request_mem or container.resources.requests["memory"],
        request_cpu=request_cpu or container.resources.requests["cpu"],
        limit_mem=limit_mem or container.resources.limits["memory"],
        limit_cpu=limit_cpu or container.resources.limits["cpu"],
        env_vars=env_vars or {ev.name: ev.value for ev in container.env}

    )
    if partial:
        response = api.patch_namespaced_deployment(name=name, namespace=namespace, body=deployment)
    else:
        response = api.replace_namespaced_deployment(name=name, namespace=namespace, body=deployment)
    time.sleep(1)
    return get_deployment(response.metadata.name, namespace)


def delete_deployment(deployment_name, namespace="default"):
    response = api.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
    return {"status": response.status}
