import time
import yaml

from autoscaler.utils.kube_resources import apps_api as api


def _get_deployment_info(deployment):
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


def create_deployment(file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.create_namespaced_deployment(namespace=namespace, body=yaml_content)
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


def get_deployment(deployment_name, namespace="default"):
    response = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    return _get_deployment_info(response)


def update_deployment(deployment_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.replace_namespaced_deployment(name=deployment_name, namespace=namespace, body=yaml_content)
        time.sleep(1)
        return get_deployment(response.metadata.name, namespace)


def partially_update_deployment(deployment_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=yaml_content)
        time.sleep(1)
        return get_deployment(response.metadata.name, namespace)


def delete_deployment(deployment_name, namespace="default"):
    response = api.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
    return {"status": response.status}
