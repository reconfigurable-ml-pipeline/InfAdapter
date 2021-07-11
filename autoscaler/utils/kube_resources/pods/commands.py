import time
import yaml

from pykube.core import core_api as api


def _get_pod_info(p):
    return {
        "kind": "Pod",
        "pod_ip": p.status.pod_ip,
        "namespace": p.metadata.namespace,
        "name": p.metadata.name,
        "node": p.spec.node_name,
        "phase": p.status.phase,
        "terminating": p.metadata.deletion_timestamp is not None,
        "container_statuses": list(map(
            lambda c: {
                "container_name": c.name,
                "image": c.image,
                "started": c.started,
                "state": {
                    "running": {
                        "started_at": c.state.running.started_at.isoformat()
                    } if c.state.running else None,
                    "terminated": {
                        "finished_at": c.state.terminated.finished_at.isoformat(),
                        "exit_code": c.state.terminated.exit_code,
                        "message": c.state.terminated.message,
                        "reason": c.state.terminated.reason,
                    } if c.state.terminated else None,
                    "waiting": {
                        "message": c.state.waiting.message,
                        "reason": c.state.waiting.reason
                    } if c.state.waiting else None,
                }
            }, p.status.container_statuses
        )),
    }


def create_pod(file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.create_namespaced_pod(
            namespace=namespace, body=yaml_content
        )
        time.sleep(1)
        return get_pod(response.metadata.name, namespace)


def get_pods(namespace="default"):
    if namespace == "all":
        pods = api.list_pod_for_all_namespaces(watch=False)
    else:
        pods = api.list_namespaced_pod(namespace, watch=False)
    return {
        "kind": pods.kind,
        "pods": list(
            map(
                lambda p: _get_pod_info(p), pods.items
            )
        )
    }


def get_pod(pod_name, namespace="default"):
    response = api.read_namespaced_pod(name=pod_name, namespace=namespace)
    return _get_pod_info(response)


def update_pod(pod_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.replace_namespaced_pod(
            name=pod_name, namespace=namespace, body=yaml_content
        )
        time.sleep(1)
        return get_pod(response.metadata.name, namespace=namespace)


def partially_update_pod(pod_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.patch_namespaced_pod(
            name=pod_name, namespace=namespace, body=yaml_content
        )
        time.sleep(1)
        return get_pod(response.metadata.name, namespace=namespace)


def delete_pod(pod_name, namespace="default"):
    response = api.delete_namespaced_pod(name=pod_name, namespace=namespace)
    return {"status": response.status}
