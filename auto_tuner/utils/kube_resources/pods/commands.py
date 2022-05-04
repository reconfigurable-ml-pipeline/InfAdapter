import time
from kubernetes.client.models import V1Pod
from auto_tuner.utils.kube_resources import core_api as api
from auto_tuner.utils.kube_resources.utils import construct_pod


def _get_pod_info(p: V1Pod):
    return {
        "kind": "Pod",
        "pod_ip": p.status.pod_ip,
        "namespace": p.metadata.namespace,
        "name": p.metadata.name,
        "node": p.spec.node_name,
        "labels": p.metadata.labels,
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


def create_pod(
        name: str,
        image: str,
        namespace="default",
        **kwargs
):
    pod = construct_pod(
        name=name,
        image=image,
        namespace=namespace,
        **kwargs
    )
    response = api.create_namespaced_pod(
        namespace=namespace, body=pod
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
        "pods": list(map(lambda p: _get_pod_info(p), pods.items))
    }


def get_pod(pod_name, namespace="default"):
    response = api.read_namespaced_pod(name=pod_name, namespace=namespace)
    return _get_pod_info(response)


def update_pod(
        name,
        labels: dict = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
        partial=False,
        namespace="default"
):
    pod = api.read_namespaced_pod(name=name, namespace=namespace)  # type: V1Pod
    resources = pod.spec.containers[0].resources
    pod = construct_pod(
        name,
        image=pod.spec.containers[0].image,
        namespace=pod.metadata.namespace,
        labels=labels or pod.metadata.labels,
        request_mem=request_mem or resources.requests["memory"],
        request_cpu=request_cpu or resources.requests["cpu"],
        limit_mem=limit_mem or resources.limits["memory"],
        limit_cpu=limit_cpu or resources.limits["cpu"],

    )
    if partial:
        response = api.patch_namespaced_pod(name=name, namespace=namespace, body=pod)
    else:
        response = api.replace_namespaced_pod(name=name, namespace=namespace, body=pod)
    time.sleep(1)
    return get_pod(response.metadata.name, namespace=namespace)


def delete_pod(pod_name, namespace="default"):
    response = api.delete_namespaced_pod(name=pod_name, namespace=namespace)
    return {"status": response.status}
