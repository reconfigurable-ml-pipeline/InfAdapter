from kubernetes.client import (
    V1Pod, V1EnvVar, V1ResourceRequirements, V1ObjectMeta, V1PodSpec, V1Container, V1Deployment, V1DeploymentStrategy,
    V1DeploymentSpec, V1LabelSelector, V1PodTemplateSpec, V1Service, V1ServiceSpec, V1ServicePort,
    V1HorizontalPodAutoscaler, V1HorizontalPodAutoscalerSpec, V1CrossVersionObjectReference
)


def construct_pod(
        name: str,
        image: str,
        namespace: str,
        labels: dict = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
        env_vars: dict = None,
) -> V1Pod:
    if labels is None:
        labels = {}
    labels.update(project="autoscaling")
    if env_vars is None:
        env_vars = {}
    env_vars = list(map(lambda k, v: V1EnvVar(k, v), env_vars.items()))

    container_kwargs = {"name": name, "image": image}
    limits = {}
    requests = {}
    if limit_mem:
        limits.update(memory=limit_mem)
    if limit_cpu:
        limits.update(cpu=limit_cpu)
    if request_mem:
        requests.update(memory=request_mem)
    if request_cpu:
        requests.update(cpu=request_cpu)
    if requests or limits:
        container_kwargs.update(resources=V1ResourceRequirements(limits=limits or None, requests=requests or None))
    if env_vars:
        container_kwargs.update(env=env_vars)
    pod = V1Pod(
        "v1",
        "Pod",
        metadata=V1ObjectMeta(name=name, namespace=namespace, labels=labels),
        spec=V1PodSpec(containers=[V1Container(**container_kwargs)])
    )
    return pod


def construct_deployment(
        name: str,
        image: str,
        namespace: str,
        replicas: int,
        labels: dict = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
        env_vars: dict = None,
) -> V1Deployment:
    pod = construct_pod(
        name,
        image,
        namespace,
        labels=labels,
        request_mem=request_mem,
        request_cpu=request_cpu,
        limit_mem=limit_mem,
        limit_cpu=limit_cpu,
        env_vars=env_vars
    )

    deployment = V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=V1ObjectMeta(name=name, namespace=namespace),
        spec=V1DeploymentSpec(
            replicas=replicas,
            selector=V1LabelSelector(match_labels=pod.metadata.labels),
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels=pod.metadata.labels),
                spec=pod.spec
            )
        )
    )
    return deployment


def construct_service(
        name: str,
        namespace: str,
        port: int,
        target_port: int,
        port_name: str,
        selector: dict,
        protocol: str = None
) -> V1Service:
    selector.update(project="autoscaling")

    if protocol is None:
        protocol = "TCP"

    service = V1Service(
        api_version="V1",
        kind="Service",
        metadata=V1ObjectMeta(name=name, namespace=namespace, labels=selector),
        spec=V1ServiceSpec(
            ports=[V1ServicePort(name=port_name, port=port, target_port=target_port, protocol=protocol)],
            selector=selector
        )
    )

    return service


def construct_hpa(
        name: str,
        namespace: str,
        target_cpu_utilization: int,
        min_replicas: int,
        max_replicas: int,
        target_api_version: str,
        target_kind: str,
        target_name: str
) -> V1HorizontalPodAutoscaler:
    hpa = V1HorizontalPodAutoscaler(
        api_version="autoscaling/v1",
        kind="HorizontalPodAutoscaler",
        metadata=V1ObjectMeta(name=name, namespace=namespace),
        spec=V1HorizontalPodAutoscalerSpec(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_target_ref=V1CrossVersionObjectReference(
                api_version=target_api_version,
                kind=target_kind,
                name=target_name
            ),
            target_cpu_utilization_percentage=target_cpu_utilization
        )
    )
    return hpa
