from typing import List

from kubernetes.client import (
    V1Pod, V1EnvVar, V1EnvVarSource, V1ConfigMapKeySelector, V1ResourceRequirements, V1ObjectMeta, V1PodSpec,
    V1Container, V1ContainerPort, V1Deployment, V1DeploymentSpec, V1LabelSelector, V1PodTemplateSpec, V1Service,
    V1ServiceSpec, V1ServicePort, V1HorizontalPodAutoscaler, V1HorizontalPodAutoscalerSpec,
    V1CrossVersionObjectReference, V1ConfigMap, V1Volume, V1VolumeMount, V1ConfigMapVolumeSource
)
from kserve import V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec, V1beta1TransformerSpec
from kserve.constants import constants


def _construct_container(
        name: str,
        image: str,
        request_mem: str,
        request_cpu: str,
        limit_mem: str,
        limit_cpu: str,
        env_vars: dict = None,
        container_ports: List[int] = None,
        command: str = None,
        args: List[str] = None,
        volume_mounts: List[dict] = None,
) -> V1Container:
    container_kwargs = {"name": name, "image": image}
    if container_ports:
        container_kwargs.update({"ports": list(map(lambda p: V1ContainerPort(container_port=p), container_ports))})
    if command:
        container_kwargs.update({"command": [command]})
    if args:
        container_kwargs.update({"args": args})
    if env_vars is None:
        env_vars = {}
    env_vars = list(map(
        lambda k, v: V1EnvVar(
            k,
            (
                lambda x: x if isinstance(x, str) else V1EnvVarSource(
                    config_map_key_ref=V1ConfigMapKeySelector(name=x["name"], key=x["key"])
                )
            )(v)
        ), env_vars.items()
    ))
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

    mounts = []
    if volume_mounts:
        for vm in volume_mounts:
            mounts.append(V1VolumeMount(**vm))
        container_kwargs.update(volume_mounts=mounts)
    return V1Container(**container_kwargs)


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
        container_ports: List[int] = None,
        command: str = None,
        args: List[str] = None,
) -> V1Pod:
    if labels is None:
        labels = {}
    pod = V1Pod(
        "v1",
        "Pod",
        metadata=V1ObjectMeta(name=name, namespace=namespace, labels=labels),
        spec=V1PodSpec(
            containers=[
                _construct_container(
                    name=name,
                    image=image,
                    request_mem=request_mem,
                    request_cpu=request_cpu,
                    limit_mem=limit_mem,
                    limit_cpu=limit_cpu,
                    env_vars=env_vars,
                    container_ports=container_ports,
                    command=command,
                    args=args
                )
            ]
        )
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
        container_ports: List[int] = None,
        command: str = None,
        args: List[str] = None,
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
        env_vars=env_vars,
        container_ports=container_ports,
        command=command,
        args=args
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
        target_port: int,
        selector: dict,
        port: int = None,
        node_port: int = None,
        port_name: str = None,
        expose_type: str = None,
        protocol: str = None
) -> V1Service:
    if protocol is None:
        protocol = "TCP"

    service = V1Service(
        api_version="v1",
        kind="Service",
        metadata=V1ObjectMeta(name=name, namespace=namespace, labels=selector),
        spec=V1ServiceSpec(
            ports=[
                V1ServicePort(
                    name=port_name, port=port, target_port=target_port, node_port=node_port, protocol=protocol
                )
            ],
            selector=selector,
            type=expose_type
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


def construct_configmap(name: str, namespace: str, data: dict, binary_data=None) -> V1ConfigMap:
    cm = V1ConfigMap(
        api_version="v1",
        metadata=V1ObjectMeta(namespace=namespace, name=name),
        data=data,
        binary_data=binary_data
    )
    return cm


def construct_inference_service(
        inference_service_name: str,
        namespace: str,
        labels: dict = None,
        predictor_image: str = None,
        transformer_image: str = None,
        predictor_request_mem: str = None,
        predictor_request_cpu: str = None,
        transformer_request_mem: str = None,
        transformer_request_cpu: str = None,
        predictor_limit_mem: str = None,
        predictor_limit_cpu: str = None,
        transformer_limit_mem: str = None,
        transformer_limit_cpu: str = None,
        predictor_env_vars: dict = None,
        transformer_env_vars: dict = None,
        predictor_container_ports: List[int] = None,
        transformer_container_ports: List[int] = None,
        predictor_command: str = None,
        transformer_command: str = None,
        predictor_args: List[str] = None,
        transformer_args: List[str] = None,
        predictor_min_replicas: int = None,
        predictor_max_replicas: int = None,
        transformer_min_replicas: int = None,
        transformer_max_replicas: int = None,
        volumes: List[dict] = None,
        predictor_volume_mounts: List[dict] = None,
) -> V1beta1InferenceService:
    assert predictor_image is not None or transformer_image is not None, "Specify predictor_image and/or" \
                                                                         " transformer_image"

    # Fixme: currently only allowing ConfigMap Volume type
    volume_list = []
    for vol in volumes:
        if vol.get("config_map") is not None:
            volume_list.append(vol)

    if predictor_image:
        predictor_spec = V1beta1PredictorSpec(
            min_replicas=predictor_min_replicas,
            max_replicas=predictor_max_replicas,
            containers=[
                _construct_container(
                    f"predictor-{inference_service_name}",
                    predictor_image,
                    request_mem=predictor_request_mem,
                    request_cpu=predictor_request_cpu,
                    limit_mem=predictor_limit_mem,
                    limit_cpu=predictor_limit_cpu,
                    env_vars=predictor_env_vars,
                    container_ports=predictor_container_ports,
                    command=predictor_command,
                    args=predictor_args
                )
            ],
            volumes=[
                V1Volume(
                    name=v["name"],
                    config_map=V1ConfigMapVolumeSource(**v["config_map"])
                ) for v in volume_list
            ]
        )
    else:
        predictor_spec = None
    if transformer_image:
        transformer_spec = V1beta1TransformerSpec(
            min_replicas=transformer_min_replicas,
            max_replicas=transformer_max_replicas,
            containers=[
                _construct_container(
                    f"transformer-{inference_service_name}",
                    transformer_image,
                    request_mem=transformer_request_mem,
                    request_cpu=transformer_request_cpu,
                    limit_mem=transformer_limit_mem,
                    limit_cpu=transformer_limit_cpu,
                    env_vars=transformer_env_vars,
                    container_ports=transformer_container_ports,
                    command=transformer_command,
                    args=transformer_args,
                    volume_mounts=predictor_volume_mounts
                )
            ]
        )
    else:
        transformer_spec = None

    return V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=V1ObjectMeta(name=inference_service_name, namespace=namespace, labels=labels),
        spec=V1beta1InferenceServiceSpec(
            predictor=predictor_spec,
            transformer=transformer_spec
        )
    )
