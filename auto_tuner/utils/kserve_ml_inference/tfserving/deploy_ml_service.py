from auto_tuner.utils.kserve_ml_inference.tfserving.serving_configuration import get_serving_configuration
from auto_tuner.utils.kube_resources.configmaps import create_configmap
from auto_tuner.utils.kube_resources.kserve import create_inference_service
from auto_tuner.utils.kube_resources.services import create_service as create_kubernetes_service


def deploy_ml_service(
        service_name: str, active_model_version: int, selector: dict, namespace="default", **kwargs
):
    configmap_name = f"{service_name}-cm"
    grpc_node_port = kwargs.pop("grpc_node_port", None)
    rest_node_port = kwargs.pop("rest_node_port", None)
    volume_name = f"{service_name}-vol"

    if not kwargs.get("predictor_args"):
        kwargs["predictor_args"] = []
    kwargs["predictor_args"].append("--model_config_file=/etc/tfserving/models.config")

    if not kwargs.get("labels"):
        kwargs["labels"] = {}
    kwargs["labels"].update(selector)

    if not kwargs.get("volumes"):
        kwargs["volumes"] = []
    kwargs["volumes"].append({"name": volume_name, "config_map": {"name": configmap_name}})

    if not kwargs.get("predictor_volume_mounts"):
        kwargs["predictor_volume_mounts"] = []
    kwargs["predictor_volume_mounts"].append({"name": volume_name, "mount_path": "/etc/tfserving"})

    create_configmap(
        configmap_name,
        namespace=namespace,
        data={
            "models.config": get_serving_configuration("resnet", "/models/resnet/", "tensorflow", active_model_version)
        }
    )
    create_inference_service(service_name, namespace, **kwargs)
    create_kubernetes_service(
        f"{service_name}-grpc", target_port=8500, namespace=namespace, node_port=grpc_node_port,
        expose_type="NodePort", selector=selector
    )
    create_kubernetes_service(
        f"{service_name}-rest", target_port=8501, namespace=namespace, node_port=rest_node_port,
        expose_type="NodePort", selector=selector
    )
