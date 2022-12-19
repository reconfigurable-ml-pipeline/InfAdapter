from kube_resources.configmaps import create_configmap
from kube_resources.kserve import create_inference_service
from kube_resources.services import create_service as create_kubernetes_service


def deploy_ml_service(
        service_name: str, active_model: str, selector: dict, namespace="default", **kwargs
):
    configmap_name = f"{service_name}-cm"
    node_port = kwargs.pop("node_port", None)
    if not kwargs.get("predictor_env_vars"):
        kwargs["predictor_env_vars"] = {}
    kwargs["predictor_env_vars"].update({"ACTIVE_MODEL": {"name": configmap_name, "key": "active_model"}})

    if not kwargs.get("predictor_args"):
        kwargs["predictor_args"] = []
    kwargs["predictor_args"].extend(["--model_variant", f"$(ACTIVE_MODEL)", "--model_name", service_name])

    if not kwargs.get("transformer_args"):
        kwargs["transformer_args"] = []
    kwargs["transformer_args"].extend(["--model_name", service_name])

    if not kwargs.get("labels"):
        kwargs["labels"] = {}
    kwargs["labels"].update(selector)

    create_configmap(configmap_name, namespace=namespace, data={"active_model": active_model})
    create_inference_service(service_name, namespace, **kwargs)
    create_kubernetes_service(
        service_name, port=8080, target_port=8080, namespace=namespace, node_port=node_port,
        expose_type="NodePort", selector=selector
    )
