from auto_tuner.utils.kube_resources.configmaps import create_configmap
from auto_tuner.utils.kube_resources.kserve import create_inference_service
from auto_tuner.utils.kube_resources.services import create_service as create_kubernetes_service
from .constants import CURRENT_MODEL_KEY


def deploy_ml_service(
        service_name: str, configmap_name: str, active_model: str, selector: dict, namespace="default", **kwargs
):
    node_port = kwargs.pop("node_port", None)
    if not kwargs.get("predictor_env_vars"):
        kwargs["predictor_env_vars"] = {}
    kwargs["predictor_env_vars"].update({CURRENT_MODEL_KEY: {"name": configmap_name, "key": CURRENT_MODEL_KEY}})

    if not kwargs.get("predictor_args"):
        kwargs["predictor_args"] = []
    kwargs["predictor_args"].extend(["--model_variant", f"$({CURRENT_MODEL_KEY})", "--model_name", service_name])

    if not kwargs.get("transformer_args"):
        kwargs["transformer_args"] = []
    kwargs["transformer_args"].extend(["--model_name", service_name])

    if not kwargs.get("labels"):
        kwargs["labels"] = {}
    kwargs["labels"].update(selector)

    create_configmap(configmap_name, namespace=namespace, data={CURRENT_MODEL_KEY: active_model})
    create_inference_service(service_name, namespace, **kwargs)
    create_kubernetes_service(
        service_name, target_port=8080, namespace=namespace, node_port=node_port,
        expose_type="NodePort", selector=selector
    )
