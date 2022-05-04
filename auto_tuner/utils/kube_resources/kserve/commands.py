import time

from kserve import KServeClient, V1beta1InferenceService

from auto_tuner.utils.kube_resources.utils import construct_inference_service

client = KServeClient()


def _get_inference_service_info(s: V1beta1InferenceService):
    return {
        "kind": "InferenceService",
        "namespace": s.metadata.namespace,
        "name": s.metadata.name,
        "terminating": s.metadata.deletion_timestamp is not None,
        "predictor": {
            "node": s.spec.predictor.node_name,
            "containers": list(map(
                lambda c: {
                    "image": c.image,
                    "ports": list(map(lambda p: p.to_dict(), c.ports)),
                    "resources": {"requests": c.resources.requests, "limits": c.resources.limits},
                    "env": list(
                        map(lambda e: {"name": e.name, "value": e.value, "value_from": str(e.value_from)}, c.env)
                    )
                },
                s.spec.predictor.containers
            ))
        },
        "transformer": {
            "node": s.spec.transformer.node_name
        }
    }


def get_inference_service(name: str, namespace="default"):
    response = client.get(name=name, namespace=namespace)
    return _get_inference_service_info(response)


def create_inference_service(inference_service_name: str, namespace="default", **kwargs):
    inference_service_obj = construct_inference_service(inference_service_name, namespace, **kwargs)
    response = client.create(inference_service_obj, namespace)
    time.sleep(1)
    return get_inference_service(response.metadata.name, namespace)
