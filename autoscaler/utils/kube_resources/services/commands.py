from autoscaler.utils.kube_resources import core_api as api
import yaml


def _get_service_info(service):
    return {
        "service_ip": service.status.service_ip,
        "namespace": service.metadata.namespace,
        "name": service.metadata.name
    }


def create_service(file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.create_namespaced_service(namespace=namespace, body=yaml_content)
        print(create_service.__name__, response)
        return get_service(response.metadata.name, namespace)


def get_services(namespace="default"):
    if namespace == "all":
        response = api.list_service_for_all_namespaces(watch=False)
    else:
        response = api.list_namespaced_service(namespace, watch=False)
    return list(
        map(
            lambda s: _get_service_info(s),
            response.items
        )
    )


def get_service(service_name, namespace="default"):
    response = api.read_namespaced_service(name=service_name, namespace=namespace)
    print(get_service.__name__, response)
    return _get_service_info(response)


def update_service(service_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.replace_namespaced_service(name=service_name, namespace=namespace, body=yaml_content)
        print(update_service.__name__, response)
        return get_service(response.metadata.name, namespace)


def partially_update_service(service_name, file_path, namespace="default"):
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        response = api.patch_namespaced_service(name=service_name, namespace=namespace, body=yaml_content)
        print(partially_update_service.__name__, response)
        return get_service(response.metadata.name, namespace)


def delete_service(service_name, namespace="default"):
    response = api.delete_namespaced_service(name=service_name, namespace=namespace)
    return {"status": response.status}
