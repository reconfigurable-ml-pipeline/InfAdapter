import grpc

from tensorflow_serving.config import model_server_config_pb2, file_system_storage_path_source_pb2
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2

from kube_resources.services import get_endpoints
from kube_resources.configmaps import update_configmap
from auto_tuner.utils.tfserving.serving_configuration import get_serving_configuration


model_platform = "tensorflow"
model_name = "resnet"
base_path = f"/models/{model_name}/"


def request_pod_to_switch_model_version(endpoint, new_model_version):
    with grpc.insecure_channel(endpoint) as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_management_pb2.ReloadConfigRequest()
        model_server_config = model_server_config_pb2.ModelServerConfig()
        config_list = model_server_config_pb2.ModelConfigList()
        config = config_list.config.add()
        config.name = model_name
        config.base_path = base_path
        config.model_platform = model_platform
        version_policy = file_system_storage_path_source_pb2.FileSystemStoragePathSourceConfig().ServableVersionPolicy()
        version_policy.specific.versions[:] = [new_model_version]
        config.model_version_policy.CopyFrom(version_policy)

        model_server_config.model_config_list.CopyFrom(config_list)

        request.config.CopyFrom(model_server_config)

        # print("request is initialized", request.IsInitialized())
        # print("request ListFields", request.ListFields())

        return stub.HandleReloadConfigRequest(request)


def switch_model(
        namespace: str,
        service_name: str,
        target_port: int,
        new_model_version: int,
):
    configmap_name = f"{service_name}-cm"
    
    def request_all_pods_to_switch_model():
        endpoints = get_endpoints(f"{service_name}-grpc", target_port, namespace=namespace)
        for endpoint in endpoints:
            r = request_pod_to_switch_model_version(endpoint, new_model_version)
            # Todo: do something in case error_code is non-zero
            print("response error code", r.status.error_code)
            print("response error message", r.status.error_message)

    update_configmap(
        configmap_name,
        namespace=namespace,
        data={
            "models.config": get_serving_configuration(
                model_name, base_path, model_platform, new_model_version
            )
        },
        partial=True
    )
 
    request_all_pods_to_switch_model()
