from auto_tuner.utils.tfserving import switch_model, deploy_ml_service


class AutoConfiguration:
    def __init__(self):
        self.service_name = "kserve_torch_resnet_service"
        self.target_port = 8080
        self.node_port = 12345
        self.active_model = "resnet34"
        self.namespace = "default"
        self.configmap_name = f"{self.service_name}-configmap"
        self.labels = {
            "inference_framework": "kserve",
            "ML_framework": "torch",
            "ML_model": "resnet"
        }
        self.predictor_request_cpu = "1"
        self.predictor_request_mem = "1Gi"
        self.transformer_request_cpu = "1"
        self.transformer_request_mem = "1Gi"

    def deploy(self):
        deploy_ml_service(
            self.service_name, self.configmap_name, self.active_model, self.labels, namespace=self.namespace,
            predictor_env_vars={"TORCH_HOME": "/app/.torch"}, transformer_env_vars={"TORCH_HOME": "/app/.torch"},
            predictor_image="kserve-torch-resnet-predictor:v1", transformer_image="kserve-torch-resnet-transformer:v1",
            predictor_request_cpu=self.predictor_request_cpu, predictor_request_mem=self.predictor_request_mem,
            transformer_request_cpu=self.transformer_request_cpu, transformer_request_mem=self.transformer_request_mem,
            predictor_container_ports=[self.target_port], transformer_container_ports=[self.target_port],
        )

    def switch_model(self, new_model_version: str):
        switch_model(
            namespace=self.namespace, service_name=self.service_name,
            target_port=self.target_port, new_model=new_model_version
        )
