import time
import requests
import os
import json
from threading import Thread, Event
import csv
from datetime import datetime

from kube_resources.services import get_service, create_service as create_kubernetes_service, delete_service, get_services
from kube_resources.deployments import create_deployment, delete_deployment, get_deployments, get_deployment
from kube_resources.configmaps import delete_configmap, create_configmap
from kube_resources.vpas import create_vpa


from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.experiments.utils import wait_till_pods_are_ready
from auto_tuner.parameters import ParamTypes
from auto_tuner.utils.prometheus import PrometheusClient
from auto_tuner.utils.tfserving.serving_configuration import get_serving_configuration
from auto_tuner.experiments.workload import generate_workload


SOLVER_TYPE_INFADAPTER = "i"
SOLVER_TYPE_MSP = "m"
SOLVER_TYPE_VPA = "v"

model_versions = [18, 34, 50, 101, 152]
base_model_name = "resnet"
base_service_name = f"tfserving-{base_model_name}"
namespace = "mehran"
baseline_accuracies = {
    18: .69758,
    34: .73314,
    50: .7613,
    101: .77374,
    152: .78312,
}


class Starter:
    def __init__(self, solver_type):
        self.solver_type = solver_type
        self.base_model_name = base_model_name
        self.base_service_name = base_service_name
        self.hardware = ParamTypes.HARDWARE_CPU
        self.namespace = namespace
        self.labels = {
            "ML_framework": "tensorflow",
            "model_server": "tfserving",
        }
        self.container_ports = [8500, 8501]
        self.model_versions = model_versions
        self.prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
        # self.baseline_accuracies = {
        #     18: 89.078,
        #     34: 91.42,
        #     50: 92.862,
        #     101: 93.546,
        #     152: 94.046,
        # }

        self.baseline_accuracies = baseline_accuracies
        self.load_times = {
            18: 6.44,
            34: 6.79,
            50: 6.82,
            101: 7.77,
            152: 8.52,
        }
        self.memory_sizes = {
            18: "1G",
            34: "1.25G",
            50: "1.5G",
            101: "2G",
            152: "2.25G"
        }
        self.node_ip = os.getenv("CLUSTER_NODE_IP")
        self.max_cpu = 20
        self.alpha = 0.04
        self.beta = 0.02
        self.warmup_count = 3
        self.env_vars = {
            "BASE_MODEL_NAME": self.base_model_name,
            "BASE_SERVICE_NAME": self.base_service_name,
            "MODEL_VERSIONS": json.dumps(self.model_versions),
            "BASELINE_ACCURACIES": json.dumps(self.baseline_accuracies),
            "MEMORY_SIZES": json.dumps(self.memory_sizes),
            "LOAD_TIMES": json.dumps(self.load_times),
            "DISPATCHER_ENDPOINT": None,
            "PROMETHEUS_ENDPOINT": f"{self.node_ip}:{self.prom_port}",
            "NFS_SERVER_IP": self.node_ip,
            "CLUSTER_NODE_IP": self.node_ip,
            "CONTAINER_PORTS": json.dumps(self.container_ports),
            "CONTAINER_LABELS": json.dumps(self.labels),
            "K8S_IN_CLUSTER_CLIENT": "true",
            "K8S_NAMESPACE": self.namespace,
            "ALPHA": str(self.alpha),
            "BETA": str(self.beta),
            "MAX_CPU": str(self.max_cpu),
            "LSTM_PREDICTION_ERROR_PERCENTAGE": 15,
            "EMW_PREDICTION_ERROR_PERCENTAGE": 15,
            "WARMUP_COUNT": self.warmup_count,
            "FIRST_DECIDE_DELAY_MINUTES": 2,
            "SOLVER_TYPE": self.solver_type,
            "STABILIZATION_INTERVAL": 6,
            "DECISION_INTERVAL": 30,
        }

    
    def get_model_name(self, version):
        return f"{self.base_model_name}-{version}"
    
    
    def get_service_name(self, version):
        return f"{self.base_service_name}-{version}"        
        
        
    def deploy_adapter(self):
        labels = {"project": "infadapter", "module": "adapter"}
        create_deployment(
            self.get_service_name("adapter"),
            replicas=1,
            namespace=self.namespace,
            labels=labels,
            containers=[
                {
                    "name": self.get_service_name("adapter") + "-container",
                    "image": "mehransi/main:adapter",
                    "limit_cpu": 2, 
                    "limit_mem": "4G",
                    "request_cpu": 1, 
                    "container_ports": [8000],
                    "env_vars" : self.env_vars
                }
            ]
        )
        create_kubernetes_service(
            self.get_service_name("adapter"),
            target_port=8000,
            port=8000,
            selector=labels,
            expose_type="NodePort",
            namespace=self.namespace
        )
        
    def create_vpa(self, model_version):
        memory = self.memory_sizes[model_version]
        service_name = self.get_service_name(model_version)
        create_vpa(
            f"{service_name}-vpa", 
            min_allowed={"cpu": 2, "memory": memory},
            max_allowed={"cpu": self.max_cpu, "memory": memory},
            target_api_version="apps/v1",
            target_kind="Deployment",
            target_name=service_name,
            target_container_name=f"{service_name}-container",
            namespace=self.namespace
        )
        # os.system(f"cat <<EOF | kubectl apply -f - {vpa} \n EOF ")

    
    def create_ml_service(self, model_version: int, size: int):
        mem = self.memory_sizes[model_version]
        service_name = self.get_service_name(model_version)
        kwargs = {"name": f"{service_name}-container", "env_vars": {}}
        kwargs["env_vars"]["MODEL_NAME"] = self.base_model_name
        volume_mount_path = "/etc/tfserving"
        configmap_name = f"{service_name}-cm"
        config_volume_name = f"{service_name}-config-vol"
        models_volume_name = f"{service_name}-models-vol"
        warmup_volume_name = f"{service_name}-warmup-vol"
        
        create_configmap(
            self.base_service_name + f"-{model_version}-cm",
            namespace=self.namespace,
            data={
                "models.config": get_serving_configuration(
                    self.base_model_name, f"/models/{self.base_model_name}/", "tensorflow", model_version
                ),
                "monitoring.config": """
                    prometheus_config {
                    enable: true,
                    path: "/monitoring/prometheus/metrics"
                    }
                """,
            }
        )
        
        selector = {
            **self.labels,
            "model": self.base_model_name + f"-{model_version}"
        }

        create_kubernetes_service(
            f"{service_name}-grpc",
            target_port=8500,
            port=8500,
            namespace=self.namespace,
            expose_type="NodePort",
            selector=selector
        )
        
        
        create_kubernetes_service(
            f"{service_name}-rest",
            target_port=8501,
            port=8501,
            namespace=self.namespace,
            expose_type="NodePort",
            selector=selector
        )
                
        kwargs.update({"request_cpu": size, "limit_cpu": size, "request_mem": mem, "limit_mem": mem})
        kwargs["container_ports"] = self.container_ports
        kwargs["image"] = "tensorflow/serving:2.8.0"

        kwargs["args"] = [
            f"--tensorflow_intra_op_parallelism=1",
            f"--tensorflow_inter_op_parallelism={size}",
            f"--rest_api_num_threads={15 * size}",
            f"--model_config_file={volume_mount_path}/models.config",
            f"--monitoring_config_file={volume_mount_path}/monitoring.config",
            "--enable_batching=false",
        ]

        labels = {**self.labels, "model": self.base_model_name + f"-{model_version}"}

        volumes = [
            {
                "name": config_volume_name,
                "config_map": {"name": configmap_name}
            },
            {
                "name": models_volume_name,
                "nfs": {"server": self.node_ip, "path": f"/fileshare/tensorflow_resnet_b64/{model_version}"}
            },
            {
                "name": warmup_volume_name,
                "empty_dir": "{}"
            }
        ]
        
        kwargs["volume_mounts"] = [
            {"name": config_volume_name, "mount_path": "/etc/tfserving"},
            {"name": models_volume_name, "mount_path": f"/models/{self.base_model_name}/{model_version}"},
            {"name": warmup_volume_name, "mount_path": "/warmup"}
        ]
        kwargs.update(readiness_probe={"exec": ["cat", "/warmup/done"], "period_seconds": 1})

        create_deployment(
            service_name, 
            containers=[
                kwargs,
                {
                    "name": "warmup",
                    "image": "mehransi/main:warmup_for_tfserving",
                    "env_vars": {
                        "WARMUP_COUNT": self.warmup_count,
                        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                        "PYTHONUNBUFFERED": "1"
                    },
                    "request_cpu": "128m",
                    "request_mem": "128Mi",
                    "volume_mounts": [{"name": warmup_volume_name, "mount_path": "/warmup"}]
                }
            ], 
            replicas=1, 
            namespace=self.namespace, 
            labels=labels, 
            volumes=volumes
        )
        wait_till_pods_are_ready(service_name)
        self.create_vpa()
        return get_service(service_name, self.namespace)["node_port"]
    
    def setup(self, model_version=None) -> int:
        if self.solver_type == SOLVER_TYPE_VPA:
            if model_version is None:
                raise Exception("model_version must be set for VPA solver_type.")
            node_port = self.create_ml_service(model_version, 15)
        else:
            # Deploy the adaptor
            self.deploy_adapter()
            time.sleep(2)
            wait_till_pods_are_ready(self.get_service_name("adapter"), self.namespace)
            adapter_k8s_service = get_service(self.get_service_name("adapter"), self.namespace)
            
            time.sleep(10)
            response = requests.post(
                f"http://{self.node_ip}:{adapter_k8s_service['node_port']}/initialize",
                json={
                    "models_config": [[101, 15]],
                }
            )
            print("adapter response", response.text)
            dispatcher_k8s_service = get_service(self.get_service_name("dispatcher"), self.namespace)
            node_port = dispatcher_k8s_service["node_port"]
            
        os.system(f"kubectl apply -f {AUTO_TUNER_DIRECTORY}/model_server_podmonitor.yaml")
        os.system(f"kubectl apply -f {AUTO_TUNER_DIRECTORY}/infadapter_podmonitor.yaml")
        return node_port
        

    def delete(self):
        services = get_services(self.namespace)
        for svc in services:
            delete_service(svc["name"], self.namespace)
        deployments = get_deployments(self.namespace)
        for deploy in deployments:
            delete_deployment(deploy["name"], self.namespace)
        for m in self.model_versions:
            delete_configmap(self.get_service_name(m) + f"-cm", self.namespace)



def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v


def query_metrics(prom_port, event: Event, solver_type: str, model_version: int):
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), prom_port)
    interval = 5  # Seconds
    while True:
        if event.is_set():
            break
        time.sleep(interval)
        percentile_99 = prom.get_instant(
            f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501", API="predict", entrypoint="REST"}}[{interval}s])) by (le)) / 1000'
        )
        percentile_99 = _get_value(percentile_99, divide_by=1000)

        if solver_type == SOLVER_TYPE_VPA:
            deploy = get_deployment(base_service_name, namespace)
            accuracy = baseline_accuracies[model_version]
            cost = None
            for container in deploy["containers"]:
                if container["name"] == "warmup":
                    continue
                cost = container["resources"]["limits"]["cpu"]
                
        else:
            cost = prom.get_instant("infadapter_cost")
            cost = _get_value(cost)
            accuracy = prom.get_instant("infadapter_accuracy")
            accuracy = _get_value(accuracy, should_round=False)

        
        if solver_type == SOLVER_TYPE_INFADAPTER:
            filename = "infadapter"
        elif solver_type == SOLVER_TYPE_MSP:
            filename = "msp"
        else:
            filename = "vpa"
        filepath = f"{AUTO_TUNER_DIRECTORY}/{filename}.csv"
        file_exists = os.path.exists(filepath)
        with open(filepath, "a") as f:
            field_names = [
                "p99_latency",
                "accuracy",
                "cost",
                "timestamp"
            ]
            writer = csv.DictWriter(f, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "p99_latency": percentile_99,
                    "accuracy": accuracy,
                    "cost": cost,
                    "timestamp": datetime.now().isoformat()
                }
            )


if __name__ == "__main__":
    solver_type = SOLVER_TYPE_INFADAPTER
    model_version = model_versions[0]
    starter = Starter(solver_type=solver_type)
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), starter.prom_port)
    node_port = starter.setup(model_version)
    
    start_time = datetime.now().timestamp()
    event = Event()
    query_task = Thread(target=query_metrics, args=(starter.prom_port, event, solver_type, model_version))
    query_task.start()
    
    # Start generating load
    
    total_requests, total_seconds = generate_workload(
        f"http://{starter.node_ip}:{node_port}/predict"
    )
    event.set()
    query_task.join()
    starter.delete()
