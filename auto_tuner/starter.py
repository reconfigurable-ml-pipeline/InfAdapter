import time
import requests
import os
import json
from threading import Thread, Event
import csv
from datetime import datetime

from kube_resources.services import get_service, create_service as create_kubernetes_service, delete_service, get_services
from kube_resources.deployments import create_deployment, delete_deployment, get_deployments
from kube_resources.configmaps import delete_configmap

from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.experiments.utils import wait_till_pods_are_ready
from auto_tuner.parameters import ParamTypes
from auto_tuner.utils.prometheus import PrometheusClient
from auto_tuner.experiments.workload import generate_workload


class Starter:
    def __init__(self, solver_type):
        self.base_model_name = "resnet"
        self.base_service_name = f"tfserving-{self.base_model_name}"
        self.hardware = ParamTypes.HARDWARE_CPU
        self.namespace = "mehran"
        self.labels = {
            "ML_framework": "tensorflow",
            "model_server": "tfserving",
        }
        self.container_ports = [8500, 8501]
        self.model_versions = [18, 34, 50, 101, 152]
        self.prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
        # self.baseline_accuracies = {
        #     18: 89.078,
        #     34: 91.42,
        #     50: 92.862,
        #     101: 93.546,
        #     152: 94.046,
        # }

        self.baseline_accuracies = {
            18: .69758,
            34: .73314,
            50: .7613,
            101: .77374,
            152: .78312,
        }
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
            "WARMUP_COUNT": 3,
            "FIRST_DECIDE_DELAY_MINUTES": 2,
            "SOLVER_TYPE": solver_type,
            "STABILIZATION_INTERVAL": 3,
            "DECISION_INTERVAL": 60,
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
    
    def setup(self):
        # Deploy the adaptor
        self.deploy_adapter()
        time.sleep(2)
        wait_till_pods_are_ready(self.get_service_name("adapter"), self.namespace)
        adapter_k8s_service = get_service(self.get_service_name("adapter"), self.namespace)
        
        # self.env_vars["DISPATCHER_ENDPOINT"] = f"{self.node_ip}:{dispatcher_k8s_service['node_port']}"
        # with open(f"{AUTO_TUNER_DIRECTORY}/adapter/envs.json", "w") as f:
        #     json.dump(self.env_vars, f)
            
        # time.sleep(1)
        # os.popen(f"python {AUTO_TUNER_DIRECTORY}/adapter/main.py")
        
        time.sleep(10)
        response = requests.post(
            f"http://{self.node_ip}:{adapter_k8s_service['node_port']}/initialize",
            json={
                "models_config": [[34, 3], [101, 7]],
            }
        )
        print("adapter response", response.text)
        # os.popen(f"python {AUTO_TUNER_DIRECTORY}/adapter/notifier.py")
        
        os.system(f"kubectl apply -f {AUTO_TUNER_DIRECTORY}/model_server_podmonitor.yaml")
        os.system(f"kubectl apply -f {AUTO_TUNER_DIRECTORY}/infadapter_podmonitor.yaml")
        

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


def query_metrics(prom_port, event: Event):
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), prom_port)
    interval = 2  # Seconds
    while True:
        if event.is_set():
            break
        time.sleep(interval)
        percentile_99 = prom.get_instant(
            f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501", API="predict", entrypoint="REST"}}[{interval}s])) by (le)) / 1000'
        )
        percentile_99 = _get_value(percentile_99, divide_by=1000)
        
        accuracy = prom.get_instant("infadapter_accuracy")
        accuracy = _get_value(accuracy, should_round=False)
        
        cost = prom.get_instant(
            "infadapter_cost"
        )
        cost = _get_value(cost)
        
        
        filepath = f"{AUTO_TUNER_DIRECTORY}/p99.csv"
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
    SOLVER_TYPE_INFADAPTER = "i"
    SOLVER_TYPE_MSP = "m"
    starter = Starter(solver_type=SOLVER_TYPE_INFADAPTER)
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), starter.prom_port)
    starter.setup()
    
    start_time = datetime.now().timestamp()
    event = Event()
    query_task = Thread(target=query_metrics, args=(starter.prom_port, event))
    query_task.start()
    
    # Start generating load
    dispatcher_k8s_service = get_service(starter.get_service_name("dispatcher"), starter.namespace)
    total_requests, total_seconds = generate_workload(
        f"http://{starter.node_ip}:{dispatcher_k8s_service['node_port']}/predict"
    )
    event.set()
    query_task.join()
    starter.delete()
