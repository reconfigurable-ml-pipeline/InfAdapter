import time
import requests
import os
import json
from threading import Thread, Event
import csv
from datetime import datetime

from kube_resources.services import get_service, create_service as create_kubernetes_service, delete_service
from kube_resources.deployments import create_deployment, delete_deployment

from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.utils.tfserving import deploy_ml_service
from auto_tuner.experiments.utils import delete_previous_deployment, wait_till_pods_are_ready
from auto_tuner.parameters import ParamTypes
from auto_tuner.utils.prometheus import PrometheusClient
from auto_tuner.experiments.workload import generate_workload


class Starter:
    def __init__(self):
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
        # self.baseline_accuracies = {
        #     18: 89.078,
        #     34: 91.42,
        #     50: 92.862,
        #     101: 93.546,
        #     152: 94.046,
        # }

        self.baseline_accuracies = {
            18: 69.758,
            34: 73.314,
            50: 76.13,
            101: 77.374,
            152: 78.312,
        }
        self.node_ip = os.getenv("CLUSTER_NODE_IP")
        self.max_cpu = 20
        self.alpha = 0.5
    
    def get_model_name(self, version):
        return f"{self.base_model_name}-{version}"
    
    def get_service_name(self, version):
        return f"{self.base_service_name}-{version}"


    def deploy_model(self, model_version: int, cpu: int, memory: str):
        if self.hardware == ParamTypes.HARDWARE_CPU:
            image = "tensorflow/serving:2.8.0"
        else:
            image = "tensorflow/serving:2.8.0-gpu"
        deploy_ml_service(
            service_name=self.get_service_name(model_version),
            image=image,
            replicas=1,
            active_model_version=model_version,
            namespace=self.namespace,
            selector={
                **self.labels,
                "model": self.get_model_name(model_version)
            },
            container_ports=self.container_ports,
            request_mem=memory,
            request_cpu=cpu,
            limit_mem=memory,
            limit_cpu=cpu,
            max_batch_size=1,
            args=[
                f"--tensorflow_intra_op_parallelism=1",
                f"--tensorflow_inter_op_parallelism={cpu}"
            ]
        )
    
    def deploy_dispatcher(self):
        labels = {"project": "infadapter", "module": "dispatcher"}
        create_deployment(
            self.get_service_name("dispatcher"), "mehransi/main:dispatcher", 1, self.namespace,
            labels=labels, limit_cpu=1, limit_mem="2G", container_ports=[8000]
        )
        create_kubernetes_service(
            self.get_service_name("dispatcher"), 
            target_port=8000,
            selector=labels, 
            expose_type="NodePort", 
            namespace=self.namespace
        )
        
    def deploy_adapter(self, dispatcher_endpoint, prometheus_endpoint):
        labels = {"project": "infadapter", "module": "adapter"}
        create_deployment(
            self.get_service_name("adapter"), "mehransi/main:adapter", 1, self.namespace,
            labels=labels, limit_cpu=2, limit_mem="4G", request_cpu=1, container_ports=[8000],
            env_vars={
                "BASE_MODEL_NAME": self.base_model_name,
                "BASE_SERVICE_NAME": self.base_service_name,
                "MODEL_VERSIONS": json.dumps(self.model_versions),
                "BASELINE_ACCURACIES": json.dumps(self.baseline_accuracies),
                "DISPATCHER_ENDPOINT": dispatcher_endpoint,
                "PROMETHEUS_ENDPOINT": prometheus_endpoint,
                "NFS_SERVER_IP": self.node_ip,
                "CONTAINER_PORTS": json.dumps(self.container_ports),
                "CONTAINER_LABELS": json.dumps(self.labels),
                "K8S_IN_CLUSTER_CLIENT": "true"
            }
        )
        create_kubernetes_service(
            self.get_service_name("adapter"),
            target_port=8000,
            selector=labels,
            expose_type="NodePort",
            namespace=self.namespace
        )
    
    def setup(self):
        # Deploy models
        for m in self.model_versions:
            self.deploy_model(m, cpu=2, memory="2G")
        time.sleep(2)
        wait_till_pods_are_ready(self.get_service_name(self.model_versions[-1]), self.namespace)
        endpoint_dict = {}
        for m in self.model_versions:
            k8s_service = get_service(self.get_service_name(m) + "-rest", self.namespace)
            endpoint_dict[self.get_model_name(m)] = f"{k8s_service['cluster_ip']}:{k8s_service['port']}"
        
        # Deploy the dispatcher
        self.deploy_dispatcher()
        time.sleep(2)
        wait_till_pods_are_ready(self.get_service_name("dispatcher"), self.namespace)
        dispatcher_k8s_service = get_service(self.get_service_name("dispatcher"), self.namespace)
        requests.post(f"{self.node_ip}:{dispatcher_k8s_service['node_port']}/initialize", json=endpoint_dict)
        
        # Deploy the adaptor
        self.deploy_adapter(
            f"{dispatcher_k8s_service['cluster_ip']}:{dispatcher_k8s_service['port']}",
            f"{self.node_ip}:{get_service('prometheus-k8s', 'monitoring')['node_port']}"
        )
        time.sleep(2)
        wait_till_pods_are_ready(self.get_service_name("adapter"), self.namespace)
        adapter_k8s_service = get_service(self.get_service_name("adapter"), self.namespace)
        requests.post(
            f"{self.node_ip}:{adapter_k8s_service['node_port']}/initialize",
            json={
                "max_cpu": self.max_cpu,
                "models_config": [[v, 2] for v in self.model_versions],
                "alpha": self.alpha    
            }
        )
        
        os.system(f"kubectl apply -f {AUTO_TUNER_DIRECTORY}/model_server_podmonitor.yaml")
        os.system(f"kubectl apply -f {AUTO_TUNER_DIRECTORY}/infadapter_podmonitor.yaml")
        

    def delete(self):
        delete_deployment(self.get_service_name("adapter"), self.namespace)
        delete_service(self.get_service_name("adapter"), self.namespace)
        
        delete_deployment(self.get_service_name("dispatcher"), self.namespace)
        delete_service(self.get_service_name("dispatcher"), self.namespace)
        
        for m in self.model_versions:
            delete_previous_deployment(self.get_service_name(m), self.namespace)


def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v

def query_metrics(prom_port, event: Event):
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), prom_port)
    while True:
        if event.is_set():
            break
        time.sleep(60)
        percentile_99 = prom.get_instant(
            f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501"}}[60s])) by (le))'
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
    starter = Starter()
    prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), prom_port)
    starter.setup()
    
    start_time = datetime.now().timestamp()
    event = Event()
    query_task = Thread(target=query_metrics, args=(prom_port, event))
    query_task.start()
    
    # Start generating load
    dispatcher_k8s_service = get_service(starter.get_service_name("dispatcher"), starter.namespace)
    total_requests, failed = generate_workload(f"{starter.node_ip}:{dispatcher_k8s_service['node_port']}")
    
    print(total_requests, failed)
    time.sleep(5)
    event.set()
    query_task.join()
    starter.delete()
