import time
import requests
import os
import json
import asyncio
from threading import Thread, Event
import csv
from datetime import datetime

from kube_resources.services import get_service, create_service as create_kubernetes_service, delete_service, get_services
from kube_resources.deployments import create_deployment, delete_deployment, get_deployments
from kube_resources.configmaps import delete_configmap
from kube_resources.vpas import create_vpa, delete_vpa


from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.experiments.utils import wait_till_pods_are_ready
from auto_tuner.parameters import ParamTypes
from auto_tuner.utils.prometheus import PrometheusClient
from auto_tuner.experiments.workload import generate_workload


GET_METRICS_INTERVAL = 2  # Seconds

SOLVER_TYPE_INFADAPTER = "i"
SOLVER_TYPE_MSP = "m"
SOLVER_TYPE_VPA = "v"

initial_states = {
    18: 4,
    34: 6,
    50: 8,
    101: 15,
    152: 20
}

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
        self.alpha = 0.05
        self.beta = 0.001
        self.warmup_count = 3
        self.first_decide_delay_minutes = 2
        self.env_vars = {
            "BASE_MODEL_NAME": self.base_model_name,
            "BASE_SERVICE_NAME": self.base_service_name,
            "MODEL_VERSIONS": json.dumps(self.model_versions),
            "BASELINE_ACCURACIES": json.dumps(self.baseline_accuracies),
            "MEMORY_SIZES": json.dumps(self.memory_sizes),
            "LOAD_TIMES": json.dumps(self.load_times),
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
            "LATENCY_SLO_MS": 750,
            "LSTM_PREDICTION_ERROR_PERCENTAGE": 10,
            "EMW_PREDICTION_ERROR_PERCENTAGE": 10,
            "WARMUP_COUNT": self.warmup_count,
            "FIRST_DECIDE_DELAY_MINUTES": self.first_decide_delay_minutes,
            "SOLVER_TYPE": self.solver_type,
            "STABILIZATION_INTERVAL": 6,
            "DECISION_INTERVAL": 30,
            "VPA_TYPE": "P",
            "CAPACITY_COEF": 0.8,
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
        time.sleep(60)
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
            controlled_resources=["cpu"],
            namespace=self.namespace,
            update_mode="Off"
        )
        # os.system(f"cat <<EOF | kubectl apply -f - {vpa} \n EOF ")

    
    def setup(self, model_version=None) -> int:
        if self.solver_type == SOLVER_TYPE_VPA:
            if model_version is None:
                raise Exception("model_version must be set for VPA solver_type.")
            self.env_vars["MODEL_VERSIONS"] = [model_version]
            init_config = [model_version, initial_states[model_version]]
        else:
            init_config = [152, initial_states[152]]
            
        # Deploy the adaptor
        self.deploy_adapter()
        time.sleep(2)
        wait_till_pods_are_ready(self.get_service_name("adapter"), self.namespace)
        adapter_k8s_service = get_service(self.get_service_name("adapter"), self.namespace)
        
        time.sleep(10)
        response = requests.post(
            f"http://{self.node_ip}:{adapter_k8s_service['node_port']}/initialize",
            json={
                "models_config": [init_config],
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
            try:
                delete_configmap(self.get_service_name(m) + f"-cm", self.namespace)
            except Exception:
                pass
            try:
                delete_vpa(self.get_service_name(m) + f"-vpa", self.namespace)
            except Exception:
                pass



def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v


def query_metrics(prom_port, event: Event, res: dict, path: str):
    async def get_metrics(prom):
        loop = asyncio.get_event_loop()
        percentiles = {
            99: None, 98: None, 97: None, 96: None, 95: None, 90: None, 85: None,
            80: None, 75: None, 70: None, 65: None, 60: None, 55: None, 50: None,
            40: None, 30: None, 20: None, 10: None
        }
        for pl in percentiles.keys():
            percentiles[pl] = loop.run_in_executor(None, lambda: prom.get_instant(
                f'histogram_quantile(0.{pl}, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501", API="predict", entrypoint="REST"}}[{GET_METRICS_INTERVAL}s])) by (le)) / 1000'
            ))

        cost = loop.run_in_executor(
            None, lambda: prom.get_instant(f"last_over_time(infadapter_cost[{GET_METRICS_INTERVAL}s])")
        )
        accuracy = loop.run_in_executor(
            None, lambda: prom.get_instant(f"last_over_time(infadapter_accuracy[{GET_METRICS_INTERVAL}s])")
        )
        rate = loop.run_in_executor(
            None, lambda: prom.get_instant(f"sum(rate(dispatcher_requests_total[{GET_METRICS_INTERVAL}s]))")
        )
        tp = loop.run_in_executor(
            None, lambda: prom.get_instant(f"sum(rate(:tensorflow:serving:request_count[{GET_METRICS_INTERVAL}s]))")
        )
        
        cost = _get_value(await cost)
        accuracy = _get_value(await accuracy, should_round=False)
        rate = _get_value(await rate, should_round=False)
        tp = _get_value(await tp, should_round=False)
        
        for pl in percentiles.keys():
            percentiles[pl] = _get_value(await percentiles[pl], divide_by=1000)
        
        return {
            **percentiles,
            "accuracy": accuracy,
            "cost": cost,
            "rate": rate,
            "tp": tp,
            "timestamp": datetime.now().isoformat()
        }
        
    prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), prom_port)
    time.sleep(60)
    t = time.perf_counter()
    os.system(f"mkdir -p {path}")
    while True:
        if event.is_set():
            break
        time.sleep(GET_METRICS_INTERVAL)
        metrics = asyncio.run(get_metrics(prom))
        filepath = f"{path}/series.csv"
        file_exists = os.path.exists(filepath)
        with open(filepath, "a") as f:
            field_names = [
                *list(metrics.keys())
            ]
            writer = csv.DictWriter(f, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()

            writer.writerow(
                metrics
            )
    
    duration = int(time.perf_counter() - t)
    percentile_99 = prom.get_instant(
        f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501", API="predict", entrypoint="REST"}}[{duration}s])) by (le)) / 1000'
    )
    percentile_99 = _get_value(percentile_99, divide_by=1000)

    cost = prom.get_instant(f"avg_over_time(infadapter_cost[{duration}s])")
    cost = _get_value(cost)
    accuracy = prom.get_instant(f"avg_over_time(infadapter_accuracy[{duration}s])")
    accuracy = _get_value(accuracy, should_round=False)
    res.update({"accuracy": accuracy, "cost": cost, "p99_latency": percentile_99})


if __name__ == "__main__":
    solver_type = SOLVER_TYPE_INFADAPTER
    for model_version in model_versions:
        if solver_type == SOLVER_TYPE_INFADAPTER:
            filename = "infadapter"
        elif solver_type == SOLVER_TYPE_MSP:
            filename = "msp"
        else:
            filename = f"vpa-{model_version}"
        path = f"{AUTO_TUNER_DIRECTORY}/../results/{filename}"
        res = {}
        starter = Starter(solver_type=solver_type)
        prom = PrometheusClient(os.getenv("CLUSTER_NODE_IP"), starter.prom_port)
        node_port = starter.setup(model_version)
        
        start_time = datetime.now().timestamp()
        event = Event()
        query_task = Thread(target=query_metrics, args=(starter.prom_port, event, res, path))
        query_task.start()
        
        vpa_starter_task = None
        if solver_type == SOLVER_TYPE_VPA:
            vpa_starter_task = Thread(target=starter.create_vpa, args=(model_version,))
            vpa_starter_task.start()
        
        # Start generating load
        
        total_requests, successful_requests = generate_workload(
            f"http://{starter.node_ip}:{node_port}/predict"
        )
        
        res["total_requests"] = int(total_requests)
        res["successful_requests"] = int(successful_requests)
        event.set()
        query_task.join()
        with open(f"{path}/whole.txt", "w") as f:
            json.dump(
                res,
                f
            )
        if vpa_starter_task:
            vpa_starter_task.join()
        starter.delete()
        time.sleep(30)
        if solver_type != SOLVER_TYPE_VPA:
            break
