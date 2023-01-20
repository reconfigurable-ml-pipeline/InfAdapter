import os
import json
from typing import List
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
import tensorflow as tf
from aiohttp import ClientSession, web
from tensorflow.keras.models import load_model
from kube_resources.deployments import create_deployment, delete_deployment, update_deployment
from kube_resources.configmaps import create_configmap
from reconfig import Reconfiguration


class Adapter:
    def __init__(self) -> None:
        self.__current_config: List[tuple] = None
        self.__current_load: int = None
        self.__base_model_name = os.environ["BASE_MODEL_NAME"]
        self.__base_service_name = os.environ["BASE_SERVICE_NAME"]
        self.__model_versions: list = json.loads(os.environ["MODEL_VERSIONS"])
        self.__dispatcher_endpoint = os.environ["DISPATCHER_ENDPOINT"]
        self.__prometheus_endpoint = os.environ["PROMETHEUS_ENDPOINT"]
        self.__nfs_server_ip = os.environ["NFS_SERVER_IP"]
        self.__container_ports = json.loads(os.environ["CONTAINER_PORTS"])
        self.__container_labels = json.loads(os.environ["CONTAINER_LABELS"])
        self.__baseline_accuracies = json.loads(os.environ["BASELINE_ACCURACIES"])
        self.__k8s_namespace = os.environ["K8S_NAMESPACE"]
        self.__lstm = os.environ["LSTM_MODEL"]
        self.__reconfiguration: Reconfiguration = None
        self.__stabilization_counter = 0
        self.__dispatcher_session = ClientSession()
        self.__prometheus_session = ClientSession()
        self.__executor = ProcessPoolExecutor(max_workers=1)
        self.__thread_executor = ThreadPoolExecutor(max_workers=len(self.__model_versions))
        
    def get_current_accuracy(self):
        acc = 0
        for mc in self.__current_config:
            model_name, _, quota = mc
            acc += self.__baseline_accuracies[model_name] * (quota / self.__current_load)
        return f"{acc:.2f}"
    
    def get_current_cost(self):
        total_size = 0
        for mc in self.__current_config:
            _, size, _ = mc
            total_size += size
        return total_size
    
    def is_initialized(self):
        return bool(self.__model_versions)
    
    @staticmethod
    def convert_config_to_dict(config):
        d = {}
        for ms in config:
            m, c, _ = ms
            d[m] = c
        return d
    
    async def initialize(self, data):
        self.__current_config = data["models_config"]
        capacity_models_paths = {}
        for v in self.__model_versions:
            capacity_models_paths[v] = f"{os.environ['CAPACITY_MODELS']}/{v}.joblib"
        self.__reconfiguration = Reconfiguration(
            self.__model_versions, data["max_cpu"], capacity_models_paths, self.__baseline_accuracies, data["alpha"]
        )
        config_with_quotas = []
        total_rate = 0
        quotas = {}
        for tup in sorted(self.__current_config, key=lambda x: x[0], reverse=True):
            m, c = tup
            rate = self.__reconfiguration.regression_model(m, c)
            config_with_quotas.append(m, c, rate)
            total_rate += rate
            quotas[self.__base_model_name + f"-{m}"] = rate
        self.__current_load = total_rate
        async with self.__dispatcher_session.post(
            f"{self.__dispatcher_endpoint}/reset", json={"quotas": quotas}
        ) as response:
            await response.text()
    
    
    async def create_ml_service(self, model_version: int, size: int):
        kwargs = {"name": f"{service_name}-container"}
        service_name = self.__base_service_name + f"-{model_version}"
        volume_mount_path = "/etc/tfserving"
        configmap_name = f"{service_name}-cm"
        config_volume_name = f"{service_name}-config-vol"
        models_volume_name = f"{service_name}-models-vol"
        
        kwargs.update({"request_cpu": size, "limit_cpu":size, "request_mem": f"{size}G", "limit_mem": f"{size}G"})
        kwargs["container_ports"] = self.__container_ports
        kwargs["image"] = "tensorflow/serving:2.8.0"

        kwargs["args"] = [
            f"--tensorflow_intra_op_parallelism=1",
            f"--tensorflow_inter_op_parallelism={size}"
            f"--model_config_file={volume_mount_path}/models.config",
            f"--monitoring_config_file={volume_mount_path}/monitoring.config",
            "--enable_batching=false",
        ]

        labels = {**self.__container_labels, "model": self.__base_model_name + f"-{model_version}"}

        volumes = [
            {
                "name": config_volume_name,
                "config_map": {"name": configmap_name}
            },
            {
                "name": models_volume_name,
                "nfs": {"server": self.__nfs_server_ip, "path": "/fileshare/tensorflow_resnet_b64"}
            }
        ]
        
        kwargs["volume_mounts"] = [
            {"name": config_volume_name, "mount_path": "/etc/tfserving"},
            {"name": models_volume_name, "mount_path": f"/models/{self.__base_model_name}"}
        ]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: create_deployment(
                service_name, 
                containers=[kwargs], 
                replicas=1, 
                namespace=self.__k8s_namespace, 
                labels=labels, 
                volumes=volumes
            )
        )
    
    async def update_ml_service(self, model_version: int, size: int):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: update_deployment(
                self.__base_service_name + f"-{model_version}",
                request_cpu=size, limit_cpu=size,
                request_mem=f"{size}G", limit_mem=f"{size}G",
                partial=True, namespace=self.__k8s_namespace
            )
        )
        
    async def delete_ml_service(self, model_version: int):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: delete_deployment(
                self.__base_service_name + f"-{model_version}",
                self.__k8s_namespace
            )
        )
        
        
        
    def predict(self, history_10m_rps):
        inp = []
        for i in range(0, len(history_10m_rps), 60):
            inp.append(max(history_10m_rps[i:i+60]))
        history_10m = tf.convert_to_tensor(np.array(inp).reshape((-1, 10, 1)), dtype=tf.float32)
        next_m = self.__lstm.predict(history_10m)
        return int(next_m)
    
    async def make_decision(self):
        loop = asyncio.get_event_loop()
        now = datetime.now().timestamp()
        async with self.__prometheus_session.post(
            f"{self.__prometheus_endpoint}/api/v1/query_range",
            params={
                "query": "sum(dispatcher_requests_total)",
                "start": now - 599,
                "end": now,
                "step": 1
            }
        ) as response:
            history_10m_rps = await response.json()["data"]["result"][0]["values"]
        print(history_10m_rps)
        print(len(history_10m_rps))
        history_10m_rps = list(map(lambda x:int(x[1]), history_10m_rps))

        next_m_rps = await loop.run_in_executor(self.__executor, self.predict, history_10m_rps)
        if next_m_rps <= self.__current_load:
            if self.__stabilization_counter < 5:
                self.__stabilization_counter += 1
                return {"success": True, "message": "stabilized"}
        self.__stabilization_counter == 0
        self.__current_load = next_m_rps
        
        
        next_config = await loop.run_in_executor(
            self.__executor, self.__reconfiguration.reconfig, self.__current_load, self.__current_config
        )
        next_config_dict = self.convert_config_to_dict(next_config)
        current_config_dict = self.convert_config_to_dict(self.__current_config)
        
        # Apply config using K8s Python client
        tasks = []
        for model in next_config_dict.keys():
            current_size = current_config_dict.get(model)
            next_size = next_config_dict[model]
            if not current_size:
                tasks.append(asyncio.create_task(self.create_ml_service(model, next_size)))
            elif current_size != next_size:
                tasks.append(asyncio.create_task(self.update_ml_service(model, next_size)))
        
        for model in current_config_dict.keys():
            if not next_config_dict.get(model):
                tasks.append(asyncio.create_task(self.delete_ml_service(model)))
        await asyncio.gather(*tasks)        
        
        # Send new quotas to the dispatcher
        quotas = {}
        for model_conf in next_config:
            model_version, _, quota = model_conf
            quotas[f"{self.__base_model_name}-{model_version}"] = quota
        
        async with self.__dispatcher_session.post(
            f"{self.__dispatcher_endpoint}/reset", json={"quotas": quotas}
        ) as response:
            response = await response.text()
        self.__current_config = next_config
        return {"success": True, "message": "reconfigured"}


adapter = Adapter()


async def initialize(request):
    data = await request.json()
    if not adapter.is_initialized():
        await adapter.initialize(data)
        return web.json_response({"success": True})
    return web.json_response({"success": False, "message": "Already initialized."})



async def export_request_count(request):
    content = "<!DOCTYPE html>"
    content += "<html>\n<head><title>Dispatcher exporter</title></head>\n<body>\n"
    content += '<pre style="word-wrap: break-word; white-space: pre-wrap;">\n'
    content += "# HELP infadapter_accuracy Weighted average of models used.\n"
    content += "# TYPE infadapter_accuracy gauge\n"
    content += f'infadapter_accuracy {adapter.get_current_accuracy()}\n'
    content += "# Help infadapter_cost Number of cpu cores used by the models.\n"
    content += "# Type infadapter_cost gauge\n"
    content += f"infadapter_cost {adapter.get_current_cost()}\n"
    content += "</pre>\n</body>\n</html>"
    return web.Response(body=content)


async def decide(request):
    return web.json_response(await adapter.make_decision())


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.get("/metrics", export_request_count),
        web.post("/decide", decide)
    ]
)
if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8000)
