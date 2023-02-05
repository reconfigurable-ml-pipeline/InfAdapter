import os
import logging
import time
import json
from typing import List
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import tensorflow as tf
from aiohttp import ClientSession, web
from tensorflow.keras.models import load_model
from kube_resources.deployments import create_deployment, delete_deployment, get_deployment
from reconfig import Reconfiguration
from aioprocessing import AioProcess, AioQueue


class Adapter:
    def __init__(self) -> None:
        self.__current_config: List[tuple] = None
        self.__current_load: int = None
        self.__base_model_name = os.environ["BASE_MODEL_NAME"]
        self.__base_service_name = os.environ["BASE_SERVICE_NAME"]
        self.__model_versions: list = sorted(json.loads(os.environ["MODEL_VERSIONS"]))
        self.__versioning = {v: 0 for v in self.__model_versions}
        self.__dispatcher_endpoint = os.environ["DISPATCHER_ENDPOINT"]
        self.__prometheus_endpoint = os.environ["PROMETHEUS_ENDPOINT"]
        self.__nfs_server_ip = os.environ["NFS_SERVER_IP"]
        self.__container_ports = json.loads(os.environ["CONTAINER_PORTS"])
        self.__container_labels = json.loads(os.environ["CONTAINER_LABELS"])
        self.__baseline_accuracies = {}
        for k, v in json.loads(os.environ["BASELINE_ACCURACIES"]).items():
            self.__baseline_accuracies[int(k)] = float(v)
        self.__load_times = {}
        for k, v in json.loads(os.environ["LOAD_TIMES"]).items():
            self.__load_times[int(k)] = float(v)
        self.__k8s_namespace = os.environ["K8S_NAMESPACE"]
        self.__lstm = None
        self.__stabilization_counter = 0
        self.__stabilization_interval = int(os.environ["STABILIZATION_INTERVAL"])
        self.__dispatcher_session = None
        self.__prometheus_session = None
        self.__max_cpu = int(os.environ["MAX_CPU"])
        self.__alpha = float(os.environ["ALPHA"])
        self.__beta = float(os.environ["BETA"])
        self.__prediction_error_percentage = int(os.environ["PREDICTION_ERROR_PERCENTAGE"])
        self.__warmup_count = int(os.environ["WARMUP_COUNT"])
        self.__solver_send_queue = AioQueue()
        self.__solver_receive_queue = AioQueue()
        self.__solver_type = os.environ["SOLVER_TYPE"]  # i=infadaptor, m=MS+
        self.__capacity_models_paths = {}
        for v in self.__model_versions:
            self.__capacity_models_paths[v] = f"{os.environ['CAPACITY_MODELS']}/{self.__base_model_name}-{v}.joblib"
        self.__solver_process = AioProcess(
            target=self.solver,
            kwargs=dict(
                send_queue=self.__solver_send_queue,
                receive_queue=self.__solver_receive_queue,
                solver_type=self.__solver_type,
                model_versions=self.__model_versions,
                max_cpu=self.__max_cpu,
                capacity_models_paths=self.__capacity_models_paths,
                baseline_accuracies=self.__baseline_accuracies,
                load_times=self.__load_times,
                alpha=self.__alpha,
                beta=self.__beta
            )
        )
        self.__solver_process.start()
        
        self.__thread_executor = ThreadPoolExecutor(max_workers=2 * len(self.__model_versions))
        
        self.logger = logging.getLogger()
        
        
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
        return bool(self.__current_config)
    
    @staticmethod
    def convert_config_to_dict(config):
        d = {}
        for ms in config:
            m, c, _ = ms
            d[m] = c
        return d
    
    async def initialize(self, data):
        self.__dispatcher_session = ClientSession()
        self.__prometheus_session = ClientSession()
        self.__lstm = load_model(os.environ["LSTM_MODEL"])
        self.__current_config = data["models_config"]
        
        reconfiguration = Reconfiguration(
            **dict(
                model_versions=self.__model_versions,
                max_cpu=self.__max_cpu,
                capacity_models_paths=self.__capacity_models_paths,
                baseline_accuracies=self.__baseline_accuracies,
                load_times=self.__load_times,
                alpha=self.__alpha,
                beta=self.__beta
            )
        )
        config_with_quotas = []
        total_rate = 0
        quotas = {f"{self.__base_model_name}-{v}": 0 for v in self.__model_versions}
        tasks = []
        for tup in sorted(self.__current_config, key=lambda x: x[0], reverse=True):
            m, c = tup
            asyncio.create_task(self.create_ml_service(m, c, self.__versioning[m]))
            tasks.append(asyncio.create_task(self.check_readiness(m, self.__versioning[m])))
            rate = reconfiguration.regression_model(m, c)
            config_with_quotas.append((m, c, rate))
            total_rate += rate
            quotas[self.__base_model_name + f"-{m}"] = rate
        await asyncio.gather(*tasks)
        self.__current_load = total_rate
        self.__current_config = config_with_quotas
        self.logger.info(f"adapter init config {str(self.__current_config)} {type(self.__current_config)}")
        
        # Warmup LSTM
        self.predict(list(range(600)))
        
        async with self.__dispatcher_session.post(
            f"http://{self.__dispatcher_endpoint}/reset", json={"quotas": quotas}
        ) as response:
            return await response.json()
    
    
    @staticmethod
    def solver(send_queue, receive_queue, solver_type, **kwargs):
        reconfiguration = Reconfiguration(**kwargs)
        while True:
            data = receive_queue.get()
            lmbda = data["lambda"]
            current_state = data["state"]
            if solver_type == "i":
                next_state = reconfiguration.reconfig(lmbda, current_state)
            else:
                next_state = reconfiguration.reconfig_msp(lmbda, current_state)
            send_queue.put(next_state)
        
        
    async def create_ml_service(self, model_version: int, size: int, deploy_version: int):
        service_name = self.__base_service_name + f"-{model_version}"
        kwargs = {"name": f"{service_name}-container", "env_vars": {}}
        kwargs["env_vars"]["MODEL_NAME"] = self.__base_model_name
        volume_mount_path = "/etc/tfserving"
        configmap_name = f"{service_name}-cm"
        config_volume_name = f"{service_name}-config-vol"
        models_volume_name = f"{service_name}-models-vol"
        warmup_volume_name = f"{service_name}-warmup-vol"
        
        kwargs.update({"request_cpu": size, "limit_cpu":size, "request_mem": f"{size}G", "limit_mem": f"{size}G"})
        kwargs["container_ports"] = self.__container_ports
        kwargs["image"] = "tensorflow/serving:2.8.0"

        kwargs["args"] = [
            f"--tensorflow_intra_op_parallelism=1",
            f"--tensorflow_inter_op_parallelism={size}",
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
                "nfs": {"server": self.__nfs_server_ip, "path": f"/fileshare/tensorflow_resnet_b64/{model_version}"}
            },
            {
                "name": warmup_volume_name,
                "empty_dir": "{}"
            }
        ]
        
        kwargs["volume_mounts"] = [
            {"name": config_volume_name, "mount_path": "/etc/tfserving"},
            {"name": models_volume_name, "mount_path": f"/models/{self.__base_model_name}/{model_version}"},
            {"name": warmup_volume_name, "mount_path": "/warmup"}
        ]
        kwargs.update(readiness_probe={"exec": ["cat", "/warmup/done"], "period_seconds": 1})

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: create_deployment(
                service_name + f"-{deploy_version}", 
                containers=[
                    kwargs,
                    {
                        "name": "warmup",
                        "image": "mehransi/main:warmup_for_tfserving",
                        "env_vars": {
                            "WARMUP_COUNT": self.__warmup_count,
                            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                            "PYTHONUNBUFFERED": "1"
                        },
                        "request_cpu": "128m",
                        "request_mem": "128Mi",
                        "volume_mounts": [{"name": warmup_volume_name, "mount_path": "/warmup"}]
                    }
                ], 
                replicas=1, 
                namespace=self.__k8s_namespace, 
                labels=labels, 
                volumes=volumes
            )
        )
        
    async def delete_ml_service(self, model_version: int, deploy_version: int):
        loop = asyncio.get_event_loop()
        deploy_name = self.__base_service_name + f"-{model_version}-{deploy_version}"
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: delete_deployment(
                deploy_name,
                self.__k8s_namespace
            )
        )
        
    async def check_readiness(self, model_version: int, deploy_version: int):
        loop = asyncio.get_event_loop()
        deployment_name = self.__base_service_name + f"-{model_version}-{deploy_version}"
        ready_replicas = 0
        while ready_replicas is None or ready_replicas < 1:
            await asyncio.sleep(0.5)
            try:
                deployment = await loop.run_in_executor(
                    self.__thread_executor,
                    lambda: get_deployment(deployment_name, namespace=self.__k8s_namespace)
                )
                ready_replicas = deployment["status"]["ready_replicas"]
            except Exception:
                pass
        
    def predict(self, history_10m_rps):
        t = time.perf_counter()
        inp = []
        for i in range(0, len(history_10m_rps), 60):
            inp.append(max(history_10m_rps[i:i+60]))
        history_10m = tf.convert_to_tensor(np.array(inp).reshape((-1, 10, 1)), dtype=tf.float32)
        next_m = self.__lstm.predict(history_10m)
        self.logger.info(f"LSTM predict took {time.perf_counter() - t} seconds")
        return int(next_m)
    
    async def make_decision(self):
        t = time.perf_counter()
        now = datetime.now().timestamp()
        async with self.__prometheus_session.get(
            f"http://{self.__prometheus_endpoint}/api/v1/query_range",
            params={
                "query": "sum(dispatcher_requests_total)",
                "start": now - 601,
                "end": now,
                "step": 1
            }
        ) as response:
            response = await response.json()
            try:
                history_10m_rps = response["data"]["result"].get("values")
            except AttributeError:
                try:
                    history_10m_rps = response["data"]["result"][0].get("values")
                except (KeyError, IndexError, AttributeError):
                    history_10m_rps = response["data"]["result"]

        history_10m_rps = history_10m_rps[:-1]
        history_10m_rps = [history_10m_rps[0]] * (601 - len(history_10m_rps)) + history_10m_rps
        history_10m_rps = list(map(lambda x: int(x[1]), history_10m_rps))
        history_10m_rps = [history_10m_rps[i] - history_10m_rps[i-1] for i in range(1, len(history_10m_rps))] 

        next_m_rps = self.predict(history_10m_rps)
        next_m_rps = int((1 + self.__prediction_error_percentage / 100) * next_m_rps)
        self.logger.info(f"adapter next_m_rps {next_m_rps}")
        if next_m_rps <= self.__current_load:
            if self.__stabilization_counter < self.__stabilization_interval:
                self.__stabilization_counter += 1
                return {"success": True, "message": "stabilized"}
        
        self.__stabilization_counter = 0
        self.__current_load = next_m_rps
        
        await self.__solver_receive_queue.coro_put({"lambda": self.__current_load, "state": self.__current_config})
        next_config = await self.__solver_send_queue.coro_get()
        
        # No config can handle this load with our max_cpu
        if next_config is None:
            next_config = [(self.__model_versions[0], self.__max_cpu, 1)]
        next_config = list(map(lambda x:(x[0], x[1], x[2] * self.__current_load), next_config))
        
        self.logger.info(f"adapter next_config {str(next_config)}")
        next_config_dict = self.convert_config_to_dict(next_config)
        current_config_dict = self.convert_config_to_dict(self.__current_config)
        
        creates = []
        deletes = []
    
        # Apply config using K8s Python client
        for model in next_config_dict.keys():
            current_size = current_config_dict.get(model)
            next_size = next_config_dict[model]
            if not current_size:
                self.__versioning[model] += 1
                creates.append((model, next_size, self.__versioning[model]))
            elif current_size != next_size:
                deletes.append((model, self.__versioning[model]))
                self.__versioning[model] += 1
                creates.append((model, next_size, self.__versioning[model]))
        for model in current_config_dict.keys():
            if not next_config_dict.get(model):
                deletes.append((model, self.__versioning[model]))
        
        tasks = []
        for create in creates:
            model, size, deploy_version = create
            asyncio.create_task(self.create_ml_service(model, size, deploy_version))
            tasks.append(asyncio.create_task(self.check_readiness(model, deploy_version)))
        
        await asyncio.gather(*tasks)
                
        # Send new quotas to the dispatcher
        quotas = {f"{self.__base_model_name}-{v}": 0 for v in self.__model_versions}
        for model_conf in next_config:
            model_version, _, quota = model_conf
            quotas[f"{self.__base_model_name}-{model_version}"] = quota
        
        async with self.__dispatcher_session.post(
            f"http://{self.__dispatcher_endpoint}/reset", json={"quotas": quotas}
        ) as response:
            response = await response.text()

        tasks = []
        for delete in deletes:
            model, deploy_version = delete
            tasks.append(asyncio.create_task(self.delete_ml_service(model, deploy_version)))
        await asyncio.gather(*tasks)
        self.__current_config = next_config
        self.logger.info(f"The whole decision making process took {time.perf_counter() - t}s")
        return {"success": True, "message": "reconfigured"}


# with open(f"/home/cc/master_project/auto_tuner/adapter/envs.json", "r") as f:
#     envs = json.load(f)

# for k, v in envs.items():
#     os.environ[k] = v
    
adapter = Adapter()


async def initialize(request):
    data = await request.json()
    if not adapter.is_initialized():
        resp = await adapter.initialize(data)
        return web.json_response({"success": True, **resp})
    return web.json_response({"success": False, "message": "Already initialized."})



async def export_request_count(request):
    content = "# HELP infadapter_accuracy Weighted average of models used.\n"
    content += "# TYPE infadapter_accuracy gauge\n"
    content += f'infadapter_accuracy {adapter.get_current_accuracy()}\n'
    content += "# HELP infadapter_cost Number of cpu cores used by the models.\n"
    content += "# TYPE infadapter_cost gauge\n"
    content += f"infadapter_cost {adapter.get_current_cost()}\n"
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
