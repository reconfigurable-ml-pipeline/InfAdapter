import numpy as np 
import requests
import asyncio
from aiohttp import ClientSession
import time
from datetime import datetime
import csv
import os
import json
from kube_resources.services import get_service
from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.experiments.parameters import ParamTypes
from auto_tuner.experiments.utils import (
    apply_config,
    wait_till_pods_are_ready,
    delete_previous_deployment,
)
from auto_tuner.utils.prometheus import PrometheusClient


images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)
)

with open(f"{AUTO_TUNER_DIRECTORY}/experiments/imagenet_idx_to_label.json", "r") as f:
    idx_to_label = json.load(f)
    
with open(f"{AUTO_TUNER_DIRECTORY}/experiments/imagenet_code_to_label.json", "r") as f:
    code_to_label = json.load(f)
    

namespace = "mehran"
service_name = "tfserving-resnet"

prev_server_count = None


async def predict(session: ClientSession, url, data):
    try:
        async with session.post(url, data=data["data"]) as response:
            response = await response.json()
            if "outputs" in response.keys():
                prediction = response['outputs'][0]
                idx = np.argmax(prediction)
                t5 = (-np.array(prediction)).argsort()[:5]
                correct_top5 = False
                for idx in t5:
                    if idx_to_label[str(idx)] == code_to_label[data["label_code"]]:
                        correct_top5 = True
                        break
                return correct_top5
            else:
                print("error", response)
    except Exception as e:
        print(e.__class__.__name__)
    return


warmup_count = 15


def warmup(url):
    print("Starting warmpu...")
    for r in range(warmup_count):
        data = images[r]
        try:
            requests.post(f"{url}", data=data["data"])
        except Exception as e:
            print("Exception in warmup", e.__class__.__name__)
    print("Warmup done.")


async def generate_load(url, lmbda, config, prom):
    global server_count
    print("starting load generation...")
    print("config", config)
    tasks = []
    current_time = 0
    count = 0
    async with ClientSession() as session:
        start_time = time.perf_counter()
        data = images[0]
        while current_time < 60:
            task = asyncio.create_task(predict(session, url, data))
            tasks.append(task)
            next_time = np.random.exponential(1/lmbda)
            await asyncio.sleep(next_time)
            current_time += next_time
            count += 1
    
        returns = await asyncio.gather(*tasks)

    def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v

    duration = time.perf_counter() - start_time
    duration_int = int(duration)
    percent_708ms = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501", le = "708235"}}[{duration_int}s]))'
        f' / sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    average_latency = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_sum{{instance=~".*:8501"}}[{duration_int}s]))'
        f' / sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    average_latency = _get_value(average_latency, divide_by=1000, should_round=False)
    percentile_99 = prom.get_instant(
        f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501"}}[{duration_int}s])) by (le))'
    )
    rate = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    queueing_delay_p99 = prom.get_instant(
        f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:batching_session:queuing_latency_bucket{{instance=~".*:8501"}}[{duration_int}s])) by(le))'
    )
    average_queueing = prom.get_instant(
        f'sum(rate(:tensorflow:serving:batching_session:queuing_latency_sum{{instance=~".*:8501"}}[{duration_int}s]))'
        f' / sum(rate(:tensorflow:serving:batching_session:queuing_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    server_count = prom.get_instant('sum(:tensorflow:serving:request_latency_count{instance=~".*:8501"})')
    server_count = _get_value(server_count)
    cpu_rate = prom.get_instant(f'sum(rate(container_cpu_usage_seconds_total{{container="{service_name}", service="kubelet"}}[{duration_int}s]))')
    cpu_throt = prom.get_instant(f'sum(rate(container_cpu_cfs_throttled_seconds_total{{container="{service_name}", service="kubelet"}}[{duration_int}s]))')
    mem_util = prom.get_instant(f'sum(container_memory_working_set_bytes{{container="{service_name}", service="kubelet"}})')
    if server_count:
        server_count -= warmup_count
        temp = server_count
        if prev_server_count:
            server_count -= prev_server_count
        prev_server_count = temp

    return {
        "percent_708ms": _get_value(percent_708ms),
        "average_latency": round(average_latency, 2) if average_latency else None,
        "p99_latency": _get_value(percentile_99, divide_by=1000),
        "rate": _get_value(rate),
        "1/avg_latency": round(1000 / average_latency, 2) if average_latency else None,
        "p99_queueing": _get_value(queueing_delay_p99, 1000),
        "average_queueing": _get_value(average_queueing, divide_by=1000),
        "client_count": count,
        "success_count": returns.count(True),
        "duration": round(duration, 2),
        "server_count": server_count,
        "cpu_rate": _get_value(cpu_rate),
        "cpu_throt": _get_value(cpu_throt),
        "mem_util": _get_value(mem_util, divide_by=1000000)
    }




def start_service(replicas, hardware, cpu, mem, arch, batch, num_batch_threads, max_enqueued_batches, intra_op, inter_op):
    config = {
        ParamTypes.REPLICA: replicas,
        ParamTypes.HARDWARE: hardware,
        ParamTypes.CPU: cpu,
        ParamTypes.MEMORY: mem,
        ParamTypes.MODEL_ARCHITECTURE: arch,
        ParamTypes.BATCH: batch,
        ParamTypes.NUM_BATCH_THREADS: num_batch_threads,
        ParamTypes.MAX_ENQUEUED_BATCHES: max_enqueued_batches,
        ParamTypes.INTRA_OP_PARALLELISM:intra_op,
        ParamTypes.INTER_OP_PARALLELISM: inter_op,
        ParamTypes.REPLICA: 1
    }
    apply_config(
        service_name,
        namespace,
        config
    )
    time.sleep(5)
    wait_till_pods_are_ready(service_name, namespace)
    time.sleep(5)
    return config


def delete_service():
    global prev_server_count
    delete_previous_deployment(service_name, namespace)
    prev_server_count = None


if __name__ == "__main__":
    # for hw in ["cpu"]:
    for hw in ["cpu"]:
        if hw == "cpu":
            ip = "192.5.86.160"
        else:
            ip = "192.5.86.155"
        prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
        prom = PrometheusClient(ip, prom_port)
        for replicas in [1]:
            for cpu in [8]:
                for mem in [f"{cpu}G"]:
                    # for arch in [18, 34, 50, 101, 152]:
                    for arch in [50]:
                        for batch in [1, 8, 64]:
                            for nbt in [cpu]:
                                for meb in [1000000]:
                                    for intra_op in [1, cpu]:
                                        for inter_op in [1, cpu]:
                                            # if inter_op == intra_op == 1:
                                            #     continue
                                            config = start_service(replicas, hw, cpu, mem, arch, batch, nbt, meb, intra_op, inter_op)
                                            port = get_service(f"tfserving-resnet-rest", "mehran")["node_port"]
                                            url = f"http://{ip}:{port}/v1/models/resnet:predict"
                                            warmup(url)
                                            time.sleep(2)
                                            for lmbda in [20, 40, 60, 80, 100, 120]:
                                                result = asyncio.run(generate_load(url, lmbda, config, prom))
                                                filepath = f'{AUTO_TUNER_DIRECTORY}/../results/throughput_result.csv'
                                                file_exists = os.path.exists(filepath)
                                                with open(
                                                    filepath, 'a', newline=''
                                                ) as csvfile:
                                                    field_names = [
                                                        *list(config.keys()),
                                                        "lambda",
                                                        *list(result.keys()),
                                                        "timestamp"
                                                    ]
                                                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                                    if not file_exists:
                                                        writer.writeheader()
                                                    
                                                    writer.writerow(
                                                        {
                                                            **config,
                                                            "lambda": lmbda,
                                                            **result,
                                                            "timestamp": datetime.now().isoformat()
                                                        }
                                                    )
                                                    time.sleep(5)
                                            delete_service()
                                            time.sleep(10)
