import numpy as np 
import requests
import asyncio
from aiohttp import ClientSession, TCPConnector
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


def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v


async def predict(session: ClientSession, url, data, delay=0):
    if delay > 0:
        await asyncio.sleep(delay)
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


async def find_saturation_throughput(prom, url, count):
    tasks = []
    data = images[0]
    current_time = 0
    conn = TCPConnector(limit=0)
    async with ClientSession(connector=conn) as session:
        start_time = time.perf_counter()
        while current_time < 1.5:
            task = asyncio.create_task(predict(session, url, data))
            tasks.append(task)
            next_time = np.random.exponential(1/count)
            await asyncio.sleep(next_time)
            current_time += next_time
        returns = await asyncio.gather(*tasks)
        
    duration = time.perf_counter() - start_time
    duration_int = int(duration)
    rate = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    return _get_value(rate)


async def find_p99_latency(prom, url, rps):
    tasks = []
    current_time = 0
    data = images[0]
    conn = TCPConnector(limit=0)
    async with ClientSession(connector=conn) as session:
        start_time = time.perf_counter()
        while current_time < 30:  # generate rps load based on exponential distribution for 30 seconds
            task = asyncio.create_task(predict(session, url, data))
            tasks.append(task)
            next_time = np.random.exponential(1/rps)
            await asyncio.sleep(next_time)
            current_time += next_time

        await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time
    duration_int = int(duration)
    percentile_99 = prom.get_instant(
        f'histogram_quantile(0.99, sum(rate(:tensorflow:serving:request_latency_bucket{{instance=~".*:8501"}}[{duration_int}s])) by (le))'
    )
    
    return _get_value(percentile_99, divide_by=1000)
    

rate_cache = {}
p99_cache = {}

def find_capacity(url, sla_ms, config, prom):
    if p99_cache.get(str(config)) is None:
        p99_cache[str(config)] = {}
    print("starting load generation...")
    print("config", config)
    
    if rate_cache.get(str(config)) is None:
        rps_list = []
        for repeat in range(5):
            rps_list.append(
                asyncio.run(
                    find_saturation_throughput(
                        prom,
                        url,
                        40 * config[ParamTypes.REPLICA] * config[ParamTypes.CPU]
                    )
                )
            )
            time.sleep(1)
        rps = max(rps_list)
        print(f"rp_list: {rps_list}")
        rate_cache[str(config)] = rps
    
    time.sleep(5)
    peak_rate = rate_cache[str(config)]
    lo, hi = max(int(peak_rate) - 15, 0), round(peak_rate) + 7
    if peak_rate > 100:
        hi += 50 * (peak_rate // 100)
        
    p99 = None
    
    final_capacity = None
    final_p99 = None
    while hi > lo + 1:
        capacity = (hi + lo) // 2
        if p99_cache.get(str(config)).get(capacity) is None:
            p99_cache[str(config)][capacity] = asyncio.run(find_p99_latency(prom, url, capacity))
        p99 = p99_cache[str(config)][capacity]
        if p99 <= sla_ms:
            lo = capacity
            final_p99 = p99
            final_capacity = capacity
        else:
            hi = capacity
    print(f"SLA: {sla_ms} ms, closest_p99: {final_p99} ms, capacity: {final_capacity} rps")
    return final_p99, final_capacity, peak_rate
        
    
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
    delete_previous_deployment(service_name, namespace)


if __name__ == "__main__":
    for hw in ["cpu"]:
        if hw == "cpu":
            ip = "192.5.86.160"
        else:
            ip = "192.5.86.155"
        prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
        prom = PrometheusClient(ip, prom_port)
        for replicas in [1]:
            for cpu in range(1, 5):
                for mem in [f"{2*cpu}G"]:
                    for arch in [18, 34, 50, 101, 152]:
                        for batch in [1]:
                            for nbt in [cpu]:
                                for meb in [1000000]:
                                    for par_tp in set([(cpu, 1)]):
                                        inter_op, intra_op = par_tp
                                        config = start_service(replicas, hw, cpu, mem, arch, batch, nbt, meb, intra_op, inter_op)
                                        port = get_service(f"tfserving-resnet-rest", "mehran")["node_port"]
                                        url = f"http://{ip}:{port}/v1/models/resnet:predict"
                                        warmup(url)
                                        time.sleep(2)
                                        for sla in [750]:
                                            p99, capacity, saturation_tp = find_capacity(url, sla, config, prom)
                                            filepath = f'{AUTO_TUNER_DIRECTORY}/../results/capacity_result.csv'
                                            file_exists = os.path.exists(filepath)
                                            with open(
                                                filepath, 'a', newline=''
                                            ) as csvfile:
                                                field_names = [
                                                    *list(config.keys()),
                                                    "SLA",
                                                    "p99_latency",
                                                    "capacity",
                                                    "saturation_tp",
                                                    "timestamp"
                                                ]
                                                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                                if not file_exists:
                                                    writer.writeheader()
                                                
                                                writer.writerow(
                                                    {
                                                        **config,
                                                        "SLA": sla,
                                                        "p99_latency": p99,
                                                        "capacity": capacity,
                                                        "saturation_tp": saturation_tp,
                                                        "timestamp": datetime.now().isoformat()
                                                    }
                                                )
                                                time.sleep(5)
                                        delete_service()
                                        time.sleep(10)
