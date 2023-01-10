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
from auto_tuner.parameters import ParamTypes
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


async def predict(session: ClientSession, url, data, delay=0):
    if delay > 0:
        await asyncio.sleep(delay)
    try:
        async with session.post(url, data=data["data"]) as response:
            response = await response.json()
            if "outputs" in response.keys():
                prediction = response['outputs'][0]
                return True
            else:
                print("error", response)
    except Exception as e:
        print(e.__class__.__name__)
    return


warmup_count = 10


def warmup(url, count):
    print("Starting warmpu...")
    for r in range(count):
        data = images[r]
        try:
            requests.post(f"{url}", data=data["data"])
        except Exception as e:
            print("Exception in warmup", e.__class__.__name__)
    print("Warmup done.")


def _get_value(prom_res, divide_by=1, should_round=True):
    for tup in prom_res:
        if tup[1] != "NaN":
            v = float(tup[1]) / divide_by
            if should_round:
                return round(v, 2)
            return v


async def find_saturation_throughput(prom, url, count):
    tasks = []
    delays = np.cumsum(np.random.exponential(1/(count * 1.5), count))
    data = images[0]
    async with ClientSession() as session:
        start_time = time.perf_counter()
        for i in range(count):
            task = asyncio.ensure_future(predict(session, url, data, delays[i]))
            tasks.append(task)
    
        returns = await asyncio.gather(*tasks)
    
    assert len(returns) == count, "No response for some of the requests"
    
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
    async with ClientSession() as session:
        start_time = time.perf_counter()
        while current_time < 60:  # generate rps load based on exponential distribution for 10 seconds
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
    average_latency = prom.get_instant(
        f'sum(rate(:tensorflow:serving:request_latency_sum{{instance=~".*:8501"}}[{duration_int}s]))'
        f' / sum(rate(:tensorflow:serving:request_latency_count{{instance=~".*:8501"}}[{duration_int}s]))'
    )
    cpu_rate = prom.get_instant(
        f'sum(rate(container_cpu_usage_seconds_total{{container="{service_name}", service="kubelet"}}[{duration_int}s]))'
    )
    cpu_throt = prom.get_instant(
        f'sum(rate(container_cpu_cfs_throttled_seconds_total{{container="{service_name}", service="kubelet"}}[{duration_int}s]))'
    )
    mem_util = prom.get_instant(
        f'sum(container_memory_working_set_bytes{{container="{service_name}", service="kubelet"}})'
    )
    return {
        "latency_p99": _get_value(percentile_99, divide_by=1000),
        "latency_avg": _get_value(average_latency, divide_by=1000),
        "cpu_rate": _get_value(cpu_rate),
        "cpu_throt": _get_value(cpu_throt),
        "mem_util": _get_value(mem_util, divide_by=1000000)
    }
    

def start_profile(url, config, prom):
    print("start profiling...")
    print("config", config)
    
    rps_list = []
    for repeat in range(5):
        rps_list.append(
            asyncio.run(
                find_saturation_throughput(prom, url, 100 * config[ParamTypes.REPLICA] * config[ParamTypes.CPU])
            )
        )
    print(f"rps_min: {min(rps_list)}. rps_max: {max(rps_list)}")
    rps = int(sum(rps_list) / len(rps_list))
    time.sleep(5)
    metrics = asyncio.run(find_p99_latency(prom, url, rps))
    print("metrics", metrics)
    
    return {
        "rate": rps,
        **metrics
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
    # for hw in ["cpu"]:
    for hw in ["cpu"]:
        if hw == "cpu":
            ip = "192.5.86.160"
        else:
            ip = "192.5.86.155"
        prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
        prom = PrometheusClient(ip, prom_port)
        for replicas in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for cpu in [1]:
                for mem in [f"2G"]:
                    for arch in [50]:
                        for batch in [1]:  # batch size of 1 means batching is disabled
                            for nbt in [cpu]:
                                for meb in [1000000]:
                                    for par_tp in set([(cpu, 1), (1, cpu), (cpu, cpu)]):
                                        inter_op, intra_op = par_tp
                                        config = start_service(replicas, hw, cpu, mem, arch, batch, nbt, meb, intra_op, inter_op)
                                        port = get_service(f"tfserving-resnet-rest", "mehran")["node_port"]
                                        url = f"http://{ip}:{port}/v1/models/resnet:predict"
                                        warmup(url, replicas*warmup_count)
                                        time.sleep(2)
                                        result = start_profile(url, config, prom)
                                        filepath = f'{AUTO_TUNER_DIRECTORY}/../results/parameter_result.csv'
                                        file_exists = os.path.exists(filepath)
                                        with open(
                                            filepath, 'a', newline=''
                                        ) as csvfile:
                                            field_names = [
                                                *list(config.keys()),
                                                *list(result.keys()),
                                                "timestamp"
                                            ]
                                            writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                            if not file_exists:
                                                writer.writeheader()
                                            
                                            writer.writerow(
                                                {
                                                    **config,
                                                    **result,
                                                    "timestamp": datetime.now().isoformat()
                                                }
                                            )
                                        delete_service()
                                        time.sleep(10)
