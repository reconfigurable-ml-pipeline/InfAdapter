import numpy as np 
import requests
import asyncio
from aiohttp import ClientSession
import time
from datetime import datetime
import csv
import os
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

namespace = "mehran"
service_name = "tfserving-resnet"


async def predict(session: ClientSession, url, data):
    try:
        async with session.post(url, data=data["data"]) as response:
            response = await response.json()
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
    print("starting load generation...")
    print("config", config)
    tasks = []
    current_time = 0
    count = 0
    async with ClientSession() as session:
        start_time = time.perf_counter()
        data = images[0]
        while current_time < 120:
            task = asyncio.create_task(predict(session, url, data))
            tasks.append(task)
            next_time = np.random.exponential(1/lmbda)
            await asyncio.sleep(next_time)
            current_time += next_time
            count += 1
    
        await asyncio.gather(*tasks)

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
    server_count = prom.get_instant('sum(:tensorflow:serving:request_latency_count{instance=~".*:8501"})')
    server_count = _get_value(server_count)
    if server_count:
        server_count -= warmup_count

    return {
        "percent_708ms": _get_value(percent_708ms),
        "average_latency": round(average_latency, 2) if average_latency else None,
        "p99_latency": _get_value(percentile_99, divide_by=1000),
        "rate": _get_value(rate),
        "1/avg_latency": round(1000 / average_latency, 2) if average_latency else None,
        "p99_queueing": _get_value(queueing_delay_p99, 1000),
        "client_count": count,
        "duration": round(duration, 2),
        "server_count": server_count
    }




def start_service(hardware, cpu, mem, arch, batch, num_batch_threads, max_enqueued_batches, intra_op, inter_op):
    config = {
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
    wait_till_pods_are_ready(f"{service_name}-predictor-default", namespace)
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
        for cpu in [8]:
            for mem in [f"{cpu}G"]:
                # for arch in [18, 34, 50, 101, 152]:
                for arch in [50]:
                    for batch in [1, 8, 64]:
                        for nbt in [cpu]:
                            for meb in [1000000]:
                                for intra_op in [1, cpu]:
                                    for inter_op in [1, cpu]:
                                        if inter_op == intra_op == 1:
                                            continue
                                        config = start_service(hw, cpu, mem, arch, batch, nbt, meb, intra_op, inter_op)
                                        port = get_service(f"tfserving-resnet-rest", "mehran")["node_port"]
                                        url = f"http://{ip}:{port}/v1/models/resnet:predict"
                                        warmup(url)
                                        for lmbda in [40, 50, 60, 70, 80, 90, 100]:
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
                                        time.sleep(5)
