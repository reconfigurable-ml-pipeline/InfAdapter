import requests
from kube_resources.services import get_service
from auto_tuner import AUTO_TUNER_DIRECTORY
import requests
import time
import numpy as np
import csv
import json
import os
from datetime import datetime
import asyncio
from aiohttp import ClientSession
from auto_tuner.experiments.parameters import ParamTypes
from auto_tuner.experiments.utils import apply_config, delete_previous_deployment, wait_till_pods_are_ready


images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)
)

async def predict(url, data, delay, session):
    await asyncio.sleep(delay)
    async with session.post(url, data=data["data"]) as response:
        response = await response.json()
        if response.get("error"):
            print(response.get("error"))
        # prediction = response["outputs"][0]
        # # idx = np.argmax(response["outputs"][0])
        # print("Actual:", code_to_label[data["label_code"]])
        # t5 = (-np.array(prediction)).argsort()[:5]
        # i = 1
        # correct_top5 = False
        # for idx in t5:
        #     #print(f"prediction{i}:", idx_to_label[str(idx)])
        #     if idx_to_label[str(idx)] == code_to_label[data["label_code"]]:
        #         correct_top5 = True
        #         break
        #     i += 1
        # print("correct in top 5:", "Yes" if correct_top5 else "No")

        #print("------------------------------------------")


        return True

async def send_requests(url, images, delay):
    tasks = []
    delays = [delay * (i+1) / len(images) for i in range(len(images))]
    async with ClientSession() as session:
        for i in range(len(images)):
            task = asyncio.ensure_future(predict(url, images[i], delays[i], session))
            tasks.append(task)
    
        return await asyncio.gather(*tasks)


def start(cpu, mem, arch, intra_op, inter_op):
    namespace = "mehran"
    service_name = "tfserving-resnet"
    apply_config(
        service_name,
        namespace,
        {
            "CPU": cpu,
            "MEM": mem,
            "ARCH": arch,
            "BATCH": 1,
            "INTRA_OP_PARALLELISM":intra_op,
            "INTER_OP_PARALLELISM": inter_op,
            "REPLICA": 1
        }
    )
    time.sleep(5)
    wait_till_pods_are_ready(f"{service_name}-predictor-default", namespace)
    ip = "192.5.86.160"
    port = get_service(f"{service_name}-rest", namespace)["node_port"]
    url = f"http://{ip}:{port}/v1/models/resnet:predict"
    for warmup in range(10):
        requests.post(url, data=images[np.random.randint(0, 200)]["data"])
    

    latencies = {}
    # for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    for batch_size in [64]:
        latencies[batch_size] = []
        batch_images = images[:batch_size]
        d = {"inputs": [json.loads(di["data"])["inputs"][0] for di in batch_images]}
        for repeat in range(40):
            t = time.perf_counter()
            delay = 0
            asyncio.run(send_requests(url, batch_images, delay))
            latencies[batch_size].append(time.perf_counter() - t)
    delete_previous_deployment(service_name, namespace)
    return {k: 
        {
            "p99": round(np.percentile(v, 99), 2),
            "min": round(min(v), 2),
            "max": round(max(v), 2),
            "avg": round(sum(v) / len(v), 2)
        }
        for k, v in latencies.items()
    }


if __name__ == "__main__":
    memory = "4G"
    for cpu in [2, 4, 16]:
        for model_version in [18, 152]:
            for inter_op in [1, cpu]:
                for intra_op in [1, cpu]:
                    if inter_op == 1 and intra_op == 1:
                        continue
                    result = start(cpu, memory, model_version, intra_op=intra_op, inter_op=inter_op)
                    filepath = f'{AUTO_TUNER_DIRECTORY}/../results/no_batch_result.csv'
                    file_exists = os.path.exists(filepath)
                    with open(
                        filepath, 'a', newline=''
                    ) as csvfile:
                        field_names = [
                            ParamTypes.CPU,
                            ParamTypes.MEMORY,
                            ParamTypes.MODEL_ARCHITECTURE,
                            ParamTypes.INTER_OP_PARALLELISM,
                            ParamTypes.INTRA_OP_PARALLELISM,
                            ParamTypes.BATCH,
                            "p99",
                            "avg",
                            "min",
                            "max",
                            "timestamp"
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        if not file_exists:
                            writer.writeheader()
                        for k, v in result.items():
                            writer.writerow(
                                {
                                    **v,
                                    ParamTypes.CPU: cpu,
                                    ParamTypes.MEMORY: memory,
                                    ParamTypes.MODEL_ARCHITECTURE: model_version,
                                    ParamTypes.INTER_OP_PARALLELISM: inter_op,
                                    ParamTypes.INTRA_OP_PARALLELISM: intra_op,
                                    ParamTypes.BATCH: k,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                    time.sleep(5)
