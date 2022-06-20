import time
import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import asyncio
from aiohttp import ClientSession
import redis

from auto_tuner import AUTO_TUNER_DIRECTORY


store = redis.Redis(db=0)

with open(f"{AUTO_TUNER_DIRECTORY}/dataset/twitter_trace/workload.txt", "r") as f:
    requests = f.read()

length = 60
requests = list(map(int, requests.split()))
requests = requests[456*length:457*length]

images = np.load(
    f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True
)

j = 1
for image in images:
    store.set(f"imagenet-{j}", json.dumps(image))
    j += 1
    if j == 100:
        break

del images


async def predict(url, data, delay):
    await asyncio.sleep(delay)
    async with ClientSession() as session:
        async with session.post(url, data=data["data"]) as response:
            response = await response.read()
            return response


def generate_load_for_second(url, count):
    loop = asyncio.get_event_loop()
    tasks = []
    data = json.loads(store.get(f"imagenet-{np.random.randint(1, 101)}"))
    delays = np.cumsum(np.random.exponential(1/count, count))
    for i in range(count):
        task = asyncio.ensure_future(predict(url, data, delays[i]))
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))


def generate_workload(ip, port):
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(requests) + 1), requests, label="request count")
    plt.legend()
    plt.savefig("load_generator.png", format="png")
    plt.close()
    ip = "192.5.86.160"
    url = f"http://{ip}:{port}/v1/models/resnet:predict"
    processes = []
    for rate in requests:
        generator_process = Process(target=generate_load_for_second, args=(url, rate))
        generator_process.daemon = True
        generator_process.start()
        processes.append(generator_process)
        time.sleep(1)
        procs = []
        for p in processes:
            if p.exitcode is None:
                procs.append(p)
        processes = procs
