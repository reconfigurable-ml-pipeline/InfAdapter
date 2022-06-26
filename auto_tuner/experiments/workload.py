import time
import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import asyncio
from aiohttp import ClientSession
from auto_tuner import AUTO_TUNER_DIRECTORY


with open(f"{AUTO_TUNER_DIRECTORY}/dataset/twitter_trace/workload.txt", "r") as f:
    requests = f.read()

length = 60
requests = list(map(int, requests.split()))
requests = requests[456*length:457*length]

images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)[:100]
)


async def predict(url, data, delay, session):
    await asyncio.sleep(delay)
    async with session.post(url, data=data["data"]) as response:
        response = await response.text()
        return json.loads(response)


async def generate_load_for_second(url, count, data):
    # data = json.loads(store.get(f"imagenet-{np.random.randint(1, 101)}"))
    async with ClientSession() as session:
        delays = np.cumsum(np.random.exponential(1 / count, count))
        tasks = []
        for i in range(count):
            task = asyncio.ensure_future(predict(url, data, delays[i], session))
            tasks.append(task)
        await asyncio.gather(*tasks)


def generate_load(url, count, data):
    asyncio.run(generate_load_for_second(url, count, data))


def generate_workload(ip, port):
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(requests) + 1), requests, label="request count")
    plt.legend()
    plt.savefig("load_generator.png", format="png")
    plt.close()
    ip = "192.5.86.160"
    url = f"http://{ip}:{port}/v1/models/resnet:predict"
    processes = []

    j = 0
    for rate in requests:
        image = images[j]
        generator_process = Process(target=generate_load, args=(url, rate, image))
        generator_process.daemon = True
        generator_process.start()
        processes.append(generator_process)
        time.sleep(1)
        procs = []
        for p in processes:
            if p.exitcode is None:
                procs.append(p)
        processes = procs
        j += 1
        if j == 100:
            j == 0
