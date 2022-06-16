import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

import redis
from settings import BASE_DIR


store = redis.Redis(db=0)

with open(f"{BASE_DIR}/auto_tuner/dataset/twitter_trace/workload.txt", "r") as f:
    requests = f.read()

length = 60
requests = list(map(int, requests.split()))
requests = requests[456*length:457*length]

images = np.load(
    "/home/mehran/my_repos/master_project/auto_tuner/experiments/saved_inputs.npy", allow_pickle=True
)

i = 1
for image in images:
    store.set(f"imagenet-{i}", json.dumps(image))
    i += 1
    if i == 100:
        break

del images

# def locust_generator(host, rate):
#     return os.system(
#         f"locust -f /home/mehran/my_repos/master_project/auto_tuner/experiments/locustfile.py --headless"
#         f" --host {host} -u {rate} -r {rate} --run-time 1 --stop-timeout 1"
#     )


def generate_workload(ip, port):
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(requests) + 1), requests, label="request count")
    plt.legend()
    plt.savefig("load_generator.png", format="png")
    plt.close()
    ip = "192.5.86.160"

    for rate in requests:
        # locust_generator_process = Process(target=locust_generator, args=(f"http://{ip}:{port}", rate))
        # locust_generator_process.start()
        os.popen(
            f"locust -f /home/mehran/my_repos/master_project/auto_tuner/experiments/locustfile.py --headless"
            f" --host http://{ip}:{port} -u {rate} -r {rate} --run-time 1 --stop-timeout 2"
        )
        time.sleep(1)
