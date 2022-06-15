import os
import time
import matplotlib.pyplot as plt

import redis
from settings import BASE_DIR


store = redis.Redis(db=0)

with open(f"{BASE_DIR}/auto_tuner/dataset/twitter_trace/workload.txt", "r") as f:
    requests = f.read()

length = 120
requests = list(map(int, requests.split()))
requests = requests[38*length:39*length]


def generate_workload(endpoint):
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(requests) + 1), requests, label="request count")
    plt.legend()
    plt.savefig("load_generator.png")
    plt.close()

    store.set("inference_endpoint", endpoint)
    rate = requests[0]
    store.set("load_test_rate", rate)
    os.popen(f"locust -f locustfile.py -u 1 -r 1 --run-time {len(requests) + 2} --stop-timeout 1")
    for rate in requests[1:]:
        time.sleep(1)
        store.set("load_test_rate", rate)
