import os
import numpy as np
import matplotlib.pyplot as plt
from barazmoon import BarAzmoon
import json
import numpy as np
import requests
from auto_tuner import AUTO_TUNER_DIRECTORY


with open(f"{AUTO_TUNER_DIRECTORY}/dataset/twitter_trace/workload.txt", "r") as f:
    workload_pattern = f.read()

day = 60 * 60 * 24
workload_pattern = list(map(int, workload_pattern.split()))

workload_pattern = workload_pattern[15 * day + 82 * 60 : 15 * day + 103 * 60]
# workload_pattern = workload_pattern[16 * day + 95 * 60 : 16 * day + 120 * 60]

workload_pattern = np.array(workload_pattern)


images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)
)


with open(f"{AUTO_TUNER_DIRECTORY}/experiments/imagenet_idx_to_label.json", "r") as f:
    idx_to_label = json.load(f)


def warmup(url):
    print("Starting warmpu...")
    for i in range(10):
        data = images[np.random.randint(0, 200)]
        requests.post(f"{url}", data=data["data"])
    print("Warmup done.")


def generate_workload(url):

    class WorkloadGenerator(BarAzmoon):
        
        @classmethod
        def get_request_data(cls) -> str:
            image = images[np.random.randint(0, 200)]
            return image["label_code"], image["data"]
        
        @classmethod
        def process_response(cls, data_id: str, response: dict):
            return response.get("error") is None
            # print("correct label:", data_id)
                
            # if "error" in response.keys():
            #     print(response["error"])
            # else:
            #     for i in range(len(response["outputs"])):
            #         idx = np.argmax(response["outputs"][i])
            #         # print("predicted label:", idx_to_label[str(idx)])

    
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(workload_pattern) + 1), workload_pattern, label="request count")
    plt.legend()
    results_dir = f"{AUTO_TUNER_DIRECTORY}/../results"
    os.system(f"mkdir -p {results_dir}")
    plt.savefig(f"{results_dir}/workload.png", format="png")
    plt.close()
    print("total number of requests being sent", sum(workload_pattern))
    counter, success_requests = WorkloadGenerator(workload=workload_pattern, endpoint=url, http_method="post").start()
    print(f"counter: {counter}, success_requests: {success_requests}")
    return counter, success_requests
