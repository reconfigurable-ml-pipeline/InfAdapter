import numpy as np
import matplotlib.pyplot as plt
from barazmoon import BarAzmoon
import json
import numpy as np
import requests
from auto_tuner import AUTO_TUNER_DIRECTORY


with open(f"{AUTO_TUNER_DIRECTORY}/dataset/twitter_trace/workload.txt", "r") as f:
    workload_pattern = f.read()

length = 60
workload_pattern = list(map(int, workload_pattern.split()))
workload_pattern = workload_pattern[length:2*length]

workload_pattern = np.array(workload_pattern) // 9

print("total number of requests being sent", sum(workload_pattern))

images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)
)


with open(f"{AUTO_TUNER_DIRECTORY}/experiments/imagenet_idx_to_label.json", "r") as f:
    idx_to_label = json.load(f)


def warmup(ip, port):
    print("Starting warmpu...")
    for i in range(3):
        data = images[i]["data"]
        requests.post(f"http://{ip}:{port}/v1/models/resnet:predict", data=data)
    print("Warmup done.")


def generate_workload(ip, port):
    url = f"http://{ip}:{port}/v1/models/resnet:predict"

    class WorkloadGenerator(BarAzmoon):
        endpoint = url 
        timeout = None
        http_method = "post"

        def get_workload(self):
            return workload_pattern
        
        @classmethod
        def get_request_data(cls) -> str:
            image = images[np.random.randint(0, 200)]
            return image["label_code"], image["data"]
        
        @classmethod
        def process_response(cls, data_id: str, response: dict):
            # print("correct label:", data_id)
            if "error" in response.keys():
                print(response["error"])
            else:
                for i in range(len(response["outputs"])):
                    idx = np.argmax(response["outputs"][i])
                    # print("predicted label:", idx_to_label[str(idx)])

    
    plt.xlabel("time (seconds)")
    plt.plot(range(1, len(workload_pattern) + 1), workload_pattern, label="request count")
    plt.legend()
    plt.savefig("load_generator.png", format="png")
    plt.close()

    counter, failed = WorkloadGenerator().start()
    print(f"counter: {counter}, failed: {failed}")
