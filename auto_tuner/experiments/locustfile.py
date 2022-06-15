import numpy as np
import json
import redis
from locust import HttpUser, task

inputs = np.load("saved_inputs.npy", allow_pickle=True)
global_idx = 0

store = redis.Redis(db=0)

with open("imagenet_idx_to_label.json", "r") as f:
    idx_to_label = json.load(f)

with open("imagenet_code_to_label.json", "r") as f:
    code_to_label = json.load(f)


class Client(HttpUser):
    def __init__(self, *args, **kwargs):
        self.endpoint = store.get("inference_endpoint")
        super().__init__(*args, **kwargs)

    def wait_time(self):
        pass

    @task
    def predict(self):
        global global_idx
        data = inputs[global_idx % len(inputs)]
        input_data = data["data"]
        global_idx += 1
        response = self.client.post(self.endpoint, data=input_data)
        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        idx = np.argmax(prediction)
        input_label_text = code_to_label.get(data["label_code"])
        response_label_text = idx_to_label.get(idx)
        # if input_label_text is not None and response_label_text is None:
        #     print(input_label_text == response_label_text)
