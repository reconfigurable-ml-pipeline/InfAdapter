import numpy as np
import json
import redis
from locust import HttpUser, task


class Client(HttpUser):
    wait_time = lambda: 1

    def on_start(self):
        self.global_idx = 1
        self.store = redis.Redis(db=0)
        self.endpoint = "/v1/models/resnet:predict"
        with open("/home/mehran/my_repos/master_project/auto_tuner/experiments/imagenet_idx_to_label.json", "r") as f:
            self.idx_to_label = json.load(f)

        with open("/home/mehran/my_repos/master_project/auto_tuner/experiments/imagenet_code_to_label.json", "r") as f:
            self.code_to_label = json.load(f)

    @task
    def predict(self):
        data = self.store.get(f"imagenet-{self.global_idx}")
        if not data:
            self.global_idx = 1
            data = self.store.get(f"imagenet-{self.global_idx}")

        data = json.loads(data)
        input_data = data["data"]
        self.global_idx += 1
        response = self.client.post(self.endpoint, data=input_data)

        print("send request", self.global_idx)

        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        idx = np.argmax(prediction)
        input_label_text = self.code_to_label.get(data["label_code"])
        response_label_text = self.idx_to_label.get(idx)

        print("sent request", self.global_idx, input_label_text, response_label_text)

        # if input_label_text is not None and response_label_text is None:
        #     print(input_label_text == response_label_text)
