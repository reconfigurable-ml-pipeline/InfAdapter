import os
import numpy as np
import json
import redis
import requests
from locust import HttpUser, task


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


class Client(HttpUser):
    wait_time = lambda: 1

    def on_start(self):
        self.client = requests
        self.store = redis.Redis(db=0)
        self.endpoint = "/v1/models/resnet:predict"

        self.data = json.loads(self.store.get(f"imagenet-{np.random.randint(1, 101)}"))

        with open(f"{CURRENT_DIR}/imagenet_idx_to_label.json", "r") as f:
            self.idx_to_label = json.load(f)

        with open(f"{CURRENT_DIR}/imagenet_code_to_label.json", "r") as f:
            self.code_to_label = json.load(f)

    @task
    def predict(self):

        input_data = self.data["data"]
        response = self.client.post(self.host + self.endpoint, json=input_data)

        print("send request")

        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        idx = np.argmax(prediction)
        input_label_text = self.code_to_label.get(self.data["label_code"])
        response_label_text = self.idx_to_label.get(idx)

        print("sent request", input_label_text, response_label_text)

        # if input_label_text is not None and response_label_text is None:
        #     print(input_label_text == response_label_text)
