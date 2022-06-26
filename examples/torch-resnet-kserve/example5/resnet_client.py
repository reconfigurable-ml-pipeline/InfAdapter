import os
import json
import numpy as np
import cv2
import base64
import requests

PORT = int(os.popen("kubectl get svc tfserving-resnet-svc -o jsonpath='{.spec.ports[0].nodePort}'").read())

IP = os.popen(
    "kubectl get node --selector='!node-role.kubernetes.io/master' -o jsonpath='{.items[0].status.addresses[0].address}'"
).read()
PREDICT_API = f'http://{IP}:{PORT}/v1/models/resnet:predict'


def main():
    with open("imagenet_labels.json", "r") as f:
        labels = json.load(f)

    im = cv2.imread("img.jpg")
    encoded = base64.b64encode(cv2.imencode(".jpg",im)[1].tobytes())
    instance =[{"b64":encoded.decode("utf-8")}]
    predict_request = json.dumps({"inputs": instance})

    total_seconds = 0
    repeat_count = 10
    for _ in range(repeat_count):
        response = requests.post(PREDICT_API, data=predict_request)
        response.raise_for_status()
        total_seconds += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]
    idx = np.argmax(prediction)
    label = labels[str(idx)]
    print(f'class: {idx}, label: {label}, average latency: {total_seconds/repeat_count} s')


if __name__ == '__main__':
    main()
