import os
import json
import numpy as np
from PIL import Image
import requests

PORT = int(os.popen("kubectl get svc tfserving-resnet-svc -o jsonpath='{.spec.ports[0].nodePort}'").read())

IP = os.popen(
    "kubectl get node --selector='!node-role.kubernetes.io/master' -o jsonpath='{.items[0].status.addresses[0].address}"
).read()
PREDICT_API = f'{IP}:{PORT}/v1/models/resnet:predict'


def main():
    with open("imagenet_labels.json", "r") as f:
        labels = json.load(f)

    image = Image.open(r"img.jpg")
    image = np.expand_dims(np.array(image) / 255, 0)
    image = np.moveaxis(image, 3, 1)
    predict_request = json.dumps({"instances": image.tolist()})

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
