import argparse
import os
import json
import math
import numpy as np
import cv2
from multiprocessing import Pipe, Process
import base64 


def get_sublists(lst: list, n: int):
    size = int(math.ceil(len(lst) / n))
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def get_data(img_path):
    im = cv2.imread(img_path)
    im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    encoded = base64.b64encode(cv2.imencode(".jpeg",im)[1].tobytes())
    instance =[{"b64": encoded.decode("utf-8")}]

    return json.dumps({"inputs": instance})



files = []
for directory in os.listdir("imagenet"):
    for img in os.listdir(f"imagenet/{directory}"):
        files.append({"label_code": directory, "path": f"imagenet/{directory}/{img}"})
        break
    # if len(files) >= 200:
    #     break


print("all files", len(files))


def task(id, conn, files):
    data = []
    i = 1
    for d in files:
        try:
            data.append(
                {"label_code": d["label_code"], "data": get_data(d["path"])}
            )
        except np.AxisError:
            continue
        if len(data) == len(files) // 2:
            i += 1
            print(f"Process{id} processed {len(data)} files.")
    print(f"Process{id} totally processed {len(data)} files.")
    conn.send(data)
    conn.close()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--processes",
    help="Number of processes for reading the dataset.",
    type=int,
    required=True
)
args = parser.parse_args()


if __name__ == "__main__":
    number_of_processes = args.processes
    file_path_shares = get_sublists(files, number_of_processes)
    processes = []
    parent_conns = []
    for i in range(1, number_of_processes + 1):
        parent_conn, child_conn = Pipe()
        p = Process(target=task, args=(i, child_conn, file_path_shares[i-1]))
        p.start()
        processes.append(p)
        parent_conns.append(parent_conn)

    inputs = []
    for i in range(number_of_processes):
        inputs.extend(parent_conns[i].recv())
    for i in range(number_of_processes):
        processes[i].join()

    print("total images:", len(inputs))
    np.save("saved_inputs", inputs, allow_pickle=True)
