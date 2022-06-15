import os
import sys
import argparse
import math
import json
import bz2
from multiprocessing import Pipe, Process
from settings import BASE_DIR, TWITTER_LOGS_DIRECTORY


def get_sublists(lst: list, n: int):
    size = int(math.ceil(len(lst) / n))
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def get_timestamp_from_record(line: json) -> int:
    if line.get("created_at"):
        dt = int(line["timestamp_ms"]) // 1000
    else:
        dt = int(line["delete"]["timestamp_ms"]) // 1000
    return dt


def process_file(path: str) -> dict:
    with bz2.open(path, "rt") as bz2_file:
        lines = bz2_file.readlines()
    requests_by_second = {}
    for line in lines:
        line = json.loads(line)
        dt = get_timestamp_from_record(line)
        requests_by_second[dt] = requests_by_second.get(dt, 0) + 1
    return requests_by_second


def task(id, conn, path_list):
    requests = {}
    i = 0
    for path in path_list:
        i += 1
        file_requests = process_file(path)
        for k, v in file_requests.items():
            requests[k] = requests.get(k, 0) + v
        if i % 60 == 0:
            print(f"Process{id} processed 60 files. last file was: {path}")
            print(f"Process{id} totally processed {i} files till now")
    conn.send(requests)
    conn.close()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--days",
    help="Day(s) of the dataset month to process. Whether specify a day (e.g. 5) or a range (e.g 2-10)",
    required=True,
    type=str
)
parser.add_argument(
    "--processes",
    help="Number of processes to use for reading the dataset.",
    type=int,
    required=True
)
args = parser.parse_args()

if __name__ == "__main__":
    if "-" in args.days:
        first, end = tuple(map(int, args.days.split("-")))
        if first >= end:
            print('Bad range. range must be like "start-end" ')
            sys.exit(1)
        days = list(range(first, end + 1))
    else:
        try:
            int(args.days)
        except ValueError:
            print("Bad input. Whether specify a day (e.g 5) or a range (e.g 2-10)")
            sys.exit(1)
        days = [int(args.days)]

    number_of_processes = int(args.processes)

    file_path_list = []
    for d in days:
        for h in range(24):
            for m in range(60):
                file_path = f"{TWITTER_LOGS_DIRECTORY}/{d:02}/{h:02}/{m:02}.json.bz2"
                if not os.path.exists(file_path):
                    continue
                file_path_list.append(file_path)

    if number_of_processes > len(file_path_list):
        print("Number of processes for this task is too many. Use a smaller number.")
        sys.exit(1)

    print("total tasks", len(file_path_list))
    file_path_shares = get_sublists(file_path_list, number_of_processes)
    print("task for process 1", len(file_path_shares[0]))
    print(f"task for process {number_of_processes}", len(file_path_shares[-1]))

    all_requests_by_second = {}

    processes = []
    parent_conns = []
    for i in range(1, number_of_processes + 1):
        parent_conn, child_conn = Pipe()
        p = Process(target=task, args=(i, child_conn, file_path_shares[i-1]))
        p.start()
        processes.append(p)
        parent_conns.append(parent_conn)

    p_data_list = []
    for i in range(number_of_processes):
        p_data_list.append(parent_conns[i].recv())
    for i in range(number_of_processes):
        processes[i].join()

    for p_data in p_data_list:
        for k, v in p_data.items():
            all_requests_by_second[k] = all_requests_by_second.get(k, 0) + v

    if len(all_requests_by_second) == 0:
        print("No file found in the twitter_trace_logs_directory")
        sys.exit(0)

    all_requests = []  # each item is a request rate. includes all seconds from dataset start to end

    seconds = sorted(all_requests_by_second.keys())
    current_dt = seconds[0]
    rate = all_requests_by_second[current_dt]
    for dt in seconds[1:]:
        all_requests.append(rate)
        for i in range(dt - current_dt - 1):
            all_requests.append(0)
        current_dt = dt
        rate = all_requests_by_second[dt]
    all_requests.append(all_requests_by_second[dt])

    print(len(all_requests) / 60 / 60 / 24)
    print("Maximum requests in a second", sorted(all_requests, reverse=True)[:100])
    with open(f"{BASE_DIR}/auto_tuner/dataset/twitter_trace/workload.txt", "w") as f:
        for count in all_requests:
            f.write(f"{count} ")
    os.system(
        f"tar -jcvf {BASE_DIR}/auto_tuner/dataset/twitter_trace/workload.tbz2 "
        f"{BASE_DIR}/auto_tuner/dataset/twitter_trace/workload.txt"
    )
