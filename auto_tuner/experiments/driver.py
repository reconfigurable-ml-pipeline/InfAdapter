import time
from datetime import datetime
import os
import json
# from ray.tune.suggest import BasicVariantGenerator
# import ray
import aiohttp
from kube_resources.services import get_service
from kube_resources.pods import get_pods

from auto_tuner.experiments import ParamTypes
from auto_tuner.experiments.utils import (
    is_config_valid,
    apply_config,
    wait_till_pods_are_ready,
    delete_previous_deployment,
    save_load_results,
    save_results
)
from auto_tuner.experiments.workload import warmup, generate_workload
from auto_tuner.experiments.load import generate_load
from auto_tuner.utils.prometheus import PrometheusClient
from auto_tuner import AUTO_TUNER_DIRECTORY

repeat_results = []
namespace = "mehran"
service_name = "tfserving-resnet"

EXPERIMENT_TYPE_STATIC = 1
EXPERIMENT_TYPE_DYNAMIC = 2

STATIC_EXPERIMENT_REPEAT_COUNT = 2


def start_experiment(config, repeat, experiment_type):
    global repeat_results

    # ip = os.popen(
    #     "kubectl get node --selector='!node-role.kubernetes.io/master'"
    #     " -o jsonpath='{.items[0].status.addresses[0].address}'"
    # ).read()
    ip = "192.5.86.160"

    prom_port = get_service("prometheus-k8s", "monitoring")["node_port"]
    prom = PrometheusClient(ip, prom_port)

    print(config)

    if repeat == 0:
        if not is_config_valid(config):  # Todo: implement
            return
        repeat_results = []
        apply_config(service_name, namespace, config)
        time.sleep(5)
        # Check if config is really applied
        wait_till_pods_are_ready(f"{service_name}-predictor-default", namespace)
        time.sleep(5)

    port = get_service(f"{service_name}-rest", namespace)["node_port"]

    print(ip, port)
    url = f"http://{ip}:{port}/v1/models/resnet:predict"

    warmup(url)
    start_time = datetime.now().timestamp()
    if experiment_type == EXPERIMENT_TYPE_STATIC:
        total_requests, failed = 300, 0
        print("total number of requests being sent", total_requests)
        result, response_times = generate_load(url, total_requests)
        fn = "-".join(map(lambda x: f"{x[0]}:{x[1]}", {**config, "repeat": repeat}.items()))
        os.system(f"mkdir -p {AUTO_TUNER_DIRECTORY}/../response_times")
        with open(f"{AUTO_TUNER_DIRECTORY}/../response_times/{fn}", "w") as f:
            f.write(json.dumps(response_times))
        repeat_results.append(result)
    else:
        total_requests, failed = generate_workload(url)

    if experiment_type == EXPERIMENT_TYPE_STATIC and repeat == STATIC_EXPERIMENT_REPEAT_COUNT:
        avg_result = {}
        for res in repeat_results:
            for k, v in res.items():
                avg_result[k] = avg_result.get(k, 0) + v
        for k, v in avg_result.items():
            avg_result[k] = round(avg_result[k] / len(repeat_results), 2)
        save_load_results(config, total=total_requests, result=avg_result)
        delete_previous_deployment(service_name, namespace)
        time.sleep(5)
    elif experiment_type == EXPERIMENT_TYPE_DYNAMIC:
        save_results(config, prom, total=total_requests, failed=failed, start_time=start_time)
        delete_previous_deployment(service_name, namespace)
        time.sleep(10)

# tune_search_space = {
#     ParamTypes.CPU: ray.tune.grid_search([4]),
#     ParamTypes.MEMORY: ray.tune.grid_search(["4G"]),
#     ParamTypes.REPLICA: ray.tune.randint(2, 3),
#     ParamTypes.BATCH: ray.tune.grid_search([16]),
#     ParamTypes.MODEL_ARCHITECTURE: ray.tune.grid_search([18]),
#     ParamTypes.INTER_OP_PARALLELISM: ray.tune.randint(1, 2),
#     ParamTypes.INTRA_OP_PARALLELISM: ray.tune.randint(4, 5),

# }

if __name__ == "__main__":
    experiment_type = EXPERIMENT_TYPE_STATIC  # Set this before running experiment
    for cpu in [2, 4, 8, 16, 32]:
        for memory in ["4G"]:
            for arch in [18, 34, 50, 101, 152]:
                for batch_size in [1]:
                    for batch_timeout in [10000]:
                        for interop in set([cpu]):
                            for intraop in set([1]):
                                if interop == 1 and intraop == 1:
                                    continue
                                if experiment_type == EXPERIMENT_TYPE_STATIC:
                                    repeat_count = STATIC_EXPERIMENT_REPEAT_COUNT
                                else:
                                    repeat_count = 0
                                for repeat in range(repeat_count + 1):
                                    try:
                                        config = {
                                            ParamTypes.CPU: cpu,
                                            ParamTypes.MEMORY: memory,
                                            ParamTypes.REPLICA: 1,
                                            ParamTypes.BATCH: batch_size,
                                            ParamTypes.BATCH_TIMEOUT: batch_timeout,
                                            ParamTypes.MODEL_ARCHITECTURE: arch,
                                            ParamTypes.INTER_OP_PARALLELISM: interop,
                                            ParamTypes.INTRA_OP_PARALLELISM: intraop
                                        }
                                        start_experiment(config, repeat=repeat, experiment_type=experiment_type)
                                    except aiohttp.client_exceptions.ServerDisconnectedError:
                                        pods = get_pods(namespace)["pods"]
                                        log = ""
                                        for pod in pods:
                                            log += f"pod: {pod['name']}"
                                            statuses = pod["container_statuses"]
                                            for status in statuses:
                                                if status["state"].get("terminated"):
                                                    log += f" - terminated: {status['state']['terminated']['reason']}"
                                                elif status["last_state"].get("terminated"):
                                                    log += f" - terminated: {status['last_state']['terminated']['reason']}"
                                        with open(f"{AUTO_TUNER_DIRECTORY}/../logs.txt", "a") as f:
                                            f.write(f"config={config} - {log}\n")
                                        delete_previous_deployment(service_name, namespace)
                                        time.sleep(5)
                                        break

    # ray.init(include_dashboard=False)
    # ray.tune.run(
    #     lambda config: start_experiment(config),
    #     config=tune_search_space,
    #     # search_alg=BasicVariantGenerator(points_to_evaluate=[
    #     #     {ParamTypes.CPU: 2, ParamTypes.BATCH: 4},
    #     #     {ParamTypes.MEMORY: "2GB"},
    #     # ]),
    #     num_samples=1
    # )
    # ray.shutdown()
