import time
from datetime import datetime
import os
import json
# from ray.tune.suggest import BasicVariantGenerator
# import ray
from kube_resources.services import get_service
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

def start_experiment(config, repeat, max_repeat):
    global repeat_results

    service_name = "tfserving-resnet"
    namespace = "mehran"

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
    # generate_workload(url)
    total_requests = 200
    result, response_times = generate_load(url, total_requests)
    fn = "-".join(map(lambda x: f"{x[0]}:{x[1]}", {**config, "repeat": repeat}.items()))
    with open(f"{AUTO_TUNER_DIRECTORY}/../response_times/{fn}", "w") as f:
        f.write(json.dumps(response_times))
    # save_results(config, prom, start_time=start_time)

    repeat_results.append(result)

    if repeat == max_repeat:
        avg_result = {}
        for res in repeat_results:
            for k, v in res.items():
                avg_result[k] = avg_result.get(k, 0) + v
        for k, v in avg_result.items():
            avg_result[k] = avg_result[k] / len(repeat_results)
        save_load_results(
            config,
            total=total_requests,
            result=avg_result
        )
        delete_previous_deployment(service_name, namespace)
        # time.sleep(10)
        time.sleep(5)


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
    repeat_count = 2
    for cpu in [2, 8, 32]:
        for memory in [f"{cpu//2}G", f"{cpu * 2}G"]:
            for batch_size in [1, 32]:
                for batch_timeout in [10000]:
                    for arch in [18, 50, 152]:
                        for interop in set([1, cpu // 2, cpu]):
                            for intraop in set([1, cpu // 2, cpu]):
                                for repeat in range(repeat_count + 1):
                                    start_experiment(
                                        {
                                            ParamTypes.CPU: cpu,
                                            ParamTypes.MEMORY: memory,
                                            ParamTypes.REPLICA: 1,
                                            ParamTypes.BATCH: batch_size,
                                            ParamTypes.BATCH_TIMEOUT: batch_timeout,
                                            ParamTypes.MODEL_ARCHITECTURE: arch,
                                            ParamTypes.INTER_OP_PARALLELISM: interop,
                                            ParamTypes.INTRA_OP_PARALLELISM: intraop
                                        },
                                        repeat=repeat,
                                        max_repeat=repeat_count
                                    )
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
