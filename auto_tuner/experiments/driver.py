import time
from datetime import datetime

import os
from ray.tune.suggest import BasicVariantGenerator
import ray
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


def start_experiment(config):
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
    print(config[ParamTypes.CPU])

    if not is_config_valid(config):  # Todo: implement
        return

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
    total_time, average, minimum, maximum = generate_load(url, total_requests)
    # save_results(config, prom, start_time=start_time)
    save_load_results(
        config,
        total=total_requests,
        total_time=total_time,
        average=average,
        minimum=minimum,
        maximum=maximum
    )
    delete_previous_deployment(service_name, namespace)
    # time.sleep(10)
    time.sleep(5)


tune_search_space = {
    ParamTypes.CPU: ray.tune.grid_search([4]),
    ParamTypes.MEMORY: ray.tune.grid_search(["4G"]),
    ParamTypes.REPLICA: ray.tune.randint(2, 3),
    ParamTypes.BATCH: ray.tune.grid_search([16]),
    ParamTypes.MODEL_ARCHITECTURE: ray.tune.grid_search([18]),
    ParamTypes.INTER_OP_PARALLELISM: ray.tune.randint(1, 2),
    ParamTypes.INTRA_OP_PARALLELISM: ray.tune.randint(4, 5),

}

if __name__ == "__main__":
    start_experiment(
        {
            ParamTypes.CPU: 2,
            ParamTypes.MEMORY: "2G",
            ParamTypes.REPLICA: 2,
            ParamTypes.BATCH: 16,
            ParamTypes.MODEL_ARCHITECTURE: 50,
            ParamTypes.INTER_OP_PARALLELISM: 1,
            ParamTypes.INTRA_OP_PARALLELISM: 2
        }
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
