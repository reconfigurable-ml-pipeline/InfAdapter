import time
from datetime import datetime

import os
from ray.tune.suggest import BasicVariantGenerator
import ray

from auto_tuner.experiments import ParamTypes
from auto_tuner.experiments.utils import (
    is_config_valid, apply_config, wait_till_pods_are_ready, delete_previous_deployment, save_results
)
from auto_tuner.experiments.workload import generate_workload
from auto_tuner.utils.prometheus import PrometheusClient


def start_experiment(config):
    service_name = "tfserving-resnet"
    namespace = "mehran"

    prom = PrometheusClient("192.5.86.160", 30090)

    print(config)
    print(config[ParamTypes.CPU])

    if not is_config_valid(config):  # Todo: implement
        return

    apply_config(service_name, namespace, config)
    wait_till_pods_are_ready(f"{service_name}-predictor-default", namespace)  # Check if config is really applied
    time.sleep(10)
    port = int(os.popen(
        f"kubectl get -n {namespace} svc {service_name}-batch -o jsonpath='{{.spec.ports[0].nodePort}}'"
    ).read())

    ip = os.popen(
        "kubectl get node --selector='!node-role.kubernetes.io/master'"
        " -o jsonpath='{.items[0].status.addresses[0].address}'"
    ).read()

    print(ip, port)

    start_time = int(datetime.now().timestamp())
    generate_workload(ip, port)
    save_results(prom, start_time=start_time)
    delete_previous_deployment(service_name, namespace)
    time.sleep(10)


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
            ParamTypes.CPU: 4,
            ParamTypes.MEMORY: "4G",
            ParamTypes.REPLICA: 2,
            ParamTypes.BATCH: 16,
            ParamTypes.MODEL_ARCHITECTURE: 18,
            ParamTypes.INTER_OP_PARALLELISM: 1,
            ParamTypes.INTRA_OP_PARALLELISM: 4
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
