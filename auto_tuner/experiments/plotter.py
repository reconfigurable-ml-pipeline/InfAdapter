import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from auto_tuner import AUTO_TUNER_DIRECTORY
from auto_tuner.experiments.parameters import ParamTypes



plots_directory = f"{AUTO_TUNER_DIRECTORY}/../plots"

EXPERIMENT_TYPE_STATIC = 1
EXPERIMENT_TYPE_DYNAMIC = 2
EXPERIMENT_TYPE_ONE_BY_ONE = 3


def plot_for_param(results_path, x_params: list, experiment_type):
    df = pd.read_csv(results_path)
    if experiment_type == EXPERIMENT_TYPE_STATIC:
        dropping_columns = [ParamTypes.REPLICA, "timestamp", ParamTypes.BATCH_TIMEOUT, "total"]
    elif experiment_type == EXPERIMENT_TYPE_DYNAMIC:
        dropping_columns = [ParamTypes.REPLICA, "timestamp", ParamTypes.BATCH_TIMEOUT]
    else:
        dropping_columns = ["timestamp"]
    df = df.drop(dropping_columns, axis=1)
    columns = list(df.columns)
    print(columns)
    d = {}
    x_axis = "-".join(x_params)
    params = [
        ParamTypes.CPU,
        ParamTypes.MEMORY,
        ParamTypes.BATCH,
        ParamTypes.NUM_BATCH_THREADS,
        ParamTypes.MAX_ENQUEUED_BATCHES,
        ParamTypes.MODEL_ARCHITECTURE,
        ParamTypes.INTRA_OP_PARALLELISM,
        ParamTypes.INTER_OP_PARALLELISM
    ]
    if experiment_type == EXPERIMENT_TYPE_ONE_BY_ONE:
        params.remove(ParamTypes.NUM_BATCH_THREADS)
        params.remove(ParamTypes.MAX_ENQUEUED_BATCHES)
    for xp in x_params:
        params.remove(xp)
    for idx, row in df.iterrows():
        config = "-".join(map(lambda p: f"{p}:{row[p]}", params))
        if d.get(config) is None:
            d[config] = {}
        if len(x_params) > 1:
            k = f"{row[x_params[0]]}-{row[x_params[1]]}"
        else:
            k = row[x_params[0]]

        if experiment_type == EXPERIMENT_TYPE_ONE_BY_ONE:
            d[config][k] = {"p99": row["p99"]}
        else:
            d[config][k] = {"p50": row["p50"], "p95": row["p95"], "p99": row["p99"]}
        
        if experiment_type == EXPERIMENT_TYPE_STATIC:
            d[config][k]["total_time"] = row["total_time"]
        elif experiment_type == EXPERIMENT_TYPE_DYNAMIC:
            d[config][k]["percent_708ms"] = row["percent_708ms"]
        else:
            d[config][k]["avg"] = row["avg"]
    os.system(f"mkdir -p {plots_directory}/{x_axis}")
    for config, data in d.items():
        x_len = len(data.keys())
        x = np.arange(x_len).astype(int)
        width = 0.18
        fig, ax = plt.subplots()
        title = config.replace(ParamTypes.INTER_OP_PARALLELISM, "INTER_OP")
        title = title.replace(ParamTypes.INTRA_OP_PARALLELISM, "INTRA_OP")
        title = title.replace(ParamTypes.NUM_BATCH_THREADS, "NBC")
        title = title.replace(ParamTypes.MAX_ENQUEUED_BATCHES, "MEB")
        ax.set_title(title.replace("-", " | "), fontsize=9)
        ax.set_xlabel(f"{x_axis}")
        if experiment_type == EXPERIMENT_TYPE_DYNAMIC:
            ax.set_ylabel("latency (ms)")
        else:
            ax.set_ylabel("latency (s)")
        if experiment_type == EXPERIMENT_TYPE_ONE_BY_ONE:
            rects1 = ax.bar(x-width/2, list(map(lambda v: v["avg"], data.values())), width, label='avg')
            rects2 = ax.bar(x+width/2, list(map(lambda v: v["p99"], data.values())), width, label='p99')
        else:
            rects1 = ax.bar(x - width, list(map(lambda v: v["p50"], data.values())), width, label='p50')
            rects2 = ax.bar(x, list(map(lambda v: v["p95"], data.values())), width, label='p95')
            if experiment_type == EXPERIMENT_TYPE_STATIC:
                rects3 = ax.bar(
                    x + width, list(map(lambda v: v["total_time"], data.values())), width, label='total_time'
                )
            else:
                rects3 = ax.bar(
                    x + width, list(map(lambda v: v["p99"], data.values())), width, label='p99'
                )
        ax.set_xticks(x)
        ax.set_xticklabels(data.keys())
        if experiment_type == EXPERIMENT_TYPE_ONE_BY_ONE:
            ax.set_ylim([0, 8])
        else:
            ax.set_ylim([0, 50])
        plt.legend()
        fig.tight_layout()
        plt.savefig(f"{plots_directory}/{x_axis}/{config}")
        plt.close()
    if experiment_type == EXPERIMENT_TYPE_DYNAMIC:
        for config, data in d.items():
            x_len = len(data.keys())
            x = np.arange(x_len).astype(int)
            width = 0.18
            fig, ax = plt.subplots()
            ax.set_title(config.replace("-", " | "), fontsize=9)
            ax.set_xlabel(f"{x_axis}")
            ax.bar(
                x,
                list(map(lambda v: v["percent_708ms"], data.values())),
                width,
                label='percent_708ms'
            )
            ax.set_xticks(x, data.keys())
            ax.set_ylim([0, 50])
            plt.legend()
            fig.tight_layout()
            plt.savefig(f"{plots_directory}/{x_axis}/percent_708ms-{config}")
            plt.close()


plot_for_param(
    f"{AUTO_TUNER_DIRECTORY}/../results/batch_result_Aug_26.csv",
    [ParamTypes.BATCH],
    EXPERIMENT_TYPE_ONE_BY_ONE
)
