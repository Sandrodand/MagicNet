path = "performance/fedcpnn"
path_dataset = [
    "datasets",
]

datasets = ["sine_federated", "weather_federated", "air_quality_federated"]
configurations = [list(range(1, 11)), list(range(1, 51)), list(range(1, 51))]

import pickle
import os
import pandas as pd
import numpy as np

rename_dict = {
    "ARF_TA_anytime": "A",
    "cLSTM_anytime": "cL",
    "cPNN_anytime": "cP",
    "FEDcPNN_anytime": "F",
}

for dataset, confs in zip(datasets, configurations):
    if "weather" in dataset:
        seq = 11
    else:
        seq = 10
    df_performance_total = pd.DataFrame()
    df_performance_averaged = pd.DataFrame()

    for conf in confs:
        print(conf)
        perf_nodes = []
        for node in range(1, 4):
            with open(
                f"{path}/{dataset}_{conf}conf_node{node}/performance_128_{seq}_it1.pkl",
                "rb",
            ) as f:
                perf_nodes.append(pickle.load(f))

        perf = {k: {} for k in perf_nodes[0] if "_anytime" in k}
        model1 = list(perf_nodes[0].keys())[0]

        drifts_len = []
        for n in range(len(perf_nodes)):
            local_drifts = perf_nodes[n][model1]["drifts"][0]
            local_drifts = (
                [0] + local_drifts + [len(perf_nodes[n][model1]["task"]["kappa"][0])]
            )
            drifts_len.append(
                [
                    local_drifts[i + 1] - local_drifts[i]
                    for i in range(len(local_drifts) - 1)
                ]
            )
        min_drifts_len = np.min(np.array(drifts_len), axis=0)
        drifts = [np.sum(min_drifts_len[0:i]) for i in range(1, len(min_drifts_len))]

        for k in perf:
            perf[k]["drifts"] = drifts

        for model in perf_nodes[0]:
            perf[model]["task"] = {}
            for metric in perf_nodes[0][model]["task"]:
                nodes_perf = []
                nodes_averaged = []
                for node in range(len(perf_nodes)):
                    local_drifts = [0] + perf_nodes[node][model]["drifts"][0] + [None]
                    local_perf = [
                        perf_nodes[node][model]["task"][metric][0][
                            local_drifts[i] : local_drifts[i + 1]
                        ]
                        for i in range(len(local_drifts) - 1)
                    ]
                    for i in range(len(local_perf)):
                        local_perf[i] = local_perf[i][: min_drifts_len[i]]

                    if metric == "kappa":
                        local_perf_tasks = local_perf.copy()[1:]
                        min_len = len(local_perf_tasks[0])
                        for i in range(1, len(local_perf_tasks)):
                            if len(local_perf_tasks[i]) < min_len:
                                min_len = len(local_perf_tasks[i])
                        for i in range(len(local_perf_tasks)):
                            local_perf_tasks[i] = local_perf_tasks[i][:min_len]
                        local_perf_tasks = np.array(local_perf_tasks)
                        local_perf_averaged = np.mean(local_perf_tasks, axis=0)
                        nodes_averaged.append(local_perf_averaged)

                    local_perf = [x for xs in local_perf for x in xs]
                    nodes_perf.append(local_perf)
                nodes_perf = np.array(nodes_perf)
                perf[model]["task"][metric] = [np.mean(nodes_perf, axis=0)]
                if metric == "kappa":
                    nodes_averaged = np.array(nodes_averaged)
                    nodes_averaged = np.mean(nodes_averaged, axis=0)
                    df_new = pd.DataFrame.from_dict(
                        {
                            "timestamp": np.arange(0, len(nodes_averaged)),
                            "model": [rename_dict[model]] * len(nodes_averaged),
                            "conf": [conf] * len(nodes_averaged),
                            "kappa": nodes_averaged,
                        }
                    )
                    df_performance_averaged = (
                        pd.concat([df_performance_averaged, df_new])
                        .reset_index()
                        .drop(columns="index")
                    )

            path_write = f"{path}/{dataset}_{conf}conf_federated"
            if not os.path.isdir(path_write):
                os.makedirs(path_write)

        with open(
            f"{path}/{dataset}_{conf}conf_federated/performance_128_{seq}.pkl", "wb"
        ) as f:
            pickle.dump(perf, f)

        n = len(perf[model1]["task"]["kappa"][0])
        for m in rename_dict:
            new_df = pd.DataFrame.from_dict(
                {
                    "timestamp": np.arange(0, n),
                    "model": [rename_dict[m]] * n,
                    "conf": [conf] * n,
                    "kappa": perf[m]["task"]["kappa"][0],
                }
            )
            df_performance_total = (
                pd.concat([df_performance_total, new_df])
                .reset_index()
                .drop(columns="index")
            )
