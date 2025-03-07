import pandas as pd
import os
import pickle
import numpy as np

path = "performance/fedcpnn"
datasets = ["sine_federated", "weather_federated", "air_quality_federated"]
confs = [list(range(1, 11)), list(range(1, 51)), list(range(1, 51))]
path_write = ["performance/fedcpnn"]

start = 50
batch_size = 128

start_dp = start * batch_size
df = pd.DataFrame()

for dataset, conf_range in zip(datasets, confs):
    seq = 11 if "weather" in dataset else 10
    for conf in conf_range:
        print(dataset, conf)
        row = {"dataset": [f"{dataset}_{conf}conf"]}
        perfs = []
        for node in range(1, 4):
            perfs.append({})
            with open(
                os.path.join(
                    path,
                    f"{dataset}_{conf}conf_node{node}",
                    f"performance_128_{seq}_it1.pkl",
                ),
                "rb",
            ) as f:
                perf = pickle.load(f)
            model1 = list(perf.keys())[0]
            drifts = [0] + perf[model1]["drifts"][0] + [0]
            for model in perf:
                model = model.replace("_anytime", "")
                if model == "F-cPNN":
                    model_wr = "cPNN-F"
                else:
                    model_wr = model
                for i in range(len(drifts) - 1):
                    s = drifts[i]
                    e = drifts[i + 1]
                    perfs[-1][f"T{i+1}_begin_{model_wr}"] = perf[f"{model}_anytime"][
                        "task"
                    ]["kappa"][0][s + start_dp]
                    perfs[-1][f"T{i+1}_end_{model_wr}"] = perf[f"{model}_anytime"][
                        "task"
                    ]["kappa"][0][e - 1]
                perfs[-1][f"begin_{model_wr}"] = np.mean(
                    [
                        perfs[-1][k]
                        for k in perfs[-1]
                        if "begin" in k
                        and k.endswith(model_wr)
                        and not (k.startswith("T1"))
                    ]
                )
                perfs[-1][f"end_{model_wr}"] = np.mean(
                    [
                        perfs[-1][k]
                        for k in perfs[-1]
                        if "end" in k
                        and k.endswith(model_wr)
                        and not (k.startswith("T1"))
                    ]
                )
        final_perf = {k: [np.mean([p[k] for p in perfs])] for k in perfs[0]}
        final_perf = dict(sorted(final_perf.items()))
        row.update(final_perf)
        row = pd.DataFrame.from_dict(row)
        df = pd.concat([df, row])
for p in path_write:
    df.to_csv(os.path.join(p, "performance.csv"), index=False)

df_excel = df.copy()
for c in df_excel.columns:
    if c != "dataset":
        df_excel[c] = df_excel[c].apply(lambda x: "{:.2f}".format(x).replace("0.", "."))

for p in path_write:
    df.to_csv(os.path.join(p, "performance.csv"), index=False)
    df_excel.to_excel(os.path.join(p, "performance.xlsx"), index=False)
