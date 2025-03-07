datasets = ["weather", "air_quality"]

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

path_read = "performance/fedcpnn"
path_write = "performance/fedcpnn"
cm = 1 / 2.54

for dataset in datasets:
    df = pd.read_csv(f"{path_read}/{dataset}_federated_averaged.csv")
    df["model_lower"] = df["model"].apply(lambda x: x.lower())
    df = df.sort_values("model_lower").drop(columns="model_lower")
    df["style"] = "solid"
    models = list(df["model"].unique())

    plt.style.use(["science", "ieee"])
    plt.rcParams.update({"font.size": 7})

    plot = sns.lineplot(
        data=df, x="timestamp", y="kappa", ci="sd", hue="model", style="style"
    )

    lines = plot.get_lines()
    colors = [lines[i].get_color() for i in range(len(lines))][: len(models)]

    plt.legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    lines = [
        Line2D([0], [0], label=m, color=c, linestyle="-")
        for c, m in zip(colors, models)
    ]
    plt.legend(handles=lines)
    plt.xlabel("N. of data points within the concept", loc="right")
    plt.ylabel("Cohen's Kappa", loc="top")

    n = len(df[(df["model"] == models[1]) & (df["conf"] == 1)])
    if "weather" in dataset:
        plt.ylim(0)
        y_text = 0.02
    else:
        y_text = -0.2
    plt.xlim(0, n)
    plt.axvline(x=50 * 128, color="grey", linestyle="dashed", linewidth=1)
    plt.text(
        50 * 128, y_text, f"start", rotation=90, va="bottom", ha="left", color="grey"
    )
    plt.text(n, y_text, f"end", rotation=90, va="bottom", ha="left", color="grey")

    fig = plt.gcf()
    fig.set_size_inches(6 * cm, 4.5 * cm)
    plt.title(dataset.replace("_", " ").title().replace(" ", ""))
    plt.minorticks_off()
    plt.tight_layout()
    plt.savefig(
        f"{path_write}/performance_averaged_{dataset}.png", transparent=True, dpi=1000
    )
