import pandas as pd

from detectors.ewma import EWMA
from evaluation.prequential_evaluation import make_dir
from evaluation.default_parameters import *
from evaluation.parameter_config import *

PATHS = [
    (
        f"/Users/federicogiannini/Library/CloudStorage/OneDrive-PolitecnicodiMilano/NN4SML/datasets/dynamic/"
        f"sine_rw10_mode5_{c}conf_train"
    )
    for c in range(6,11)
]  # a list containing the paths of the data streams (without the extension)
PATH_PERFORMANCE = "detector_no_adwin"


def build_detector_config(delta_values):
    configs = []
    for delta in delta_values:
        configs += [
            {
                "name": f"continuous_arf_{delta}",
                "params": {
                    "detector": ADWIN(delta=delta, clock=1),
                    "create_model_func": CFG.create_arf_no_adwin
                },
            },
            {
                "name": f"fixed_arf_{delta}",
                "params": {
                    "detector": ADWIN(delta=delta, clock=1),
                    "training_data_points": 50*128,
                    "create_model_func": CFG.create_arf_no_adwin
                },
            },
            # {
            #     "name": f"fixed_arf_ta_{delta}",
            #     "params": {
            #         "detector": ADWIN(delta=delta, clock=1),
            #         "training_data_points": 50*128,
            #         "create_model_func": CFG.create_arf_ta_no_adwin
            #     },
            # },
            {
                "name": f"fixed_arf_ta_feat_{delta}",
                "params": {
                    "detector": ADWIN(delta=delta, clock=1),
                    "training_data_points": 50 * 128,
                    "create_model_func": CFG.create_arf_ta_features_no_adwin
                },
            },
            {
                "name": f"fixed_clstm_{delta}",
                "params": {
                    "detector": ADWIN(delta=delta, clock=1),
                    "training_data_points": 50 * 128,
                    "create_model_func": CFG.create_acpnn_clstm,
                    "numeric_model": True
                },
            },
        ]
    return configs


if not PATH_PERFORMANCE.startswith("/"):
    PATH_PERFORMANCE = os.path.join("performance", PATH_PERFORMANCE)
CFG = Config()

for path in PATHS:
    dataset = path.split("/")[-1]
    deltas = set_deltas_test_detector(dataset)
    detectors_config = build_detector_config(deltas)
    df = pd.read_csv(f"{path}.csv", nrows=1)
    columns = list(df.columns)
    columns.remove("target")
    columns.remove("task")
    converters = {c: float for c in columns}
    converters["target"] = int
    converters["task"] = int
    current_path_performance = os.path.join(PATH_PERFORMANCE, path.split("/")[-1])
    make_dir(current_path_performance)
    seq_len = set_seq_len(dataset)
    batch_size = set_batch_size(dataset)
    output_size = set_output_size(dataset)
    old_labels_ta = min(seq_len - 1, 20)
    num_features = len(columns)
    CFG.set_params(
        ta_order=old_labels_ta,
        seq_len=seq_len,
        num_features=num_features,
        batch_size=batch_size,
        output_size=output_size
    )
    data = stream.iter_csv(str(path) + ".csv", converters=converters, target="target")

    print("DETECTORS:")
    for d in detectors_config:
        print(d["name"])
    detectors = {d["name"]: Detector(**d["params"]) for d in detectors_config}
    drifts = {d["name"]: [] for d in detectors_config}
    for idx, (x, y) in enumerate(data):
        if "task" in x:
            del x["task"]
        for d in detectors_config:
            if detectors[d["name"]].update(x, y):
                drifts[d["name"]].append(idx)
                print(f"\n{idx+1} detection {d['name']}")
        if (idx + 1) % 100 == 0:
            print(
                path.split("/")[-1],
                idx + 1,
                end="\r",
            )
        if (idx + 1) % 5000 == 0:
            with open(
                os.path.join(current_path_performance, "detections.pkl"), "wb"
            ) as f:
                pickle.dump(drifts, f)
    with open(os.path.join(current_path_performance, "detections.pkl"), "wb") as f:
        pickle.dump(drifts, f)
