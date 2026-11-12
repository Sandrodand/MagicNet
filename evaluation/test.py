import argparse
import json
from detectors.detector_simulator import DetectorSimulator
from evaluation.cl_evaluation import EvaluateContinualLearning
from evaluation.prequential_evaluation import EvaluatePrequential, make_dir
import pandas as pd
import sys
import traceback
from evaluation.parameter_config import *
from evaluation.default_parameters import *

# Settings for python -m evaluation.test.
PARAMETERS = {
    "MODE": "local",
    "PATHS": [f"datasets/weather_{conf}conf" for conf in range(1, 11)],
    "PATH_PERFORMANCE": "debug_ensemble",
    "USE_DETECTOR": True,
    # Use None for both rates to run Sentinel.
    "DETECTOR_SIMULATOR_PRECISION": 0.7,
    "DETECTOR_SIMULATOR_RECALL": 1,
    "DETECTOR_SIMULATOR_MIN_DELAY": 0,
    "DETECTOR_SIMULATOR_MAX_DELAY": 1000,
    "LOAD_DETECTIONS": False,
    "PATH_LOAD_DETECTIONS": None,
    "SAVE_BASE_LEARNER": True,
    "PATH_BASE_LEARNER": "",
    "UPLOAD_BASE_LEARNER": True,
    "WRITE_CHECKPOINTS": True,
    "BASE_RNN": ["gru"],
    # None selects the dataset defaults for these dimensions and label delay.
    "SEQ_LEN_PARAM": None,
    "BATCH_SIZE_PARAM": None,
    "HIDDEN_SIZE_PARAM": {"gru": None, "lstm": None},
    "MAX_OLD_LABELS": 20,
    "DO_CL": True,
    "NUM_CLASSES": None,
    "PREPROCESS": None,
    "DELAY_PARAM": None,
    "ENSEMBLE_BATCHES": {"gru": 50, "lstm": 50},
    "SELECTION_BATCHES": {"gru": 40, "lstm": 40},
    "DYNCPNN_SELECTION_BATCHES": {"gru": 40, "lstm": 40},
    "HIDDEN_MULT": {"gru": None, "lstm": None},
    "ENSEMBLE_TH": {"gru": 1.05, "lstm": 1.05},
    "ITERATIONS": 1,
    "CALLBACK_FUNC": None,
    "OUTPUT_FILE": None,
    "suffix": "",
    "RNN_LEARNERS": ["MAGIC_Net"],
    "STREAM_LEARNERS": [],
    # The launcher supplies the shared name for the result and log directories.
    "JOB_NAME": None,
}


def main(params=None):
    """Run prequential and continual-learning evaluation with the given parameters."""
    if params is None:
        params = PARAMETERS
    CFG = Config(base_rnn=params["BASE_RNN"])
    learners = CFG.create_learners(params["RNN_LEARNERS"], params["STREAM_LEARNERS"])
    paths = params["PATHS"]
    mode = params["MODE"]
    use_detector = params["USE_DETECTOR"]
    precision = params["DETECTOR_SIMULATOR_PRECISION"]
    recall = params["DETECTOR_SIMULATOR_RECALL"]
    do_cl = params["DO_CL"]
    preprocessor = params["PREPROCESS"]
    label_delay = params["DELAY_PARAM"]
    max_delay = params["DETECTOR_SIMULATOR_MAX_DELAY"]
    seq_len_override = params["SEQ_LEN_PARAM"]
    batch_size_override = params["BATCH_SIZE_PARAM"]
    num_classes = params["NUM_CLASSES"]
    hidden_sizes = params["HIDDEN_SIZE_PARAM"]
    hidden_additions = params["HIDDEN_MULT"]
    run_suffix = params["suffix"]
    job_name = params["JOB_NAME"]
    for name, values in (
        ("HIDDEN_SIZE_PARAM", hidden_sizes),
        ("HIDDEN_MULT", hidden_additions),
        ("ENSEMBLE_BATCHES", params["ENSEMBLE_BATCHES"]),
        ("SELECTION_BATCHES", params["SELECTION_BATCHES"]),
        ("DYNCPNN_SELECTION_BATCHES", params["DYNCPNN_SELECTION_BATCHES"]),
        ("ENSEMBLE_TH", params["ENSEMBLE_TH"]),
    ):
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a dictionary keyed by RNN")
        missing = set(CFG.base_rnn) - values.keys()
        if missing:
            raise ValueError(f"Missing {name} for: {', '.join(sorted(missing))}")
    path_performance = params["PATH_PERFORMANCE"]
    path_base_learner = params["PATH_BASE_LEARNER"]
    path_load_detections = params["PATH_LOAD_DETECTIONS"]
    output_file = params["OUTPUT_FILE"]
    callback_func = params["CALLBACK_FUNC"]

    detector_type = None
    precision_recall_suffix = ""
    if use_detector:
        if precision is not None and recall is not None:
            detector_type = "simulator"
            precision_recall_suffix = f"_{int(precision * 100)}prec_{int(recall * 100)}rec"
        else:
            detector_type = "sentinel"
    if output_file is None:
        output_file = os.path.basename(paths[0]) + precision_recall_suffix
    eval_cl = None
    if not path_performance.startswith("/"):
        path_performance = os.path.join("performance", path_performance)
    if not path_base_learner.startswith("/"):
        path_base_learner = os.path.join("base_learner", path_base_learner)
    make_dir(path_base_learner)

    orig_stdout = sys.stdout
    log_file = None
    if mode == "aws" and job_name is None:
        for path in paths:
            print(os.path.basename(path))
        make_dir("outputs")
        log_file = open(f"outputs/{output_file}.txt", "w", buffering=1)
        sys.stdout = log_file

    try:
        for path in paths:
            result_name = job_name or os.path.basename(path) + precision_recall_suffix
            current_path_performance = os.path.join(path_performance, result_name)
            make_dir(current_path_performance)

            if do_cl:
                path_cl = path + "_test"
                path = path + "_train"
            else:
                path_cl = None
            df = pd.read_csv(f"{path}.csv", nrows=1)
            dataset = path.split("/")[-1].lower()

            columns = list(df.columns)
            initial_task = df.iloc[0]["task"]
            columns.remove("target")
            columns.remove("task")
            if "batch" in columns:
                columns.remove("batch")
            converters = {c: float for c in columns}
            converters["target"] = int
            converters["task"] = int
            num_features = len(columns)
            if seq_len_override is None:
                seq_len = set_seq_len(dataset)
            else:
                seq_len = seq_len_override
            hidden_size = {
                rnn: set_hidden_size(dataset, rnn)
                if hidden_sizes[rnn] is None else hidden_sizes[rnn]
                for rnn in CFG.base_rnn
            }
            if batch_size_override is None:
                batch_size = set_batch_size(dataset)
            else:
                batch_size = batch_size_override
            if num_classes is None:
                output_size = set_output_size(dataset)
            else:
                output_size = num_classes
            if label_delay is None:
                delay = set_delay(dataset)
            else:
                delay = label_delay
            hidden_mult = {
                rnn: int(0.5 * hidden_size[rnn])
                if hidden_additions[rnn] is None else hidden_additions[rnn]
                for rnn in CFG.base_rnn
            }
            if detector_type == "sentinel":
                drift_delay = set_drift_delay(dataset)
                save_column_freq = set_save_column_freq(dataset)
            elif detector_type == "simulator":
                drift_delay = int(
                    max_delay + 0.15 * max_delay
                )
                save_column_freq = int(drift_delay / 2)
            else:
                drift_delay = None
                save_column_freq = None
            old_labels_ta = min(seq_len - 1, params["MAX_OLD_LABELS"])
            delta = set_adwin_delta(dataset)

            CFG.set_params(
                ta_order=old_labels_ta,
                seq_len=seq_len,
                num_features=num_features,
                batch_size=batch_size,
                iterations=params["ITERATIONS"],
                path=path,
                converters=converters,
                delta=delta,
                output_size=output_size,
                hidden_size=hidden_size,
                hidden_mult=hidden_mult,
                ensemble_batches=params["ENSEMBLE_BATCHES"],
                selection_batches=params["SELECTION_BATCHES"],
                dyncpnn_selection_batches=params["DYNCPNN_SELECTION_BATCHES"],
                ensemble_th=params["ENSEMBLE_TH"],
                save_column_freq=save_column_freq,
                drift_delay=drift_delay,
            )
            print(path)
            CFG.prepare_base_learners(
                path_base_learner, path.split("/")[-1],
                load=params["UPLOAD_BASE_LEARNER"], save=params["SAVE_BASE_LEARNER"],
            )
            data_stream = CFG.create_iter_csv

            print("BATCH SIZE, SEQ LEN:", batch_size, seq_len)
            print("HIDDEN SIZE PER RNN:", hidden_size)
            print("NUM OLD LABELS:", old_labels_ta)
            print("ANYTIME LEARNERS:", [m.name for m in learners])
            print("SUFFIX:", run_suffix)
            print("DETECTOR:", use_detector)
            print("DELAY:", delay)
            if detector_type == "simulator":
                print("DETECTOR PRECISION:", precision)
                print("DETECTOR RECALL:", recall)
            print()

            if params["LOAD_DETECTIONS"] and (use_detector or job_name is not None):
                detection_root = path_load_detections or path_performance
                detection_name = job_name or dataset.replace("_train", "") + precision_recall_suffix
                filename = "drifts.pkl"
                if job_name is not None and os.path.abspath(detection_root) == os.path.abspath(path_performance):
                    filename = "prepared_drifts.pkl"
                with open(os.path.join(detection_root, detection_name, filename), "rb") as file:
                    detections = pickle.load(file)[0]
                drift_detector = DetectorSimulator(
                    detections=detections, after_prediction=detector_type == "sentinel"
                )
                print("Detections loaded:", detections)
            elif detector_type == "sentinel":
                drift_detector = CFG.create_drift_detector()
            elif detector_type == "simulator":
                drift_detector = DetectorSimulator(
                    recall=recall,
                    precision=precision,
                    dataset=path,
                    min_delay=params["DETECTOR_SIMULATOR_MIN_DELAY"],
                    max_delay=max_delay,
                )
            else:
                drift_detector = None

            eval_preq = EvaluatePrequential(
                batch_size=batch_size,
                metrics=["accuracy", "kappa"],
                learners=learners,
                data_stream=data_stream,
                path_write=current_path_performance,
                suffix=run_suffix,
                write_checkpoints=params["WRITE_CHECKPOINTS"],
                iterations=params["ITERATIONS"],
                dataset_name=dataset,
                mode=mode,
                drift_detector=drift_detector,
                preprocessing_func=None if preprocessor is None else preprocessor.preprocess,
                delay=delay,
                iteration_str=precision_recall_suffix + run_suffix,
            )

            if path_cl is not None and do_cl:
                eval_cl = EvaluateContinualLearning(
                    path=path_cl,
                    checkpoint=eval_preq.checkpoint,
                    learners_config=learners,
                    path_write=current_path_performance,
                    suffix=run_suffix,
                    print_suffix=precision_recall_suffix + run_suffix,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    mode=mode,
                    delay=delay,
                )
                callback_func = eval_cl.evaluate

            CFG.initialize_callback(eval_cl, eval_preq)

            eval_preq.evaluate(callback=callback_func, initial_task=initial_task)

            if preprocessor is not None:
                preprocessor.reset()
            print()
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        if log_file is not None:
            sys.stdout = orig_stdout
            log_file.close()
    print("\n\nEND.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", help="Read evaluation parameters from this JSON file.")
    args = parser.parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as file:
            params = json.load(file)
        main(params)
    else:
        main()
