from detectors.detector_simulator import DetectorSimulator
from evaluation.cl_evaluation import EvaluateContinualLearning
from evaluation.prequential_evaluation import EvaluatePrequential, make_dir
import pandas as pd
import sys
import traceback
from evaluation.parameter_config import *
from evaluation.default_parameters import *

# __________________
# PARAMETERS
# __________________
MODE = "local"
# 'local' or 'aws'. If 'aws', the messages will be written in a specific txt file in the output_file dir
#TODO
PATHS = [
    f"datasets/{d}_{c}conf"
    for c in range(1, 2)
    for d in ["weather"]
]  # a list containing the paths of the data streams (without the extension)
PATH_PERFORMANCE = ""
# the path in which to save the results. In the case of a relative path, the performance folder is automatically
# created
#TODO
USE_DETECTOR = False
# True if you want to use a detector, False if you want to use the supervised drift information (the "task" column in
# csv file)
DETECTOR_SIMULATOR_PRECISION = 0.75
DETECTOR_SIMULATOR_RECALL = 0.75
# Set both to None if you want to use the automatic drift detector with ADWIN (Sentinel)
# Set both to a specific value if you want to simulate a drift detector with a specific recall and precision
# This configuration works only when setting USE_DETECTOR = True
DETECTOR_SIMULATOR_MIN_DELAY = 0
DETECTOR_SIMULATOR_MAX_DELAY = 1000
# In the case of detector's simulation they represent the minimum and maximum number of data points following
# a real drift after which to generate true positives.
LOAD_DETECTIONS = False
# Set to true if you want to load the detections of a previously run detector (if available)
PATH_LOAD_DETECTIONS = None
# the path from which to load the detections if LOAD_DETECTIONS is set to True.
# Set it to None to use PATH_PERFORMANCE

SAVE_BASE_LEARNER = True
# If True it saves in the base_learner folder the initialized base learner. This is useful when you want to simulate
# a new detector's configuration with the same model's initialization.
PATH_BASE_LEARNER = ""
# the path in which to save the base learner. In the case of a relative path, the base_learner folder is automatically
# created
UPLOAD_BASE_LEARNER = True
# If True it uploads the base learner from the base_learner folder instead of initializing it random
# This configuration can be used when you want to use an already initialized base learner. For instance, when you want
# to compare performance with different detector's configurations and use the same base learner initialization.
WRITE_CHECKPOINTS = False
# True if you want to write the pickle files of the models after each supervised concept's end.

BASE_RNN = ["gru", "lstm"]
# Architectures evaluated together on the same stream and detections.
SEQ_LEN_PARAM = None
# The size of the sliding window. If None the default value for the dataset is set.
BATCH_SIZE_PARAM = None
# the batch size of periodic learners and classifiers. If None the default value for the dataset is set.
HIDDEN_SIZE_PARAM = {"gru": None, "lstm": None}
# the hidden size of the cRNN (cLSTM or cGRU). Set None to use the default value for the dataset.
MAX_OLD_LABELS = 20
# the number of the maximum temporal order of SML temporal augmentation.
DO_CL = True
# True if you want to perform the CL evaluation. There must be a training dataset for prequential evaluation
# that ends with '_train' and a test dataset for CL evaluation that ends with '_test'.
NUM_CLASSES = None
# number of the classes of the classification problem. If None the default value for the dataset is set.
PREPROCESS = None
# the preprocessing function to apply to the data. Use None to not apply anything.
DELAY_PARAM = None
# the labeling delay. Set None to use the default values (0 for present prediction, temporal order for future
# prediction)

###---MAGIC Net---
ENSEMBLE_BATCHES = {"gru": 30, "lstm": 30}
HIDDEN_MULT = {"gru": None, "lstm": None}
ENSEMBLE_TH = {"gru": 1.05, "lstm": 1.05}

ITERATIONS = 1
# number of experiments for each dataset
CALLBACK_FUNC = None
# function to call after each iteration (set it to None). It will use the CL evaluation if DO_CL = True.
OUTPUT_FILE = None
# the name of the output file in outputs dir. If None, it will use the name of the current data stream.
suffix = ""
# Optional common output suffix. RNN types are distinguished by learner names.

# LEARNERS
RNN_LEARNERS = ["cRNN", "DYNcPNN", "MAGIC_Net", "MAGIC_Net_sh", "MAGIC_Net_binary_sh", "MAGIC_Net_binary_mh", "MultiHeadCRNN"]
# Available names: cRNN, cPNN, DYNcPNN, MAGIC_Net, MAGIC_Net_sh, MAGIC_Net_binary_sh, MAGIC_Net_binary_mh, MultiHeadCRNN
# MAGIC_Net_binary_sh, MAGIC_Net_binary_mh.
STREAM_LEARNERS = []
# Optional architecture-independent methods, created once: ARF, ARF_TA, HAT, HAT_TA.

# __________________
# CODE
# __________________
def main():
    CFG = Config(base_rnn=BASE_RNN)
    learners = CFG.create_learners(RNN_LEARNERS, STREAM_LEARNERS)
    for name, values in (
        ("HIDDEN_SIZE_PARAM", HIDDEN_SIZE_PARAM),
        ("HIDDEN_MULT", HIDDEN_MULT),
        ("ENSEMBLE_BATCHES", ENSEMBLE_BATCHES),
        ("ENSEMBLE_TH", ENSEMBLE_TH),
    ):
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a dictionary keyed by RNN")
        missing = set(CFG.base_rnn) - values.keys()
        if missing:
            raise ValueError(f"Missing {name} for: {', '.join(sorted(missing))}")
    path_performance = PATH_PERFORMANCE
    path_base_learner = PATH_BASE_LEARNER
    path_load_detections = PATH_LOAD_DETECTIONS
    output_file = OUTPUT_FILE
    callback_func = CALLBACK_FUNC

    PREC_REC_SFX = ""
    if (
        USE_DETECTOR
        and DETECTOR_SIMULATOR_PRECISION is not None
        and DETECTOR_SIMULATOR_RECALL is not None
    ):
        PREC_REC_SFX = (
            f"_{int(DETECTOR_SIMULATOR_PRECISION * 100)}prec_"
            f"{int(DETECTOR_SIMULATOR_RECALL * 100)}rec"
        )

    METRICS = ["accuracy", "kappa"]
    if output_file is None:
        output_file = PATHS[0].split("/")[-1]
        if (
            USE_DETECTOR
            and DETECTOR_SIMULATOR_PRECISION is not None
            and DETECTOR_SIMULATOR_RECALL is not None
        ):
            output_file = (
                f"{output_file}_"
                f"{int(DETECTOR_SIMULATOR_PRECISION*100)}prec_"
                f"{int(DETECTOR_SIMULATOR_RECALL*100)}rec"
            )
    eval_cl = None
    if not path_performance.startswith("/"):
        path_performance = os.path.join("performance", path_performance)
    if not path_base_learner.startswith("/"):
        path_base_learner = os.path.join("base_learner", path_base_learner)
    make_dir(path_base_learner)

    if MODE == "aws":
        for p in PATHS:
            print(p.split("/")[-1])
    orig_stdout = sys.stdout
    log_file = None
    if MODE == "aws":
        make_dir(f"outputs")
        log_file = open(f"outputs/{output_file}.txt", "w", buffering=1)
        sys.stdout = log_file

    try:
        for path in PATHS:
            current_path_performance = os.path.join(path_performance, path.split("/")[-1])
            DETECTOR_TYPE = None
            if USE_DETECTOR:
                if (
                    DETECTOR_SIMULATOR_PRECISION is not None
                    and DETECTOR_SIMULATOR_RECALL is not None
                ):
                    current_path_performance = (
                        f"{current_path_performance}"
                        f"_{int(DETECTOR_SIMULATOR_PRECISION*100)}prec"
                        f"_{int(DETECTOR_SIMULATOR_RECALL*100)}rec"
                    )
                    DETECTOR_TYPE = "simulator"
                else:
                    DETECTOR_TYPE = "sentinel"
            make_dir(current_path_performance)

            if DO_CL:
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
            if SEQ_LEN_PARAM is None:
                seq_len = set_seq_len(dataset)
            else:
                seq_len = SEQ_LEN_PARAM
            hidden_size = {
                rnn: set_hidden_size(dataset, rnn)
                if HIDDEN_SIZE_PARAM[rnn] is None else HIDDEN_SIZE_PARAM[rnn]
                for rnn in CFG.base_rnn
            }
            if BATCH_SIZE_PARAM is None:
                batch_size = set_batch_size(dataset)
            else:
                batch_size = BATCH_SIZE_PARAM
            if NUM_CLASSES is None:
                output_size = set_output_size(dataset)
            else:
                output_size = NUM_CLASSES
            if DELAY_PARAM is None:
                delay = set_delay(dataset)
            else:
                delay = DELAY_PARAM
            hidden_mult = {
                rnn: int(0.5 * hidden_size[rnn])
                if HIDDEN_MULT[rnn] is None else HIDDEN_MULT[rnn]
                for rnn in CFG.base_rnn
            }
            if DETECTOR_TYPE == "sentinel":
                drift_delay = set_drift_delay(dataset)
                save_column_freq = set_save_column_freq(dataset)
            elif DETECTOR_TYPE == "simulator":
                drift_delay = int(
                    DETECTOR_SIMULATOR_MAX_DELAY + 0.15 * DETECTOR_SIMULATOR_MAX_DELAY
                )
                save_column_freq = int(drift_delay / 2)
            else:
                drift_delay = None
                save_column_freq = None
            old_labels_ta = min(seq_len - 1, MAX_OLD_LABELS)
            delta = set_adwin_delta(dataset)

            CFG.set_params(
                ta_order=old_labels_ta,
                seq_len=seq_len,
                num_features=num_features,
                batch_size=batch_size,
                iterations=ITERATIONS,
                path=path,
                converters=converters,
                delta=delta,
                output_size=output_size,
                hidden_size=hidden_size,
                hidden_mult=hidden_mult,
                ensemble_batches=ENSEMBLE_BATCHES,
                ensemble_th=ENSEMBLE_TH,
                save_column_freq=save_column_freq,
                drift_delay=drift_delay,
            )
            print(path)
            CFG.prepare_base_learners(
                path_base_learner, path.split("/")[-1],
                load=UPLOAD_BASE_LEARNER, save=SAVE_BASE_LEARNER,
            )
            data_stream = CFG.create_iter_csv

            print("BATCH SIZE, SEQ LEN:", batch_size, seq_len)
            print("HIDDEN SIZE PER RNN:", hidden_size)
            print("NUM OLD LABELS:", old_labels_ta)
            print("ANYTIME LEARNERS:", [m.name for m in learners])
            print("SUFFIX:", suffix)
            print("DETECTOR:", USE_DETECTOR)
            print("DELAY:", delay)
            if (
                USE_DETECTOR
                and DETECTOR_SIMULATOR_PRECISION is not None
                and DETECTOR_SIMULATOR_RECALL is not None
            ):
                print("DETECTOR PRECISION:", DETECTOR_SIMULATOR_PRECISION)
                print("DETECTOR RECALL:", DETECTOR_SIMULATOR_RECALL)
            print()

            if USE_DETECTOR:
                if LOAD_DETECTIONS:
                    if path_load_detections is None or path_load_detections == "":
                        path_load_detections = path_performance
                    with open(
                        os.path.join(
                            path_load_detections,
                            f"{dataset.replace('_train','')}{PREC_REC_SFX}",
                            "drifts.pkl",
                        ),
                        "rb",
                    ) as f:
                        detections = pickle.load(f)[0]
                    drift_detector = DetectorSimulator(detections=detections)
                    print("Detections loaded:", detections)
                elif (
                    DETECTOR_SIMULATOR_PRECISION is None
                    or DETECTOR_SIMULATOR_RECALL is None
                ):
                    drift_detector = CFG.create_drift_detector()
                else:
                    drift_detector = DetectorSimulator(
                        recall=DETECTOR_SIMULATOR_RECALL,
                        precision=DETECTOR_SIMULATOR_PRECISION,
                        dataset=path,
                        min_delay=DETECTOR_SIMULATOR_MIN_DELAY,
                        max_delay=DETECTOR_SIMULATOR_MAX_DELAY,
                    )
            else:
                drift_detector = None

            eval_preq = EvaluatePrequential(
                batch_size=batch_size,
                metrics=METRICS,
                learners=learners,
                data_stream=data_stream,
                path_write=current_path_performance,
                suffix=suffix,
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=ITERATIONS,
                dataset_name=dataset,
                mode=MODE,
                drift_detector=drift_detector,
                preprocessing_func=None if PREPROCESS is None else PREPROCESS.preprocess,
                delay=delay,
                iteration_str=PREC_REC_SFX + suffix,
            )

            if path_cl is not None and DO_CL:
                eval_cl = EvaluateContinualLearning(
                    path=path_cl,
                    checkpoint=eval_preq.checkpoint,
                    learners_config=learners,
                    path_write=current_path_performance,
                    suffix=suffix,
                    print_suffix=PREC_REC_SFX + suffix,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    mode=MODE,
                    delay=delay,
                )
                callback_func = eval_cl.evaluate

            CFG.initialize_callback(eval_cl, eval_preq)

            eval_preq.evaluate(callback=callback_func, initial_task=initial_task)

            if PREPROCESS is not None:
                PREPROCESS.reset()
            print()
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        if MODE == "aws":
            sys.stdout = orig_stdout
            log_file.close()
    print("\n\nEND.")


if __name__ == "__main__":
    main()
