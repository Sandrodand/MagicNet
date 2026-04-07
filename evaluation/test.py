from detectors.detector_simulator import DetectorSimulator
from evaluation.cl_evaluation import EvaluateContinualLearning
from evaluation.learner_config import LearnerConfig
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
PATHS = [
    (
        f"datasets/weather_{c}conf"
    )
    for c in range(1,51)
]  # a list containing the paths of the data streams (without the extension)
PATH_PERFORMANCE = ""
# the path in which to save the results. In the case of a relative path, the performance folder is automatically
# created
PATH_BASE_LEARNER = ""
# the path in which to save the base learner. In the case of a relative path, the base_learner folder is automatically
# created
PATH_DETECTIONS = ""
# the path in which to load the detections if LOAD_DETECTIONS is set to True.
USE_DETECTOR = False
# True if you want to use a detector, False if you want to use the supervised drift information (the "task" column in
# csv file)
DETECTOR_SIMULATOR_PRECISION = None
DETECTOR_SIMULATOR_RECALL = None
# Set both to None if you want to use the automatic drift detector with ADWIN
# Set both to a specific value if you want to simulate a drift detector with a specific recall and precision
# This configuration works only when setting USE_DETECTOR = True
DETECTOR_SIMULATOR_MIN_DELAY = 0
DETECTOR_SIMULATOR_MAX_DELAY = 1000
# In the case of detector's simulation they represent the minimum and maximum number of data points following
# a real drift after which to generate true positives.
LOAD_DETECTIONS = False
# Set to true if you want to load the detections of a previously run detector.
SAVE_BASE_LEARNER = False
# If True it saves in the base_learner folder the initialized base learner
UPLOAD_BASE_LEARNER = False
# If True it uploads the base learner from the base_learner folder instead of initializing it random
# This configuration can be used when you want to use an already initialized base learner. For instance, when you want
# to compare performance with different detector's configurations and use the same base learner initialization.
WRITE_CHECKPOINTS = False
# True if you want to write the pickle files of the models after each supervised concept's end.
BASE_RNN = "gru"
# the base learner. it can be gru or lstm
SEQ_LEN_PARAM = None
# The size of the sliding window. If None the default value for the dataset is set.
BATCH_SIZE_PARAM = None
# the batch size of periodic learners and classifiers. If None the default value for the dataset is set.
HIDDEN_SIZE_PARAM = None
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
MASK_INIT = "1s"
ENSEMBLE_BATCHES = 30
THRESHOLD_FN = "sigmoid"
HIDDEN_MULT = None
ENSEMBLE_TH = 1.05

ITERATIONS = 1
# number of experiments for each dataset
CALLBACK_FUNC = None
# function to call after each iteration (set it to None). It will use the CL evaluation if DO_CL = True.
OUTPUT_FILE = None
# the name of the output file in outputs dir. If None, it will use the name of the current data stream.
suffix = f"_{HIDDEN_SIZE_PARAM}hs" if HIDDEN_SIZE_PARAM is not None else ""
# the suffix to add the files containing the evaluation results.

# LEARNERS
CFG = Config(base_learner=BASE_RNN)
learners = [
    LearnerConfig(
        name="ARF",
        model=CFG.create_arf,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
        temp_dep=False,
    ),
    LearnerConfig(
        name="ARF_TA",
        model=CFG.create_arf_ta,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
        temp_dep=True,
    ),
    LearnerConfig(
        name="cGRU",
        model=CFG.base_learner.get_cpnn,
        numeric=True,
        batch_learner=False,
        drift=False,
        cpnn=True,
    ),
    LearnerConfig(
        name="cPNN",
        model=CFG.base_learner.get_cpnn,
        numeric=True,
        batch_learner=False,
        drift=True,
        cpnn=True,
    ),
    LearnerConfig(
        name="MAGIC_Net",
        model=CFG.create_magic_net,
        numeric=True,
        batch_learner=False,
        drift=True,
        cpnn=False,
        magic=True,
    ),
]

# __________________
# CODE
# __________________
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
if OUTPUT_FILE is None:
    OUTPUT_FILE = PATHS[0].split("/")[-1]
    if (
        USE_DETECTOR
        and DETECTOR_SIMULATOR_PRECISION is not None
        and DETECTOR_SIMULATOR_RECALL is not None
    ):
        OUTPUT_FILE = (
            f"{OUTPUT_FILE}_"
            f"{int(DETECTOR_SIMULATOR_PRECISION*100)}prec_"
            f"{int(DETECTOR_SIMULATOR_RECALL*100)}rec"
        )
eval_cl = None
if not PATH_PERFORMANCE.startswith("/"):
    PATH_PERFORMANCE = os.path.join("performance", PATH_PERFORMANCE)
if not PATH_BASE_LEARNER.startswith("/"):
    PATH_BASE_LEARNER = os.path.join("base_learner", PATH_BASE_LEARNER)
make_dir(PATH_BASE_LEARNER)

if MODE == "aws":
    for p in PATHS:
        print(p.split("/")[-1])
orig_stdout = sys.stdout
f = None
if MODE == "aws":
    make_dir(f"outputs")
    f = open(f"outputs/{OUTPUT_FILE}.txt", "w", buffering=1)
    sys.stdout = f

try:
    for path in PATHS:
        current_path_performance = os.path.join(PATH_PERFORMANCE, path.split("/")[-1])
        if (
            USE_DETECTOR
            and DETECTOR_SIMULATOR_PRECISION is not None
            and DETECTOR_SIMULATOR_RECALL is not None
        ):
            current_path_performance = (
                f"{current_path_performance}"
                f"_{int(DETECTOR_SIMULATOR_PRECISION*100)}prec"
                f"_{int(DETECTOR_SIMULATOR_RECALL*100)}rec"
            )
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
        if HIDDEN_SIZE_PARAM is None:
            hidden_size = set_hidden_size(dataset, BASE_RNN)
        else:
            hidden_size = HIDDEN_SIZE_PARAM
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
        if HIDDEN_MULT is None:
            hidden_mult = int(0.5 * hidden_size)
        else:
            hidden_mult = HIDDEN_MULT
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
            threshold_fn=THRESHOLD_FN,
            mask_init=MASK_INIT,
            hidden_mult=hidden_mult,
            ensemble_batches=ENSEMBLE_BATCHES,
            ensemble_th=ENSEMBLE_TH,
        )
        print(path)
        write = True
        if UPLOAD_BASE_LEARNER:
            try:
                with open(
                    os.path.join(PATH_BASE_LEARNER, f'{path.split("/")[-1]}.pkl'), "rb"
                ) as f:
                    base_learner = pickle.load(f)
                CFG.base_learner.set_base_learner(base_learner)
                write = False
                print("Base learner loaded.")
            except:
                print("Base learner not found.")
        if write:
            CFG.base_learner.reset_base_learner()
            print("Base learner initialized.")
            if SAVE_BASE_LEARNER:
                with open(
                    os.path.join(PATH_BASE_LEARNER, f'{path.split("/")[-1]}.pkl'), "wb"
                ) as f:
                    pickle.dump(CFG.base_learner.get_base_learner(), f)
                print("Base learner stored.")
        data_stream = CFG.create_iter_csv

        print("BATCH SIZE, SEQ LEN, HIDDEN SIZE:", batch_size, seq_len, hidden_size)
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
                with open(os.path.join(PATH_DETECTIONS, f"{dataset.replace('_train','')}{PREC_REC_SFX}", "drifts.pkl"), "rb") as f:
                    detections = pickle.load(f)[0]
                drift_detector = DetectorSimulator(
                    detections=detections
                )
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
            anytime_learners=learners,
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
            CALLBACK_FUNC = eval_cl.evaluate

        CFG.initialize_callback(eval_cl, eval_preq)

        eval_preq.evaluate(callback=CALLBACK_FUNC, initial_task=initial_task)

        if PREPROCESS is not None:
            PREPROCESS.reset()
        print()
except Exception:
    print(traceback.format_exc())
    if MODE == "aws":
        sys.stdout = orig_stdout
        f.close()
        print(traceback.format_exc())
print("\n\nEND.")
if MODE == "aws":
    sys.stdout = orig_stdout
    f.close()
