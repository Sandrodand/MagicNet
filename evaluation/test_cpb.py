from river import stream
from evaluation.prequential_evaluation import EvaluatePrequential, make_dir
import pandas as pd
import sys
import traceback
from evaluation.test_utils import *

# __________________
# PARAMETERS
# __________________
PATHS = [
    "datasets/sine_rw10_mode5_extended_16-16_1234",
]  # a list containing the paths of the data streams (without the extension)
ITERATIONS = 10  # number of experiments
PATH_PERFORMANCE = "cpb"  # path to write the outputs of the evaluation
MODE = "local"  # 'local' or 'aws'. If 'aws', the messages will be written in a specific txt file in the output_file dir
OUTPUT_FILE = None
# the name of the output file in outputs dir. If None, it will use the name of the current data stream.
suffix = f""  # the suffix to add the files containing the evaluation results.
BATCH_SIZE = 128  # the batch size of periodic learners and classifiers.

# __________________
# CODE
# __________________


class ModelLoader:
    def __init__(self):
        self.iteration = 1
        self.datastream = None

    def return_base_learner_pretrained(self):
        with open(
            f"saved_models/{self.datastream}/pretrained_cgru_iteration{self.iteration}.pkl",
            "rb",
        ) as f:
            model = pickle.load(f)
        return model

    def return_cpb(self):
        with open(
            f"saved_models/{self.datastream}/pretrained_cgru_iteration{self.iteration}.pkl",
            "rb",
        ) as f:
            model = pickle.load(f)
        return cPB(
            model
        )  # create a cPB model with the specific pretrained base learner

    def return_base_learner(self):
        with open(
            f"saved_models/{self.datastream}/cgru_iteration{self.iteration}.pkl", "rb"
        ) as f:
            model = pickle.load(f)
        return model

    def next_iteration(self):
        self.iteration += 1

    def set_path(self, datastream):
        self.datastream = datastream


MODEL_LOADER = ModelLoader()

anytime_learners = [
    LearnerConfig(
        name="ARF_TA",
        model=create_arf_ta,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    LearnerConfig(
        name="ARF",
        model=create_arf,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
]
batch_learners = [
    LearnerConfig(
        name="cGRU",
        model=MODEL_LOADER.return_base_learner,
        numeric=True,
        batch_learner=True,
        drift=False,
        cpnn=True,
    ),
    LearnerConfig(
        name="cGRU_pretrained",
        model=MODEL_LOADER.return_base_learner_pretrained,
        numeric=True,
        batch_learner=True,
        drift=False,
        cpnn=True,
    ),
    LearnerConfig(
        name="cPB",
        model=MODEL_LOADER.return_cpb,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    ),
]

if SEQ_LEN is None:
    if "sine" in PATHS[0].lower():
        SEQ_LEN = 10
    elif "weather" in PATHS[0].lower():
        SEQ_LEN = 11
    elif "air" in PATHS[0].lower():
        SEQ_LEN = 10
    else:
        SEQ_LEN = 10
NUM_FEATURES = 2
NUM_CLASSES = 2
NUM_OLD_LABELS = SEQ_LEN - 1
METRICS = ["accuracy", "kappa"]
MAX_SAMPLES = None
WRITE_CHECKPOINTS = False
ANYTIME_SCENARIO = True
PERIODIC_SCENARIO = False

if OUTPUT_FILE is None:
    OUTPUT_FILE = PATHS[0].split("/")[-1]

initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS)
EVAL_CL = None


def create_iter_csv():
    return stream.iter_csv(str(PATH) + ".csv", converters=CONVERTERS, target="target")


def next_iteration_callback(**kwargs):
    MODEL_LOADER.next_iteration()


CALLBACK_FUNC = next_iteration_callback
PATH = ""
if not PATH_PERFORMANCE.startswith("/"):
    PATH_PERFORMANCE = os.path.join("performance", PATH_PERFORMANCE)

orig_stdout = sys.stdout
f = None
if MODE == "aws":
    make_dir(f"outputs")
    f = open(f"outputs/{OUTPUT_FILE}.txt", "w", buffering=1)
    sys.stdout = f

try:
    for path in PATHS:
        PATH = path
        MODEL_LOADER.set_path(PATH.split("/")[-1])
        current_path_performance = os.path.join(PATH_PERFORMANCE, PATH.split("/")[-1])
        make_dir(current_path_performance)

        df = pd.read_csv(f"{PATH}.csv", nrows=1)
        columns = list(df.columns)
        initial_task = df.iloc[0]["task"]
        columns.remove("target")
        columns.remove("task")
        CONVERTERS = {c: float for c in columns}
        CONVERTERS["target"] = int
        CONVERTERS["task"] = int
        NUM_FEATURES = len(columns)
        data_stream = create_iter_csv

        initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS)
        print(PATH)
        print("BATCH SIZE, SEQ LEN:", BATCH_SIZE, SEQ_LEN)
        print("NUM OLD LABELS:", NUM_OLD_LABELS)
        print("ANYTIME LEARNERS:", [m.name for m in anytime_learners])
        print("BATCH LEARNERS:", [(m.name, m.drift) for m in batch_learners])
        print("SUFFIX:", suffix)
        print()

        EVAL_PREQ = EvaluatePrequential(
            max_data_points=MAX_SAMPLES,
            batch_size=BATCH_SIZE,
            metrics=METRICS,
            anytime_learners=anytime_learners,
            batch_learners=batch_learners,
            data_stream=data_stream,
            path_write=current_path_performance,
            suffix=suffix,
            write_checkpoints=WRITE_CHECKPOINTS,
            iterations=ITERATIONS,
            dataset_name=PATH.split("/")[-1],
            mode=MODE,
            anytime_scenario=ANYTIME_SCENARIO,
            periodic_scenario=PERIODIC_SCENARIO,
        )

        initialize_callback(EVAL_CL, EVAL_PREQ)

        EVAL_PREQ.evaluate(callback=CALLBACK_FUNC, initial_task=initial_task)
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
