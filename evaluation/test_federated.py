from river import stream
from models.federated_cpnn import *
from evaluation.prequential_evaluation import EvaluatePrequential, make_dir
import pandas as pd
import sys
import traceback
from evaluation.test_utils import *
import threading

# __________________
# PARAMETERS
# __________________
PATHS = [
    f"datasets/federated/air_quality_federated_{c}conf" for c in range(1, 51)
]  # a list containing the paths of the data streams (without the extension)
N_TASKS = 5
NUM_BATCHES = 50
MAX_MODELS = 10
LOCAL_MODELS_PROP = 0.3
SEQ_LEN = None  # length of the sequence. If None is set to 11 for Weather, to 10 for Sine and AirQuality
ITERATIONS = 1  # number of experiments
PATH_PERFORMANCE = "fedcpnn"  # subdirectory of the folder 'performance' where write the outputs of the evaluation
MODE = "aws"  # 'local' or 'aws'. If 'aws', the messages will be written in a specific txt file in the output_file dir
OUTPUT_FILE = None
# the name of the output file in outputs dir. If None, it will use the name of the current data stream.
BATCH_SIZE = 128  # the batch size of periodic learners and classifiers.
PATH = None
HANDLE_RECURSION = False


# __________________
# CODE
# __________________
if SEQ_LEN is None:
    if "sine" in PATHS[0]:
        SEQ_LEN = 10
    elif "weather" in PATHS[0]:
        SEQ_LEN = 11
    elif "air_quality" in PATHS[0]:
        SEQ_LEN = 10
    else:
        SEQ_LEN = 10
NUM_FEATURES = 2
NUM_CLASSES = 2
NUM_OLD_LABELS = SEQ_LEN - 1
METRICS = ["accuracy", "kappa"]
MAX_SAMPLES = None
WRITE_CHECKPOINTS = False
INITIAL_TASK = 1


class BaseLearner:
    def __init__(self):
        self.model: cPNN = None

    def initialize_model(self):
        self.model: cPNN = create_qcpnn_clstm()

    def get_model_task1(self):
        model = pickle.loads(pickle.dumps(self.model))
        model.set_initial_task(1)
        return model

    def get_model_not_quantized_task1(self):
        model = pickle.loads(pickle.dumps(self.model))
        model.set_initial_task(1)
        model.set_quantized(False)
        return model

    def get_model_task2(self):
        model = pickle.loads(pickle.dumps(self.model))
        model.set_initial_task(2)
        return model

    def get_model_not_quantized_task2(self):
        model = pickle.loads(pickle.dumps(self.model))
        model.set_initial_task(2)
        model.set_quantized(False)
        return model


def create_iter_csv(node, task, start=False):
    start = "_start" if start else ""
    return lambda: stream.iter_csv(
        str(f"{PATH}_node{node}_task{task}{start}") + ".csv",
        converters=CONVERTERS,
        target="target",
    )


CONVERTERS = None
BASE_LEARNER = BaseLearner()

if OUTPUT_FILE is None:
    OUTPUT_FILE = PATHS[0].split("/")[-1]

initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS)
EVAL_CL = None


def create_fcpnn_initial_task1():
    federated = FederatedCPNN(
        BASE_LEARNER.get_model_task1,
        num_batches=NUM_BATCHES,
        max_models=MAX_MODELS,
        local_models_prop=LOCAL_MODELS_PROP,
        initial_task=1,
        handle_recursion=HANDLE_RECURSION,
    )
    return federated


def create_fcpnn_initial_task2():
    federated = FederatedCPNN(
        BASE_LEARNER.get_model_task2,
        num_batches=NUM_BATCHES,
        initial_task=2,
        max_models=MAX_MODELS,
        local_models_prop=LOCAL_MODELS_PROP,
        handle_recursion=HANDLE_RECURSION,
    )
    return federated


def thread_evaluation(
    node: int,
    evaluator: EvaluatePrequential,
    model: FederatedCPNN,
    external_models: dict,
    node_to_start: threading.Thread = None,
):
    if node_to_start is not None:
        data_stream = create_iter_csv(node, 0, True)
        evaluator.evaluate(
            datastream=data_stream,
            reset_checkpoints=False,
            iteration_str=f"task0_start",
        )
        node_to_start.start()

    for task in range(0, N_TASKS):
        data_stream = create_iter_csv(node, task)
        evaluator.evaluate(
            datastream=data_stream, reset_checkpoints=False, iteration_str=f"task{task}"
        )
        for n_id in external_models:
            models = external_models[n_id].send_local_models()
            models_to_print = [x.model for x in models]
            print(models_to_print)
            print(
                f""
                f"Local node {node}, \n"
                f"current task number: {task}, \n"
                f"External node: {n_id}, \n"
                f"N models: {len(models_to_print)}, \n"
                f"Task dict: {[m.task_ids for m in models_to_print]}, \n"
                f"Cols: {len(models_to_print[0].columns.columns)} \n"
                f""
            )
            model.add_models(models, n_id)
    if node_to_start is not None:
        node_to_start.join()


PATH = ""
if not PATH_PERFORMANCE.startswith("/"):
    PATH_PERFORMANCE = os.path.join("performance", PATH_PERFORMANCE)

print("PATHS:", PATHS[0], "...")
orig_stdout = sys.stdout
f = None
if MODE == "aws":
    make_dir(f"outputs")
    f = open(f"outputs/{OUTPUT_FILE}.txt", "w", buffering=1)
    sys.stdout = f

try:
    for path in PATHS:
        for it in range(1, ITERATIONS + 1):
            PATH = path
            df = pd.read_csv(f"{path}_node1_task0.csv", nrows=1)
            columns = list(df.columns)
            columns.remove("target")
            columns.remove("task")
            CONVERTERS = {c: float for c in columns}
            CONVERTERS["target"] = int
            CONVERTERS["task"] = int
            NUM_FEATURES = len(columns)

            anytime_learners_node1 = [
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
                LearnerConfig(
                    name="HAT",
                    model=create_hat,
                    numeric=False,
                    batch_learner=False,
                    drift=False,
                    cpnn=False,
                ),
                LearnerConfig(
                    name="HAT_TA",
                    model=create_hat_ta,
                    numeric=False,
                    batch_learner=False,
                    drift=False,
                    cpnn=False,
                ),
                LearnerConfig(
                    name="cLSTM",
                    model=BASE_LEARNER.get_model_not_quantized_task1,
                    numeric=True,
                    batch_learner=False,
                    drift=False,
                    cpnn=True,
                ),
                LearnerConfig(
                    name="cPNN",
                    model=BASE_LEARNER.get_model_not_quantized_task1,
                    numeric=True,
                    batch_learner=False,
                    drift=True,
                    cpnn=True,
                ),
                LearnerConfig(
                    name="FEDcPNN",
                    model=create_fcpnn_initial_task1,
                    numeric=True,
                    batch_learner=False,
                    drift=True,
                    cpnn=True,
                ),
            ]

            anytime_learners_node2 = pickle.loads(pickle.dumps(anytime_learners_node1))
            anytime_learners_node2[
                -3
            ].model = BASE_LEARNER.get_model_not_quantized_task1
            anytime_learners_node2[
                -2
            ].model = BASE_LEARNER.get_model_not_quantized_task1

            anytime_learners_node3 = pickle.loads(pickle.dumps(anytime_learners_node1))
            anytime_learners_node3[
                -3
            ].model = BASE_LEARNER.get_model_not_quantized_task2
            anytime_learners_node3[
                -2
            ].model = BASE_LEARNER.get_model_not_quantized_task2
            anytime_learners_node3[-1].model = create_fcpnn_initial_task2
            current_path_performance1 = os.path.join(
                PATH_PERFORMANCE, f'{path.split("/")[-1]}_node1'
            )
            make_dir(current_path_performance1)
            current_path_performance2 = os.path.join(
                PATH_PERFORMANCE, f'{path.split("/")[-1]}_node2'
            )
            make_dir(current_path_performance2)
            current_path_performance3 = os.path.join(
                PATH_PERFORMANCE, f'{path.split("/")[-1]}_node3'
            )
            make_dir(current_path_performance3)

            initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, iterations_=1)
            BASE_LEARNER.initialize_model()

            eval_preq_node1 = EvaluatePrequential(
                max_data_points=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                metrics=METRICS,
                anytime_learners=anytime_learners_node1,
                batch_learners=[],
                data_stream=None,
                path_write=current_path_performance1,
                suffix=f"_it{it}",
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=1,
                dataset_name=f'{path.split("/")[-1]}_node1',
                mode=MODE,
                anytime_scenario=True,
                periodic_scenario=False,
                initial_task=1,
            )
            model_node1: FederatedCPNN = eval_preq_node1._eval[
                f"{anytime_learners_node1[-1].name}_anytime"
            ]["alg"][it - 1]

            eval_preq_node2 = EvaluatePrequential(
                max_data_points=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                metrics=METRICS,
                anytime_learners=anytime_learners_node2,
                batch_learners=[],
                data_stream=None,
                path_write=current_path_performance2,
                suffix=f"_it{it}",
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=1,
                dataset_name=f'{path.split("/")[-1]}_node2',
                mode=MODE,
                anytime_scenario=True,
                periodic_scenario=False,
                initial_task=1,
            )
            model_node2: FederatedCPNN = eval_preq_node2._eval[
                f"{anytime_learners_node2[-1].name}_anytime"
            ]["alg"][it - 1]

            eval_preq_node3 = EvaluatePrequential(
                max_data_points=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                metrics=METRICS,
                anytime_learners=anytime_learners_node3,
                batch_learners=[],
                data_stream=None,
                path_write=current_path_performance3,
                suffix=f"_it{it}",
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=1,
                dataset_name=f'{path.split("/")[-1]}_node3',
                mode=MODE,
                anytime_scenario=True,
                periodic_scenario=False,
                initial_task=2,
            )
            model_node3: FederatedCPNN = eval_preq_node3._eval[
                f"{anytime_learners_node3[-1].name}_anytime"
            ]["alg"][it - 1]

            node3 = threading.Thread(
                target=thread_evaluation,
                args=(
                    3,
                    eval_preq_node3,
                    model_node3,
                    {1: model_node1, 2: model_node2},
                ),
            )
            node2 = threading.Thread(
                target=thread_evaluation,
                args=(
                    2,
                    eval_preq_node2,
                    model_node2,
                    {1: model_node1, 3: model_node3},
                    node3,
                ),
            )
            node1 = threading.Thread(
                target=thread_evaluation,
                args=(
                    1,
                    eval_preq_node1,
                    model_node1,
                    {2: model_node2, 3: model_node3},
                    node2,
                ),
            )
            node1.start()
            node1.join()

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
