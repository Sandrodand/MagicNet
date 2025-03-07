from evaluation.cl_evaluation import *
import os
import pickle

from models.dynamic_cpnn import DynamicCPNN
from evaluation.test_utils import *



PATHS = [
    (
        f"/Users/federicogiannini/Library/CloudStorage/OneDrive-PolitecnicodiMilano/NN4SML/datasets/dynamic/"
        f"air_quality_{c}conf"
    )
    for c in range(1, 7)
]  # a list containing the paths of the data streams (without the extension)
PATH_PERFORMANCE = "dynamic"
SEQ_LEN_PARAM = None
BATCH_SIZE = 128
ANYTIME_LEARNERS = [
    LearnerConfig(
        name="ARF_TA",
        model=lambda : print(end=""),
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    LearnerConfig(
        name="cPNN",
        model=lambda : print(end=""),
        numeric=True,
        batch_learner=False,
        drift=True,
        cpnn=True,
    ),
    LearnerConfig(
        name="cLSTM",
        model=lambda : print(end=""),
        numeric=True,
        batch_learner=False,
        drift=False,
        cpnn=True,
    ),
    LearnerConfig(
        name="DyncPNN",
        model=lambda : print(end=""),
        numeric=True,
        batch_learner=False,
        drift=True,
        cpnn=True,
        dyn_cpnn=True,
    ),
]

SEQ_LEN = None
NUM_FEATURES = None
if not PATH_PERFORMANCE.startswith("/"):
    PATH_PERFORMANCE = os.path.join("performance", PATH_PERFORMANCE)

for PATH in PATHS:
    print(PATH)
    dataset = PATH.split("/")[-1].lower()
    df = pd.read_csv(f"{PATH}_train.csv", nrows=1)
    columns = list(df.columns)
    columns.remove("target")
    columns.remove("task")
    CONVERTERS = {c: float for c in columns}
    CONVERTERS["target"] = int
    CONVERTERS["task"] = int
    NUM_FEATURES = len(columns)

    if SEQ_LEN_PARAM is None:
        SEQ_LEN = 11 if "weather" in dataset else 10
    else:
        SEQ_LEN = SEQ_LEN_PARAM
    current_path_performance = os.path.join(PATH_PERFORMANCE, dataset)
    initialize(seq_len_=SEQ_LEN, num_old_labels_=SEQ_LEN-1, batch_size_=BATCH_SIZE, num_features_=NUM_FEATURES)

    with open(os.path.join(current_path_performance, f"checkpoints_{BATCH_SIZE}_{SEQ_LEN}_it0.pkl"), "rb") as f:
        checkpoint_file = pickle.load(f)
    checkpoint = {}
    for model in ANYTIME_LEARNERS:
        model_name = f"{model.name}_anytime"
        if not model.cpnn:
            checkpoint[model_name] = [checkpoint_file[model_name]]
            continue
        checkpoint[model_name] = [[]]
        if not model.dyn_cpnn:
            for c in checkpoint_file[model_name]:
                cpnn = create_cpnn_for_dynamic()
                cpnn.set_save_column_freq(None)
                cpnn.columns.columns = c
                checkpoint[model_name][-1].append(cpnn)
        else:
            for c in checkpoint_file[model_name]:
                dyn_cpnns = [create_cpnn_for_dynamic() for _ in range(len(c))]
                for cpnn, cols in zip(dyn_cpnns, c):
                    cpnn.columns.columns = cols
                checkpoint[model_name][-1].append(DynamicCPNN(models=dyn_cpnns))


    eval_cl = EvaluateContinualLearning(
        path=f"{PATH}_test",
        checkpoint=checkpoint,
        learners_config=ANYTIME_LEARNERS,
        path_write=current_path_performance,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN
    )

    eval_cl.evaluate()