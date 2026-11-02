import pickle
import datetime
from typing import List, Iterable, Callable

import river.metrics as river_metrics
import river.utils as river_utils
import os


from detectors.detector_simulator import DetectorSimulator
from detectors.sentinel import Sentinel
from evaluation.buffer import Buffer
from evaluation.learner_config import LearnerConfig


def make_dir(path):
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)


# noinspection DuplicatedCode
class EvaluatePrequential:
    """
    Class that implements the comparison on a specific data stream between different models. See the method 'evaluate'
    for details
    """

    def __init__(
        self,
        max_data_points: int = None,
        batch_size: int = 128,
        metrics: Iterable = ("accuracy", "kappa"),
        learners: List[LearnerConfig] = None,
        data_stream: Callable = None,
        path_write: str = None,
        suffix: str = "",
        write_checkpoints: bool = False,
        iterations: int = 1,
        dataset_name: str = "",
        mode: str = "local",
        writing_frequency: int = 5000,
        drift_detector=None,
        iteration_str=None,
        initial_task=1,
        preprocessing_func=None,
        delay: int = 0,
        rolling_window_batches=[10, 50],
    ):
        """
        Parameters
        ----------
        max_data_points: int, default: None
            If not None it indicates the number of data points to take from the data stream.
        batch_size: int, default: 128
            Reference batch size for rolling metric windows.
        metrics: Iterable, default: ("accuracy", "kapp")
            The list of metrics to be computed. Available metrics: {'accuracy', 'kappa'}.
        learners: List[LearnerConfig], default: ()
            A list of LearnerConfig that contains the models able to learn from single data points.
            They must implement the method learn_one(x, y) and predict_one(x).
        data_stream: Callable, default: None.
            A function that returns the river iterator (e.g. iter_csv or iter_pandas) representing the data stream.
        path_write: str, default: None.
            The path to which write the evaluation outputs.
        suffix: str, default: ''.
            The suffix to add the evaluation outputs' file names.
        write_checkpoints: bool, default: False.
            True if you want to write the pickles of the models after each task.
        iterations: int, default: 1.
            Number of experiments that you want to run.
        dataset_name: str, default: ''.
            The name of the dataset.
        mode: str, default: 'local'.
            'local' if you are running the experiment on your local machine.
            'aws' if you are running them on aws machines. In this case, it will write the messages in a specific file.
        writing_frequency: int, default: 5000.
            The frequency, specified as the number data points, at which the process writes the pickles files.
        drift_detector: detector, default: None.
            If provided, the automatic drift detector. The supervised drift are detected using the task labels of the
            data stream. This detector is only used to manage concept drift in the models. If None, it uses
            the task labels also to manage concept drift in the models. Sentinel is
            updated after all learners predict and before training. Simulated drift
            timestamps and supervised boundaries are applied before prediction.
        iteration_str: str, default: None.
            The string to append to each line when printing the outputs. If None, it prints the iteration index.
        initial_task: int, default: 1.
            The id of the first task. It should be equal to the first data point's task label.
        preprocessing_func: function, default: None.
            If not None it represents the function to call before running the model training.
        delay: int, default: 0.
            The number of timestamps of delay between the moment when the model receive feature vector X_t of
            the data point d_t and the moment when it receives the real label y_t. The original data stream contains
            both X_t and y_t in the same row. The class simulates the delay.
        """

        super().__init__()
        self.max_data_points = max_data_points
        self.batch_size = batch_size
        self.metrics = metrics
        self.learners = learners
        self._write_checkpoints = write_checkpoints
        self._predictions = {}
        self._iterations = iterations
        self.dataset_name = dataset_name
        self.data_stream = data_stream
        self.path_write = path_write
        self.drift_supervised = False
        self.drift_detected = False
        self.suffix = suffix
        self.rolling_window_batches = rolling_window_batches
        if rolling_window_batches is not None:
            self.rolling_metrics = [
                f"rolling_{w*self.batch_size}" for w in self.rolling_window_batches
            ]
        else:
            self.rolling_metrics = []
        self._create_eval()
        self._mode = mode
        self.writing_freq = writing_frequency
        self.drift_detector = drift_detector
        self.detected_drifts = []
        self.iteration_str = iteration_str
        self.prev_task = initial_task
        self.preprocessing_func = preprocessing_func
        self.buffer_x = Buffer(delay)
        self.buffer_x_preproc = Buffer(delay)
        self.buffer_y = Buffer(delay)
        self.delay = delay
        make_dir(self.path_write)

    def _create_eval(self):
        self._eval = dict()
        self._perf = dict()
        self.checkpoint = dict()
        # Streaming Machine Learning models
        self._init_structure()

    def reset_checkpoints(self):
        for k in self.checkpoint:
            self.checkpoint[k] = [[] for i in range(0, self._iterations)]

    def _init_structure(self):
        for model in self.learners:
            name = model.name + "_anytime"
            self.checkpoint[name] = [[] for _ in range(self._iterations)]
            self._eval[name] = {
                "alg": [model.model() for _ in range(self._iterations)],
                "metrics": {},
            }
            self._perf[name] = {
                key: [[] for _ in range(self._iterations)]
                for key in ("drifts", "time", "memory")
            }
            self._predictions[name] = [[] for _ in range(self._iterations)]
            for period in ["all", "task"] + self.rolling_metrics:
                self._eval[name]["metrics"][period] = {}
                self._perf[name][period] = {}
                for metric in self.metrics:
                    self._perf[name][period][metric] = [
                        [] for _ in range(self._iterations)
                    ]
                    self._eval[name]["metrics"][period][metric] = [
                        self._get_metric(metric, period)
                        for _ in range(self._iterations)
                    ]

    @staticmethod
    def _get_metric(metric, metric_type):
        rolling = None
        if "rolling_" in metric_type:
            rolling = int(metric_type.replace("rolling_", ""))
        if rolling is None:
            if metric == "accuracy":
                return river_metrics.BalancedAccuracy()
            if metric == "kappa":
                return river_metrics.CohenKappa()
        else:
            if metric == "kappa":
                return river_utils.Rolling(
                    river_metrics.CohenKappa(), window_size=rolling
                )
            if metric == "accuracy":
                return river_utils.Rolling(
                    river_metrics.BalancedAccuracy(), window_size=rolling
                )

    def _start_task(self, iteration, idx):
        """Store the preceding task before adaptation and reset task metrics."""
        print(" " * 50, end="\r")
        self._write_pickles(iteration, write_checkpoints=True)
        print("SUPERVISED DRIFT:", idx)
        for model in self.learners:
            model_name = model.name + "_anytime"
            for metric in self.metrics:
                for period in ["task"] + self.rolling_metrics:
                    self._eval[model_name]["metrics"][period][metric][
                        iteration
                    ] = self._get_metric(metric, period)
            self._perf[model_name]["drifts"][iteration].append(idx)
            learner = self._eval[model_name]["alg"][iteration]
            checkpoint = (
                learner.get_cl_checkpoint()
                if model.dyn_cpnn
                else pickle.loads(pickle.dumps(learner))
            )
            self.checkpoint[model_name][iteration].append(checkpoint)

    def _apply_drift(self, iteration, idx):
        """Notify all learners at the point when the detector signal is available."""
        self.detected_drifts[iteration].append(idx)
        print(" " * 50, end="\r")
        print("DETECTED DRIFT:", idx)
        self._write_pickles(iteration, write_checkpoints=True)
        for model in self.learners:
            if model.drift:
                model_name = model.name + "_anytime"
                learner = self._eval[model_name]["alg"][iteration]
                learner.add_new_column()
                if model.cpnn or model.magic:
                    self._perf[model_name]["memory"][iteration].append(
                        learner.get_size()
                    )

    def _predict(self, x, x_preproc, y, iteration, idx):
        for model in self.learners:
            x_ = list(x_preproc.values()) if model.numeric else x
            model_name = model.name + "_anytime"
            if model.cpnn or model.magic:
                y_hat = self._eval[model_name]["alg"][iteration].predict_one(
                    x_, timestamp=idx
                )
            else:
                y_hat = self._eval[model_name]["alg"][iteration].predict_one(x_)
            self._predictions[model_name][iteration] += [y_hat]
            y_hat = 0 if y_hat is None else y_hat
            for metric in self.metrics:
                for t in ["all", "task"] + self.rolling_metrics:
                    # update the metrics
                    self._eval[model_name]["metrics"][t][metric][
                        iteration
                    ] = self._eval[model_name]["metrics"][t][metric][iteration].update(
                        y, y_hat
                    )
                    # get metrics
                    self._perf[model_name][t][metric][iteration].append(
                        self._eval[model_name]["metrics"][t][metric][iteration].get()
                    )

    def _learn(self, x_train, x_train_preproc, y_train, iteration, idx):
        for model in self.learners:
            model_name = model.name + "_anytime"
            start = datetime.datetime.now()
            if x_train is not None:
                if model.numeric:
                    x_ = list(x_train_preproc.values())
                else:
                    x_ = x_train
                if model.cpnn or model.magic:
                    self._eval[model_name]["alg"][iteration].learn_one(
                        x_, y_train, timestamp=idx - self.delay
                    )
                else:
                    self._eval[model_name]["alg"][iteration].learn_one(x_, y_train)
            end = datetime.datetime.now()
            self._perf[model_name]["time"][iteration].append((end - start).microseconds)

    def _write_pickles(self, iteration=0, write_checkpoints=False):
        if self.learners:
            for i, m in enumerate(self.learners):
                m_name = m.name.replace("_anytime", "").lower()
                if m.magic:
                    with open(
                        os.path.join(
                            self.path_write, f"choices_{m_name}{self.suffix}.pkl"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(
                            self._eval[self.learners[i].name + "_anytime"]["alg"][
                                iteration
                            ].manager.ensemble_choices,
                            f,
                        )
                    with open(
                        os.path.join(
                            self.path_write, f"ensemble_perf_{m_name}{self.suffix}.pkl"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(
                            self._eval[self.learners[i].name + "_anytime"]["alg"][
                                iteration
                            ].manager.ensemble_perf,
                            f,
                        )
                if m.dyn_cpnn:
                    with open(
                        os.path.join(
                            self.path_write, f"choices_{m_name}{self.suffix}.pkl"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(
                            self._eval[self.learners[i].name + "_anytime"]["alg"][
                                iteration
                            ].choices,
                            f,
                        )
        with open(
            os.path.join(
                self.path_write,
                f"performance{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self._perf, f)
        with open(
            os.path.join(
                self.path_write,
                f"predictions{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self._predictions, f)
        with open(
            os.path.join(
                self.path_write,
                f"drifts{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.detected_drifts, f)
        if write_checkpoints and self._write_checkpoints:
            with open(
                os.path.join(
                    self.path_write,
                    f"checkpoints{self.suffix}_it{iteration}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(self._build_checkpoints(iteration), f)

    def _build_checkpoints(self, iteration):
        check = {}
        for i, model in enumerate(self.checkpoint):
            check[model] = []
            if not self.learners[i].cpnn:
                check[model] = self.checkpoint[model][iteration]
            elif not self.learners[i].dyn_cpnn:
                for c in self.checkpoint[model][iteration]:
                    check[model].append(c.columns.columns)
            else:
                for c in self.checkpoint[model][iteration]:
                    check[model].append(c.get_cl_checkpoint())
        return check

    def evaluate(
        self,
        callback: Callable = None,
        initial_task: int = None,
        reset_checkpoints: bool = True,
        datastream: Callable = None,
        path_write=None,
        iteration_str=None,
    ):
        """
        It evaluates the learners by predicting each point before training
        on the corresponding labeled point. Labels can be delayed. Supervised
        task boundaries create CL checkpoints; detected drifts trigger adaptation.

        Parameters
        ----------
        callback: Callable, default: None.
            A callback function to be called after each iteration. It will receive the following parameters:
            -   iteration: int. the number of the iteration (starting from 0)
            -   learners_dict: list. The list of LearnersConfig representing the learners.
            -   path: str. The path_write parameter of the constructor.
            -   suffix: str. The suffix parameter of the constructor.
            -   models: dict. A dictionary containing for each name of the models in learners_dict the object
                representing the model (at the end of the iteration).
        initial_task: int, default: 1.
            The id of the first task. If you don't specify it and the first task id is not 1, it can raise an error.
        reset_checkpoints: bool, default: True.
            True if you want to reset the checkpoint after each iteration
        datastream: Callable, default: None.
            The function that creates the iterator on the da stream. If None, it uses the ones given in the
            constructor method.
        path_write: str, default: None.
            The path to which write the results. If None, it uses the ones given in the constructor method.
        iteration_str: str, default: None.
            The string to append when printing the outputs. If None, it uses the ones given in the constructor method.
        Returns
        -------
        It writes the following pickle files in the path_write/dataset_name path.
        -   performance{suffix}.pkl:
            A dict d indexed by '{name}_anytime'. The 'drifts' entry contains
            task boundary indices; 'time' and 'memory' contain per-iteration
            measurements. The 'all', 'task', and rolling metric entries contain
            per-point metric values, indexed by metric and iteration.
        - predictions{suffix}.pkl:
            A dict d where d['{name}_anytime'][it] contains the predictions
            made by the model during iteration 'it'.
        """
        if datastream is not None:
            self.data_stream = datastream
        if path_write is not None:
            self.path_write = path_write
        if iteration_str is not None:
            self.iteration_str = iteration_str
        for iteration in range(self._iterations):
            self.buffer_x = Buffer(self.delay)
            self.buffer_x_preproc = Buffer(self.delay)
            self.buffer_y = Buffer(self.delay)
            with open(os.path.join(self.path_write, "time.txt"), "a") as f:
                f.write(
                    f"START {iteration}/{self._iterations}: {datetime.datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}\n"
                )
            data_stream = self.data_stream()
            if self._iterations == 1 and self.iteration_str is not None:
                iteration_str = self.iteration_str
            else:
                iteration_str = f"{iteration+1}/{self._iterations}"

            if self.max_data_points is None:
                stream = data_stream
            else:
                stream = data_stream.take(self.max_data_points)
            if initial_task is not None:
                self.prev_task = initial_task
            batch = 1
            self.drift_supervised = False
            self.drift_detected = False
            drift_detector = pickle.loads(pickle.dumps(self.drift_detector))
            self.detected_drifts.append([])
            for model in self.learners:
                model_name = model.name + "_anytime"
                if model.cpnn or model.magic:
                    self._perf[model_name]["memory"][iteration].append(
                        self._eval[model_name]["alg"][iteration].get_size()
                    )
            for idx, (x, y) in enumerate(stream):
                if (idx + 1) % 100 == 0:
                    print(
                        self.dataset_name,
                        f"{iteration_str}",
                        idx + 1,
                        end=None if self._mode == "aws" else "\r",
                    )

                supervised = False
                if "task" in x:
                    if self.prev_task != int(x["task"]):
                        self.drift_supervised = True
                    self.prev_task = int(x["task"])
                    del x["task"]
                    supervised = True
                if "batch" in x:
                    current_batch = x["batch"]
                    del x["batch"]
                    if current_batch != batch:
                        batch = current_batch
                        for model in self.learners:
                            model_name = model.name + "_anytime"
                            try:
                                self._eval[model_name]["alg"][iteration].reset_buffer()
                            except:
                                pass

                if self.preprocessing_func is not None:
                    x_preproc = self.preprocessing_func(x)
                else:
                    x_preproc = x
                x_train = self.buffer_x.enqueue(x)
                x_train_preproc = self.buffer_x_preproc.enqueue(x_preproc)
                y_train = self.buffer_y.enqueue(y)

                self.drift_detected = False
                if drift_detector is None:
                    self.drift_detected = self.drift_supervised
                elif isinstance(drift_detector, DetectorSimulator):
                    # Simulated timestamps refer to stream points, independently of label delay.
                    self.drift_detected = drift_detector.update(x, y)
                elif not isinstance(drift_detector, Sentinel) and x_train is not None:
                    self.drift_detected = drift_detector.update(x_train, y_train)
                if not supervised and not isinstance(drift_detector, Sentinel):
                    self.drift_supervised = self.drift_detected

                if self.drift_supervised:
                    self._start_task(iteration, idx)
                if self.drift_detected:
                    self._apply_drift(iteration, idx)

                self._predict(x, x_preproc, y, iteration, idx)

                # Label-based detection must follow inference by every learner.
                if isinstance(drift_detector, Sentinel) and x_train is not None:
                    self.drift_detected = drift_detector.update(x_train, y_train)
                    if self.drift_detected:
                        if not supervised:
                            self.drift_supervised = True
                            self._start_task(iteration, idx)
                        self._apply_drift(iteration, idx)

                self._learn(x_train, x_train_preproc, y_train, iteration, idx)

                if self.drift_supervised:
                    self.drift_supervised = False

                if (idx + 1) % self.writing_freq == 0:
                    self._write_pickles(iteration)

            self.drift_supervised = True
            for model in self.learners:
                model_name = model.name + "_anytime"
                learner = self._eval[model_name]["alg"][iteration]
                checkpoint = (
                    learner.get_cl_checkpoint()
                    if model.dyn_cpnn
                    else pickle.loads(pickle.dumps(learner))
                )
                self.checkpoint[model_name][iteration].append(checkpoint)

            self._write_pickles(iteration, write_checkpoints=True)

            if callback is not None:
                callback(
                    iteration=iteration,
                    learners_dict=self.learners,
                    path=self.path_write,
                    suffix=self.suffix,
                    models={
                        m.name: self._eval[m.name + "_anytime"]["alg"][iteration]
                        for m in self.learners
                    },
                )
            if reset_checkpoints:
                self.reset_checkpoints()
            print()
            with open(os.path.join(self.path_write, "time.txt"), "a") as f:
                f.write(
                    f"END {iteration}/{self._iterations}: {datetime.datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}\n"
                )
            print()
        return self._perf
