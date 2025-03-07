import datetime
import os
import pickle
from typing import List

import pandas as pd
import numpy as np
from river import metrics

from evaluation.buffer import Buffer
from evaluation_gin.learner_config import LearnerConfig
from evaluation_gin.inference_cpnn import InferenceCPNN
from evaluation_gin.inference_gin import InferenceGIN

import torch


def get_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size

def get_size_gin(models):
    size = 0
    for model in models[:-1]:
        torch.save(model.state_dict(), "temp.p")
        size = size +  os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
    torch.save(models[-1].state_dict(), "temp.p")
    size2 = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return (size,size2)

class EvaluateContinualLearning:
    def __init__(
        self,
        path,
        checkpoint,
        learners_config,
        path_write,
        batch_size,
        seq_len,
        suffix="",
        mode="local",
        delay=0,
    ):
        self.dataset = pd.read_csv(f"{path}.csv")
        self.dataset_name = path.split("/")[-1].replace("_test", "")
        self.learners_config: List[LearnerConfig] = learners_config
        self.checkpoint = checkpoint
        self.feature_names = list(self.dataset.columns)[:-2]
        self._iterations = len(self.checkpoint[list(self.checkpoint.keys())[0]])
        self.delay = delay
        self.X = []
        self.Y = []
        if mode == "local":
            self.print_end = "\r"
        else:
            self.print_end = "\n"
        for task in range(1, self.dataset["task"].max() + 1):
            df_task = self.dataset[self.dataset["task"] == task].drop(columns="task")
            self.X.append(df_task.iloc[:, :-1].values)
            self.Y.append(df_task.iloc[:, -1].values)
        self.metric_names = ["kappa", "accuracy", "time", "memory"]
        model_names = [a.name for a in self.learners_config]
        self.metric_tables = {
            model: {
                metric: [[] for _ in range(self._iterations)]
                for metric in self.metric_names
            }
            for model in model_names
        }
        self.cl_metrics = {}
        for model_name in model_names:
            self.cl_metrics[model_name] = {}
            for metric in self.metric_names:
                self.cl_metrics[model_name][metric] = [
                    {} for _ in range(self._iterations)
                ]
        self.predictions = {}
        for model_name in model_names:
            self.predictions[model_name] = [[] for _ in range(self._iterations)]
        self.path_write = path_write
        if suffix != "" and not suffix.startswith("_"):
            suffix = "_" + suffix
        self.suffix = suffix
        self.batch_size = batch_size
        self.seq_len = seq_len

    def _compute_cl_metrics(self, model_name, metric, iteration=0):
        n = len(self.metric_tables[model_name][metric][iteration])
        self.cl_metrics[model_name][metric][iteration] = {
            "average": np.mean(self.metric_tables[model_name][metric][iteration][-1]),
            "a_metric": np.sum(
                [
                    self.metric_tables[model_name][metric][iteration][i][j]
                    for i in range(n)
                    for j in range(i + 1)
                ]
            )
            / (n * (n + 1) / 2),
            "bwt": np.sum(
                [
                    (
                        self.metric_tables[model_name][metric][iteration][i][j]
                        - self.metric_tables[model_name][metric][iteration][j][j]
                    )
                    for i in range(1, n)
                    for j in range(i)
                ]
            )
            / (n * (n - 1) / 2),
        }

    def _convert_to_dict(self, x):
        return {self.feature_names[i]: x[i] for i in range(len(x))}
    
    def retrieve_linear_biases(self, checkpoints):
        biases = []
        for model in checkpoints:
            biases.append(model.manager.model.classifier[1].bias)
        return biases

    def evaluate(self, iteration=0, **kwargs):
        print("\nCL evaluation STARTED")
        for model_dict in self.learners_config:
            model_name = model_dict.name
            model_name_perf = model_dict.name + "_anytime"
            if model_dict.gin:
                    linear_biases = self.retrieve_linear_biases(self.checkpoint[model_name_perf][iteration])
            for task_train, model_task in enumerate(
                self.checkpoint[model_name_perf][iteration]
            ):
                self.predictions[model_name][iteration].append([])
                if model_dict.cpnn and not model_dict.gin and not model_dict.dyn_cpnn and model_dict.drift:
                    model_task = InferenceCPNN(model_task)
                if model_dict.gin:
                    model_task = InferenceGIN(model_task, linear_biases= linear_biases)
                    size = get_size_gin(model_task.models)
                if model_dict.cpnn and not model_dict.drift:
                    model_task.columns.to(torch.device("cpu"))
                    
                for metric_name in self.metric_names:
                    self.metric_tables[model_name][metric_name][iteration].append([])
                for task_test in range(len(self.X)):
                    self.predictions[model_name][iteration][task_train].append([])
                    update_inference = False
                    if model_dict.temp_dep:
                        model_task.reset_previous_data_points()
                    if (model_dict.cpnn or model_dict.gin) and not model_dict.dyn_cpnn and model_dict.drift:
                        update_inference = True
                    elif model_dict.dyn_cpnn:
                        model_task.inference_mode(False)
                        model_task.inference_mode(True)
                        update_inference = True
                    accuracy = metrics.BalancedAccuracy()
                    kappa = metrics.CohenKappa()
                    count = 1
                    start = datetime.datetime.now()
                    buffer_y = Buffer(self.delay)
                    for idx, (x, y) in enumerate(
                        zip(self.X[task_test], self.Y[task_test])
                    ):
                        if count % 5 == 0:
                            print(
                                f"{self.dataset_name}, {model_name}, train {task_train + 1}, test {task_test + 1}, "
                                f"{'{:04d}'.format(count)}/{len(self.X[task_test])}",
                                end=self.print_end,
                            )
                        count += 1
                        if not model_dict.numeric:
                            x = self._convert_to_dict(x)
                        if model_dict.cpnn or model_dict.gin:
                            y_hat = model_task.predict_one(x, timestamp=idx)
                        else:
                            y_hat = model_task.predict_one(x)
                        self.predictions[model_name][iteration][task_train][
                            task_test
                        ].append(y_hat)
                        y_hat = 0 if y_hat is None else y_hat
                        accuracy.update(y, y_hat)
                        kappa.update(y, y_hat)
                        if update_inference:
                            y_update = buffer_y.enqueue(y)
                            if y_update is not None:
                                model_task.update_inference(
                                    y_update, timestamp=idx - self.delay
                                )
                    print(
                        f"{self.dataset_name}, {model_name}, train {task_train + 1}, test {task_test + 1}, "
                        f"{'{:04d}'.format(count-1)}/{len(self.X[task_test])}",
                        end=self.print_end,
                    )
                    end = datetime.datetime.now()
                    self.metric_tables[model_name]["kappa"][iteration][-1].append(
                        kappa.get()
                    )
                    self.metric_tables[model_name]["accuracy"][iteration][-1].append(
                        accuracy.get()
                    )
                    self.metric_tables[model_name]["time"][iteration][-1].append(
                        (end - start).microseconds
                    )
                if not model_dict.cpnn and not model_dict.gin:
                    size = 0
                elif model_dict.gin:
                    size = size
                elif model_dict.dyn_cpnn:
                    size = np.sum(
                        [get_size(m.model.columns) for m in model_task.models]
                    )
                elif not model_dict.drift:
                    size = get_size(model_task.columns)
                    print(size)
                else:
                    size = get_size(model_task.model.columns)
                self.metric_tables[model_name]["memory"][iteration][-1] = size
            if self.print_end == "\r":
                print()
            for metric in ["accuracy", "kappa"]:
                self.metric_tables[model_name][metric][iteration] = np.array(
                    self.metric_tables[model_name][metric][iteration]
                )
                self._compute_cl_metrics(model_name, metric, iteration)

        with open(
            os.path.join(
                self.path_write,
                f"metric_tables_{self.batch_size}_{self.seq_len}{self.suffix}_size.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.metric_tables, f)
        with open(
            os.path.join(
                self.path_write,
                f"cl_metrics_{self.batch_size}_{self.seq_len}{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.cl_metrics, f)
        with open(
            os.path.join(
                self.path_write,
                f"cl_predictions_{self.batch_size}_{self.seq_len}{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.predictions, f)

        print("CL evaluation ENDED")
