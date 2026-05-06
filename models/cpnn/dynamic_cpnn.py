from typing import List

from models.cpnn.cpnn import *
import pickle
from river import metrics
import numpy as np
import os
import time


class SingleModel:
    """
    It implements the class that represents a single model stored in a DYNcPNN.
    """

    def __init__(self, model: cPNN, timestamp: int = time.time()):
        """
        Parameters
        ----------
        model: cPNN
            The cPNN model
        timestamp: int, default: now.
            The model's last selection timestamp. If not expressed it's equal to the current timestamp.
        """
        self.model: cPNN = model
        self.timestamp: int = timestamp

    def __str__(self):
        return (
            f"len: {len(self.model.columns.columns)}, timestamp: {self.timestamp}, "
            f"columns: {self.model.columns.columns}"
        )


class ModelEnsemble:
    def __init__(
        self,
        model: cPNN,
        model_idx: int = 0,
        train: bool = True,
        column: int = None,
        model_type: int = -1,
    ):
        """
        It implements the class to express a model within the DYNcPNN ensemble.

        Parameters
        ----------
        model: cPNN
            the cPNN model.
        model_idx: int, default: 0.
            The index of the associated model in the models attribute of DynamicCPNN.
        train: bool, default: true.
            True if the model must be trained, False otherwise.
            This parameter should not be True when using a column different with respect to the last.
        column: int, default: None
            The column to use to perform prediction. If None the last column is used.
        model_type: int, default: -1
            -1 indicates that the model represents the cPNN without adding a new column after the detected drift.
            0 indicates that the model represents the cPNN of the immediately preceding concept to which a new column
                has been added after the detected drift.
            1 indicates that the model represents the cPNN of a previous concept (other than the immediately preceding
                one) to which a new column has been added after the detected drift.
            2 indicates that you are using an already trained column. In this case train should be False.
        """
        self.model: cPNN = model
        self.train: bool = train
        if column is None:
            column = len(model.columns.columns) - 1
        self.column: int = column
        self.model_type = model_type
        self.model_idx = model_idx
        if self.column < len(model.columns.columns) - 1:
            self.train = False
            self.model_type = 2
        if self.model_type == 2:
            assert column < len(model.columns.columns)
            assert not self.train

    def __str__(self):
        return (
            f"idx: {self.model_idx}, len: {len(self.model.columns.columns)}, column: {self.column}, "
            f"train: {self.train}, type: {self.model_type}"
        )


class DynamicCPNN:
    def __init__(
        self,
        models: List[cPNN],
        drift_window_batches: int = 30,
        grace_period: int = 5000,
        store_models: bool = True,
        max_models: int = 4,
        data_points_previous_concept: int = 1000,
        minimum_k_difference: float = 0.04,
        verbose: bool = True,
        inference_ensemble_data_points: int = 500,
        recurrent_concepts: bool = False,
    ):
        """
        It implements the Dynamic cPNN (DYNcPNN) model. The model starts with a single cPNN. After a drift detection,
        it evaluates if the model can adapt to the new concept without adding a new column or not. To do so, it builds
        an ensemble containing the cPNN that does not add a new column and the cPNN that adds it. After a configurable
        number of data points, if the cPNN that does not add a new column performs worse it accepts the drift.
        If store_models==True, in the case of drift rejection, it stores the model's associated with the old concept
        when drift causes its forgetting. The stored models are added to the ensemble after a drift detection.

        Parameters
        ----------
        models: list of cPNN.
            The initial cPNN models.
        drift_window_batches: int, default: 50.
            Number of training mini-batches at the end of which to choose whether accepting or rejecting the drift.
        grace_period: int, default: 10k.
            Number of data points following the last accepted drift in which the model does not take into consideration
            new drifts.
        store_models: bool, default: True.
            Two situations could cause the rejection of a drift:
            1) The model can adapt quickly to the new concept without adding a column.
            2) The drift is not significant.
            In the first case, when adapting to the new concept, the model may lose the predictive ability on the
            previous. To avoid this situation, if store_models is set to True, we compare the performance
            (on the previous concept) of the current model and the model stored before the drift. If the latter performs
             better we save it into 'models' to not forget the previous concept.
        max_models: int, default: 4.
            Maximum number of models to store in the case of store_models==True. The ones with higher last selection
            timestamp are selected.
        data_points_previous_concept: int, default: 1000.
            In the case of store_models==True, it represents the number of data points of the previous concept to use
            to evaluate if the concept preceding the rejected drift is forgotten or not.
        minimum_k_difference: float, default: 0.04.
            Minimum difference in Cohen's K values between the best model in the ensemble and the model that does not
            add a column to accept drift.
        verbose: bool, default: True.
            If True, it prints the messages regarding the acceptance or rejection of the drifts.
        inference_ensemble_data_points: int, default: 2*128.
            Number of data points after which to choose the best model in the ensemble during the inference mode.
            Use -1 to keep the ensemble during the entire inference phase.
        recurrent_concepts: bool, default: False.
            True if you want to add to the ensemble the columns of all the models in the pool without training them.
            This is useful in the case of recurrent concepts. A previously trained column may be enough for an
            already-seen concept.
        """
        self.models = [
            SingleModel(m, timestamp=time.time() - i) for i, m in enumerate(models)
        ]
        self.ensemble = [ModelEnsemble(models[0])]
        self.metrics = [metrics.CohenKappa()]
        self.selected = 0
        self.drift_cont = -1
        # -1 drift not in evaluation,
        # >=0 number of data points following the currently evaluating drift.
        self.data_points_previous_concept = data_points_previous_concept
        self.stream_start = True
        self.predictions = {}
        self.drift_window = drift_window_batches * self.ensemble[0].model.batch_size + 1
        self.grace_period = grace_period
        self.grace_period_cont = 0
        self.cont = 0
        self.choices = []
        self.max_models = max_models
        self.concept_x = {"old": [], "new": []}
        self.concept_y = {"old": [], "new": []}
        self.old_model = None
        self.store_models = store_models
        self._previous_data_points = None
        self.minimum_k_difference = minimum_k_difference
        self.verbose = verbose
        self.inference = False
        self._ensemble_old = self.ensemble.copy()
        self._selected_old = 0
        self._previous_data_points_old = None
        self._metrics_old = []
        self.inference_ensemble_data_points = inference_ensemble_data_points
        self.inference_count = 0
        self.features = {}
        self.last_ts_predict = -1
        self.last_ts_train = -1
        self.recurrent = recurrent_concepts

    def learn_one(self, x, y, timestamp=-1):
        """
        It trains DYNcPNN on a single data point. On each cPNN, the training is performed after filling up a mini-batch
        of data points. During the drift_window_batches mini-batches following the drift detection, it trains all the
        cPNNs in the ensemble and updates the best-performing one using the prediction made during the last predict_one
        call.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        y: int
            The target value of the single data point.
        timestamp: int, default: -1
            The timestamp associated with the data point. Use -1 in case of no delay.
        """
        if self.inference:
            return
        self.last_ts_train = timestamp
        if self.drift_cont > -1:
            self.drift_cont += 1
            if (
                self.store_models
                and len(self.concept_x) < self.data_points_previous_concept
            ):
                self.concept_x["new"].append(x)
                self.concept_y["new"].append(y)
        elif (
            self.stream_start
            and self.store_models
            and len(self.concept_x) < self.data_points_previous_concept
        ):
            self.concept_x["new"].append(x)
            self.concept_y["new"].append(y)
        if self.grace_period_cont > -1:
            self.grace_period_cont += 1
            if self.grace_period_cont == self.grace_period:
                if self.verbose:
                    print("\nReset grace period")
                self.grace_period_cont = -1
        if timestamp in self.predictions:
            for i in range(len(self.ensemble)):
                if self.drift_cont > -1:
                    self.metrics[i].update(y, self.predictions[timestamp][i])
        for i in range(len(self.ensemble)):
            if self.ensemble[i].train:
                self.ensemble[i].model.learn_one(x, y, timestamp=timestamp)
        if self.drift_cont > -1:
            ks = [m.get() for m in self.metrics]
            self.selected = np.argmax(ks)
            if self.drift_cont == self.drift_window:
                if ks[self.selected] - ks[0] < self.minimum_k_difference:
                    self.selected = 0
                selected_model: ModelEnsemble = self.ensemble[self.selected]
                choice = {
                    "cont": self.cont - self.drift_cont,
                    "kappas": ks,
                    "selected_idx": self.selected,
                    "selected_model_idx": selected_model.model_idx,
                    "selected_column": selected_model.column,
                    "selected_type": selected_model.model_type,
                    "model_idx": [m.model_idx for m in self.ensemble],
                    "columns": [m.column for m in self.ensemble],
                    "types": [m.model_type for m in self.ensemble],
                    "grace_period": False,
                    "already_in_evaluation": False,
                }
                if selected_model.model_type != -1:
                    self.grace_period_cont = self.drift_cont
                    # If we accept the drift, we can discard the model that doesn't introduce the new column and
                    # substitute it with its copy that adds a new column. This way, when we will remove the last
                    # column from all the non-selected models, we will have a column representing the last concept.
                    model_new_column = [m for m in self.ensemble if m.model_type == 0][
                        0
                    ]
                    model_new_column_idx = [
                        i
                        for i in range(len(self.ensemble))
                        if self.ensemble[i].model_type == 0
                    ][0]
                    self.models[0].model = model_new_column.model
                    self.ensemble[0] = ModelEnsemble(
                        model=model_new_column.model,
                        model_idx=model_new_column.model_idx,
                        train=model_new_column.train,
                        column=model_new_column.column,
                        model_type=0,
                    )
                    self.metrics[0] = self.metrics[model_new_column_idx]

                    if selected_model.model_type == 0:
                        # we moved the model with model_type==0 to the first position
                        self.selected = 0
                        selected_model = self.ensemble[0]
                    self.ensemble[model_new_column_idx].model_type = 2
                    # Otherwise, when looping on the ensemble, it would remove the last column twice
                    # (one time for the first position and one time for the original position)
                if selected_model.model_type == 0 or selected_model.model_type == 1:
                    choice["accepted"] = True
                    choice["recurrent"] = False
                    if self.verbose:
                        print("\nDrift ACCEPTED: ", [np.round(k, 3) for k in ks])
                elif selected_model.model_type == 2:
                    # we are using a previous frozen column
                    choice["accepted"] = True
                    choice["recurrent"] = True
                    if self.verbose:
                        print("\nRECURRENT concept: ", [np.round(k, 3) for k in ks])
                else:  # selected_model.model_type == -1
                    # we rejected the drift
                    choice["accepted"] = False
                    choice["recurrent"] = False
                    if self.verbose:
                        print("\nDrift REJECTED: ", [np.round(k, 3) for k in ks])
                    if self.store_models:
                        # Two situations could cause the rejection of a drift:
                        # 1) The model can adapt quickly to the new concept without adding a column.
                        # 2) The drift is not significant.
                        # In the first case, when adapting to the new concept, the model may lose the predictive ability
                        # on the previous. To avoid this situation, we compare the performance (on the previous concept)
                        # of the current model and the model stored before the drift. If the latter performs better
                        # we must save it into 'models' in order to not forget the concept.
                        current_model: cPNN = pickle.loads(
                            pickle.dumps(selected_model.model)
                        )
                        current_model.reset_previous_data_points()
                        current_k = metrics.CohenKappa()
                        old_model: cPNN = pickle.loads(pickle.dumps(self.old_model))
                        old_model.reset_previous_data_points()
                        old_k = metrics.CohenKappa()
                        for x, y in zip(self.concept_x["old"], self.concept_y["old"]):
                            old_k = old_k.update(
                                y, old_model.predict_one(x, timestamp=-1)
                            )
                            current_k = current_k.update(
                                y, current_model.predict_one(x, timestamp=-1)
                            )
                        if old_k.get() - current_k.get() >= self.minimum_k_difference:
                            self.models.append(SingleModel(self.old_model))
                            choice["old_model"] = True
                            if self.verbose:
                                print(
                                    "STORED old model:",
                                    np.round(old_k.get(), 3),
                                    np.round(current_k.get(), 3),
                                )
                        else:
                            choice["old_model"] = False
                            if self.verbose:
                                print(
                                    "DISCARDED old model:",
                                    np.round(old_k.get(), 3),
                                    np.round(current_k.get(), 3),
                                )
                        choice["old_model_k"] = old_k.get()
                        choice["current_model_k"] = current_k.get()
                    self.concept_x["old"].clear()
                    self.concept_y["old"].clear()
                for i in range(0, len(self.ensemble)):
                    if i != self.selected and self.ensemble[i].model_type != 2:
                        self.ensemble[i].model.remove_last_column()
                self.models[selected_model.model_idx].timestamp = time.time()
                self.models = sorted(
                    self.models, key=lambda m: m.timestamp, reverse=True
                )[: self.max_models]
                self.ensemble = [selected_model]
                self.metrics = [self.metrics[self.selected]]
                self.choices.append(choice)
                self.choices = sorted(self.choices, key=lambda c: c["cont"])
                self.predictions = {}
                self.selected = 0
                self.drift_cont = -1
                self.old_model = None

                if self.verbose:
                    print(f"\nModels:")
                    for m in self.models:
                        print(str(m), end=";\n")
                    print(f"\nEnsemble:")
                    for m in self.ensemble:
                        print(str(m), end=";\n")
                    print()
        self.cont += 1
        if timestamp in self.predictions:
            del self.predictions[timestamp]

    def predict_one(self, x, timestamp=-1):
        """
        It performs prediction on a single data point. During the drift_window_batches mini-batches following the
        drift detection, it returns the prediction of the current best-performing model in the ensemble.

        Parameters
        ----------
        x: numpy.array or list
           The features values of the single data point.
        timestamp: int, default: -1.
            The timestamp associated with the data point. Use -1 in case of no delay between test and train.
        Returns
        -------
        prediction : int
           The predicted int label of x.
        """
        self.last_ts_predict = timestamp
        self.predictions[timestamp] = []
        for i in range(len(self.ensemble)):
            self.predictions[timestamp].append(
                self.ensemble[i].model.predict_one(
                    x,
                    column_id=self.ensemble[i].column,
                    previous_data_points=self._previous_data_points,
                    timestamp=timestamp,
                )
            )
        if self._previous_data_points is None:
            self._previous_data_points = np.array(x).reshape(1, -1)
        else:
            self._previous_data_points = np.concatenate(
                [self._previous_data_points, np.array(x).reshape(1, -1)]
            )[-(self.get_seq_len() - 1) :]
        return self.predictions[timestamp][self.selected]

    def add_new_column(self, task_id: int = None):
        """
        Call it to manage a drift detection. It builds the ensemble containing the current model without adding a new
        column, the current model with a new column, the stored models and all the previous columns of the current and
        stored models. If a drift is already in evaluation it ignores the new drift detection. The same happens during
        the grace period (a configurable number of data points following the last accepted drift detection).

        Parameters
        ----------
        task_id: int, default: None
            The id of the new task. If None it increments the last one.
        """
        self.stream_start = False
        if self.grace_period_cont != -1:
            if self.verbose:
                print("\nDrift requested during GRACE PERIOD")
            self.choices.append(
                {
                    "cont": self.cont + 1,
                    "accepted": False,
                    "recurrent": False,
                    "grace_period": True,
                    "already_in_evaluation": False,
                }
            )
            self.choices = sorted(self.choices, key=lambda c: c["cont"])
        elif self.drift_cont != -1:
            if self.verbose:
                print("\nDrift ALREADY IN EVALUATION")
            self.choices.append(
                {
                    "cont": self.cont + 1,
                    "accepted": False,
                    "recurrent": False,
                    "grace_period": False,
                    "already_in_evaluation": True,
                }
            )
            self.choices = sorted(self.choices, key=lambda c: c["cont"])
        else:
            if self.store_models:
                # We store the model before the drift to check if in future we will lose the predictive ability
                # on that concept.
                self.old_model: cPNN = pickle.loads(
                    pickle.dumps(self.ensemble[0].model)
                )
                # We take the version some data points before the drift to be sure that the model is not trained on
                # data points belonging to the new concept.
                self.old_model.restore_previous_state()
                self.old_model.reset_previous_data_points()

            for m in self.models[1:]:
                m.model.add_new_column(task_id)
                m.model.reset_previous_data_points()

            # We add a new model to the ensemble that is a copy current one with a new column.
            model_new = pickle.loads(pickle.dumps(self.ensemble[0].model))
            model_new.add_new_column()

            # we build the ensemble:
            # the first model is the current model without adding a new column
            # then, we add all the other stored models (with the addition of a new column)
            # then, we add the current model with the addition of a new column
            # finally, we add all the columns of the stored models. These columns are frozen and won't be trained
            self.ensemble = (
                [
                    ModelEnsemble(
                        self.models[0].model, model_type=-1, train=True, model_idx=0
                    )
                ]
                + [
                    ModelEnsemble(m.model, model_type=1, train=True, model_idx=i + 1)
                    for i, m in enumerate(self.models[1:])
                ]
                + [ModelEnsemble(model_new, model_type=0, train=True, model_idx=0)]
            )
            if self.recurrent:
                for i, m in enumerate(self.models):
                    for col in range(0, len(m.model.columns.columns) - 1):
                        self.ensemble.append(
                            ModelEnsemble(
                                m.model,
                                model_type=2,
                                train=False,
                                model_idx=i,
                                column=col,
                            )
                        )
            self.predictions = {}
            self.metrics = [metrics.CohenKappa() for _ in self.ensemble]
            if self.store_models:
                self.concept_x["old"] = self.concept_x["new"].copy()
                self.concept_x["new"].clear()
                self.concept_y["old"] = self.concept_y["new"].copy()
                self.concept_y["new"].clear()
            self.drift_cont = 0
            self.selected = 0
            if self.verbose:
                print(f"\nModels:")
                for m in self.models:
                    print(str(m), end=";\n")
                print(f"\nEnsemble:")
                for m in self.ensemble:
                    print(str(m), end=";\n")
                print()

    def inference_mode(self, mode: bool = True):
        """
        Call this method before running the model in inference mode or after the end of the inference mode to recover
        the training mode. The attribute inference_mode indicates if the model is in inference mode or not.
        During the inference mode it builds an ensemble that considers all the columns of all the models within the
        ensemble and the pool. On the i-th data point of the test set, it considers the prediction made by the
        best-performing model from the first data point of the test set to the (i-1)-th.
        After self.inference_ensemble_data_points data points it chooses the best model in the ensemble.

        Parameters
        ----------
        mode: bool, default: True.
            True if you want to activate the inference mode. False if you want di deactivate it.
        """
        if self.inference and not mode:
            self.ensemble = self._ensemble_old
            self.selected = self._selected_old
            self.metrics = self._metrics_old
            if self._previous_data_points_old is not None:
                self._previous_data_points = self._previous_data_points_old.copy()
            else:
                self._previous_data_points = None
            self.predictions = {}
        if mode and not self.inference:
            self._ensemble_old = self.ensemble.copy()
            self._selected_old = self.selected
            self._metrics_old = [pickle.loads(pickle.dumps(m)) for m in self.metrics]
            if self._previous_data_points is not None:
                self._previous_data_points_old = self._previous_data_points.copy()
            else:
                self._previous_data_points_old = None
            self.ensemble = [
                ModelEnsemble(
                    m.model, model_type=2, train=False, model_idx=i, column=col
                )
                for i, m in enumerate(self.models)
                for col in range(0, len(m.model.columns.columns))
            ]
            self.selected = 0
            self.metrics = [metrics.CohenKappa() for _ in self.ensemble]
            self._previous_data_points = None
            self.inference_count = 0
            self.predictions = {}
        self.inference = mode

    def update_inference(self, y, timestamp=-1):
        """
        Call this method during the inference mode to input the model the label of the previous data point.

        Parameters
        ----------
        y: int
            The label of the data point.
        timestamp: int, default: -1.
            The timestamp associated with the data point (in the case of delayed labels). Use -1 if there is no delay
            between Xt and yt and the label refers to the current data point.
        """
        if timestamp in self.predictions:
            for p, m in zip(self.predictions[timestamp], self.metrics):
                m.update(y, p)
            self.selected = np.argmax([m.get() for m in self.metrics])
            del self.predictions[timestamp]
        self.inference_count += 1
        if self.inference_count == self.inference_ensemble_data_points:
            self.ensemble = [self.ensemble[self.selected]]
            self.metrics = [self.metrics[self.selected]]
            self.selected = 0

    def reset_previous_data_points(self):
        """
        Call this method to reset the buffer containing the last W-1 data points representing the previous data points
        on which to build the RNN sequence.
        """
        self._previous_data_points = None
        self.predictions = {}
        for m in self.models:
            m.model.reset_previous_data_points()

    def get_seq_len(self):
        return self.ensemble[0].model.get_seq_len()

    def get_size(self):
        return 1
