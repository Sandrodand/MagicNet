from models.magic.magic_net import MagicNet
import numpy as np
import torch
from river import metrics
import copy


class InferenceMagicNet:
    def __init__(self, model: MagicNet, ensemble_data_points=128 * 10):
        """
        It implements a wrapper on a MAGIC Net model to perform inference when the task label is not known.
        It builds an ensemble that considers all the saved PiggyMasks of a given GIN model. On the i-th data point of the test
        set,it considers the prediction made by the best-performing model from the first data point of the test set
        to the (i-1)-th.

        Parameters
        ----------
        model: MagicNet.
            The MagicNet model.
        ensemble_data_points: int, default: 128*2.
            Number of data points after which to choose the best model in the ensemble during the inference mode.
            Use -1 to keep the ensemble during the entire inference phase.
        """
        self.model: MagicNet = copy.deepcopy(model)
        self._previous_data_points = None
        self.metrics = None
        self.no_preparation = False
        self.selected = None
        self.models = []
        self.reset_previous_data_points()
        self.ensemble_data_points = ensemble_data_points
        self.count = 0
        self.predictions = {}

    def predict_one(self, x, timestamp=-1):
        """
        It performs prediction on a single data point. It returns the prediction of the current best-performing Mask
        from the first data point onwards.

        Parameters
        ----------
        x: numpy.array or list
           The features values of the single data point.
        Returns
        -------
        prediction : int
           The predicted int label of x.
        timestamp: int, default -1.
            The timestamp associated with the data point. Use -1 in case of no delay between features and labels.
        """
        self.predictions[timestamp] = []
        for m in self.models:
            self.predictions[timestamp].append(
                m.predict_one(
                    x,
                    previous_data_points=self._previous_data_points,
                )
            )
        if self._previous_data_points is None:
            self._previous_data_points = np.array(x).reshape(1, -1)
        else:
            self._previous_data_points = np.concatenate(
                [self._previous_data_points, np.array(x).reshape(1, -1)]
            )[-(self.model.get_seq_len() - 1) :]
        return self.predictions[timestamp][self.selected]

    def update_inference(self, y, timestamp=-1):
        """
        It updates the best-performing Mask using the real label. Call this method after predict_one on the same
        data point.

        Parameters
        ----------
        y: int.
            The real label of the last predicted data point.
        timestamp: int, default -1.
            The timestamp associated with the data point. Use -1 in case of no delay between features and labels.

        Returns
        -------

        """
        if timestamp in self.predictions:
            for p, m in zip(self.predictions[timestamp], self.metrics):
                m.update(y, p)
            self.selected = np.argmax([m.get() for m in self.metrics])
            del self.predictions[timestamp]
        self.count += 1
        if self.count == self.ensemble_data_points:
            self.models = [self.models[self.selected]]
            self.metrics = [self.metrics[self.selected]]
            self.selected = 0
            self.predictions = {}

    def prepare_task_models(self):
        """
        Crea una copia indipendente del modello per ogni task storico,
        applica la maschera corrispondente una volta sola e lo congela.
        """
        models = []
        num_tasks = len(self.model.manager.mask_list)

        for task_id in range(1, num_tasks + 1):
            task_model = copy.deepcopy(self.model)

            if task_id in task_model.manager.forgotten_models:
                task_model.manager.model = copy.deepcopy(task_model.manager.forgotten_models[task_id])
                task_model.manager.model.eval()
            else:
                task_model.manager.model.eval()
                idx = task_id - 1
                if idx < len(task_model.manager.piggymask_list):
                    historical_mask = task_model.manager.piggymask_list[idx]
                    task_model.manager.model.reinit_piggymask(mask_init="random", masks=historical_mask)
            task_model.manager.curr_task_idx = task_id
            models.append(task_model)
        return models

    def initialize(self):
        self.predictions = {}

        self.models = self.prepare_task_models()
        self.metrics = [
            metrics.CohenKappa() for _ in range(len(self.models))
        ]

        self.selected = len(self.models) - 1
        self.count = 0

    def reset_previous_data_points(self):
        for m in self.models:
            m.reset_previous_data_points()
        self._previous_data_points = None
        self.predictions = {}
        self.initialize()
