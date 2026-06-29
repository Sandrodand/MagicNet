from models.magic.magic_net import MagicNet
import numpy as np
import torch
from river import metrics
import river.utils as river_utils
import pickle


class InferenceMagicNet:
    def __init__(self, model: MagicNet, ensemble_data_points=500, rolling_window=None):
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
        self.model: MagicNet = pickle.loads(pickle.dumps(model))
        self._previous_data_points = None
        self.metrics = None
        self.no_preparation = False
        self.selected = None
        self.models = []
        self.rolling_window = rolling_window
        self.ensemble_predictions = {}
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
        for i, m in enumerate(self.models):
            pred = m.predict_one(
                x,
                previous_data_points=self._previous_data_points,
            )
            if pred is None:
                pred = 0
            pred = int(pred)
            self.predictions[timestamp].append(pred)
            self.ensemble_predictions[m.manager.curr_task_idx].append(pred)
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
        y = int(y)
        if timestamp in self.predictions:
            for i in range(len(self.predictions[timestamp])):
                self.metrics[i] = self.metrics[i].update(
                    y, self.predictions[timestamp][i]
                )
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
        model_copy = pickle.loads(pickle.dumps(self.model))
        if model_copy.manager.in_expansion:
            model_copy.manager.force_decision()
        model_copy.manager.in_grace_period = False
        model_copy.manager.in_expansion = False
        for task_id in range(1, self.model.manager.curr_task_idx + 1):
            task_model = pickle.loads(pickle.dumps(model_copy))
            if task_id in task_model.manager.forgotten_models:
                task_model.manager.model = pickle.loads(
                    pickle.dumps(task_model.manager.forgotten_models[task_id])
                )
                task_model.manager.model.eval()
            else:
                task_model.manager.model.eval()
                if task_id in task_model.manager.piggymask_list:
                    historical_mask = pickle.loads(
                        pickle.dumps(task_model.manager.piggymask_list[task_id])
                    )
                    task_model.manager.model.reinit_piggymask(
                        mask_init="random", masks=historical_mask
                    )
                if (
                    not task_model.manager.multi_head
                    and task_id in task_model.manager.biases
                ):
                    historical_bias = pickle.loads(
                        pickle.dumps(task_model.manager.biases[task_id])
                    )
                    if historical_bias is not None:
                        # Lo ricolleghiamo forzatamente come parametro sul device corretto
                        task_model.manager.model.classifier[
                            1
                        ].bias = torch.nn.Parameter(
                            historical_bias.clone().to(task_model.manager.model.device)
                        )
            task_model.manager.curr_task_idx = task_id
            models.append(task_model)
        return models

    def initialize(self):
        self.predictions = {}

        self.models = self.prepare_task_models()
        if self.rolling_window is not None:
            self.metrics = [
                river_utils.Rolling(metrics.CohenKappa, window_size=self.rolling_window)
                for _ in range(len(self.models))
            ]
        else:
            self.metrics = [metrics.CohenKappa() for _ in range(len(self.models))]
        self.selected = len(self.models) - 1
        for m in self.models:
            self.ensemble_predictions[m.manager.curr_task_idx] = []
        self.count = 0

    def reset_previous_data_points(self):
        for m in self.models:
            m.reset_previous_data_points()
        self._previous_data_points = None
        self.predictions = {}
        self.initialize()
