from models.gin.GIN import GIN
import numpy as np
import torch
from river import metrics
import copy
from models.gin.cgru_from_masks import cGRUFromMasks


class InferenceGIN:
    def __init__(self, model: GIN, ensemble_data_points=128 * 2, quantize=False):
        """
        It implements a wrapper on a cPNN model to perform inference when the task label is not known.
        It builds an ensemble that considers all the columns of a given cPNN model. On the i-th data point of the test
        set,it considers the prediction made by the best-performing model from the first data point of the test set
        to the (i-1)-th.

        Parameters
        ----------
        model: cPNN.
            The cpNN model.
        ensemble_data_points: int, default: 128*2.
            Number of data points after which to choose the best model in the ensemble during the inference mode.
            Use -1 to keep the ensemble during the entire inference phase.
        """
        self.model: GIN = model
        self.inference_model = copy.deepcopy(model)
        self._previous_data_points = None
        self.metrics = None
        self.no_preparation = False
        self.selected = None
        self.models = []
        self.reset_previous_data_points()
        self.ensemble_data_points = ensemble_data_points
        self.count = 0
        self.predictions = {}
        self.quantize = quantize

    def predict_one(self, x, timestamp=-1):
        """
        It performs prediction on a single data point. It returns the prediction of the current best-performing column
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
            self.inference_model.manager.model = m
            self.predictions[timestamp].append(
                self.inference_model.predict_one(
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
        It updates the best-performing column using the real label. Call this method after predict_one on the same
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

    def prepare_masks(self):
        sigmoid = torch.nn.Sigmoid()
        conf = {torch.nn.GRU, torch.nn.Linear}
        models = []
        if len(self.model.manager.mask_list) == 1:
            return [self.model.manager.model]
        weights = {}
        newmasks = {}
        for name, mask in self.model.manager.mask_list[1].items():
            size = mask.size()
            if name == "weight":
                module = 1
            else:
                module = 0
            if len(size) == 1:
                weights[name] = getattr(
                    self.model.manager.model.classifier[module], name
                )[: size[0]]
            else:
                weights[name] = getattr(
                    self.model.manager.model.classifier[module], name
                )[: size[0], : size[1]]
            newmasks[name] = sigmoid(mask)
        cGRU = cGRUFromMasks(
            weights=weights,
            masks=newmasks,
            bias=self.model.manager.biases[0],
            device=self.model.device,
        )
        if self.quantize:
            cGRU = torch.quantization.quantize_dynamic(cGRU, conf, dtype=torch.qint8)
        models.append(cGRU)
        weights = {}
        newmasks = {}
        for biases, masks in zip(
            self.model.manager.biases[1:], self.model.manager.piggymask_list[1:]
        ):
            for name, mask in masks.items():
                name = name.replace("mask_real_", "")
                size = mask.size()
                if name == "weight":
                    module = 1
                else:
                    module = 0
                if len(size) == 1:
                    weights[name] = getattr(
                        self.model.manager.model.classifier[module], name
                    )[: size[0]]
                else:
                    weights[name] = getattr(
                        self.model.manager.model.classifier[module], name
                    )[: size[0], : size[1]]
                newmasks[name] = mask

            cGRU = cGRUFromMasks(
                weights=weights, masks=newmasks, bias=biases, device=self.model.device
            )
            if self.quantize:
                cGRU = torch.quantization.quantize_dynamic(cGRU, conf, dtype=torch.qint8)
            models.append(cGRU)
        models.append(self.model.manager.model)

        return models

    def initialize(self):
        self.predictions = {}

        self.models = self.prepare_masks()
        self.metrics = [
            metrics.CohenKappa() for _ in range(len(self.model.manager.mask_list))
        ]
        # self.masks = [mask for mask in range(len(self.model.manager.piggymask_list))]
        self.selected = len(self.models) - 1
        self.count = 0

    def reset_previous_data_points(self):
        self.inference_model.reset_previous_data_points()
        self._previous_data_points = None
        self.predictions = {}
        self.initialize()
