from abc import ABC
from river import base
from collections import deque
import numpy as np


class TemporallyAugmentedFeaturesClassifier(base.Classifier, ABC):
    def __init__(
        self,
        base_learner: base.Classifier = None,
        ta_order: int = 0,
    ):
        """

        Parameters
        ----------
        base_learner: base.Classifier.
            The base learner to which apply the temporal augmentation.
        ta_order: int, default: 0.
            The order of the temporal augmentation.
        """
        self._base_learner = base_learner
        self.ta_order = ta_order
        self._old_features_training = []
        self._old_features_inference = []

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        x_ext = self._extend_with_old_features(x, True)
        self._base_learner.learn_one(x_ext, y)
        self._update_past_features(x, True)
        return self._base_learner

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        x_ext = self._extend_with_old_features(x, False)
        y_hat = self._base_learner.predict_one(x_ext)
        self._update_past_features(x, False)
        return y_hat

    def predict_many(self, x_batch: list):
        return np.array([self.predict_one(item) for item in x_batch])

    def learn_many(self, x_batch: list, y_batch: list):
        for x_item, y_item in zip(x_batch, y_batch):
            self.learn_one(x_item, y_item)
        return self._base_learner

    def _update_past_features(self, x, training=True):
        if training:
            old_features = self._old_features_training
        else:
            old_features = self._old_features_inference
        old_features.append(x)
        if len(old_features) > self.ta_order:
            old_features = old_features[1:]
        if training:
            self._old_features_training = old_features
        else:
            self._old_features_inference = old_features

    def _extend_with_old_features(self, x, training=True):
        x_ext = x.copy()
        if training:
            old_features = self._old_features_training
        else:
            old_features = self._old_features_inference
        l = len(old_features)
        for i in range(0, l):
            j = l - 1 - i
            for k in old_features[j]:
                x_ext[f"{k}_{i}"] = old_features[j][k]
        return x_ext

    def reset_previous_data_points(self):
        self._old_features_training = []
        self._old_features_inference = []
