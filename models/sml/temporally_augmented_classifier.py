from abc import ABC
from river import base
from collections import deque
import numpy as np


class TemporallyAugmentedClassifier(base.Classifier, ABC):
    """Augment an online classifier's inputs with past labels or predictions."""
    def __init__(
        self,
        base_learner: base.Classifier = None,
        num_old_labels: int = 0,
        use_predictions: str = "",
    ):
        """

        Parameters
        ----------
        base_learner: base.Classifier.
            The base learner to which apply the temporal augmentation.
        num_old_labels: int, default: 0.
            The number of old labels to use as temporal augmentation.
        use_predictions: str, default: "".
            Use past predictions in training and inference for "train",
            "train_test", or "both". Other values use past true labels.
        """
        self._base_learner = base_learner
        self.ta_order = num_old_labels
        self._old_labels = deque([0] * self.ta_order)
        self._old_predictions = deque([0] * self.ta_order)
        use_predictions = use_predictions.lower()
        if use_predictions == "test":
            self._use_predictions_train = False
            self._use_predictions_test = True
        if (
            use_predictions == "train"
            or use_predictions == "train_test"
            or use_predictions == "both"
        ):
            self._use_predictions_train = True
            self._use_predictions_test = True
        else:
            self._use_predictions_train = False
            self._use_predictions_test = False
        self._y_hat = 0

    def set_use_predictions(self, use_predictions: str):
        """Use past predictions for train, train_test, or both, and past labels otherwise."""
        use_predictions = use_predictions.lower()
        if use_predictions == "test":
            self._use_predictions_train = False
            self._use_predictions_test = True
        if (
            use_predictions == "train"
            or use_predictions == "train_test"
            or use_predictions == "both"
        ):
            self._use_predictions_train = True
            self._use_predictions_test = True
        else:
            self._use_predictions_train = False
            self._use_predictions_test = False

    def get_use_predictions(self):
        """Return the training and inference prediction-history flags."""
        return self._use_predictions_train, self._use_predictions_test

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        """Train on one example with label or prediction history and return the base learner."""
        x = self._extend_with_old_labels(x, use_predictions=self._use_predictions_train)
        self._base_learner.learn_one(x, y)
        if not self._use_predictions_train:
            self._update_past_labels(y)
        else:
            self._update_past_predictions(self._y_hat)
        return self._base_learner

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        """Predict one label using the configured history."""
        x = self._extend_with_old_labels(x, use_predictions=self._use_predictions_test)
        self._y_hat = self._base_learner.predict_one(x)
        if self._use_predictions_test and not self._use_predictions_train:
            self._update_past_predictions(self._y_hat)
        return self._y_hat

    def update_inference(self, y: base.typing.ClfTarget, **kwargs):
        """Record a true label for subsequent inference."""
        self._update_past_labels(y)

    def reset_previous_data_points(self):
        """Reset both label and prediction histories to zeros."""
        self._old_labels = deque([0] * self.ta_order)
        self._old_predictions = deque([0] * self.ta_order)

    def predict_many(self, x_batch: list):
        """Predict each example in stream order and return a NumPy array."""
        return np.array([self.predict_one(item) for item in x_batch])

    def learn_many(self, x_batch: list, y_batch: list):
        """Train on feature-label pairs in stream order and return the base learner."""
        for x_item, y_item in zip(x_batch, y_batch):
            self.learn_one(x_item, y_item)
        return self._base_learner

    def _update_past_labels(self, y):
        if y is None:
            y = 0
        self._old_labels.append(y)
        self._old_labels.popleft()

    def _update_past_predictions(self, y_hat):
        if y_hat is None:
            y_hat = 0
        self._old_predictions.append(y_hat)
        self._old_predictions.popleft()

    def _extend_with_old_labels(self, x, use_predictions=False):
        x_ext = x.copy()
        if not use_predictions:
            old_labels = self._old_labels
        else:
            old_labels = self._old_predictions
        ext = range(len(x_ext.keys()), len(x_ext.keys()) + self.ta_order)
        for el, old_label in zip(ext, list(old_labels)):
            # check on type of keys, if string or int
            if isinstance(list(x_ext.keys())[0], type("str")):
                x_ext[str(el)] = old_label
            else:
                x_ext[el] = old_label
        return x_ext

    def reset_previous_data_points(self):
        """Reset both label and prediction histories to zeros."""
        self._old_labels = deque([0] * self.ta_order)
        self._old_predictions = deque([0] * self.ta_order)
