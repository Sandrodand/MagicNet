from audioop import error

from river import forest
import pickle
from river.drift import ADWIN


class Sentinel:
    """
    It implements the DYNcPNN Sentinel
    """

    def __init__(
        self,
        error_rate_monitor,
        evaluator=None,
        numeric_evaluator=False,
        training_data_points=-1,
    ):
        """
        Parameters
        ----------
        error_rate_monitor: default: None
            The error rate monitor to use. If None, it uses ADWIN with delta=0.002 and clock=1.
        evaluator: default: None
            The evaluator to use. If None, it uses a forest of Hoeffding Trees.
        numeric_evaluator: bool, default: False.
            If True the evaluator requires an array as input, otherwise it requires a dictionary (like ARF).
        training_data_points: int, default: -1.
            Number of training data points during which the evaluator is trained. In this case the model is trained
            only on the first training_data_points of the concept and then run only in inference mode.
            If -1 the evaluator is trained continuously on all the data points in test then train mode.
        """
        self.count = 0
        self.drift_detected = False
        if error_rate_monitor is None:
            error_rate_monitor = ADWIN(delta=0.002, clock=1)
        self.error_rate_monitor = pickle.loads(pickle.dumps(error_rate_monitor))
        self.error_rate_monitor_copy = pickle.loads(pickle.dumps(error_rate_monitor))
        self.numeric_evaluator = numeric_evaluator
        if evaluator is None:
            self.evaluator = create_arf_no_adwin()
        else:
            self.evaluator = pickle.loads(pickle.dumps(evaluator))
        self.model_copy = pickle.loads(pickle.dumps(evaluator))
        self.training_data_points = training_data_points

    def update(self, x, y):
        """
        It updates the Sentinel on the current data point Xt, yt and returns True if it detects a possible drift.

        If training_data_points == -1, it first makes a prediction on Xt and then trains on Xt and yt.
        It uses the prediction and the real value yt to update the error_rate_monitor and returns True if it detects
        a possible drift.

        If training_data_points != -1, if the counter of the number of data points is less than training_data_points,
        it trains the model on Xt and yt without producing detections.
        When the counter is greater or equal than training_data_points it makes a prediction on Xt (without training
        after), compares it with yt, updates the error_rate_monitor and returns True if the error_rate_monitor detects
        a possible drift.

        Parameters
        ----------
        x: dictionary, list, or array
            The feature vector Xt of the current data point.
        y: int
            The real label associated with Xt.

        Returns
        -------
        drift_detected: True
            True if the Sentinel detects a drift after the update, False otherwise.
        """
        if not self.numeric_evaluator and type(x) != dict:
            x = {f"feat_{i}": x[i] for i in range(len(x))}
        if self.numeric_evaluator and type(x) == dict:
            x = list(x.values())
        if self.count >= self.training_data_points:
            y_hat = self.evaluator.predict_one(x)
            if y_hat is not None:
                self.error_rate_monitor.update(int(y == y_hat))
                self.drift_detected = self.error_rate_monitor.drift_detected
            else:
                self.drift_detected = False
        else:
            self.drift_detected = False
        if self.drift_detected:
            self.evaluator = pickle.loads(pickle.dumps(self.model_copy))
            self.count = 0
        if self.training_data_points == -1 or self.count < self.training_data_points:
            self.evaluator.learn_one(x, y)
        self.count += 1
        return self.drift_detected

    def reset(self):
        """
        After a detection call this method to reset the evaluator and the error rate monitor.
        """
        self.evaluator = pickle.loads(pickle.dumps(self.model_copy))
        self.error_rate_monitor = pickle.loads(
            pickle.dumps(self.error_rate_monitor_copy)
        )


def create_arf_no_adwin():
    """
    It creates a forest of Hoeffding Trees.

    Returns
    -------
    model: river.forest.ARFClassifier
    """
    return forest.ARFClassifier(
        leaf_prediction="nb", drift_detector=None, warning_detector=None
    )
