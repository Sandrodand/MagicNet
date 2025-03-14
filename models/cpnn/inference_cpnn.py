from models.cpnn.cpnn import cPNN
import numpy as np
from river import metrics


class InferenceCPNN:
    def __init__(self, model: cPNN, ensemble_data_points=128 * 2):
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
        self.model: cPNN = model
        self._previous_data_points = None
        self.metrics = None
        self.selected = None
        self.reset_previous_data_points()
        self.ensemble_data_points = ensemble_data_points
        self.columns = []
        self.count = 0
        self.predictions = {}

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
        for col in self.columns:
            self.predictions[timestamp].append(
                self.model.predict_one(
                    x,
                    column_id=col,
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
            self.columns = [self.columns[self.selected]]
            self.metrics = [self.metrics[self.selected]]
            self.selected = 0
            self.predictions = {}

    def initialize(self):
        self.predictions = {}
        self.metrics = [
            metrics.CohenKappa() for _ in range(len(self.model.columns.columns))
        ]
        self.columns = [col for col in range(len(self.model.columns.columns))]
        self.selected = len(self.columns) - 1
        self.count = 0

    def reset_previous_data_points(self):
        self.model.reset_previous_data_points()
        self._previous_data_points = None
        self.predictions = {}
        self.initialize()
