import pandas as pd
import numpy as np


class DetectorSimulator:
    """
    It simulates a detector with a specific precision and recall
    """

    def __init__(
        self,
        detections=None,
        precision=None,
        recall=None,
        dataset=None,
        min_delay=0,
        max_delay=128 * 50,
        margin=None,
    ):
        """

        Parameters
        ----------
        detections: list
            If not None, represents the timestamps of the detections.
        precision: float
            If detections is None, it represents the desired precision of the detector.
        recall: float
            If detections is None, it represents the desired recall of the detector.
        dataset: str or path
            If detections is None, it represents the path containing the csv representing the entire data stream.
        min_delay: int, default: 0
            If detections is None, it represents the minimum number of data points following a true drift after
             which to generate the true positives.
        max_delay: int, default: 128*50
            If detections is None, it represents the maximum number of data points following a true drift after
            which to generate the true positives.
        margin: int, default: None
            If detections is None, it represents the additional data points following a real drift and the max_delay
            from which to generate the false positives. If None it considers max_delay * 2.
        """
        self.cont = 0
        if detections is not None:
            self.drifts = detections
            return
        precision = min(1.0, precision)
        recall = min(1.0, recall)
        self.precision = precision
        self.recall = recall
        self.min_delay = min_delay
        self.max_delay = max_delay
        if margin is None:
            self.margin = self.max_delay * 2
        else:
            self.margin = margin

        self.dataset_len = 0
        df = pd.read_csv(f"{dataset}.csv")
        self.drifts_real = sorted(
            [df[df["task"] == i].iloc[0].name for i in list(df["task"].unique())[1:]]
        )
        self.dataset_len = len(df)
        self.correct_drifts_ranges = [
            (d + self.min_delay, d + self.max_delay) for d in self.drifts_real
        ]

        self.wrong_drifts_ranges = (
            [(self.max_delay + self.margin, self.drifts_real[0])]
            + [
                (
                    self.drifts_real[i] + self.max_delay + self.margin,
                    self.drifts_real[i + 1],
                )
                for i in range(len(self.drifts_real) - 1)
            ]
            + [(self.drifts_real[-1] + self.max_delay + self.margin, self.dataset_len)]
        )
        self.true_positives = round(self.recall * len(self.drifts_real))
        self.detections = round(self.true_positives / self.precision)
        self.false_positives = self.detections - self.true_positives
        self.drifts = []
        correct_drifts_ranges_to_choose = self.correct_drifts_ranges.copy()
        for _ in range(self.true_positives):
            i = np.random.choice(np.arange(0, len(correct_drifts_ranges_to_choose)))
            d = correct_drifts_ranges_to_choose[i]
            correct_drifts_ranges_to_choose.pop(i)
            self.drifts.append(np.random.randint(d[0], d[1]))
            if len(correct_drifts_ranges_to_choose) == 0:
                correct_drifts_ranges_to_choose = self.correct_drifts_ranges.copy()
        wrong_drifts_ranges_to_choose = self.wrong_drifts_ranges.copy()
        for _ in range(self.false_positives):
            i = np.random.choice(np.arange(0, len(wrong_drifts_ranges_to_choose)))
            d = wrong_drifts_ranges_to_choose[i]
            wrong_drifts_ranges_to_choose.pop(i)
            self.drifts.append(np.random.randint(d[0], d[1]))
            if len(wrong_drifts_ranges_to_choose) == 0:
                wrong_drifts_ranges_to_choose = self.wrong_drifts_ranges.copy()
        self.drift_detected = False

    def update(self, *args, **kwargs):
        """
        Call this method after each data point to check for detections. It returns True in the case of a detection,
        False otherwise.
        """
        self.drift_detected = False
        if self.cont in self.drifts:
            self.drift_detected = True
        self.cont += 1
        return self.drift_detected
