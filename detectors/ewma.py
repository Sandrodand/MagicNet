import numpy as np


class EWMA:
    def __init__(self, lambda_param: float = 0.2, arl0: int = 400, invert: bool = True):
        self.lambda_param = lambda_param
        self.arl0 = arl0
        self.errors = [[]]
        self.z_values = [[0.0]]
        self.p0_values = [[0.0]]
        self.l_values = [[]]
        self.thresholds = [[]]
        self.warn_thresholds = [[]]
        self.z = 0
        self.p0 = 0
        self.warnings = []
        self.drifts = []
        self.t = 0
        self.drift_detected = False
        self.warning_detected = False
        self.invert = invert

    def _return_l(self, p0):
        if self.lambda_param == 0.2:
            if self.arl0 == 100:
                return (
                    2.76
                    - 6.23 * p0
                    + 18.12 * (p0**3)
                    - 312.45 * (p0**5)
                    + 1002.18 * (p0**7)
                )
            if self.arl0 == 400:
                return (
                    3.97
                    - 6.56 * p0
                    + 48.73 * (p0**3)
                    - 330.13 * (p0**5)
                    + 848.18 * (p0**7)
                )
            if self.arl0 == 1000:
                return (
                    1.17
                    + 7.56 * p0
                    - 21.24 * (p0**3)
                    + 112.12 * (p0**5)
                    - 987.23 * (p0**7)
                )

    def update(self, e_t):
        if self.invert:
            e_t = 1 if e_t == 0 else 0
        self.errors[-1].append(e_t)
        self.t += 1
        p0 = self.t / (self.t + 1) * self.p0_values[-1][-1] + 1 / (self.t + 1) * e_t
        self.p0_values[-1].append(p0)
        sigma_e = np.sqrt(p0 * (1 - p0))
        sigma_z = (
            np.sqrt(
                self.lambda_param
                / (2 - self.lambda_param)
                * (1 - (1 - self.lambda_param) ** (2 * self.t))
            )
            * sigma_e
        )
        l_t = self._return_l(p0)
        self.l_values[-1].append(l_t)
        z_t = (1 - self.lambda_param) * self.z_values[-1][-1] + self.lambda_param * e_t
        self.z_values[-1].append(z_t)

        threshold = p0 + l_t * sigma_z
        self.thresholds[-1].append(threshold)
        if z_t > threshold:
            self.drift_detected = True
            self.drifts.append(self.t - 1)
            self.z_values.append([0.0])
            self.p0_values.append([0.0])
            self.l_values.append([])
            self.errors.append([])
            self.thresholds.append([])
            self.warn_thresholds.append([])
            self.t = 0
        else:
            self.drift_detected = False

        warn_threshold = p0 + 0.5 * l_t * sigma_z
        self.warn_thresholds[-1].append(warn_threshold)
        if z_t > warn_threshold:
            self.warning_detected = True
            self.warnings.append(self.t - 1)
        else:
            self.warning_detected = False

        return self.drift_detected
