from river import preprocessing


class AdaptiveStandardScaler:
    def __init__(self):
        self.scaler = preprocessing.AdaptiveStandardScaler()

    def preprocess(self, x):
        self.scaler.learn_one(x)
        return self.scaler.transform_one(x)

    def reset(self):
        self.scaler = preprocessing.AdaptiveStandardScaler()
