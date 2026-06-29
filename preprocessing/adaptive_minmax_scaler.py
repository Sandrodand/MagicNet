from river import preprocessing


class AdaptiveMinMaxScaler:
    def __init__(self):
        self.scaler = preprocessing.MinMaxScaler()

    def preprocess(self, x):
        self.scaler.learn_one(x)
        return self.scaler.transform_one(x)

    def reset(self):
        self.scaler = preprocessing.MinMaxScaler()
