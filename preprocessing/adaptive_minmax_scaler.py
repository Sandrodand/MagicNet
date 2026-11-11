from river import preprocessing


class AdaptiveMinMaxScaler:
    """Scale streaming feature dictionaries using their running minimum and maximum."""
    def __init__(self):
        """Initialize the online scaler with empty statistics."""
        self.scaler = preprocessing.MinMaxScaler()

    def preprocess(self, x):
        """Update scaling statistics with the current example and return its transformed features."""
        self.scaler.learn_one(x)
        return self.scaler.transform_one(x)

    def reset(self):
        """Discard the accumulated scaling statistics."""
        self.scaler = preprocessing.MinMaxScaler()
