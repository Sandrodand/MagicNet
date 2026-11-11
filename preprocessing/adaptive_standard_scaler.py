from river import preprocessing


class AdaptiveStandardScaler:
    """Scale streaming feature dictionaries using their adaptive mean and variance."""
    def __init__(self):
        """Initialize the online scaler with empty statistics."""
        self.scaler = preprocessing.AdaptiveStandardScaler()

    def preprocess(self, x):
        """Update scaling statistics with the current example and return its transformed features."""
        self.scaler.learn_one(x)
        return self.scaler.transform_one(x)

    def reset(self):
        """Discard the accumulated scaling statistics."""
        self.scaler = preprocessing.AdaptiveStandardScaler()
