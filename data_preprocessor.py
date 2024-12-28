import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def normalize(self):
        self.data = (self.data - np.mean(self.data, axis=0)) / np.std(self.data, axis=0)
        return self.data

    def get_features_and_labels(self):
        X = self.data[:, :-1]
        y = self.data[:, -1]
        return X, y