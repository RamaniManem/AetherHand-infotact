
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class SignalPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, data):

        data = np.asarray(data, dtype=np.float32)

        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("Data must be a non-empty 2D array")

        if not np.isfinite(data).all():
            raise ValueError("Training data contains missing or invalid values")

        self.scaler.fit(data)
        self.is_fitted = True

        return self

    def transform(self, data):

        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        data = np.asarray(data, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError("Data must be a 2D array")

        if not np.isfinite(data).all():
            raise ValueError("Input data contains missing or invalid values")

        processed = self.scaler.transform(data)

        return torch.tensor(processed, dtype=torch.float32)

    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)
