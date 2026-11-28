
import numpy as np

from preprocessing.preprocess import SignalPreprocessor

# Temporary sample signal features
sample_data = np.array([
    [10, 100, 1000],
    [20, 200, 2000],
    [30, 300, 3000],
    [40, 400, 4000],
    [50, 500, 5000]
], dtype=np.float32)

# Initialize preprocessor
preprocessor = SignalPreprocessor()

# Fit and transform sample data
processed_data = preprocessor.fit_transform(sample_data)

print("Original Data:")
print(sample_data)

print("\nProcessed Data:")
print(processed_data)

print("\nProcessed Shape:", processed_data.shape)
print("Data Type:", processed_data.dtype)

# Check whether features are approximately standardized
print("\nFeature Means:")
print(processed_data.mean(dim=0))

print("\nFeature Standard Deviations:")
print(processed_data.std(dim=0, unbiased=False))

print("\nPreprocessing test successful!")
