
import numpy as np
import torch

from datasets.synthetic_data import generate_synthetic_data
from models.autoencoder import Autoencoder
from preprocessing.preprocess import SignalPreprocessor
from training.train_autoencoder import train_autoencoder
from detection.anomaly_score import calculate_anomaly_score
from detection.detector import calculate_threshold, classify_anomaly
from evaluation.evaluate_model import evaluate_model


torch.manual_seed(42)

# Generate synthetic RF dataset
data, labels = generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="rf",
    random_state=42
)

# Separate normal samples
normal_data = data[labels == 0]

# Split normal data into training and validation
train_data = normal_data[:640]
validation_data = normal_data[640:]

# Fit preprocessing using training data only
preprocessor = SignalPreprocessor()

train_tensor = preprocessor.fit_transform(train_data)
validation_tensor = preprocessor.transform(validation_data)

# Create Autoencoder
model = Autoencoder(input_size=100)

print("\n===== TRAINING AUTOENCODER =====")

model, history = train_autoencoder(
    model,
    train_tensor,
    validation_data=validation_tensor,
    epochs=50,
    batch_size=32,
    patience=5
)

# Calculate threshold from normal validation scores
model.eval()

with torch.no_grad():
    validation_reconstructed = model(validation_tensor)

    validation_scores = calculate_anomaly_score(
        validation_tensor,
        validation_reconstructed
    ).numpy()

threshold = calculate_threshold(
    validation_scores,
    percentile=95
)

print("\nDetection Threshold:", threshold)

# Prepare all test samples
test_data = data[labels == 0]
test_anomalies = data[labels == 1]

test_samples = np.vstack((test_data, test_anomalies))
actual_labels = np.concatenate((
    np.zeros(len(test_data), dtype=int),
    np.ones(len(test_anomalies), dtype=int)
))

processed_test = preprocessor.transform(test_samples)

# Generate predictions
with torch.no_grad():
    reconstructed = model(processed_test)

    test_scores = calculate_anomaly_score(
        processed_test,
        reconstructed
    ).numpy()

predictions = classify_anomaly(
    test_scores,
    threshold
)

predicted_labels = np.array([
    1 if result == "Anomalous" else 0
    for result in predictions
])

# Evaluate detection
results = evaluate_model(
    actual_labels,
    predicted_labels
)

print("\n===== SYNTHETIC RF DETECTION RESULTS =====")
print("Accuracy:", round(results["accuracy"], 4))
print("Precision:", round(results["precision"], 4))
print("Recall:", round(results["recall"], 4))
print("F1 Score:", round(results["f1_score"], 4))

print("\nConfusion Matrix:")

print(results["confusion_matrix"])

print(
    "False Positive Rate:",
    round(results["false_positive_rate"], 4)
)

print(
    "False Negative Rate:",
    round(results["false_negative_rate"], 4)
)

print("\nSynthetic RF detection test completed!")
