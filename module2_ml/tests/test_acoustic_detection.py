
import json
import os

import numpy as np
import torch

from models.autoencoder import Autoencoder
from datasets.synthetic_data import generate_synthetic_data
from preprocessing.preprocess import SignalPreprocessor
from training.train_autoencoder import train_autoencoder
from detection.anomaly_score import calculate_anomaly_score
from detection.detector import calculate_threshold, classify_anomaly
from evaluation.evaluate_model import evaluate_model


# 1. Generate training and validation data
train_val_data, train_val_labels = generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="acoustic",
    random_state=42
)

normal_data = train_val_data[train_val_labels == 0]

train_data = normal_data[:640]
validation_data = normal_data[640:]

# 2. Generate independent test data
test_data, test_labels = generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="acoustic",
    random_state=44
)

# 3. Preprocess datasets
preprocessor = SignalPreprocessor()

train_processed = preprocessor.fit_transform(train_data)
validation_processed = preprocessor.transform(validation_data)
test_processed = preprocessor.transform(test_data)

# 4. Train Autoencoder
model = Autoencoder(input_size=100)

print("\n===== TRAINING ACOUSTIC AUTOENCODER =====")

model, history = train_autoencoder(
    model,
    train_processed,
    validation_data=validation_processed,
    epochs=50,
    batch_size=32,
    patience=5
)

# 5. Calculate validation reconstruction errors
model.eval()

with torch.no_grad():
    validation_reconstructed = model(validation_processed)

    validation_scores = calculate_anomaly_score(
        validation_processed,
        validation_reconstructed
    )

# 6. Calculate test reconstruction errors
with torch.no_grad():
    test_reconstructed = model(test_processed)

    test_scores = calculate_anomaly_score(
        test_processed,
        test_reconstructed
    )

validation_scores = validation_scores.cpu().numpy()
test_scores = test_scores.cpu().numpy()

# 7. Test different threshold percentiles
percentiles = [90, 95, 97, 99]

optimization_results = {}

print("\n===== ACOUSTIC THRESHOLD OPTIMIZATION =====")

for percentile in percentiles:

    threshold = calculate_threshold(
        validation_scores,
        percentile=percentile
    )

    predictions = classify_anomaly(
        test_scores,
        threshold
    )

    predicted_labels = np.array([
        1 if prediction == "Anomalous" else 0
        for prediction in predictions
    ])

    results = evaluate_model(
        test_labels,
        predicted_labels
    )

    optimization_results[str(percentile)] = {
        "threshold": threshold,
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "false_positive_rate": results["false_positive_rate"],
        "false_negative_rate": results["false_negative_rate"],
        "confusion_matrix": np.asarray(
            results["confusion_matrix"]
        ).tolist()
    }

    print(f"\nThreshold Percentile: {percentile}")
    print("Threshold Value:", threshold)
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1 Score:", results["f1_score"])
    print("False Positive Rate:", results["false_positive_rate"])
    print("False Negative Rate:", results["false_negative_rate"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

# 8. Save optimization results
results_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation",
    "acoustic_threshold_results.json"
)

os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w") as file:
    json.dump(optimization_results, file, indent=4)

print("\nAll acoustic threshold results saved successfully!")
