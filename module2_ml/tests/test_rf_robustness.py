
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


# 1. Generate normal training and validation data
train_val_data, train_val_labels = generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="rf",
    random_state=42
)

normal_data = train_val_data[train_val_labels == 0]

train_data = normal_data[:640]
validation_data = normal_data[640:]

# 2. Preprocess normal training and validation data
preprocessor = SignalPreprocessor()

train_processed = preprocessor.fit_transform(train_data)
validation_processed = preprocessor.transform(validation_data)

# 3. Train Autoencoder
model = Autoencoder(input_size=100)

print("\n===== TRAINING RF ROBUSTNESS MODEL =====")

model, history = train_autoencoder(
    model,
    train_processed,
    validation_data=validation_processed,
    epochs=50,
    batch_size=32,
    patience=5
)

# 4. Calculate threshold from normal validation data
model.eval()

with torch.no_grad():
    validation_reconstructed = model(validation_processed)

    validation_scores = calculate_anomaly_score(
        validation_processed,
        validation_reconstructed
    )

threshold = calculate_threshold(
    validation_scores.cpu().numpy(),
    percentile=99
)

print("\nDetection Threshold:", threshold)

# 5. Test different anomaly strengths
anomaly_strengths = [0.5, 1.0, 2.0, 4.0]

robustness_results = {}

print("\n===== RF ANOMALY STRENGTH ROBUSTNESS TEST =====")

for strength in anomaly_strengths:

    test_data, test_labels = generate_synthetic_data(
        samples=1000,
        features=100,
        anomaly_ratio=0.2,
        signal_type="rf",
        random_state=100,
        anomaly_strength=strength
    )

    test_processed = preprocessor.transform(test_data)

    with torch.no_grad():
        reconstructed = model(test_processed)

        test_scores = calculate_anomaly_score(
            test_processed,
            reconstructed
        )

    predictions = classify_anomaly(
        test_scores.cpu().numpy(),
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

    robustness_results[str(strength)] = {
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

    print(f"\nAnomaly Strength: {strength}")
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1 Score:", results["f1_score"])
    print("False Positive Rate:", results["false_positive_rate"])
    print("False Negative Rate:", results["false_negative_rate"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

# 6. Save robustness results
results_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation",
    "rf_robustness_results.json"
)

os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w") as file:
    json.dump(robustness_results, file, indent=4)

print("\nRF robustness results saved successfully!")
