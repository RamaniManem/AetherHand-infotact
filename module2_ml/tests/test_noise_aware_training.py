
import json
import os

import numpy as np
import torch

from datasets.synthetic_data import generate_synthetic_data
from models.autoencoder import Autoencoder
from preprocessing.preprocess import SignalPreprocessor
from training.train_autoencoder import train_autoencoder
from detection.anomaly_score import calculate_anomaly_score
from detection.detector import calculate_threshold, classify_anomaly
from evaluation.evaluate_model import evaluate_model


NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0]
TRAINING_NOISE_LEVELS = [0.5, 1.0, 1.5]


def generate_normal_data(samples, noise_level, random_state):
    data, labels = generate_synthetic_data(
        samples=samples,
        features=100,
        anomaly_ratio=0.0,
        signal_type="rf",
        random_state=random_state,
        anomaly_strength=2.0,
        noise_level=noise_level
    )

    return data[labels == 0]


def prepare_baseline_data():
    # Model A: all training and validation data use noise level 1.0.
    data = generate_normal_data(
        samples=900,
        noise_level=1.0,
        random_state=10
    )

    return data[:720], data[720:]


def prepare_noise_aware_data():
    # Model B: equal amounts of data from three noise levels.
    training_parts = []
    validation_parts = []

    for index, noise in enumerate(TRAINING_NOISE_LEVELS):
        data = generate_normal_data(
            samples=300,
            noise_level=noise,
            random_state=20 + index
        )

        training_parts.append(data[:240])
        validation_parts.append(data[240:])

    train_data = np.vstack(training_parts)
    validation_data = np.vstack(validation_parts)

    # Shuffle the combined datasets.
    rng = np.random.default_rng(42)

    train_data = train_data[rng.permutation(len(train_data))]
    validation_data = validation_data[
        rng.permutation(len(validation_data))
    ]

    return train_data, validation_data


def train_and_calibrate(train_data, validation_data, model_name):
    print(f"\nTraining {model_name}...")

    # Fit scaler only on training data.
    preprocessor = SignalPreprocessor()
    preprocessor.fit(train_data)

    processed_train = preprocessor.transform(train_data)
    processed_validation = preprocessor.transform(validation_data)

    torch.manual_seed(42)

    model = Autoencoder(input_size=100)

    train_autoencoder(
        model,
        processed_train,
        validation_data=processed_validation,
        epochs=50,
        batch_size=32,
        patience=5
    )

    model.eval()

    # Calculate threshold using normal validation samples only.
    with torch.no_grad():
        reconstructed = model(processed_validation)

        validation_scores = calculate_anomaly_score(
            processed_validation,
            reconstructed
        ).cpu().numpy()

    threshold = calculate_threshold(
        validation_scores,
        percentile=99
    )

    print(f"{model_name} threshold: {threshold:.6f}")

    return model, preprocessor, threshold


def evaluate_on_noise_levels(model, preprocessor, threshold, model_name):
    results = {}

    for index, noise in enumerate(NOISE_LEVELS):
        print(f"\nTesting {model_name} at noise level {noise}...")

        test_data, test_labels = generate_synthetic_data(
            samples=1000,
            features=100,
            anomaly_ratio=0.2,
            signal_type="rf",
            random_state=100,
            anomaly_strength=2.0,
            noise_level=noise
        )

        processed_test = preprocessor.transform(test_data)

        with torch.no_grad():
            reconstructed = model(processed_test)

            scores = calculate_anomaly_score(
                processed_test,
                reconstructed
            ).cpu().numpy()

        predictions = classify_anomaly(scores, threshold)

        predicted_labels = np.array([
            1 if prediction == "Anomalous" else 0
            for prediction in predictions
        ])

        metrics = evaluate_model(test_labels, predicted_labels)

        results[str(noise)] = {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
            "false_positive_rate": float(
                metrics["false_positive_rate"]
            ),
            "false_negative_rate": float(
                metrics["false_negative_rate"]
            ),
            "confusion_matrix": np.asarray(
                metrics["confusion_matrix"]
            ).tolist()
        }

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"FPR: {metrics['false_positive_rate']:.4f}"
        )

    return results


def main():
    print("=" * 60)
    print("STAGE 7: NOISE-AWARE AUTOENCODER EXPERIMENT")
    print("=" * 60)

    # Prepare equal-sized training and validation datasets.
    baseline_train, baseline_validation = prepare_baseline_data()

    aware_train, aware_validation = prepare_noise_aware_data()

    print("\nDataset sizes:")
    print(f"Baseline training: {len(baseline_train)}")
    print(f"Baseline validation: {len(baseline_validation)}")
    print(f"Noise-aware training: {len(aware_train)}")
    print(f"Noise-aware validation: {len(aware_validation)}")

    # Train Model A.
    baseline_model, baseline_preprocessor, baseline_threshold = (
        train_and_calibrate(
            baseline_train,
            baseline_validation,
            "Model A - Baseline"
        )
    )

    # Train Model B.
    aware_model, aware_preprocessor, aware_threshold = (
        train_and_calibrate(
            aware_train,
            aware_validation,
            "Model B - Noise-Aware"
        )
    )

    # Evaluate both models on identical test conditions.
    baseline_results = evaluate_on_noise_levels(
        baseline_model,
        baseline_preprocessor,
        baseline_threshold,
        "Model A - Baseline"
    )

    aware_results = evaluate_on_noise_levels(
        aware_model,
        aware_preprocessor,
        aware_threshold,
        "Model B - Noise-Aware"
    )

    output = {
        "experiment": "Noise-Aware Autoencoder Comparison",
        "training_noise_levels": TRAINING_NOISE_LEVELS,
        "test_noise_levels": NOISE_LEVELS,
        "anomaly_strength": 2.0,
        "threshold_percentile": 99,
        "baseline_threshold": baseline_threshold,
        "noise_aware_threshold": aware_threshold,
        "baseline": baseline_results,
        "noise_aware": aware_results
    }

    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "evaluation",
        "noise_aware_results.json"
    )

    output_path = os.path.abspath(output_path)

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
