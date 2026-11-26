
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


OLD_NOISE_LEVELS = [0.5, 1.0, 1.5]
EXTENDED_NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0]
TEST_NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0]


def generate_normal_data(samples, noise, seed):
    data, labels = generate_synthetic_data(
        samples=samples,
        features=100,
        anomaly_ratio=0.0,
        signal_type="rf",
        random_state=seed,
        anomaly_strength=2.0,
        noise_level=noise
    )

    return data[labels == 0]


def prepare_datasets(noise_levels, samples_per_level, train_per_level, seed_start):
    training_parts = []
    validation_parts = {}

    for index, noise in enumerate(noise_levels):
        data = generate_normal_data(
            samples=samples_per_level,
            noise=noise,
            seed=seed_start + index
        )

        training_parts.append(data[:train_per_level])
        validation_parts[noise] = data[train_per_level:]

    train_data = np.vstack(training_parts)

    rng = np.random.default_rng(42)
    train_data = train_data[rng.permutation(len(train_data))]

    return train_data, validation_parts


def get_scores(model, data):
    with torch.no_grad():
        reconstructed = model(data)
        scores = calculate_anomaly_score(data, reconstructed)

    return scores.cpu().numpy()


def train_and_calibrate(name, noise_levels, samples_per_level,
                        train_per_level, seed_start):

    print(f"\n{'=' * 60}")
    print(f"Training {name}")
    print(f"{'=' * 60}")

    train_data, validation_parts = prepare_datasets(
        noise_levels,
        samples_per_level,
        train_per_level,
        seed_start
    )

    preprocessor = SignalPreprocessor()
    preprocessor.fit(train_data)

    processed_train = preprocessor.transform(train_data)

    processed_validation = {
        noise: preprocessor.transform(data)
        for noise, data in validation_parts.items()
    }

    combined_validation = torch.cat(
        list(processed_validation.values()),
        dim=0
    )

    torch.manual_seed(42)
    model = Autoencoder(input_size=100)

    train_autoencoder(
        model,
        processed_train,
        validation_data=combined_validation,
        epochs=50,
        batch_size=32,
        patience=5
    )

    model.eval()

    thresholds = {}

    print("\nCalibrating thresholds:")

    for noise, validation_data in processed_validation.items():
        scores = get_scores(model, validation_data)

        threshold = calculate_threshold(
            scores,
            percentile=99
        )

        thresholds[str(noise)] = threshold

        print(f"Noise {noise}: {threshold:.6f}")

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Validation samples: {len(combined_validation)}")

    return model, preprocessor, thresholds


def estimate_noise_level(data):
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("Input must be a 2D batch with at least 2 samples")

    if not np.isfinite(data).all():
        raise ValueError("Input contains invalid values")

    feature_medians = np.median(data, axis=0)

    deviations = np.abs(data - feature_medians)

    feature_mads = np.median(deviations, axis=0)

    feature_noise = feature_mads / 0.67448975

    return float(np.median(feature_noise))


def evaluate_configuration(name, model, preprocessor,
                           thresholds, calibration_levels):

    results = {}

    print(f"\n{'=' * 60}")
    print(f"Evaluating {name}")
    print(f"{'=' * 60}")

    for noise in TEST_NOISE_LEVELS:

        test_data, test_labels = generate_synthetic_data(
            samples=1000,
            features=100,
            anomaly_ratio=0.2,
            signal_type="rf",
            random_state=100,
            anomaly_strength=2.0,
            noise_level=noise
        )

        estimated_noise = estimate_noise_level(test_data)

        selected_noise = min(
            calibration_levels,
            key=lambda level: abs(level - estimated_noise)
        )

        selected_threshold = thresholds[str(selected_noise)]

        processed_test = preprocessor.transform(test_data)

        scores = get_scores(model, processed_test)

        predictions = classify_anomaly(
            scores,
            selected_threshold
        )

        predicted_labels = np.array([
            1 if item == "Anomalous" else 0
            for item in predictions
        ])

        metrics = evaluate_model(
            test_labels,
            predicted_labels
        )

        results[str(noise)] = {
            "actual_noise": noise,
            "estimated_noise": estimated_noise,
            "selected_noise": selected_noise,
            "threshold": selected_threshold,
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

        print(f"\nActual noise: {noise}")
        print(f"Estimated noise: {estimated_noise:.4f}")
        print(f"Selected calibration: {selected_noise}")
        print(f"Threshold: {selected_threshold:.6f}")

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
    print("STAGE 10: EXTENDED NOISE CALIBRATION")
    print("=" * 60)

    # Model A: Original three-level calibration.
    old_model, old_preprocessor, old_thresholds = (
        train_and_calibrate(
            "Model A - Three Noise Levels",
            OLD_NOISE_LEVELS,
            samples_per_level=300,
            train_per_level=240,
            seed_start=20
        )
    )

    # Model B: Extended four-level calibration.
    extended_model, extended_preprocessor, extended_thresholds = (
        train_and_calibrate(
            "Model B - Four Noise Levels",
            EXTENDED_NOISE_LEVELS,
            samples_per_level=225,
            train_per_level=180,
            seed_start=30
        )
    )

    # Evaluate both configurations on identical test data.
    old_results = evaluate_configuration(
        "Model A - Three Noise Levels",
        old_model,
        old_preprocessor,
        old_thresholds,
        OLD_NOISE_LEVELS
    )

    extended_results = evaluate_configuration(
        "Model B - Four Noise Levels",
        extended_model,
        extended_preprocessor,
        extended_thresholds,
        EXTENDED_NOISE_LEVELS
    )

    output = {
        "experiment": "Three-Level vs Four-Level Noise Calibration",
        "test_noise_levels": TEST_NOISE_LEVELS,
        "anomaly_strength": 2.0,
        "threshold_percentile": 99,
        "model_a": {
            "calibration_levels": OLD_NOISE_LEVELS,
            "thresholds": old_thresholds,
            "results": old_results
        },
        "model_b": {
            "calibration_levels": EXTENDED_NOISE_LEVELS,
            "thresholds": extended_thresholds,
            "results": extended_results
        }
    }

    output_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "extended_noise_calibration_results.json"
        )
    )

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 60)
    print("STAGE 10 COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
