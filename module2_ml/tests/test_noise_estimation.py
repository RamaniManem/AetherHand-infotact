
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


CALIBRATION_NOISE_LEVELS = [0.5, 1.0, 1.5]
TEST_NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0]

MAD_NORMALIZER = 0.67448975


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


def prepare_training_data():
    training_parts = []
    validation_parts = {}

    for index, noise in enumerate(CALIBRATION_NOISE_LEVELS):
        data = generate_normal_data(
            samples=300,
            noise=noise,
            seed=20 + index
        )

        training_parts.append(data[:240])
        validation_parts[noise] = data[240:]

    train_data = np.vstack(training_parts)

    rng = np.random.default_rng(42)
    train_data = train_data[rng.permutation(len(train_data))]

    return train_data, validation_parts


def estimate_noise_level(data):
    """
    Estimate Gaussian noise using feature-wise MAD.

    The median across features reduces the influence of
    anomalous RF features affecting the first quarter of
    the feature vector.

    Input must be raw, unscaled feature data.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("Input must be a 2D batch with at least 2 samples")

    if not np.isfinite(data).all():
        raise ValueError("Input contains invalid values")

    feature_medians = np.median(data, axis=0)

    absolute_deviations = np.abs(
        data - feature_medians
    )

    feature_mads = np.median(
        absolute_deviations,
        axis=0
    )

    feature_noise_estimates = feature_mads / MAD_NORMALIZER

    estimated_noise = float(
        np.median(feature_noise_estimates)
    )

    return estimated_noise


def select_noise_condition(estimated_noise):
    """
    Select the nearest calibrated noise condition.
    """
    selected_noise = min(
        CALIBRATION_NOISE_LEVELS,
        key=lambda level: abs(level - estimated_noise)
    )

    return selected_noise


def get_anomaly_scores(model, data):
    with torch.no_grad():
        reconstructed = model(data)

        scores = calculate_anomaly_score(
            data,
            reconstructed
        )

    return scores.cpu().numpy()


def main():
    print("=" * 60)
    print("STAGE 9: AUTOMATIC NOISE-LEVEL ESTIMATION")
    print("=" * 60)

    # Prepare training and validation data.
    train_data, validation_parts = prepare_training_data()

    # Fit preprocessing only on training data.
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

    # Train noise-aware Autoencoder.
    torch.manual_seed(42)

    model = Autoencoder(input_size=100)

    print("\nTraining noise-aware Autoencoder...")

    train_autoencoder(
        model,
        processed_train,
        validation_data=combined_validation,
        epochs=50,
        batch_size=32,
        patience=5
    )

    model.eval()

    # Calibrate thresholds for each supported noise condition.
    thresholds = {}

    print("\nCalibrating thresholds...")

    for noise, validation_data in processed_validation.items():
        scores = get_anomaly_scores(
            model,
            validation_data
        )

        thresholds[str(noise)] = calculate_threshold(
            scores,
            percentile=99
        )

        print(
            f"Noise {noise}: "
            f"Threshold = {thresholds[str(noise)]:.6f}"
        )

    results = {}

    # Test automatic noise estimation.
    for noise in TEST_NOISE_LEVELS:
        print(f"\nTesting actual noise level: {noise}")

        test_data, test_labels = generate_synthetic_data(
            samples=1000,
            features=100,
            anomaly_ratio=0.2,
            signal_type="rf",
            random_state=100,
            anomaly_strength=2.0,
            noise_level=noise
        )

        # Estimate noise from raw data before scaling.
        estimated_noise = estimate_noise_level(test_data)

        # Select nearest calibrated noise condition.
        selected_noise = select_noise_condition(
            estimated_noise
        )

        selected_threshold = thresholds[str(selected_noise)]

        print(f"Estimated noise: {estimated_noise:.4f}")
        print(f"Selected noise condition: {selected_noise}")
        print(f"Selected threshold: {selected_threshold:.6f}")

        # Preprocess and detect anomalies.
        processed_test = preprocessor.transform(test_data)

        scores = get_anomaly_scores(
            model,
            processed_test
        )

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
            "selected_noise_condition": selected_noise,
            "selected_threshold": selected_threshold,
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

    # Save results.
    output = {
        "experiment": "Automatic Noise-Level Estimation",
        "calibration_noise_levels": CALIBRATION_NOISE_LEVELS,
        "test_noise_levels": TEST_NOISE_LEVELS,
        "estimation_method": "Feature-wise Median Absolute Deviation",
        "threshold_percentile": 99,
        "thresholds": thresholds,
        "results": results
    }

    output_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "noise_estimation_results.json"
        )
    )

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 60)
    print("NOISE ESTIMATION EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
