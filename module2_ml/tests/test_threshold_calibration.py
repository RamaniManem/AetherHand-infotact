
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


TRAINING_NOISE_LEVELS = [0.5, 1.0, 1.5]
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


def prepare_training_data():
    training_parts = []
    validation_parts = {}

    for index, noise in enumerate(TRAINING_NOISE_LEVELS):
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
    print("STAGE 8: THRESHOLD CALIBRATION COMPARISON")
    print("=" * 60)

    # Prepare training and separate validation datasets.
    train_data, validation_parts = prepare_training_data()

    # Fit scaler using training data only.
    preprocessor = SignalPreprocessor()
    preprocessor.fit(train_data)

    processed_train = preprocessor.transform(train_data)

    processed_validation = {
        noise: preprocessor.transform(data)
        for noise, data in validation_parts.items()
    }

    # Combine validation data for model early stopping.
    combined_validation = torch.cat(
        list(processed_validation.values()),
        dim=0
    )

    # Train one noise-aware Autoencoder.
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

    # Calculate validation anomaly scores separately.
    validation_scores = {}

    for noise, data in processed_validation.items():
        validation_scores[noise] = get_anomaly_scores(
            model,
            data
        )

    # Method A: one global threshold from all normal validation scores.
    all_validation_scores = np.concatenate(
        list(validation_scores.values())
    )

    global_threshold = calculate_threshold(
        all_validation_scores,
        percentile=99
    )

    # Method B: one threshold for each known noise condition.
    noise_thresholds = {
        str(noise): calculate_threshold(scores, percentile=99)
        for noise, scores in validation_scores.items()
    }

    print("\nCalibrated thresholds:")
    print(f"Global threshold: {global_threshold:.6f}")

    for noise, threshold in noise_thresholds.items():
        print(f"Noise {noise}: {threshold:.6f}")

    global_results = {}
    calibrated_results = {}

    # Test both methods on identical independent datasets.
    for noise in TEST_NOISE_LEVELS:
        print(f"\nTesting at noise level {noise}...")

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

        scores = get_anomaly_scores(model, processed_test)

        # Method A: global threshold.
        global_predictions = classify_anomaly(
            scores,
            global_threshold
        )

        global_labels = np.array([
            1 if item == "Anomalous" else 0
            for item in global_predictions
        ])

        global_metrics = evaluate_model(
            test_labels,
            global_labels
        )

        global_results[str(noise)] = {
            "threshold": global_threshold,
            "accuracy": float(global_metrics["accuracy"]),
            "precision": float(global_metrics["precision"]),
            "recall": float(global_metrics["recall"]),
            "f1_score": float(global_metrics["f1_score"]),
            "false_positive_rate": float(
                global_metrics["false_positive_rate"]
            ),
            "false_negative_rate": float(
                global_metrics["false_negative_rate"]
            ),
            "confusion_matrix": np.asarray(
                global_metrics["confusion_matrix"]
            ).tolist()
        }

        # Method B: use noise-specific threshold when available.
        # Noise 2.0 is unseen, so use the global threshold as fallback.
        selected_threshold = noise_thresholds.get(
            str(noise),
            global_threshold
        )

        calibrated_predictions = classify_anomaly(
            scores,
            selected_threshold
        )

        calibrated_labels = np.array([
            1 if item == "Anomalous" else 0
            for item in calibrated_predictions
        ])

        calibrated_metrics = evaluate_model(
            test_labels,
            calibrated_labels
        )

        calibrated_results[str(noise)] = {
            "threshold": selected_threshold,
            "threshold_type": (
                "noise_specific"
                if str(noise) in noise_thresholds
                else "global_fallback"
            ),
            "accuracy": float(calibrated_metrics["accuracy"]),
            "precision": float(calibrated_metrics["precision"]),
            "recall": float(calibrated_metrics["recall"]),
            "f1_score": float(calibrated_metrics["f1_score"]),
            "false_positive_rate": float(
                calibrated_metrics["false_positive_rate"]
            ),
            "false_negative_rate": float(
                calibrated_metrics["false_negative_rate"]
            ),
            "confusion_matrix": np.asarray(
                calibrated_metrics["confusion_matrix"]
            ).tolist()
        }

        print(
            f"Global: Accuracy={global_metrics['accuracy']:.4f}, "
            f"Recall={global_metrics['recall']:.4f}, "
            f"FPR={global_metrics['false_positive_rate']:.4f}"
        )

        print(
            f"Calibrated: Accuracy={calibrated_metrics['accuracy']:.4f}, "
            f"Recall={calibrated_metrics['recall']:.4f}, "
            f"FPR={calibrated_metrics['false_positive_rate']:.4f}"
        )

    # Save experiment results.
    output = {
        "experiment": "Global vs Noise-Specific Threshold Calibration",
        "training_noise_levels": TRAINING_NOISE_LEVELS,
        "test_noise_levels": TEST_NOISE_LEVELS,
        "threshold_percentile": 99,
        "anomaly_strength": 2.0,
        "global_threshold": global_threshold,
        "noise_specific_thresholds": noise_thresholds,
        "global_threshold_results": global_results,
        "noise_specific_results": calibrated_results
    }

    output_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "threshold_calibration_results.json"
        )
    )

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
