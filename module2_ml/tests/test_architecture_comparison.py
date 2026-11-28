
import json
import os

import numpy as np
import torch
import torch.nn as nn

from datasets.synthetic_data import generate_synthetic_data
from preprocessing.preprocess import SignalPreprocessor
from training.train_autoencoder import train_autoencoder
from detection.anomaly_score import calculate_anomaly_score
from detection.detector import calculate_threshold, classify_anomaly
from evaluation.evaluate_model import evaluate_model


NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0]
TEST_NOISE_LEVELS = [1.0, 2.0]
ANOMALY_STRENGTHS = [0.5, 1.0, 1.5, 2.0]


# Model A: Existing architecture
class BaselineAutoencoder(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Model B: Deeper architecture
class DeepAutoencoder(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Model C: Smaller bottleneck
class CompactAutoencoder(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


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

    for index, noise in enumerate(NOISE_LEVELS):
        data = generate_normal_data(
            samples=225,
            noise=noise,
            seed=30 + index
        )

        training_parts.append(data[:180])
        validation_parts[noise] = data[180:]

    train_data = np.vstack(training_parts)

    rng = np.random.default_rng(42)
    train_data = train_data[rng.permutation(len(train_data))]

    return train_data, validation_parts


def get_scores(model, data):
    with torch.no_grad():
        reconstructed = model(data)

        scores = calculate_anomaly_score(
            data,
            reconstructed
        )

    return scores.cpu().numpy()


def train_and_calibrate(model_class, model_name,
                        processed_train, processed_validation):

    print("\n" + "=" * 60)
    print(f"TRAINING {model_name}")
    print("=" * 60)

    combined_validation = torch.cat(
        list(processed_validation.values()),
        dim=0
    )

    torch.manual_seed(42)

    model = model_class(input_size=100)

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

    return model, thresholds


def evaluate_model_architecture(model, thresholds,
                                preprocessor, model_name):

    results = {}

    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)

    for noise in TEST_NOISE_LEVELS:
        results[str(noise)] = {}

        selected_threshold = thresholds[str(noise)]

        for strength in ANOMALY_STRENGTHS:

            test_data, test_labels = generate_synthetic_data(
                samples=1000,
                features=100,
                anomaly_ratio=0.2,
                signal_type="rf",
                random_state=100,
                anomaly_strength=strength,
                noise_level=noise
            )

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

            results[str(noise)][str(strength)] = {
                "noise_level": noise,
                "anomaly_strength": strength,
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

            print(
                f"Noise {noise} | Strength {strength} | "
                f"Accuracy {metrics['accuracy']:.4f} | "
                f"Recall {metrics['recall']:.4f} | "
                f"F1 {metrics['f1_score']:.4f} | "
                f"FPR {metrics['false_positive_rate']:.4f}"
            )

    return results


def main():
    print("=" * 60)
    print("STAGE 12: AUTOENCODER ARCHITECTURE COMPARISON")
    print("=" * 60)

    # Prepare one shared training and validation dataset.
    train_data, validation_parts = prepare_training_data()

    preprocessor = SignalPreprocessor()
    preprocessor.fit(train_data)

    processed_train = preprocessor.transform(train_data)

    processed_validation = {
        noise: preprocessor.transform(data)
        for noise, data in validation_parts.items()
    }

    print("\nShared dataset sizes:")
    print(f"Training: {len(processed_train)}")
    print(
        "Validation:",
        sum(len(data) for data in processed_validation.values())
    )

    architectures = [
        ("Baseline", BaselineAutoencoder),
        ("Deep", DeepAutoencoder),
        ("Compact", CompactAutoencoder)
    ]

    all_results = {}

    for model_name, model_class in architectures:

        model, thresholds = train_and_calibrate(
            model_class,
            model_name,
            processed_train,
            processed_validation
        )

        results = evaluate_model_architecture(
            model,
            thresholds,
            preprocessor,
            model_name
        )

        all_results[model_name] = {
            "thresholds": thresholds,
            "results": results
        }

    output = {
        "experiment": "Autoencoder Architecture Comparison",
        "noise_levels": TEST_NOISE_LEVELS,
        "anomaly_strengths": ANOMALY_STRENGTHS,
        "threshold_percentile": 99,
        "architectures": {
            "Baseline": "100-64-32-16-32-64-100",
            "Deep": "100-64-48-32-16-32-48-64-100",
            "Compact": "100-64-32-8-32-64-100"
        },
        "results": all_results
    }

    output_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "architecture_comparison_results.json"
        )
    )

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 60)
    print("STAGE 12 COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
