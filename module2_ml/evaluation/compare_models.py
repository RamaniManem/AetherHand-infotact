
import json
import os


def load_results(file_path):
    """Load model evaluation results from JSON."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Results file not found: {file_path}. "
            "Run both detection experiments first."
        )

    with open(file_path, "r") as file:
        return json.load(file)


def compare_models(rf_results, acoustic_results):
    """Compare RF and acoustic anomaly detection models."""

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("False Positive Rate", "false_positive_rate"),
        ("False Negative Rate", "false_negative_rate")
    ]

    print("\n")
    print("=" * 70)
    print("          RF vs ACOUSTIC MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<25}"
        f"{'RF Model':<20}"
        f"{'Acoustic Model':<20}"
    )

    print("-" * 70)

    for label, key in metrics:
        rf_value = rf_results[key]
        acoustic_value = acoustic_results[key]

        print(
            f"{label:<25}"
            f"{rf_value:<20.4f}"
            f"{acoustic_value:<20.4f}"
        )

    print("-" * 70)

    print("\nRF CONFUSION MATRIX")
    print(rf_results["confusion_matrix"])

    print("\nACOUSTIC CONFUSION MATRIX")
    print(acoustic_results["confusion_matrix"])

    print("\nComparison completed successfully!")


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)

    rf_path = os.path.join(
        base_dir,
        "rf_results.json"
    )

    acoustic_path = os.path.join(
        base_dir,
        "acoustic_results.json"
    )

    rf_results = load_results(rf_path)
    acoustic_results = load_results(acoustic_path)

    compare_models(rf_results, acoustic_results)
