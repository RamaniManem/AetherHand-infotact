
import json
import os


def load_results(filename):
    """Load threshold optimization results."""

    path = os.path.join(
        os.path.dirname(__file__),
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    with open(path, "r") as file:
        return json.load(file)


def compare_thresholds():
    rf_results = load_results("rf_threshold_results.json")
    acoustic_results = load_results("acoustic_threshold_results.json")

    percentiles = ["90", "95", "97", "99"]

    print("\n" + "=" * 95)
    print("          RF AND ACOUSTIC THRESHOLD COMPARISON")
    print("=" * 95)

    print(
        f"{'Percentile':<12}"
        f"{'Signal':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
        f"{'FPR':<12}"
    )

    print("-" * 95)

    for percentile in percentiles:

        for signal_name, results in [
            ("RF", rf_results),
            ("Acoustic", acoustic_results)
        ]:

            item = results[percentile]

            print(
                f"{percentile + '%':<12}"
                f"{signal_name:<12}"
                f"{item['accuracy']:<12.4f}"
                f"{item['precision']:<12.4f}"
                f"{item['recall']:<12.4f}"
                f"{item['f1_score']:<12.4f}"
                f"{item['false_positive_rate']:<12.4f}"
            )

        print("-" * 95)

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    compare_thresholds()
