
from evaluation.evaluate_model import evaluate_model

# Actual labels (ground truth)
# 0 = Normal, 1 = Anomalous
actual_labels = [
    0, 0, 0, 1, 1,
    1, 0, 1, 0, 1
]

# Predictions made by our detection system
predicted_labels = [
    0, 0, 1, 1, 0,
    1, 0, 1, 0, 1
]

# Evaluate the predictions
results = evaluate_model(actual_labels, predicted_labels)

print("\n===== AETHERHOUND MODEL EVALUATION =====")

print(f"Accuracy: {results['accuracy']:.2f}")
print(f"Precision: {results['precision']:.2f}")
print(f"Recall: {results['recall']:.2f}")
print(f"F1 Score: {results['f1_score']:.2f}")

print("\nConfusion Matrix:")
print(results["confusion_matrix"])

print(
    f"\nFalse Positive Rate: "
    f"{results['false_positive_rate']:.2f}"
)

print(
    f"False Negative Rate: "
    f"{results['false_negative_rate']:.2f}"
)

print("\nEvaluation completed!")
