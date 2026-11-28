
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(actual_labels, predicted_labels):

    actual_labels = np.asarray(actual_labels)
    predicted_labels = np.asarray(predicted_labels)

    if actual_labels.shape != predicted_labels.shape:
        raise ValueError("Actual and predicted labels must have the same shape")

    if actual_labels.size == 0:
        raise ValueError("Evaluation data cannot be empty")

    # 0 = Normal, 1 = Anomalous
    accuracy = accuracy_score(actual_labels, predicted_labels)

    precision = precision_score(
        actual_labels, predicted_labels, zero_division=0
    )

    recall = recall_score(
        actual_labels, predicted_labels, zero_division=0
    )

    f1 = f1_score(
        actual_labels, predicted_labels, zero_division=0
    )

    matrix = confusion_matrix(
        actual_labels, predicted_labels, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate
    }
