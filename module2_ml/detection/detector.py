
import numpy as np


def calculate_threshold(normal_scores, percentile=95):

    if len(normal_scores) == 0:
        raise ValueError("Normal scores cannot be empty")

    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    threshold = np.percentile(normal_scores, percentile)

    return float(threshold)


def classify_anomaly(anomaly_scores, threshold):

    results = []

    for score in anomaly_scores:

        if score > threshold:
            results.append("Anomalous")
        else:
            results.append("Normal")

    return results

