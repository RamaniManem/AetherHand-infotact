
from detection.detector import (
    calculate_threshold,
    classify_anomaly
)

# Simulated reconstruction errors from normal samples
normal_scores = [
    0.012, 0.015, 0.018, 0.020, 0.025, 0.030
]

# Calculate threshold using normal samples
threshold = calculate_threshold(normal_scores)

print("Detection Threshold:", threshold)

# New sample anomaly scores
test_scores = [0.010, 0.022, 0.035, 0.050]

# Classify samples
results = classify_anomaly(test_scores, threshold)

for score, result in zip(test_scores, results):
    print(f"Score: {score:.4f} -> {result}")

print("\nDetection test completed!")

