
import torch
from detection.anomaly_score import calculate_anomaly_score

# Original signal samples
original = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 4.0, 6.0, 8.0]
])

# Reconstructed signal samples
reconstructed = torch.tensor([
    [1.1, 2.1, 2.9, 4.2],
    [2.5, 4.5, 6.5, 8.5]
])

scores = calculate_anomaly_score(original, reconstructed)

print("Anomaly Scores:", scores)
print("Number of scores:", len(scores))
print("Anomaly scoring test successful!")
