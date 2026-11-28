
import torch
import torch.nn.functional as F


def calculate_anomaly_score(original, reconstructed):

    if original.shape != reconstructed.shape:
        raise ValueError("Input and reconstructed shapes must match")

    # Calculate reconstruction error for each sample
    error = F.mse_loss(
        reconstructed,
        original,
        reduction="none"
    )

    # Average error across signal features
    anomaly_scores = error.mean(dim=1)

    return anomaly_scores
