
import numpy as np


def generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="rf",
    random_state=42,
    anomaly_strength=4.0,
    noise_level=1.0
):
    """
    Generate synthetic RF or acoustic feature data.

    Labels:
        0 = Normal
        1 = Anomalous

    anomaly_strength controls anomaly intensity.
    noise_level controls the standard deviation of
    the generated background noise.
    """

    if samples <= 0 or features <= 0:
        raise ValueError("Samples and features must be positive")

    if not 0 <= anomaly_ratio <= 1:
        raise ValueError("Anomaly ratio must be between 0 and 1")

    if signal_type not in ("rf", "acoustic"):
        raise ValueError("Signal type must be 'rf' or 'acoustic'")

    if anomaly_strength < 0:
        raise ValueError("Anomaly strength cannot be negative")

    if noise_level <= 0:
        raise ValueError("Noise level must be positive")

    rng = np.random.default_rng(random_state)

    anomaly_count = int(samples * anomaly_ratio)
    normal_count = samples - anomaly_count

    # Generate normal background features
    normal_data = rng.normal(
        0.0,
        noise_level,
        (normal_count, features)
    )

    # Generate anomalous features
    anomalous_data = rng.normal(
        0.0,
        noise_level,
        (anomaly_count, features)
    )

    if signal_type == "rf":
        anomalous_data[:, :features // 4] += anomaly_strength

    else:
        anomalous_data[:, features // 2:] *= anomaly_strength

    # Combine data and labels
    data = np.vstack((normal_data, anomalous_data))

    labels = np.concatenate((
        np.zeros(normal_count, dtype=int),
        np.ones(anomaly_count, dtype=int)
    ))

    # Shuffle samples
    indices = rng.permutation(samples)

    data = data[indices]
    labels = labels[indices]

    return data.astype(np.float32), labels
