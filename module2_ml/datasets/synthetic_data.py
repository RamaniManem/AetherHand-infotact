
import numpy as np


def generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="rf",
    random_state=42
):
    """
    Generate simulated RF or acoustic feature data.

    Labels:
        0 = Normal
        1 = Anomalous

    This is synthetic test data, not real signal recordings.
    """

    if samples <= 0 or features <= 0:
        raise ValueError("Samples and features must be positive")

    if not 0 <= anomaly_ratio <= 1:
        raise ValueError("Anomaly ratio must be between 0 and 1")

    if signal_type not in ("rf", "acoustic"):
        raise ValueError("Signal type must be 'rf' or 'acoustic'")

    rng = np.random.default_rng(random_state)

    anomaly_count = int(samples * anomaly_ratio)
    normal_count = samples - anomaly_count

    # Generate normal feature patterns
    if signal_type == "rf":
        normal_data = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(normal_count, features)
        )

        # Simulate unusual frequency-band activity
        anomalous_data = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(anomaly_count, features)
        )

        anomalous_data[:, :features // 4] += 4.0

    else:
        normal_data = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(normal_count, features)
        )

        # Simulate unusual acoustic feature patterns
        anomalous_data = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(anomaly_count, features)
        )

        anomalous_data[:, features // 2:] *= 3.0

    # Combine samples
    data = np.vstack((normal_data, anomalous_data))

    # Assign labels
    labels = np.concatenate((
        np.zeros(normal_count, dtype=int),
        np.ones(anomaly_count, dtype=int)
    ))

    # Shuffle dataset
    indices = rng.permutation(samples)

    data = data[indices]
    labels = labels[indices]

    return data.astype(np.float32), labels

