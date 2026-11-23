
from datasets.synthetic_data import generate_synthetic_data

# Generate RF samples
rf_data, rf_labels = generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="rf"
)

# Generate acoustic samples
acoustic_data, acoustic_labels = generate_synthetic_data(
    samples=1000,
    features=100,
    anomaly_ratio=0.2,
    signal_type="acoustic"
)

print("===== SYNTHETIC DATASET SUMMARY =====")

print("\nRF Dataset:")
print("Shape:", rf_data.shape)
print("Normal samples:", (rf_labels == 0).sum())
print("Anomalous samples:", (rf_labels == 1).sum())

print("\nAcoustic Dataset:")
print("Shape:", acoustic_data.shape)
print("Normal samples:", (acoustic_labels == 0).sum())
print("Anomalous samples:", (acoustic_labels == 1).sum())

print("\nSynthetic dataset generation successful!")
