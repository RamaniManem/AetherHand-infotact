
import numpy as np

from models.autoencoder import Autoencoder
from detection.inference import predict_with_saved_model

# Generate new temporary samples
np.random.seed(100)

new_samples = np.random.normal(
    loc=10,
    scale=2,
    size=(5, 100)
).astype(np.float32)

# Create model architecture
model = Autoencoder(input_size=100)

# Predict using saved model
results = predict_with_saved_model(
    model,
    new_samples
)

print("\n===== SAVED MODEL INFERENCE =====")

for index, (score, prediction) in enumerate(
    zip(results["anomaly_scores"], results["predictions"])
):
    print(
        f"Sample {index + 1}: "
        f"Score = {score:.6f}, "
        f"Prediction = {prediction}"
    )

print("\nSaved model inference successful!")
