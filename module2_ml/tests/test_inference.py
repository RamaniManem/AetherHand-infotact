
import numpy as np
import torch

from models.autoencoder import Autoencoder
from preprocessing.preprocess import SignalPreprocessor
from detection.inference import predict_anomalies

# Create temporary normal training samples
np.random.seed(42)

normal_data = np.random.normal(
    loc=10,
    scale=2,
    size=(200, 100)
).astype(np.float32)

# Fit preprocessing using normal training samples
preprocessor = SignalPreprocessor()
train_data = preprocessor.fit_transform(normal_data)

# Create and train the Autoencoder
model = Autoencoder(input_size=100)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

model.train()

for epoch in range(20):
    optimizer.zero_grad()

    reconstructed = model(train_data)
    loss = criterion(reconstructed, train_data)

    loss.backward()
    optimizer.step()

# Calculate threshold from normal training scores
model.eval()

with torch.no_grad():
    reconstructed = model(train_data)

    from detection.anomaly_score import calculate_anomaly_score

    normal_scores = calculate_anomaly_score(
        train_data,
        reconstructed
    ).tolist()

threshold = float(np.percentile(normal_scores, 95))

print("Calculated Threshold:", threshold)

# Create new samples for inference
new_samples = np.random.normal(
    loc=10,
    scale=2,
    size=(5, 100)
).astype(np.float32)

# Run inference
results = predict_anomalies(
    model,
    preprocessor,
    new_samples,
    threshold
)

print("\n===== AETHERHOUND INFERENCE RESULTS =====")

for index, (score, prediction) in enumerate(
    zip(results["anomaly_scores"], results["predictions"])
):
    print(
        f"Sample {index + 1}: "
        f"Score = {score:.6f}, "
        f"Prediction = {prediction}"
    )

print("\nInference test completed!")
