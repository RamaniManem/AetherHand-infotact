
import numpy as np
import torch

from models.autoencoder import Autoencoder
from preprocessing.preprocess import SignalPreprocessor
from training.model_manager import save_model, load_model

# Generate temporary normal training data
np.random.seed(42)

sample_data = np.random.normal(
    loc=10,
    scale=2,
    size=(200, 100)
).astype(np.float32)

# Fit preprocessing
preprocessor = SignalPreprocessor()
processed_data = preprocessor.fit_transform(sample_data)

# Create Autoencoder
model = Autoencoder(input_size=100)

# Train the model briefly
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

model.train()

for epoch in range(10):
    optimizer.zero_grad()

    reconstructed = model(processed_data)
    loss = criterion(reconstructed, processed_data)

    loss.backward()
    optimizer.step()

# Example threshold for this test
threshold = 0.5

# Save complete model package
save_model(model, preprocessor, threshold)

# Create a fresh model instance
loaded_model = Autoencoder(input_size=100)

# Load saved components
loaded_model, loaded_scaler, loaded_threshold = load_model(
    loaded_model
)

print("\n===== MODEL MANAGER TEST =====")
print("Model loaded:", loaded_model is not None)
print("Scaler loaded:", loaded_scaler is not None)
print("Threshold:", loaded_threshold)

print("\nModel manager test successful!")
