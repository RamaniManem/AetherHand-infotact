
import torch
from models.autoencoder import Autoencoder

# Create model
model = Autoencoder(input_size=100)

# Create sample input (not a signal dataset)
sample = torch.randn(1, 100)

# Get reconstructed output
output = model(sample)

print("Input shape:", sample.shape)
print("Output shape:", output.shape)
print("Model test successful!")
