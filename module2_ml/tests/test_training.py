
import torch

from models.autoencoder import Autoencoder
from training.train_autoencoder import train_autoencoder

# Create the Autoencoder
model = Autoencoder(input_size=100)

# Generate temporary sample data
# 200 samples, each containing 100 features
train_data = torch.randn(200, 100)

print("Starting Autoencoder training...")

# Train the model
trained_model, history = train_autoencoder(
    model,
    train_data,
    epochs=20,
    batch_size=32
)

print("\nTraining completed!")
print("Initial loss:", history[0])
print("Final loss:", history[-1])

# Save the trained model
torch.save(
    trained_model.state_dict(),
    "saved_models/autoencoder.pth"
)

print("Model saved successfully!")
