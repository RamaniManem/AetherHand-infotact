
import torch

from models.autoencoder import Autoencoder
from training.train_autoencoder import train_autoencoder

# Generate temporary normal training samples
torch.manual_seed(42)

train_data = torch.randn(160, 100)
validation_data = torch.randn(40, 100)

# Create Autoencoder
model = Autoencoder(input_size=100)

print("Starting Autoencoder training...")

# Train with validation and early stopping
trained_model, history = train_autoencoder(
    model,
    train_data,
    validation_data=validation_data,
    epochs=50,
    batch_size=32,
    patience=5
)

print("\n===== TRAINING SUMMARY =====")

print("Training epochs completed:", len(history["train_loss"]))
print("Final training loss:", history["train_loss"][-1])
print("Best validation loss:", min(history["validation_loss"]))

# Save model weights
torch.save(
    trained_model.state_dict(),
    "saved_models/autoencoder.pth"
)

print("\nModel saved successfully!")
print("Training test completed!")
