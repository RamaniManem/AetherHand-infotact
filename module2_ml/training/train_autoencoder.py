
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_autoencoder(
    model,
    train_data,
    validation_data=None,
    epochs=50,
    batch_size=32,
    patience=5
):

    if train_data.ndim != 2 or train_data.shape[0] == 0:
        raise ValueError("Training data must be a non-empty 2D tensor")

    if validation_data is not None:
        if validation_data.ndim != 2 or validation_data.shape[0] == 0:
            raise ValueError("Validation data must be a non-empty 2D tensor")

        if train_data.shape[1] != validation_data.shape[1]:
            raise ValueError("Training and validation features must match")

    dataset = TensorDataset(train_data)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    best_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):

        # Training phase
        model.train()
        total_train_loss = 0

        for batch in loader:

            inputs = batch[0]

            optimizer.zero_grad()

            reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(loader)
        history["train_loss"].append(average_train_loss)

        # Validation phase
        if validation_data is not None:

            model.eval()

            with torch.no_grad():
                reconstructed = model(validation_data)
                validation_loss = criterion(
                    reconstructed,
                    validation_data
                ).item()

            history["validation_loss"].append(validation_loss)

            monitored_loss = validation_loss

        else:
            monitored_loss = average_train_loss

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_train_loss:.6f}"
        )

        if validation_data is not None:
            print(f"Validation Loss: {validation_loss:.6f}")

        # Save best weights
        if monitored_loss < best_loss:

            best_loss = monitored_loss
            best_weights = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Restore best model weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    print("\nTraining completed!")

    return model, history
