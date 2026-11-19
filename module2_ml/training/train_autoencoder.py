
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_autoencoder(model, train_data, epochs=20, batch_size=32):

    # Check input data
    if train_data.ndim != 2:
        raise ValueError("Training data must have shape [samples, features]")

    if train_data.shape[0] == 0:
        raise ValueError("Training data cannot be empty")

    # Prepare dataset
    dataset = TensorDataset(train_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    history = []

    # Training loop
    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

            inputs = batch[0]

            optimizer.zero_grad()

            reconstructed = model(inputs)

            loss = criterion(reconstructed, inputs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        history.append(average_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.6f}"
        )

    return model, history
