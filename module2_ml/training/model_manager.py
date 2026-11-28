
import os
import joblib
import torch


def save_model(model, preprocessor, threshold, save_dir="saved_models"):

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save Autoencoder weights
    model_path = os.path.join(save_dir, "autoencoder.pth")

    torch.save(model.state_dict(), model_path)

    # Save fitted preprocessing scaler
    scaler_path = os.path.join(save_dir, "scaler.pkl")

    joblib.dump(preprocessor.scaler, scaler_path)

    # Save detection threshold
    threshold_path = os.path.join(save_dir, "threshold.txt")

    with open(threshold_path, "w") as file:
        file.write(str(threshold))

    print("Model saved successfully!")


def load_model(model, load_dir="saved_models"):

    # Load Autoencoder weights
    model_path = os.path.join(load_dir, "autoencoder.pth")

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Load preprocessing scaler
    scaler_path = os.path.join(load_dir, "scaler.pkl")

    scaler = joblib.load(scaler_path)

    # Load detection threshold
    threshold_path = os.path.join(load_dir, "threshold.txt")

    with open(threshold_path, "r") as file:
        threshold = float(file.read())

    print("Model loaded successfully!")

    return model, scaler, threshold
