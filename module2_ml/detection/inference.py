
import torch

from detection.anomaly_score import calculate_anomaly_score
from detection.detector import classify_anomaly
from preprocessing.preprocess import SignalPreprocessor
from training.model_manager import load_model


def predict_anomalies(model, preprocessor, input_data, threshold):

    # Preprocess incoming features
    processed_data = preprocessor.transform(input_data)

    device = next(model.parameters()).device
    processed_data = processed_data.to(device)

    model.eval()

    with torch.no_grad():
        reconstructed = model(processed_data)

        scores = calculate_anomaly_score(
            processed_data,
            reconstructed
        )

    scores = scores.cpu().tolist()

    predictions = classify_anomaly(scores, threshold)

    return {
        "anomaly_scores": scores,
        "predictions": predictions
    }


def predict_with_saved_model(model, input_data, load_dir="saved_models"):

    # Load trained model and saved components
    model, scaler, threshold = load_model(
        model,
        load_dir
    )

    # Restore the fitted preprocessing object
    preprocessor = SignalPreprocessor()
    preprocessor.scaler = scaler
    preprocessor.is_fitted = True

    # Perform inference
    results = predict_anomalies(
        model,
        preprocessor,
        input_data,
        threshold
    )

    return results
