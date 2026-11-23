
import json
from datetime import datetime
from pathlib import Path


# Store logs in the project's root logs folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
ALERT_FILE = LOG_DIR / "alerts.json"


def classify_anomaly(score):
    """Classify an anomaly score between 0 and 1."""

    if score >= 0.80:
        return "CRITICAL"

    if score >= 0.50:
        return "WARNING"

    return "NORMAL"


def create_alert(signal_type, score):
    """Create one alert record."""

    score = float(score)

    return {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "signal_type": str(signal_type),
        "anomaly_score": round(score, 3),
        "status": classify_anomaly(score)
    }


def load_alerts():
    """Load saved alerts from the JSON file."""

    if not ALERT_FILE.exists():
        return []

    try:
        with ALERT_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        return []


def save_alerts(alerts):
    """Save the complete alert history."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with ALERT_FILE.open("w", encoding="utf-8") as file:
        json.dump(alerts, file, indent=4)


def append_alerts(new_alerts):
    """Append new alerts without deleting existing history."""

    existing_alerts = load_alerts()
    existing_alerts.extend(new_alerts)

    save_alerts(existing_alerts)

    return existing_alerts


def generate_alert_summary(alerts):
    """Calculate alert counts."""

    return {
        "total": len(alerts),
        "critical": sum(
            alert["status"] == "CRITICAL"
            for alert in alerts
        ),
        "warning": sum(
            alert["status"] == "WARNING"
            for alert in alerts
        ),
        "normal": sum(
            alert["status"] == "NORMAL"
            for alert in alerts
        )
    }
