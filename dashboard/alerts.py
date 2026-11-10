
from datetime import datetime


def classify_anomaly(score):
    """
    Classify an anomaly score between 0 and 1.
    These are initial prototype thresholds.
    """
    if score >= 0.80:
        return "CRITICAL"
    elif score >= 0.50:
        return "WARNING"
    return "NORMAL"


def create_alert(signal_type, score):
    status = classify_anomaly(score)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_type": signal_type,
        "anomaly_score": round(float(score), 3),
        "status": status
    }


def generate_alert_summary(alerts):
    return {
        "total": len(alerts),
        "critical": sum(a["status"] == "CRITICAL" for a in alerts),
        "warning": sum(a["status"] == "WARNING" for a in alerts),
        "normal": sum(a["status"] == "NORMAL" for a in alerts)
    }
