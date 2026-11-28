
import csv
import math
import random
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "simulated_signals.csv"


# --------------------------------------------------
# DATASET SETTINGS
# --------------------------------------------------

NUMBER_OF_EVENTS = 40
SAMPLES_PER_EVENT = 200
SAMPLE_RATE = 200

RANDOM_SEED = 42


# --------------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------------

def generate_signal(signal_type, is_anomaly):
    """Generate one synthetic RF or acoustic signal window."""

    signal_values = []

    frequency = 8 if signal_type == "RF" else 4

    for sample_index in range(SAMPLES_PER_EVENT):
        time_value = sample_index / SAMPLE_RATE

        # Base periodic signal
        signal_value = math.sin(
            2 * math.pi * frequency * time_value
        )

        # Add a second harmonic
        signal_value += 0.25 * math.sin(
            2 * math.pi * frequency * 2 * time_value
        )

        # Add synthetic background noise
        signal_value += random.gauss(0, 0.12)

        # Inject an artificial anomaly into some samples
        if is_anomaly and 80 <= sample_index <= 120:
            signal_value += random.uniform(1.5, 2.5)

        signal_values.append(signal_value)

    return signal_values


# --------------------------------------------------
# DATASET CREATION
# --------------------------------------------------

def create_dataset():
    """Create and save the simulated signal dataset."""

    random.seed(RANDOM_SEED)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    for event_number in range(1, NUMBER_OF_EVENTS + 1):

        signal_type = random.choice(
            ["RF", "Acoustic"]
        )

        # Approximately 25% anomaly events
        is_anomaly = random.random() < 0.25

        label = "ANOMALY" if is_anomaly else "NORMAL"

        signal_values = generate_signal(
            signal_type,
            is_anomaly,
        )

        for sample_index, signal_value in enumerate(
            signal_values
        ):

            rows.append(
                {
                    "event_id": f"EVT-{event_number:03d}",
                    "timestamp": generated_at,
                    "signal_type": signal_type,
                    "sample_index": sample_index,
                    "signal_value": round(
                        signal_value,
                        6,
                    ),
                    "label": label,
                }
            )

    # Save dataset as CSV
    fieldnames = [
        "event_id",
        "timestamp",
        "signal_type",
        "sample_index",
        "signal_value",
        "label",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Print dataset summary
    normal_samples = sum(
        1 for row in rows
        if row["label"] == "NORMAL"
    )

    anomaly_samples = sum(
        1 for row in rows
        if row["label"] == "ANOMALY"
    )

    print("\nAetherHound Simulated Dataset Created")
    print("------------------------------------")
    print(f"Events generated : {NUMBER_OF_EVENTS}")
    print(f"Samples per event: {SAMPLES_PER_EVENT}")
    print(f"Total samples    : {len(rows)}")
    print(f"Normal samples   : {normal_samples}")
    print(f"Anomaly samples  : {anomaly_samples}")
    print(f"Output file      : {OUTPUT_FILE}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    create_dataset()
