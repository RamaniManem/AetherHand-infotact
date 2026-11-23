
import json

import streamlit as st
import pandas as pd
import numpy as np

from alerts import (
    create_alert,
    generate_alert_summary,
    load_alerts,
    append_alerts
)

from visualization import (
    generate_signal,
    create_signal_chart
)


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AetherHound | Security Monitor",
    page_icon="🛡️",
    layout="wide"
)


# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
.stApp {
    background-color: #101820;
    color: #F2F5F7;
}

[data-testid="stMetric"] {
    background-color: #1B2935;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #34495E;
}

h1, h2, h3 {
    color: #E8F1F5;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("🛡️ AETHERHOUND")

st.subheader(
    "Air-Gapped RF & Acoustic Covert Channel Monitoring System"
)

st.caption(
    "Software prototype | Synthetic signal monitoring | "
    "Machine-learning integration under development"
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("Control Panel")

st.sidebar.info(
    "This prototype uses simulated signals and "
    "illustrative anomaly scores."
)

signal_mode = st.sidebar.selectbox(
    "Select Signal Type",
    ["Both", "RF", "Acoustic"]
)

generate_new = st.sidebar.button(
    "Generate New Simulation",
    use_container_width=True
)


# ---------------- SIMULATION FUNCTION ----------------

def generate_simulated_events(number_of_events=12):
    """Generate illustrative RF and acoustic events."""

    rng = np.random.default_rng()

    events = []

    for _ in range(number_of_events):

        signal_type = str(
            rng.choice(["RF", "Acoustic"])
        )

        score = float(
            rng.uniform(0.05, 0.98)
        )

        event = create_alert(
            signal_type,
            score
        )

        events.append(event)

    return events


# ---------------- INITIAL EVENT CREATION ----------------

if not st.session_state.get("initial_batch_created", False):

    saved_events = load_alerts()

    # Create demonstration data only if no history exists
    if not saved_events:

        initial_events = generate_simulated_events()

        append_alerts(initial_events)

    st.session_state.initial_batch_created = True


# ---------------- GENERATE NEW EVENTS ----------------

if generate_new:

    new_events = generate_simulated_events()

    append_alerts(new_events)

    st.sidebar.success(
        f"{len(new_events)} new simulated events saved."
    )


# ---------------- LOAD PERSISTENT HISTORY ----------------

all_events = load_alerts()


# ---------------- FILTER EVENTS ----------------

if signal_mode == "Both":

    visible_events = all_events

else:

    visible_events = [
        event for event in all_events
        if event["signal_type"] == signal_mode
    ]


# ---------------- SYSTEM OVERVIEW ----------------

summary = generate_alert_summary(
    visible_events
)

st.header("System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Events",
    summary["total"]
)

col2.metric(
    "Normal",
    summary["normal"]
)

col3.metric(
    "Warnings",
    summary["warning"]
)

col4.metric(
    "Critical",
    summary["critical"]
)

st.divider()


# ---------------- SIGNAL MONITORING ----------------

st.header("Signal Monitoring")

chart_col1, chart_col2 = st.columns(2)


# RF monitoring

if signal_mode in ["Both", "RF"]:

    with chart_col1:

        st.subheader("📡 RF Signal Monitor")

        rf_event = next(
            (
                event for event in reversed(visible_events)
                if event["signal_type"] == "RF"
            ),
            None
        )

        rf_anomaly = (
            rf_event is not None
            and rf_event["status"] != "NORMAL"
        )

        time_values, signal_values = generate_signal(
            "RF",
            anomaly=rf_anomaly
        )

        rf_chart = create_signal_chart(
            time_values,
            signal_values,
            "Simulated RF Waveform"
        )

        st.plotly_chart(
            rf_chart,
            use_container_width=True
        )


# Acoustic monitoring

if signal_mode in ["Both", "Acoustic"]:

    with chart_col2:

        st.subheader("🔊 Acoustic Signal Monitor")

        audio_event = next(
            (
                event for event in reversed(visible_events)
                if event["signal_type"] == "Acoustic"
            ),
            None
        )

        audio_anomaly = (
            audio_event is not None
            and audio_event["status"] != "NORMAL"
        )

        time_values, signal_values = generate_signal(
            "Acoustic",
            anomaly=audio_anomaly
        )

        audio_chart = create_signal_chart(
            time_values,
            signal_values,
            "Simulated Acoustic Waveform"
        )

        st.plotly_chart(
            audio_chart,
            use_container_width=True
        )


st.divider()


# ---------------- ANOMALY DETECTION RESULTS ----------------

st.header("Anomaly Detection Results")

if visible_events:

    dataframe = pd.DataFrame(
        visible_events
    )

    dataframe = dataframe.rename(
        columns={
            "timestamp": "Timestamp",
            "signal_type": "Signal Type",
            "anomaly_score": "Anomaly Score",
            "status": "Status"
        }
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No events available for this signal type."
    )


st.divider()


# ---------------- PERSISTENT ALERT HISTORY ----------------

st.header("🚨 Persistent Alert History")

st.caption(
    "These alerts are loaded from the project's "
    "logs/alerts.json file."
)

abnormal_events = [
    event for event in visible_events
    if event["status"] in ["WARNING", "CRITICAL"]
]

if abnormal_events:

    # Display the most recent 20 abnormal alerts
    recent_alerts = abnormal_events[-20:]

    for event in reversed(recent_alerts):

        status = event["status"]

        if status == "CRITICAL":

            st.error(
                f"🔴 CRITICAL | {event['signal_type']} | "
                f"Score: {event['anomaly_score']} | "
                f"{event['timestamp']}"
            )

        else:

            st.warning(
                f"🟠 WARNING | {event['signal_type']} | "
                f"Score: {event['anomaly_score']} | "
                f"{event['timestamp']}"
            )

else:

    st.success(
        "No warning or critical events in saved history."
    )


# ---------------- DOWNLOAD ALERT LOG ----------------

st.subheader("Export Alert History")

alert_json = json.dumps(
    all_events,
    indent=4
)

st.download_button(
    label="Download Alert History (JSON)",
    data=alert_json,
    file_name="aetherhound_alert_history.json",
    mime="application/json",
    use_container_width=True
)


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AetherHound | Development Prototype | "
    "All current signals and anomaly scores are simulated. "
    "No real-world covert-channel detection is claimed."
)
