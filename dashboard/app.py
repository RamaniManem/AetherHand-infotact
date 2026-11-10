
import streamlit as st
import pandas as pd
import numpy as np

from alerts import create_alert, generate_alert_summary
from visualization import generate_signal, create_signal_chart


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

.status-normal {
    color: #35D07F;
    font-weight: bold;
}

.status-warning {
    color: #FFB84D;
    font-weight: bold;
}

.status-critical {
    color: #FF5C5C;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("🛡️ AETHERHOUND")
st.subheader("Air-Gapped RF & Acoustic Covert Channel Monitoring System")

st.caption(
    "Software prototype | Synthetic signal monitoring | "
    "Machine-learning integration under development"
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("Control Panel")

st.sidebar.info(
    "This prototype currently uses simulated signals and "
    "illustrative anomaly scores."
)

signal_mode = st.sidebar.selectbox(
    "Select Signal Type",
    ["Both", "RF", "Acoustic"]
)

if st.sidebar.button("Generate New Simulation", use_container_width=True):
    st.session_state.pop("simulation", None)
    st.rerun()


# ---------------- SIMULATED EVENT GENERATION ----------------

if "simulation" not in st.session_state:

    rng = np.random.default_rng()

    events = []

    for i in range(12):

        signal_type = rng.choice(["RF", "Acoustic"])

        score = float(rng.uniform(0.05, 0.98))

        alert = create_alert(signal_type, score)

        events.append(alert)

    st.session_state.simulation = events


all_events = st.session_state.simulation

if signal_mode == "Both":
    visible_events = all_events
else:
    visible_events = [
        event for event in all_events
        if event["signal_type"] == signal_mode
    ]


# ---------------- ALERT SUMMARY ----------------

summary = generate_alert_summary(visible_events)

st.header("System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Events", summary["total"])
col2.metric("Normal", summary["normal"])
col3.metric("Warnings", summary["warning"])
col4.metric("Critical", summary["critical"])

st.divider()


# ---------------- SIGNAL MONITORING ----------------

st.header("Signal Monitoring")

chart_col1, chart_col2 = st.columns(2)

if signal_mode in ["Both", "RF"]:

    with chart_col1:

        st.subheader("📡 RF Signal Monitor")

        rf_event = next(
            (e for e in reversed(visible_events)
             if e["signal_type"] == "RF"),
            None
        )

        rf_anomaly = (
            rf_event is not None
            and rf_event["status"] != "NORMAL"
        )

        time, signal = generate_signal(
            "RF",
            anomaly=rf_anomaly
        )

        st.plotly_chart(
            create_signal_chart(
                time,
                signal,
                "Simulated RF Waveform"
            ),
            use_container_width=True
        )

if signal_mode in ["Both", "Acoustic"]:

    with chart_col2:

        st.subheader("🔊 Acoustic Signal Monitor")

        audio_event = next(
            (e for e in reversed(visible_events)
             if e["signal_type"] == "Acoustic"),
            None
        )

        audio_anomaly = (
            audio_event is not None
            and audio_event["status"] != "NORMAL"
        )

        time, signal = generate_signal(
            "Acoustic",
            anomaly=audio_anomaly
        )

        st.plotly_chart(
            create_signal_chart(
                time,
                signal,
                "Simulated Acoustic Waveform"
            ),
            use_container_width=True
        )


st.divider()


# ---------------- ANOMALY DETECTION ----------------

st.header("Anomaly Detection Results")

if visible_events:

    df = pd.DataFrame(visible_events)

    df = df.rename(columns={
        "timestamp": "Timestamp",
        "signal_type": "Signal Type",
        "anomaly_score": "Anomaly Score",
        "status": "Status"
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No events available for this signal type.")


# ---------------- ALERT HISTORY ----------------

st.header("🚨 Alert History")

abnormal_events = [
    event for event in visible_events
    if event["status"] in ["WARNING", "CRITICAL"]
]

if abnormal_events:

    for event in reversed(abnormal_events):

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
    st.success("No warning or critical events in this simulation.")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AetherHound | Development Prototype | "
    "All current signals and anomaly scores are simulated. "
    "No real-world covert-channel detection is claimed."
)
