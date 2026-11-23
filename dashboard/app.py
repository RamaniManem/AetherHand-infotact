
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alerts import (
    create_alert,
    generate_alert_summary,
    load_alerts,
    append_alerts,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AetherHound | Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #263449;
        padding: 18px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    .status-note {
        background-color: #172033;
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIMULATION FUNCTIONS
# --------------------------------------------------

def generate_simulated_events(number_of_events=12):
    """Generate demo events using synthetic scores."""

    events = []

    for _ in range(number_of_events):
        signal_type = random.choice(["RF", "Acoustic"])

        # Synthetic anomaly score between 0 and 1
        score = random.random()

        event = create_alert(
            signal_type=signal_type,
            score=score,
        )

        events.append(event)

    return events


def generate_signal(signal_type):
    """Generate a synthetic RF or acoustic waveform."""

    time_values = np.linspace(0, 1, 500)

    if signal_type == "RF":
        signal_values = (
            np.sin(2 * np.pi * 8 * time_values)
            + 0.35 * np.sin(2 * np.pi * 20 * time_values)
            + np.random.normal(0, 0.15, 500)
        )
    else:
        signal_values = (
            np.sin(2 * np.pi * 4 * time_values)
            + 0.25 * np.sin(2 * np.pi * 12 * time_values)
            + np.random.normal(0, 0.12, 500)
        )

    return time_values, signal_values


def create_signal_chart(signal_type):
    """Create a Plotly chart for the selected signal."""

    time_values, signal_values = generate_signal(signal_type)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=time_values,
            y=signal_values,
            mode="lines",
            name=f"{signal_type} Signal",
        )
    )

    figure.update_layout(
        title=f"Simulated {signal_type} Signal",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return figure


# --------------------------------------------------
# INITIALIZE SAVED ALERTS
# --------------------------------------------------

all_events = load_alerts()

if not all_events:
    initial_events = generate_simulated_events(12)
    append_alerts(initial_events)
    all_events = load_alerts()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ AetherHound")

st.sidebar.caption(
    "Air-Gapped RF & Acoustic Covert Channel Detector"
)

st.sidebar.divider()

signal_mode = st.sidebar.selectbox(
    "Filter by Signal Type",
    ["Both", "RF", "Acoustic"],
)

severity_mode = st.sidebar.selectbox(
    "Filter by Severity",
    ["All", "NORMAL", "WARNING", "CRITICAL"],
)

st.sidebar.divider()

if st.sidebar.button(
    "Generate New Simulation",
    use_container_width=True,
):
    new_events = generate_simulated_events(12)
    append_alerts(new_events)

    st.rerun()

st.sidebar.caption(
    "Demo mode: all generated events are synthetic."
)


# --------------------------------------------------
# FILTER ALERTS
# --------------------------------------------------

if signal_mode == "Both":
    visible_events = all_events
else:
    visible_events = [
        event
        for event in all_events
        if event.get("signal_type") == signal_mode
    ]


if severity_mode == "All":
    display_events = visible_events
else:
    display_events = [
        event
        for event in visible_events
        if event.get("status") == severity_mode
    ]


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("🛡️ AetherHound Monitoring Dashboard")

st.markdown(
    """
    <div class="status-note">
    Monitoring simulated RF and acoustic signals for
    demonstration of anomaly detection and alert management.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

summary = generate_alert_summary(display_events)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Total Alerts",
    summary["total"],
)

metric2.metric(
    "Critical",
    summary["critical"],
)

metric3.metric(
    "Warning",
    summary["warning"],
)

metric4.metric(
    "Normal",
    summary["normal"],
)


# --------------------------------------------------
# SIGNAL VISUALIZATION
# --------------------------------------------------

st.subheader("Signal Visualization")

if signal_mode == "Both":
    chart1, chart2 = st.columns(2)

    with chart1:
        st.plotly_chart(
            create_signal_chart("RF"),
            use_container_width=True,
        )

    with chart2:
        st.plotly_chart(
            create_signal_chart("Acoustic"),
            use_container_width=True,
        )

elif signal_mode == "RF":
    st.plotly_chart(
        create_signal_chart("RF"),
        use_container_width=True,
    )

else:
    st.plotly_chart(
        create_signal_chart("Acoustic"),
        use_container_width=True,
    )


# --------------------------------------------------
# ALERT TABLE
# --------------------------------------------------

st.subheader("Anomaly Detection Results")

if display_events:
    dataframe = pd.DataFrame(display_events)

    required_columns = [
        "timestamp",
        "signal_type",
        "score",
        "status",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in dataframe.columns
    ]

    dataframe = dataframe[available_columns]

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("No alerts match the selected filters.")


# --------------------------------------------------
# ALERT HISTORY
# --------------------------------------------------

st.subheader("Alert History")

abnormal_events = [
    event
    for event in display_events
    if event.get("status") in ["WARNING", "CRITICAL"]
]

if abnormal_events:
    latest_alerts = abnormal_events[-20:][::-1]

    history_dataframe = pd.DataFrame(latest_alerts)

    st.dataframe(
        history_dataframe,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.success(
        "No warning or critical alerts match the current filters."
    )


# --------------------------------------------------
# DOWNLOAD ALERTS
# --------------------------------------------------

st.subheader("Export Alert Data")

download_col1, download_col2 = st.columns(2)

with download_col1:
    json_data = json.dumps(
        display_events,
        indent=4,
    )

    st.download_button(
        label="Download Filtered Alerts (JSON)",
        data=json_data,
        file_name="aetherhound_filtered_alerts.json",
        mime="application/json",
        use_container_width=True,
    )

with download_col2:
    if display_events:
        csv_dataframe = pd.DataFrame(display_events)

        csv_data = csv_dataframe.to_csv(
            index=False,
        ).encode("utf-8")

        st.download_button(
            label="Download Filtered Alerts (CSV)",
            data=csv_data,
            file_name="aetherhound_filtered_alerts.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.button(
            "Download Filtered Alerts (CSV)",
            disabled=True,
            use_container_width=True,
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AetherHound | Internship Prototype | "
    "Synthetic data is used for demonstration. "
    "This dashboard does not represent real RF or acoustic detection."
)
