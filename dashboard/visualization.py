
import numpy as np
import plotly.graph_objects as go


def generate_signal(signal_type="RF", anomaly=False, samples=500):
    """
    Generate illustrative synthetic signals.
    These are not real RF or acoustic measurements.
    """
    time = np.linspace(0, 1, samples)

    if signal_type == "RF":
        signal = (
            np.sin(2 * np.pi * 12 * time)
            + 0.35 * np.sin(2 * np.pi * 25 * time)
        )
    else:
        signal = (
            np.sin(2 * np.pi * 5 * time)
            + 0.25 * np.sin(2 * np.pi * 11 * time)
        )

    signal += np.random.normal(0, 0.08, samples)

    if anomaly:
        signal += 0.8 * np.sin(2 * np.pi * 45 * time)

    return time, signal


def create_signal_chart(time, signal, title):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            name="Signal"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (simulated seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=30)
    )

    return fig
