from __future__ import annotations

"""Streamlit UI for CPR analysis.

Features:
- Local video and live camera mode.
- In-place metric updates (no growing lines).
- Optional camera index selection for external cameras.
- Live graph of wrist compression motion.
- Suppress CPM/regularity during detected ventilation pauses.
"""

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from cpr_analyser.analyzer import CPRAnalyzer, CPRMetrics

st.set_page_config(page_title="CPR Analyser", layout="wide")
st.title("CPR Analyse – Frequenz, Regelmäßigkeit, Haltung")

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("Quelle")
    source = st.radio("Videoquelle", ["Lokales Video", "Live Kamera"], index=0)

    st.header("Performance")
    target_fps = st.slider("Auswertungs-FPS", min_value=8, max_value=30, value=15, step=1)
    frame_skip = st.slider("Jede N-te Frame auswerten", min_value=1, max_value=4, value=1, step=1)

    st.header("Signalparameter")
    prominence = st.slider("Peak-Prominenz", min_value=0.001, max_value=0.02, value=0.003, step=0.001)

    # Camera selection only relevant for live mode.
    camera_index = 0
    if source == "Live Kamera":
        st.header("Kamera")
        camera_index = st.number_input(
            "Kamera-Index (0=intern, 1/2=extern)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

# -----------------------------
# UI placeholders (in-place updates)
# -----------------------------
video_ph = st.empty()
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
cpm_ph = metrics_col1.empty()
reg_ph = metrics_col2.empty()
comp_ph = metrics_col3.empty()
vent_ph = metrics_col4.empty()
status_ph = st.empty()
feedback_ph = st.empty()
chart_title_ph = st.empty()
chart_ph = st.empty()


def render_metrics(metrics: CPRMetrics) -> None:
    """Render CPR metrics and feedback without creating new rows."""
    cpm_ph.metric("CPM", "--" if metrics.cpm is None else f"{metrics.cpm:.1f}")
    reg_ph.metric("Regelmäßigkeit", "--" if metrics.regularity is None else f"{metrics.regularity:.2f}")
    comp_ph.metric("Kompressionen", metrics.compression_count)
    vent_ph.metric("Beatmungen", metrics.ventilation_count)

    if metrics.in_ventilation_pause:
        status_ph.caption("Status: Beatmungspause erkannt – CPM/Regelmäßigkeit pausiert.")
    else:
        status_ph.caption(
            f"Rate-Qualität: {metrics.compression_quality} | Haltung: {metrics.posture_label}"
            f" ({'--' if metrics.posture_score is None else f'{metrics.posture_score:.2f}'}) | 30:2-Score:"
            f" {'--' if metrics.ratio_30_2_score is None else f'{metrics.ratio_30_2_score:.2f}'}"
        )

    # Color-coded compression speed feedback.
    if metrics.cpm is None:
        feedback_ph.markdown(
            "<span style='color:gray; font-weight:700'>Keine aktive Druckphase erkannt.</span>",
            unsafe_allow_html=True,
        )
    elif metrics.cpm < 100:
        feedback_ph.markdown(
            "<span style='color:#d62728; font-weight:700'>Bitte schneller drücken (unter 100 CPM).</span>",
            unsafe_allow_html=True,
        )
    elif metrics.cpm > 120:
        feedback_ph.markdown(
            "<span style='color:#d62728; font-weight:700'>Bitte langsamer drücken (über 120 CPM).</span>",
            unsafe_allow_html=True,
        )
    else:
        feedback_ph.markdown(
            "<span style='color:#2ca02c; font-weight:700'>Druckfrequenz OK (100–120 CPM).</span>",
            unsafe_allow_html=True,
        )


def update_motion_chart(signal_history: list[float]) -> None:
    """Draw live wrist-motion graph (sinus-like during compressions)."""
    chart_title_ph.markdown("**Verlauf der Druckbewegung (live)**")
    if not signal_history:
        chart_ph.line_chart(pd.DataFrame({"Signal": []}))
        return
    df = pd.DataFrame({"Signal": signal_history})
    chart_ph.line_chart(df)


if source == "Lokales Video":
    upload = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi", "mkv"])
    run = st.button("Analyse starten", type="primary", disabled=upload is None)

    if run and upload is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0

        analyzer = CPRAnalyzer(fps=fps, prominence=prominence)
        frame_idx = 0
        wall_start = time.perf_counter()
        last_metrics = CPRMetrics()
        signal_history: list[float] = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (960, 540))
                t_sec = frame_idx / fps

                if frame_idx % frame_skip == 0:
                    frame_out, last_metrics = analyzer.process_frame(frame, t_sec)
                else:
                    frame_out = frame

                # Graph behavior:
                # - During compressions: append signal values.
                # - During ventilation pause: reset graph.
                if last_metrics.in_ventilation_pause:
                    signal_history = []
                elif last_metrics.signal_value is not None:
                    signal_history.append(last_metrics.signal_value)
                    signal_history = signal_history[-240:]  # keep latest samples

                video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                render_metrics(last_metrics)
                update_motion_chart(signal_history)

                # Keep original video speed.
                target_time = wall_start + (frame_idx / fps)
                sleep_time = target_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                frame_idx += 1
        finally:
            analyzer.close()
            cap.release()
            st.success("Analyse abgeschlossen")

else:
    run_live = st.button("Live starten", type="primary")
    stop_live = st.button("Live stoppen")

    if "live_stop" not in st.session_state:
        st.session_state.live_stop = False

    if run_live:
        st.session_state.live_stop = False
        cap = cv2.VideoCapture(int(camera_index))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else float(target_fps)

        analyzer = CPRAnalyzer(fps=fps, prominence=prominence)
        frame_idx = 0
        signal_history: list[float] = []
        last_metrics = CPRMetrics()

        try:
            while cap.isOpened() and not st.session_state.live_stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Keine Kamera-Frames verfügbar. Prüfe den Kamera-Index.")
                    break

                frame = cv2.resize(frame, (960, 540))
                t_sec = frame_idx / fps

                if frame_idx % frame_skip == 0:
                    frame_out, last_metrics = analyzer.process_frame(frame, t_sec)
                else:
                    frame_out = frame

                if last_metrics.in_ventilation_pause:
                    signal_history = []
                elif last_metrics.signal_value is not None:
                    signal_history.append(last_metrics.signal_value)
                    signal_history = signal_history[-240:]

                video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                render_metrics(last_metrics)
                update_motion_chart(signal_history)

                time.sleep(max(0.0, 1.0 / target_fps))
                frame_idx += 1

                if stop_live:
                    st.session_state.live_stop = True
        finally:
            analyzer.close()
            cap.release()
            if st.session_state.live_stop:
                st.info("Live-Analyse gestoppt")
