from __future__ import annotations

import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import streamlit as st

from cpr_analyser.analyzer import CPRAnalyzer, CPRMetrics

st.set_page_config(page_title="CPR Analyser", layout="wide")
st.title("CPR Analyse – Frequenz, Regelmäßigkeit, Haltung")

with st.sidebar:
    st.header("Quelle")
    source = st.radio("Videoquelle", ["Lokales Video", "Live Kamera"], index=0)

    st.header("Performance")
    target_fps = st.slider("Anzeige-FPS Ziel", min_value=8, max_value=30, value=24, step=1)
    analysis_fps = st.slider("Analyse-FPS Ziel", min_value=4, max_value=30, value=12, step=1)
    ui_update_fps = st.slider("UI-Update-FPS", min_value=4, max_value=20, value=10, step=1)
    frame_skip = st.slider("Zusätzliches Frame-Skip", min_value=1, max_value=4, value=1, step=1)

    st.header("Signalparameter")
    prominence = st.slider("Peak-Prominenz", min_value=0.001, max_value=0.02, value=0.003, step=0.001)

    st.header("Anzeige")
    show_motion_graph = st.checkbox("Druckbewegung als Live-Graph anzeigen", value=False)

    camera_index = 0
    if source == "Live Kamera":
        st.header("Kamera")
        camera_index = st.number_input("Kamera-Index", min_value=0, max_value=10, value=0, step=1)

video_ph = st.empty()
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
cpm_ph = metrics_col1.empty()
reg_ph = metrics_col2.empty()
comp_ph = metrics_col3.empty()
vent_ph = metrics_col4.empty()
status_ph = st.empty()
feedback_ph = st.empty()
chart_ph = st.empty()


def render_metrics(metrics: CPRMetrics) -> None:
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

    if metrics.cpm is None:
        feedback_ph.markdown("<span style='color:gray;font-weight:700'>Keine aktive Druckphase erkannt.</span>", unsafe_allow_html=True)
    elif metrics.cpm < 100:
        feedback_ph.markdown("<span style='color:#d62728;font-weight:700'>Bitte schneller drücken (unter 100 CPM).</span>", unsafe_allow_html=True)
    elif metrics.cpm > 120:
        feedback_ph.markdown("<span style='color:#d62728;font-weight:700'>Bitte langsamer drücken (über 120 CPM).</span>", unsafe_allow_html=True)
    else:
        feedback_ph.markdown("<span style='color:#2ca02c;font-weight:700'>Druckfrequenz OK (100–120 CPM).</span>", unsafe_allow_html=True)


def maybe_render_graph(signal_hist: deque[float]) -> None:
    if not show_motion_graph:
        chart_ph.empty()
        return
    if signal_hist:
        chart_ph.line_chart(list(signal_hist))
    else:
        chart_ph.line_chart([])


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
        signal_hist: deque[float] = deque(maxlen=240)
        last_metrics = CPRMetrics()

        analysis_step = max(1, int(round(fps / max(analysis_fps, 1)))) * frame_skip

        frame_idx = -1
        wall_start = time.perf_counter()
        next_ui_time = 0.0

        try:
            while cap.isOpened():
                elapsed = time.perf_counter() - wall_start
                desired_idx = int(elapsed * fps)

                # Catch up when processing is slower than source video speed.
                while frame_idx + 1 < desired_idx:
                    if not cap.grab():
                        break
                    frame_idx += 1

                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                frame = cv2.resize(frame, (960, 540))
                t_sec = frame_idx / fps

                if frame_idx % analysis_step == 0:
                    frame_out, last_metrics = analyzer.process_frame(frame, t_sec)
                else:
                    frame_out = frame

                if last_metrics.in_ventilation_pause:
                    signal_hist.clear()
                elif last_metrics.signal_value is not None:
                    signal_hist.append(last_metrics.signal_value)

                # Decouple UI update rate from processing rate (big speed gain in Streamlit).
                if elapsed >= next_ui_time:
                    video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                    render_metrics(last_metrics)
                    maybe_render_graph(signal_hist)
                    next_ui_time = elapsed + (1.0 / ui_update_fps)

                # Tiny sleep to avoid busy-looping at 100% single core.
                time.sleep(0.001)
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
        signal_hist: deque[float] = deque(maxlen=240)
        last_metrics = CPRMetrics()

        analysis_step = max(1, int(round(fps / max(analysis_fps, 1)))) * frame_skip
        frame_idx = 0
        loop_start = time.perf_counter()
        next_ui_time = 0.0
        next_frame_time = 0.0

        try:
            while cap.isOpened() and not st.session_state.live_stop:
                now = time.perf_counter() - loop_start
                ret, frame = cap.read()
                if not ret:
                    st.warning("Keine Kamera-Frames verfügbar. Prüfe den Kamera-Index.")
                    break

                frame = cv2.resize(frame, (960, 540))
                t_sec = frame_idx / fps

                if frame_idx % analysis_step == 0:
                    frame_out, last_metrics = analyzer.process_frame(frame, t_sec)
                else:
                    frame_out = frame

                if last_metrics.in_ventilation_pause:
                    signal_hist.clear()
                elif last_metrics.signal_value is not None:
                    signal_hist.append(last_metrics.signal_value)

                if now >= next_ui_time:
                    video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                    render_metrics(last_metrics)
                    maybe_render_graph(signal_hist)
                    next_ui_time = now + (1.0 / ui_update_fps)

                frame_idx += 1

                if stop_live:
                    st.session_state.live_stop = True

                # Pace only if we are ahead; if behind do not sleep.
                frame_interval = 1.0 / max(target_fps, 1)
                next_frame_time += frame_interval
                lag = next_frame_time - (time.perf_counter() - loop_start)
                if lag > 0:
                    time.sleep(lag)
        finally:
            analyzer.close()
            cap.release()
            if st.session_state.live_stop:
                st.info("Live-Analyse gestoppt")
