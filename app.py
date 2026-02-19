from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from cpr_analyser.analyzer import CPRAnalyzer

st.set_page_config(page_title="CPR Analyser", layout="wide")
st.title("CPR Analyse – Frequenz, Regelmäßigkeit, Haltung")

with st.sidebar:
    st.header("Quelle")
    source = st.radio("Videoquelle", ["Lokales Video", "Live Kamera"], index=0)

    st.header("Performance")
    target_fps = st.slider("Auswertungs-FPS", min_value=8, max_value=30, value=15, step=1)
    frame_skip = st.slider("Jede N-te Frame auswerten", min_value=1, max_value=4, value=1, step=1)

    st.header("Signalparameter")
    prominence = st.slider("Peak-Prominenz", min_value=0.001, max_value=0.02, value=0.003, step=0.001)

video_ph = st.empty()
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

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

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                frame = cv2.resize(frame, (960, 540))
                t_sec = frame_idx / fps
                frame_out, metrics = analyzer.process_frame(frame, t_sec)

                video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                metrics_col1.metric("CPM", "--" if metrics.cpm is None else f"{metrics.cpm:.1f}")
                metrics_col2.metric("Regelmäßigkeit", "--" if metrics.regularity is None else f"{metrics.regularity:.2f}")
                metrics_col3.metric("Kompressionen", metrics.compression_count)
                metrics_col4.metric("Beatmungen", metrics.ventilation_count)
                st.caption(
                    f"Rate-Qualität: {metrics.compression_quality} | Haltung: {metrics.posture_label}"
                    f" ({'--' if metrics.posture_score is None else f'{metrics.posture_score:.2f}'}) | 30:2-Score:"
                    f" {'--' if metrics.ratio_30_2_score is None else f'{metrics.ratio_30_2_score:.2f}'}"
                )

                time.sleep(max(0.0, (1.0 / target_fps)))
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
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else float(target_fps)

        analyzer = CPRAnalyzer(fps=fps, prominence=prominence)
        frame_idx = 0

        try:
            while cap.isOpened() and not st.session_state.live_stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Keine Kamera-Frames verfügbar.")
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                frame = cv2.resize(frame, (960, 540))
                t_sec = frame_idx / fps
                frame_out, metrics = analyzer.process_frame(frame, t_sec)

                video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                metrics_col1.metric("CPM", "--" if metrics.cpm is None else f"{metrics.cpm:.1f}")
                metrics_col2.metric("Regelmäßigkeit", "--" if metrics.regularity is None else f"{metrics.regularity:.2f}")
                metrics_col3.metric("Kompressionen", metrics.compression_count)
                metrics_col4.metric("Beatmungen", metrics.ventilation_count)
                st.caption(
                    f"Rate-Qualität: {metrics.compression_quality} | Haltung: {metrics.posture_label}"
                    f" ({'--' if metrics.posture_score is None else f'{metrics.posture_score:.2f}'}) | 30:2-Score:"
                    f" {'--' if metrics.ratio_30_2_score is None else f'{metrics.ratio_30_2_score:.2f}'}"
                )

                time.sleep(max(0.0, (1.0 / target_fps)))
                frame_idx += 1

                if stop_live:
                    st.session_state.live_stop = True
        finally:
            analyzer.close()
            cap.release()
            if st.session_state.live_stop:
                st.info("Live-Analyse gestoppt")
