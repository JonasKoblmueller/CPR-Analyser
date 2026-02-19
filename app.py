from __future__ import annotations

"""Streamlit UI for CPR analysis.

Wichtige Design-Entscheidung:
- Video (Datei) wird in *Originalzeit* angezeigt.
- Analyse kann mit niedrigerer Rate laufen (analysis_fps), damit Rechenlast sinkt.
- Dadurch bleibt die Zeitdarstellung korrekt, selbst wenn nicht jedes Frame analysiert wird.
"""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from cpr_analyser.analyzer import CPRAnalyzer, CPRMetrics


# -----------------------------
# UI Grundaufbau
# -----------------------------
st.set_page_config(page_title="CPR Analyser", layout="wide")
st.title("CPR Analyse – Frequenz, Regelmäßigkeit, Haltung")

with st.sidebar:
    st.header("Quelle")
    source = st.radio("Videoquelle", ["Lokales Video", "Live Kamera"], index=0)

    st.header("Performance")
    # Zielrate für Anzeige (primär Live-Kamera relevant)
    target_fps = st.slider("Anzeige-FPS (Live)", min_value=8, max_value=30, value=20, step=1)
    # Entkoppelte Analysefrequenz: reduziert Rechenlast bei gleicher Videozeit
    analysis_fps = st.slider("Analyse-FPS", min_value=4, max_value=30, value=12, step=1)
    prominence = st.slider("Peak-Prominenz", min_value=0.001, max_value=0.02, value=0.003, step=0.001)

# Persistente Platzhalter -> verhindert neue Zeilen pro Frame
video_ph = st.empty()
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
cpm_ph = metrics_col1.empty()
reg_ph = metrics_col2.empty()
comp_ph = metrics_col3.empty()
vent_ph = metrics_col4.empty()
status_ph = st.empty()
feedback_ph = st.empty()


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def render_metrics(metrics: CPRMetrics) -> None:
    """Aktualisiert Metriken in-place (ohne neue UI-Elemente anzulegen)."""
    cpm_ph.metric("CPM", "--" if metrics.cpm is None else f"{metrics.cpm:.1f}")
    reg_ph.metric("Regelmäßigkeit", "--" if metrics.regularity is None else f"{metrics.regularity:.2f}")
    comp_ph.metric("Kompressionen", metrics.compression_count)
    vent_ph.metric("Beatmungen", metrics.ventilation_count)

    status_ph.caption(
        f"Rate-Qualität: {metrics.compression_quality} | Haltung: {metrics.posture_label}"
        f" ({'--' if metrics.posture_score is None else f'{metrics.posture_score:.2f}'}) | 30:2-Score:"
        f" {'--' if metrics.ratio_30_2_score is None else f'{metrics.ratio_30_2_score:.2f}'}"
    )

    # Explizites Echtzeit-Feedback basierend auf CPM
    if metrics.cpm is None:
        feedback_ph.markdown("<span style='color:gray'>Feedback: Warte auf genügend Daten…</span>", unsafe_allow_html=True)
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
            "<span style='color:#2ca02c; font-weight:700'>Druckgeschwindigkeit OK (100–120 CPM).</span>",
            unsafe_allow_html=True,
        )


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
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = (frame_count / fps) if frame_count and frame_count > 0 else None

        analyzer = CPRAnalyzer(fps=fps, prominence=prominence)
        last_metrics = CPRMetrics()
        last_analysis_ts = -1.0

        # Referenzzeit: elapsed wall-time == gewünschte Videozeit
        wall_start = time.perf_counter()

        try:
            while cap.isOpened():
                elapsed = time.perf_counter() - wall_start

                # Stoppe am Videoende (falls Dauer bekannt)
                if duration_sec is not None and elapsed >= duration_sec:
                    break

                # Zeitsynchrone Frame-Selektion: springt auf aktuelle Videozeit
                # Vorteil: keine Zeitverzerrung, selbst wenn Analyse langsamer ist.
                cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (960, 540))

                # Analyse nur mit analysis_fps -> reduziert Last
                if (elapsed - last_analysis_ts) >= (1.0 / analysis_fps):
                    frame_out, last_metrics = analyzer.process_frame(frame, elapsed)
                    last_analysis_ts = elapsed
                else:
                    # Ohne neue Analyse nur Frame anzeigen, letzte Metriken beibehalten
                    frame_out = frame

                video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                render_metrics(last_metrics)

                # Kleine Pause, damit Streamlit/UI nicht 100% CPU zieht
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
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else float(target_fps)

        analyzer = CPRAnalyzer(fps=fps, prominence=prominence)
        last_metrics = CPRMetrics()
        last_analysis_ts = -1.0
        loop_start = time.perf_counter()

        try:
            while cap.isOpened() and not st.session_state.live_stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Keine Kamera-Frames verfügbar.")
                    break

                now = time.perf_counter()
                t_sec = now - loop_start
                frame = cv2.resize(frame, (960, 540))

                if (t_sec - last_analysis_ts) >= (1.0 / analysis_fps):
                    frame_out, last_metrics = analyzer.process_frame(frame, t_sec)
                    last_analysis_ts = t_sec
                else:
                    frame_out = frame

                video_ph.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), channels="RGB")
                render_metrics(last_metrics)

                # Live-Ansicht begrenzen
                time.sleep(max(0.0, (1.0 / target_fps)))

                if stop_live:
                    st.session_state.live_stop = True
        finally:
            analyzer.close()
            cap.release()
            if st.session_state.live_stop:
                st.info("Live-Analyse gestoppt")
