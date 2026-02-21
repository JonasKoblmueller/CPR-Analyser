from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from cpr_analyser.analyzer import CPRAnalyzer, CPRMetrics

DEFAULT_ANALYSIS_FPS = 15.0
PROCESS_SIZE = (640, 360)
DISPLAY_SIZE = (640, 360)
FPS_UPDATE_SEC = 0.5
MAX_ALLOWED_LAG_SEC = 0.20
SIGNAL_HISTORY_LEN = 240
METRICS_UPDATE_SEC = 0.10
CHART_UPDATE_SEC = 0.25


class LatestFrameBuffer:
    """Thread-safe latest-frame container for live capture."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._seq = -1

    def push(self, frame, timestamp: float) -> None:
        with self._lock:
            self._seq += 1
            self._frame = frame
            self._timestamp = timestamp

    def latest(self):
        with self._lock:
            return self._frame, self._timestamp, self._seq


def camera_capture_worker(cap: cv2.VideoCapture, buffer: LatestFrameBuffer, stop_event: threading.Event) -> None:
    """Continuously read camera frames and keep only the latest one."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        buffer.push(frame, time.perf_counter())


def advance_visual_fps(
    window_start: float,
    window_frames: int,
    visual_fps: float | None,
) -> tuple[float, int, float | None]:
    """Update measured visualization FPS in a rolling time window."""
    window_frames += 1
    now = time.perf_counter()
    elapsed = now - window_start
    if elapsed >= FPS_UPDATE_SEC:
        visual_fps = window_frames / elapsed
        return now, 0, visual_fps
    return window_start, window_frames, visual_fps


def sync_analysis_step(analysis_step: int, wall_start: float, analysis_fps: float) -> tuple[int, float, float]:
    """Return analysis step, target time and lag; drop steps when lag grows too large."""
    now = time.perf_counter()
    target_t = analysis_step / analysis_fps
    lag = now - (wall_start + target_t)

    if lag > MAX_ALLOWED_LAG_SEC:
        realtime_step = int((now - wall_start) * analysis_fps)
        if realtime_step > analysis_step:
            analysis_step = realtime_step
            target_t = analysis_step / analysis_fps
            lag = now - (wall_start + target_t)

    return analysis_step, target_t, lag


st.set_page_config(page_title="CPR Analyser", layout="wide")
st.title("CPR Analyse - Frequenz, Regelmaessigkeit, Haltung")

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("Quelle")
    source = st.radio("Videoquelle", ["Lokales Video", "Live Kamera"], index=0)

    st.header("Performance")
    analysis_fps = st.slider(
        "Analyse-FPS",
        min_value=8,
        max_value=30,
        value=int(DEFAULT_ANALYSIS_FPS),
        step=1,
    )
    st.caption(
        f"Analyse-FPS: {analysis_fps:.0f} | Prozessauflösung: "
        f"{PROCESS_SIZE[0]}x{PROCESS_SIZE[1]} | Max-Lag: {MAX_ALLOWED_LAG_SEC:.2f}s"
    )

    st.header("Signalparameter")
    prominence = st.slider("Peak-Prominenz", min_value=0.001, max_value=0.02, value=0.003, step=0.001)

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
metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5, metrics_col6 = st.columns(6)
cpm_ph = metrics_col1.empty()
reg_ph = metrics_col2.empty()
comp_ph = metrics_col3.empty()
vent_ph = metrics_col4.empty()
src_fps_ph = metrics_col5.empty()
vis_fps_ph = metrics_col6.empty()
status_ph = st.empty()
feedback_ph = st.empty()
runtime_debug_ph = st.empty()
chart_title_ph = st.empty()
chart_ph = st.empty()


def render_metrics(metrics: CPRMetrics) -> None:
    """Render CPR metrics and feedback without creating new rows."""
    cpm_ph.metric("CPM", "--" if metrics.cpm is None else f"{metrics.cpm:.1f}")
    reg_ph.metric("Regelmaessigkeit", "--" if metrics.regularity is None else f"{metrics.regularity:.2f}")
    comp_ph.metric("Kompressionen", metrics.compression_count)
    vent_ph.metric("Beatmungen (aus)", "--")
    status_ph.caption(
        f"Rate-Qualitaet: {metrics.compression_quality} | Haltung: {metrics.posture_label}"
        f" ({'--' if metrics.posture_score is None else f'{metrics.posture_score:.2f}'})"
    )

    if metrics.cpm is None:
        feedback_ph.markdown(
            "<span style='color:gray; font-weight:700'>Keine aktive Druckphase erkannt.</span>",
            unsafe_allow_html=True,
        )
    elif metrics.cpm < 100:
        feedback_ph.markdown(
            "<span style='color:#d62728; font-weight:700'>Bitte schneller druecken (unter 100 CPM).</span>",
            unsafe_allow_html=True,
        )
    elif metrics.cpm > 120:
        feedback_ph.markdown(
            "<span style='color:#d62728; font-weight:700'>Bitte langsamer druecken (ueber 120 CPM).</span>",
            unsafe_allow_html=True,
        )
    else:
        feedback_ph.markdown(
            "<span style='color:#2ca02c; font-weight:700'>Druckfrequenz OK (100-120 CPM).</span>",
            unsafe_allow_html=True,
        )


def update_motion_chart(signal_history: list[float]) -> None:
    """Draw live wrist-motion graph."""
    chart_title_ph.markdown("**Verlauf der Druckbewegung (live)**")
    if not signal_history:
        chart_ph.line_chart(pd.DataFrame({"Signal": []}))
        return
    chart_ph.line_chart(pd.DataFrame({"Signal": signal_history}))


def render_fps(source_label: str, source_fps: float | None, visual_fps: float | None) -> None:
    """Render source FPS and measured visualization FPS."""
    src_fps_ph.metric(source_label, "--" if source_fps is None else f"{source_fps:.2f}")
    vis_fps_ph.metric("Visualisierungs-FPS", "--" if visual_fps is None else f"{visual_fps:.1f}")


def render_runtime_debug(lag_sec: float | None, dropped_frames: int | None) -> None:
    lag_text = "--" if lag_sec is None else f"{max(0.0, lag_sec) * 1000:.0f}"
    dropped_text = "--" if dropped_frames is None else str(dropped_frames)
    runtime_debug_ph.caption(f"Timing-Lag: {lag_text} ms | Dropped Frames: {dropped_text}")


if source == "Lokales Video":
    render_fps("Original-Video-FPS", None, None)
    render_runtime_debug(None, None)
    upload = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi", "mkv"])
    run = st.button("Analyse starten", type="primary", disabled=upload is None)

    if run and upload is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if source_fps and source_fps > 0 else 30.0
        effective_analysis_fps = max(1.0, min(float(analysis_fps), float(source_fps)))

        analyzer = CPRAnalyzer(
            fps=effective_analysis_fps,
            prominence=prominence,
            min_valid_cpm=20.0,
            max_valid_cpm=170.0,
            enable_ventilation_detection=False,
        )
        source_frame_idx = -1
        analysis_step = 0
        wall_start = time.perf_counter()
        last_metrics = CPRMetrics()
        signal_history: list[float] = []
        visual_fps: float | None = None
        fps_window_start = time.perf_counter()
        fps_window_frames = 0
        dropped_source_frames = 0
        last_metrics_render = 0.0
        last_chart_render = 0.0

        try:
            while cap.isOpened():
                analysis_step, target_t, lag_sec = sync_analysis_step(analysis_step, wall_start, effective_analysis_fps)
                target_src_idx = max(0, int(round(target_t * source_fps)))

                has_frame = True
                while source_frame_idx + 1 < target_src_idx:
                    if not cap.grab():
                        has_frame = False
                        break
                    source_frame_idx += 1
                    dropped_source_frames += 1

                if not has_frame:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                source_frame_idx += 1

                frame = cv2.resize(frame, PROCESS_SIZE)
                t_sec = source_frame_idx / source_fps
                frame_out, last_metrics = analyzer.process_frame(frame, t_sec)

                if last_metrics.signal_value is not None:
                    signal_history.append(last_metrics.signal_value)
                    signal_history = signal_history[-SIGNAL_HISTORY_LEN:]

                fps_window_start, fps_window_frames, visual_fps = advance_visual_fps(
                    fps_window_start, fps_window_frames, visual_fps
                )

                frame_display = frame_out if DISPLAY_SIZE == PROCESS_SIZE else cv2.resize(frame_out, DISPLAY_SIZE)
                video_ph.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")
                now_ui = time.perf_counter()
                if now_ui - last_metrics_render >= METRICS_UPDATE_SEC:
                    render_metrics(last_metrics)
                    render_fps("Original-Video-FPS", source_fps, visual_fps)
                    render_runtime_debug(lag_sec, dropped_source_frames)
                    last_metrics_render = now_ui
                if now_ui - last_chart_render >= CHART_UPDATE_SEC:
                    update_motion_chart(signal_history)
                    last_chart_render = now_ui

                analysis_step += 1
                next_tick = wall_start + (analysis_step / effective_analysis_fps)
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            analyzer.close()
            cap.release()
            st.success("Analyse abgeschlossen")

else:
    render_fps("Kamera-FPS", None, None)
    render_runtime_debug(None, None)
    run_live = st.button("Live starten", type="primary")
    stop_live = st.button("Live stoppen")

    if "live_stop" not in st.session_state:
        st.session_state.live_stop = False

    if run_live:
        st.session_state.live_stop = False
        cap = cv2.VideoCapture(int(camera_index))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error("Kamera konnte nicht geoeffnet werden. Bitte Kamera-Index pruefen.")
        else:
            camera_fps = cap.get(cv2.CAP_PROP_FPS)
            camera_fps = camera_fps if camera_fps and camera_fps > 0 else 30.0
            effective_analysis_fps = max(1.0, min(float(analysis_fps), float(camera_fps)))

            analyzer = CPRAnalyzer(
                fps=effective_analysis_fps,
                prominence=prominence,
                min_valid_cpm=20.0,
                max_valid_cpm=170.0,
                enable_ventilation_detection=False,
            )
            frame_buffer = LatestFrameBuffer()
            stop_event = threading.Event()
            capture_thread = threading.Thread(
                target=camera_capture_worker, args=(cap, frame_buffer, stop_event), daemon=True
            )
            capture_thread.start()

            analysis_step = 0
            live_start_wall = time.perf_counter()
            wall_start = live_start_wall
            last_metrics = CPRMetrics()
            signal_history: list[float] = []
            visual_fps: float | None = None
            fps_window_start = time.perf_counter()
            fps_window_frames = 0
            last_seq = -1
            dropped_live_frames = 0
            no_frame_deadline = time.perf_counter() + 2.0
            last_metrics_render = 0.0
            last_chart_render = 0.0

            try:
                while not st.session_state.live_stop and not stop_event.is_set():
                    if stop_live:
                        st.session_state.live_stop = True
                        break

                    analysis_step, _, lag_sec = sync_analysis_step(analysis_step, wall_start, effective_analysis_fps)

                    frame, _, seq = frame_buffer.latest()
                    now = time.perf_counter()
                    if frame is None:
                        if now > no_frame_deadline:
                            st.warning("Keine Kamera-Frames verfuegbar. Pruefe den Kamera-Index.")
                            break
                        time.sleep(0.005)
                        continue
                    no_frame_deadline = now + 2.0

                    if last_seq >= 0 and seq == last_seq:
                        analysis_step += 1
                        next_tick = wall_start + (analysis_step / effective_analysis_fps)
                        sleep_time = next_tick - time.perf_counter()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue

                    if last_seq >= 0 and seq > (last_seq + 1):
                        dropped_live_frames += seq - last_seq - 1
                    last_seq = seq

                    frame = cv2.resize(frame, PROCESS_SIZE)
                    t_sec = max(0.0, time.perf_counter() - live_start_wall)
                    frame_out, last_metrics = analyzer.process_frame(frame, t_sec)

                    if last_metrics.signal_value is not None:
                        signal_history.append(last_metrics.signal_value)
                        signal_history = signal_history[-SIGNAL_HISTORY_LEN:]

                    fps_window_start, fps_window_frames, visual_fps = advance_visual_fps(
                        fps_window_start, fps_window_frames, visual_fps
                    )

                    frame_display = frame_out if DISPLAY_SIZE == PROCESS_SIZE else cv2.resize(frame_out, DISPLAY_SIZE)
                    video_ph.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")
                    now_ui = time.perf_counter()
                    if now_ui - last_metrics_render >= METRICS_UPDATE_SEC:
                        render_metrics(last_metrics)
                        render_fps("Kamera-FPS", camera_fps, visual_fps)
                        render_runtime_debug(lag_sec, dropped_live_frames)
                        last_metrics_render = now_ui
                    if now_ui - last_chart_render >= CHART_UPDATE_SEC:
                        update_motion_chart(signal_history)
                        last_chart_render = now_ui

                    analysis_step += 1
                    next_tick = wall_start + (analysis_step / effective_analysis_fps)
                    sleep_time = next_tick - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                stop_event.set()
                capture_thread.join(timeout=1.0)
                analyzer.close()
                cap.release()
                if st.session_state.live_stop:
                    st.info("Live-Analyse gestoppt")
