from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import streamlit as st

from cpr_analyser.analyzer import CPRAnalyzer, CPRMetrics

# Streamlit-UI fuer die CPR-Analyse.
# Wichtige Idee:
# - Analyse und Anzeige laufen mit getrennten Takten.
# - Upload-Videos werden deterministisch auf der Quell-Zeitachse analysiert.
# - Live-Video nutzt einen Capture-Thread, damit kein Kamera-Backlog entsteht.

# -----------------------------
# Projekt-Defaults
# -----------------------------
# Analyse-FPS: Wie oft pro Sekunde die KI ausgewertet wird.
# Anzeige-FPS: Wie oft das Bild im UI aktualisiert wird.
# Beide sind getrennt, damit die Analyse stabil bleibt, auch wenn das UI langsamer ist.
DEFAULT_ANALYSIS_FPS = 15.0
DEFAULT_DISPLAY_FPS = 12.0
PROCESS_SIZE = (640, 360)
DISPLAY_SIZE = (640, 360)
FPS_UPDATE_SEC = 0.5
MAX_ALLOWED_LAG_SEC = 0.20
SIGNAL_HISTORY_LEN = 240
METRICS_HZ = 10.0
CHART_HZ = 4.0
SLEEP_CAP_SEC = 0.01


@dataclass
class FramePacket:
    """Ein einzelnes Roh-Frame aus der Quelle (Video/Kamera)."""

    frame: Optional[object] = None
    t_sec: float = 0.0
    seq: int = -1
    capture_ms: float = 0.0


@dataclass
class AnalysisPacket:
    """Ein bereits analysiertes Frame inkl. Metriken."""

    frame: Optional[object] = None
    metrics: Optional[CPRMetrics] = None
    t_sec: float = 0.0
    seq: int = -1
    analysis_ms: float = 0.0


class LatestFrameBuffer:
    """Thread-sicherer Puffer, der immer nur das neueste Rohframe speichert.

    Es gibt bewusst keine Queue:
    Alte Frames werden ueberschrieben, damit Live-Analyse nicht hinterherlaeuft.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet = FramePacket()

    def push(self, frame, t_sec: float, seq: int, capture_ms: float) -> None:
        with self._lock:
            self._packet = FramePacket(frame=frame, t_sec=t_sec, seq=seq, capture_ms=capture_ms)

    def latest(self) -> FramePacket:
        with self._lock:
            return self._packet


class LatestAnalysisBuffer:
    """Thread-sicherer Puffer fuer das letzte Analyse-Ergebnis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet = AnalysisPacket()

    def push(self, frame, metrics: CPRMetrics, t_sec: float, seq: int, analysis_ms: float) -> None:
        with self._lock:
            self._packet = AnalysisPacket(
                frame=frame, metrics=metrics, t_sec=t_sec, seq=seq, analysis_ms=analysis_ms
            )

    def latest(self) -> AnalysisPacket:
        with self._lock:
            return self._packet


def ema(previous: Optional[float], value: float, alpha: float = 0.2) -> float:
    """Glaettet Zeitmessungen, damit die Diagnosezeile nicht springt."""
    return value if previous is None else ((1.0 - alpha) * previous + alpha * value)


def consume_tick(next_tick: float, now: float, hz: float) -> tuple[bool, float]:
    """Einfacher Scheduler fuer periodische Tasks (Analyse/Anzeige).

    Rueckgabe:
    - `due=True`, wenn der Task jetzt ausgefuehrt werden soll
    - `next_tick` bereits auf den naechsten Sollzeitpunkt vorgezogen
    """
    step = 1.0 / max(hz, 1e-6)
    if now < next_tick:
        return False, next_tick
    while next_tick <= now:
        next_tick += step
    return True, next_tick


def sync_analysis_step(analysis_step: int, wall_start: float, analysis_fps: float) -> tuple[int, float, float]:
    """Hilfsfunktion fuer wall-clock-basierte Analyse-Loops mit Lag-Schutz.

    Wird in der aktuellen App vor allem fuer Echtzeit-Szenarien genutzt:
    Wenn der Loop zu weit hinter dem Soll liegt, werden Schritte uebersprungen,
    damit die Zeitachse nicht verlangsamt wird.
    """
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


def advance_visual_fps(
    window_start: float,
    window_frames: int,
    visual_fps: Optional[float],
) -> tuple[float, int, Optional[float]]:
    """Misst die tatsaechlich erreichte Anzeige-FPS ueber ein Zeitfenster."""
    window_frames += 1
    now = time.perf_counter()
    elapsed = now - window_start
    if elapsed >= FPS_UPDATE_SEC:
        visual_fps = window_frames / elapsed
        return now, 0, visual_fps
    return window_start, window_frames, visual_fps


def camera_capture_worker(
    cap: cv2.VideoCapture,
    buffer: LatestFrameBuffer,
    stop_event: threading.Event,
    start_wall: float,
) -> None:
    """Capture-Thread fuer Live-Modus.

    Aufgabe:
    - Kamera kontinuierlich lesen
    - auf Verarbeitungsaufloesung skalieren
    - immer nur das neueste Frame im Puffer halten
    """
    seq = -1
    while not stop_event.is_set():
        start = time.perf_counter()
        ret, frame = cap.read()
        capture_ms = (time.perf_counter() - start) * 1000.0
        if not ret:
            time.sleep(0.005)
            continue
        frame = cv2.resize(frame, PROCESS_SIZE)
        seq += 1
        buffer.push(frame=frame, t_sec=max(0.0, time.perf_counter() - start_wall), seq=seq, capture_ms=capture_ms)


st.set_page_config(page_title="CPR Analyser", layout="wide")
st.title("CPR Analyse - Frequenz, Regelmaessigkeit, Haltung")

with st.sidebar:
    st.header("Quelle")
    source = st.radio("Videoquelle", ["Lokales Video", "Live Kamera"], index=0)

    st.header("Pipeline")
    analysis_fps = st.slider("Analyse-FPS", min_value=8, max_value=30, value=int(DEFAULT_ANALYSIS_FPS), step=1)
    display_fps = st.slider("Anzeige-FPS", min_value=10, max_value=30, value=int(DEFAULT_DISPLAY_FPS), step=1)

    st.header("Signal")
    prominence = st.slider("Peak-Prominenz", min_value=0.001, max_value=0.02, value=0.003, step=0.001)
    signal_source_label = st.selectbox("Signalquelle", ["Palm-Center", "Wrist-0"], index=0)
    pose_active = st.checkbox("Pose aktiv", value=False)

    st.header("Anzeige")
    light_mode = st.checkbox("Light-Mode (ohne Chart)", value=False)
    opencv_fallback = st.checkbox("OpenCV High-FPS Fallback", value=False)

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

    st.caption(
        f"Analyse: {analysis_fps} FPS | Anzeige: {display_fps} FPS | "
        f"Aufloesung: {PROCESS_SIZE[0]}x{PROCESS_SIZE[1]}"
    )
    st.caption("Hinweis: Bei lokalem Video ist die Analyse deterministisch und quellenbasiert (reproduzierbar).")

# -----------------------------
# UI-Elemente
# -----------------------------
signal_mode_map = {"Palm-Center": "palm", "Wrist-0": "wrist"}
hand_signal_mode = signal_mode_map[signal_source_label]

video_ph = st.empty()
metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
cpm_ph = metrics_col1.empty()
reg_ph = metrics_col2.empty()
comp_ph = metrics_col3.empty()
vent_ph = metrics_col4.empty()
vis_fps_ph = metrics_col5.empty()
status_ph = st.empty()
feedback_ph = st.empty()
runtime_debug_ph = st.empty()
chart_title_ph = st.empty()
chart_ph = st.empty()


def render_metrics(metrics: CPRMetrics) -> None:
    """Aktualisiert die sichtbaren Kennzahlen und das Text-Feedback im UI."""
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


def render_fps(visual_fps: Optional[float]) -> None:
    """Zeigt nur die relevante Anzeige-FPS (Original-FPS ist absichtlich ausgeblendet)."""
    vis_fps_ph.metric("Visualisierungs-FPS", "--" if visual_fps is None else f"{visual_fps:.1f}")


def update_motion_chart(signal_history: list[float]) -> None:
    """Zeigt den Verlauf des Bewegungssignals (vereinfachte Trendanzeige)."""
    chart_title_ph.markdown("**Verlauf der Druckbewegung (live)**")
    if not signal_history:
        chart_ph.line_chart(pd.DataFrame({"Signal": []}))
        return
    chart_ph.line_chart(pd.DataFrame({"Signal": signal_history}))


def render_runtime_debug(
    profile: dict[str, Optional[float]],
    dropped_capture: int,
    dropped_analysis: int,
    metrics: Optional[CPRMetrics],
) -> None:
    """Kompakte Diagnosezeile fuer Timing- und Signal-Qualitaet."""
    sample_fps = "--" if metrics is None or metrics.sample_fps_est is None else f"{metrics.sample_fps_est:.1f}"
    valid_intervals = "--" if metrics is None else str(metrics.valid_intervals)
    hold_age = "--" if metrics is None or metrics.hold_age_ms is None else f"{metrics.hold_age_ms:.0f}"
    confidence = "--" if metrics is None or metrics.signal_confidence is None else f"{metrics.signal_confidence:.2f}"
    capture_ms = "--" if profile["capture_ms"] is None else f"{profile['capture_ms']:.1f}"
    analysis_ms = "--" if profile["analysis_ms"] is None else f"{profile['analysis_ms']:.1f}"
    render_ms = "--" if profile["render_ms"] is None else f"{profile['render_ms']:.1f}"
    sleep_ms = "--" if profile["sleep_ms"] is None else f"{profile['sleep_ms']:.1f}"
    runtime_debug_ph.caption(
        f"capture_ms={capture_ms} | analysis_ms={analysis_ms} | render_ms={render_ms} | sleep_ms={sleep_ms} | "
        f"sample_fps={sample_fps} | valid_intervals={valid_intervals} | hold_age_ms={hold_age} | "
        f"signal_conf={confidence} | dropped_capture={dropped_capture} | dropped_analysis={dropped_analysis}"
    )


# -----------------------------
# Upload-Modus (deterministisch / reproduzierbar)
# -----------------------------
if source == "Lokales Video":
    render_fps(None)
    render_runtime_debug(
        {"capture_ms": None, "analysis_ms": None, "render_ms": None, "sleep_ms": None},
        0,
        0,
        None,
    )
    upload = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi", "mkv"])
    run = st.button("Analyse starten", type="primary", disabled=upload is None)

    if run and upload is not None:
        upload_temp_path: Optional[Path] = None
        # Streamlit liefert ein in-memory Upload-Objekt.
        # OpenCV arbeitet stabiler mit einem echten Dateipfad -> temporaere Datei.
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.read())
            video_path = tmp.name
            upload_temp_path = Path(video_path)

        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if source_fps and source_fps > 0 else 30.0
        # Wichtig fuer Reproduzierbarkeit:
        # Bei Upload analysieren wir immer auf der Quell-Zeitachse (alle Frames),
        # damit gleiche Videos bei gleichem Setup wiederholbar gleiche Ergebnisse liefern.
        effective_analysis_fps = float(source_fps)
        effective_display_fps = max(1.0, min(float(display_fps), float(source_fps)))

        analyzer = CPRAnalyzer(
            fps=effective_analysis_fps,
            prominence=prominence,
            min_valid_cpm=20.0,
            max_valid_cpm=170.0,
            enable_ventilation_detection=False,
            enable_pose=pose_active,
            pose_every_n=3,
            hand_signal_mode=hand_signal_mode,
            display_debug=True,
        )

        profile: dict[str, Optional[float]] = {
            "capture_ms": None,
            "analysis_ms": None,
            "render_ms": None,
            "sleep_ms": None,
        }
        signal_history: list[float] = []
        visual_fps: Optional[float] = None
        fps_window_start = time.perf_counter()
        fps_window_frames = 0
        dropped_capture = 0  # Upload: bleibt 0, da wir jedes Quellframe lesen.
        dropped_analysis = 0  # Upload: bleibt 0, da wir jedes Quellframe analysieren.
        source_frame_idx = -1
        display_step = 0
        next_display_src_idx = 0
        last_metrics = CPRMetrics()
        next_metrics_tick = time.perf_counter() + (1.0 / METRICS_HZ)
        next_chart_tick = time.perf_counter() + (1.0 / CHART_HZ)
        opencv_window_ok = opencv_fallback
        opencv_warned = False
        wall_start = time.perf_counter()

        try:
            # Deterministische Upload-Pipeline:
            # - Frames werden in fester Reihenfolge gelesen
            # - jedes Frame wird analysiert (reproduzierbar)
            # - Anzeige erfolgt mit eigener, begrenzter Ziel-FPS
            while True:
                # 1) Naechstes Quellframe lesen (ohne Dropping im Upload-Modus).
                capture_start = time.perf_counter()
                ret, frame = cap.read()
                capture_ms = (time.perf_counter() - capture_start) * 1000.0
                profile["capture_ms"] = ema(profile["capture_ms"], capture_ms)
                if not ret:
                    break

                source_frame_idx += 1
                # Quellzeitstempel kommt direkt aus der Frame-Nummer -> reproduzierbar.
                t_sec = source_frame_idx / source_fps
                frame = cv2.resize(frame, PROCESS_SIZE)

                # 2) Analyse auf einer Kopie: `frame` bleibt fuer Debug/Fallback sauber.
                analysis_start = time.perf_counter()
                frame_out, metrics = analyzer.process_frame(frame.copy(), t_sec)
                analysis_ms = (time.perf_counter() - analysis_start) * 1000.0
                profile["analysis_ms"] = ema(profile["analysis_ms"], analysis_ms)
                last_metrics = metrics

                if metrics.signal_value is not None:
                    signal_history.append(metrics.signal_value)
                    signal_history = signal_history[-SIGNAL_HISTORY_LEN:]

                # Anzeige entkoppelt: wir zeigen nur die Frames, die zum Anzeige-Zeitraster passen.
                if source_frame_idx >= next_display_src_idx:
                    # Fuer Upload "spielen" wir das Video in Echtzeit ab (Originalzeitachse).
                    target_display_time = wall_start + t_sec
                    wait_sec = target_display_time - time.perf_counter()
                    if wait_sec > 0:
                        sleep_sec = min(wait_sec, SLEEP_CAP_SEC)
                        time.sleep(sleep_sec)
                        profile["sleep_ms"] = ema(profile["sleep_ms"], sleep_sec * 1000.0)
                    else:
                        profile["sleep_ms"] = ema(profile["sleep_ms"], 0.0)

                    render_start = time.perf_counter()
                    frame_display = (
                        frame_out if DISPLAY_SIZE == PROCESS_SIZE else cv2.resize(frame_out, DISPLAY_SIZE)
                    )
                    # Streamlit-Rendering ist meist der eigentliche FPS-Flaschenhals.
                    video_ph.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")
                    fps_window_start, fps_window_frames, visual_fps = advance_visual_fps(
                        fps_window_start, fps_window_frames, visual_fps
                    )

                    if opencv_window_ok:
                        try:
                            cv2.imshow("CPR Analyzer High FPS", frame_display)
                            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                opencv_window_ok = False
                                cv2.destroyAllWindows()
                        except cv2.error:
                            opencv_window_ok = False
                            if not opencv_warned:
                                st.warning("OpenCV-Fallback konnte nicht gestartet werden.")
                                opencv_warned = True

                    now_ui = time.perf_counter()
                    # Metriken und Chart werden bewusst seltener aktualisiert als das Video,
                    # damit die Anzeige-FPS moeglichst stabil bleibt.
                    if now_ui >= next_metrics_tick:
                        render_metrics(last_metrics)
                        render_fps(visual_fps)
                        render_runtime_debug(profile, dropped_capture, dropped_analysis, last_metrics)
                        next_metrics_tick = now_ui + (1.0 / METRICS_HZ)

                    if (not light_mode) and now_ui >= next_chart_tick:
                        update_motion_chart(signal_history)
                        next_chart_tick = now_ui + (1.0 / CHART_HZ)

                    profile["render_ms"] = ema(profile["render_ms"], (time.perf_counter() - render_start) * 1000.0)
                    display_step += 1
                    # Mappt den naechsten Anzeige-Schritt auf ein Quellframe.
                    next_display_src_idx = int(round(display_step * source_fps / effective_display_fps))
        finally:
            analyzer.close()
            cap.release()
            # Aufraeumen der temporaeren Upload-Datei, damit der Temp-Ordner nicht volllaeuft.
            if upload_temp_path is not None and upload_temp_path.exists():
                try:
                    upload_temp_path.unlink()
                except OSError:
                    st.warning(f"Temp-Datei konnte nicht geloescht werden: {upload_temp_path}")
            if opencv_fallback:
                cv2.destroyAllWindows()
            st.success("Analyse abgeschlossen")

# -----------------------------
# Live-Modus (Capture-Thread + getrennte Analyse/Anzeige)
# -----------------------------
else:
    render_fps(None)
    render_runtime_debug(
        {"capture_ms": None, "analysis_ms": None, "render_ms": None, "sleep_ms": None},
        0,
        0,
        None,
    )
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
            effective_display_fps = max(1.0, float(display_fps))

            analyzer = CPRAnalyzer(
                fps=effective_analysis_fps,
                prominence=prominence,
                min_valid_cpm=20.0,
                max_valid_cpm=170.0,
                enable_ventilation_detection=False,
                enable_pose=pose_active,
                pose_every_n=3,
                hand_signal_mode=hand_signal_mode,
                display_debug=True,
            )

            frame_buffer = LatestFrameBuffer()
            analysis_buffer = LatestAnalysisBuffer()
            stop_event = threading.Event()
            live_start_wall = time.perf_counter()
            capture_thread = threading.Thread(
                target=camera_capture_worker, args=(cap, frame_buffer, stop_event, live_start_wall), daemon=True
            )
            capture_thread.start()

            profile: dict[str, Optional[float]] = {
                "capture_ms": None,
                "analysis_ms": None,
                "render_ms": None,
                "sleep_ms": None,
            }
            signal_history: list[float] = []
            visual_fps: Optional[float] = None
            fps_window_start = time.perf_counter()
            fps_window_frames = 0
            dropped_capture = 0
            dropped_analysis = 0
            last_analyzed_seq = -1
            last_display_time = time.perf_counter()
            last_metrics = CPRMetrics()
            next_metrics_tick = time.perf_counter()
            next_chart_tick = time.perf_counter()
            no_frame_deadline = time.perf_counter() + 2.0
            opencv_window_ok = opencv_fallback
            opencv_warned = False
            next_analysis_tick = time.perf_counter()
            next_display_tick = time.perf_counter()

            try:
                # Live-Pipeline:
                # - Capture laeuft im Thread (immer neuestes Frame)
                # - Analyse und Anzeige laufen getrennt im Hauptloop
                while not stop_event.is_set() and not st.session_state.live_stop:
                    if stop_live:
                        st.session_state.live_stop = True
                        break

                    now = time.perf_counter()

                    due_analysis, next_analysis_tick = consume_tick(next_analysis_tick, now, effective_analysis_fps)
                    if due_analysis:
                        # Analyse nimmt immer den neuesten Kamerastand (kein Backlog).
                        frame_pkt = frame_buffer.latest()
                        profile["capture_ms"] = ema(profile["capture_ms"], frame_pkt.capture_ms)
                        if frame_pkt.frame is None:
                            if now > no_frame_deadline:
                                st.warning("Keine Kamera-Frames verfuegbar. Pruefe den Kamera-Index.")
                                break
                        else:
                            no_frame_deadline = now + 2.0
                            if frame_pkt.seq > (last_analyzed_seq + 1) and last_analyzed_seq >= 0:
                                dropped_capture += frame_pkt.seq - last_analyzed_seq - 1

                            if frame_pkt.seq != last_analyzed_seq:
                                # Nur neue Frames analysieren. Wenn keines da ist, wird dieser Tick ausgelassen.
                                analysis_start = time.perf_counter()
                                frame_out, metrics = analyzer.process_frame(frame_pkt.frame.copy(), frame_pkt.t_sec)
                                analysis_ms = (time.perf_counter() - analysis_start) * 1000.0
                                profile["analysis_ms"] = ema(profile["analysis_ms"], analysis_ms)
                                analysis_buffer.push(
                                    frame=frame_out,
                                    metrics=metrics,
                                    t_sec=frame_pkt.t_sec,
                                    seq=frame_pkt.seq,
                                    analysis_ms=analysis_ms,
                                )
                                last_analyzed_seq = frame_pkt.seq
                                last_metrics = metrics
                                if metrics.signal_value is not None:
                                    signal_history.append(metrics.signal_value)
                                    signal_history = signal_history[-SIGNAL_HISTORY_LEN:]
                            else:
                                # Analyse-Tick ohne neues Kamera-Frame -> bewusst als Drop zaehlen.
                                dropped_analysis += 1

                    due_display, next_display_tick = consume_tick(next_display_tick, now, effective_display_fps)
                    if due_display:
                        # Anzeige nutzt das letzte fertige Analyseframe; falls noch keines da ist,
                        # zeigen wir direkt das letzte Rohframe.
                        render_start = time.perf_counter()
                        analysis_pkt = analysis_buffer.latest()
                        frame_to_show = analysis_pkt.frame
                        metrics_to_show = analysis_pkt.metrics if analysis_pkt.metrics is not None else last_metrics

                        if frame_to_show is None:
                            frame_to_show = frame_buffer.latest().frame

                        if frame_to_show is not None:
                            frame_display = (
                                frame_to_show
                                if DISPLAY_SIZE == PROCESS_SIZE
                                else cv2.resize(frame_to_show, DISPLAY_SIZE)
                            )
                            video_ph.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")
                            fps_window_start, fps_window_frames, visual_fps = advance_visual_fps(
                                fps_window_start, fps_window_frames, visual_fps
                            )

                            if opencv_window_ok:
                                try:
                                    cv2.imshow("CPR Analyzer High FPS", frame_display)
                                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                        st.session_state.live_stop = True
                                        break
                                except cv2.error:
                                    opencv_window_ok = False
                                    if not opencv_warned:
                                        st.warning("OpenCV-Fallback konnte nicht gestartet werden.")
                                        opencv_warned = True

                        now_ui = time.perf_counter()
                        # UI-Subbereiche wieder getrennt takten (Video > Metriken > Chart).
                        if now_ui >= next_metrics_tick:
                            render_metrics(metrics_to_show)
                            render_fps(visual_fps)
                            render_runtime_debug(profile, dropped_capture, dropped_analysis, metrics_to_show)
                            next_metrics_tick = now_ui + (1.0 / METRICS_HZ)

                        if (not light_mode) and now_ui >= next_chart_tick:
                            update_motion_chart(signal_history)
                            next_chart_tick = now_ui + (1.0 / CHART_HZ)

                        profile["render_ms"] = ema(profile["render_ms"], (time.perf_counter() - render_start) * 1000.0)
                        last_display_time = now_ui

                    now_sleep = time.perf_counter()
                    next_event = min(next_analysis_tick, next_display_tick)
                    # Kurzes Sleep verhindert Busy-Wait, ohne die Reaktionszeit stark zu verschlechtern.
                    sleep_sec = max(0.0, min(next_event - now_sleep, SLEEP_CAP_SEC))
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)
                    profile["sleep_ms"] = ema(profile["sleep_ms"], sleep_sec * 1000.0)
            finally:
                stop_event.set()
                capture_thread.join(timeout=1.0)
                analyzer.close()
                cap.release()
                if opencv_fallback:
                    cv2.destroyAllWindows()
                if st.session_state.live_stop or (time.perf_counter() - last_display_time) > 0.1:
                    st.info("Live-Analyse gestoppt")
