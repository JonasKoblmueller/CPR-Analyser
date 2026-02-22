from __future__ import annotations

"""Core CPR analysis logic."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from cpr_analyser.metrics import rate_quality_label, ratio_score_30_2

HandSignalMode = Literal["palm", "wrist", "hybrid"]
_PALM_INDICES = np.array([0, 5, 9, 13, 17], dtype=int)


@dataclass
class CPRMetrics:
    """Output metrics for one processed frame."""

    cpm: Optional[float] = None
    regularity: Optional[float] = None
    compression_count: int = 0
    ventilation_count: int = 0
    compression_quality: str = "unknown"
    posture_score: Optional[float] = None
    posture_label: str = "unknown"
    ratio_30_2_score: Optional[float] = None
    in_ventilation_pause: bool = False
    signal_value: Optional[float] = None
    sample_fps_est: Optional[float] = None
    valid_intervals: int = 0
    signal_confidence: Optional[float] = None
    hold_age_ms: Optional[float] = None


@dataclass
class _HandCandidate:
    signal_y: float
    anchor_px: tuple[int, int]
    anchor_norm: tuple[float, float]
    track_points_px: np.ndarray


class CPRAnalyzer:
    """Frame-basierter CPR-Analyzer.

    Signalidee:
    - Aus Hand-Landmarks (bevorzugt Palm-Center) wird ein vertikales Bewegungssignal gewonnen.
    - Optical Flow stabilisiert das Signal bei kurzen Landmark-Aussetzern.
    - Aus dem zeitbasierten Signal wird die CPR-Frequenz (CPM) geschaetzt.
    """

    def __init__(
        self,
        fps: float,
        live_window_sec: float = 8.0,
        min_peak_distance_sec: float = 0.35,
        prominence: float = 0.003,
        smoothing_window: int = 5,
        bp_low: float = 20.0 / 60.0,
        bp_high: float = 170.0 / 60.0,
        bp_order: int = 3,
        ventilation_pause_sec: float = 0.8,
        min_valid_cpm: float = 20.0,
        max_valid_cpm: float = 170.0,
        enable_ventilation_detection: bool = False,
        enable_pose: bool = False,
        pose_every_n: int = 3,
        hand_signal_mode: str = "palm",
        display_debug: bool = False,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.live_window_sec = live_window_sec
        self.min_peak_distance_sec = min_peak_distance_sec
        self.prominence = prominence
        self.smoothing_window = smoothing_window
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.bp_order = bp_order
        self.ventilation_pause_sec = ventilation_pause_sec
        self.min_valid_cpm = min_valid_cpm
        self.max_valid_cpm = max_valid_cpm
        self.enable_ventilation_detection = enable_ventilation_detection
        self.enable_pose = enable_pose
        self.pose_every_n = max(1, int(pose_every_n))
        self.display_debug = display_debug

        mode = hand_signal_mode.strip().lower()
        self.hand_signal_mode: HandSignalMode = "palm" if mode not in {"palm", "wrist", "hybrid"} else mode

        self.max_hist_sec = live_window_sec + 2.0
        self.max_hist_samples = max(1, int(self.max_hist_sec * self.fps))
        self.frame_counter = 0

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_track_points: Optional[np.ndarray] = None
        self.tracked_signal_y: Optional[float] = None

        self.last_valid_cpm: Optional[float] = None
        self.last_valid_regularity: Optional[float] = None
        self.last_valid_t: Optional[float] = None
        self.last_hold_age_ms: Optional[float] = None
        self.cpm_hold_sec = 1.5

        self._last_pose_landmarks = None
        self._cpm_ema: Optional[float] = None
        self._cpm_alpha = 0.35

        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6,
        )

        self.y_positions: list[float] = []
        self.timestamps: list[float] = []
        self.peak_times_global: list[float] = []
        self.vent_times: list[float] = []

    def close(self) -> None:
        self.hands.close()
        self.pose.close()

    @staticmethod
    def _robust_mean(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return float("nan")
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        if mad <= 1e-6:
            return float(np.mean(values))
        mask = np.abs(values - med) <= (3.0 * mad)
        if int(np.sum(mask)) == 0:
            return med
        return float(np.mean(values[mask]))

    @staticmethod
    def _estimate_sample_fps(valid_timestamps: np.ndarray, fallback_fps: float) -> float:
        if len(valid_timestamps) < 3:
            return fallback_fps
        dt = np.diff(valid_timestamps)
        dt = dt[dt > 1e-6]
        if len(dt) == 0:
            return fallback_fps
        median_dt = float(np.median(dt))
        if median_dt <= 1e-6:
            return fallback_fps
        return float(max(1.0, 1.0 / median_dt))

    def _bandpass_filter(self, signal: np.ndarray, sample_fps: float) -> np.ndarray:
        if sample_fps <= 0:
            return signal
        low_hz = max(self.bp_low, self.min_valid_cpm / 60.0, 0.3)
        high_hz = min(self.bp_high, self.max_valid_cpm / 60.0, 3.0)
        nyq = 0.5 * sample_fps
        low = max(low_hz / nyq, 1e-4)
        high = min(high_hz / nyq, 0.999)
        if low >= high:
            return signal
        b, a = butter(self.bp_order, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b))
        if len(signal) <= padlen:
            return signal
        try:
            return filtfilt(b, a, signal)
        except ValueError:
            return signal

    def _smooth_signal_same_length(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) == 0:
            return signal
        window = min(int(self.smoothing_window), len(signal))
        if window < 2:
            return signal
        kernel = np.ones(window, dtype=float) / float(window)
        smoothed = np.convolve(signal, kernel, mode="same")
        if len(smoothed) != len(signal):
            return smoothed[: len(signal)]
        return smoothed

    @staticmethod
    def _sync_signal_and_time_lengths(signal: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = min(len(signal), len(timestamps))
        if n <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        return signal[:n], timestamps[:n]

    def _apply_cpm_hold(
        self, cpm: Optional[float], regularity: Optional[float], t_sec: float
    ) -> tuple[Optional[float], Optional[float]]:
        if cpm is not None:
            self.last_valid_cpm = cpm
            self.last_valid_regularity = regularity
            self.last_valid_t = t_sec
            self.last_hold_age_ms = 0.0
            return cpm, regularity

        if self.last_valid_t is None:
            self.last_hold_age_ms = None
            return None, None

        hold_age_ms = (t_sec - self.last_valid_t) * 1000.0
        if hold_age_ms <= (self.cpm_hold_sec * 1000.0):
            self.last_hold_age_ms = hold_age_ms
            return self.last_valid_cpm, self.last_valid_regularity

        self.last_hold_age_ms = None
        return None, None

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = a - b
        cb = c - b
        denom = np.linalg.norm(ab) * np.linalg.norm(cb)
        if denom == 0:
            return 0.0
        cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def _posture_from_pose(self, pose_landmarks) -> Tuple[Optional[float], str]:
        if pose_landmarks is None:
            return None, "no-pose"

        lm = pose_landmarks.landmark
        ls = np.array([lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        le = np.array([lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].y])
        lw = np.array([lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x, lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y])
        rs = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        re = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        rw = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y])

        left_angle = self._angle(ls, le, lw)
        right_angle = self._angle(rs, re, rw)
        arm_ext_score = np.clip(((left_angle + right_angle) / 2.0 - 140.0) / 35.0, 0.0, 1.0)

        shoulder_mid_x = (ls[0] + rs[0]) / 2.0
        wrist_mid_x = (lw[0] + rw[0]) / 2.0
        over_chest_score = np.clip(1.0 - abs(shoulder_mid_x - wrist_mid_x) * 4.0, 0.0, 1.0)

        score = 0.7 * arm_ext_score + 0.3 * over_chest_score
        label = "good" if score >= 0.65 else "improve"
        return float(score), label

    def _pose_chest_roi(self, pose_landmarks) -> Optional[tuple[float, float, float, float]]:
        if pose_landmarks is None:
            return None
        lm = pose_landmarks.landmark
        ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        chest_x = float((ls.x + rs.x) / 2.0)
        shoulder_y = float((ls.y + rs.y) / 2.0)
        hip_y = float((lh.y + rh.y) / 2.0)
        x_half = float(max(0.12, abs(ls.x - rs.x) * 1.6))
        y_min = shoulder_y - 0.18
        y_max = hip_y + 0.12
        return chest_x, y_min, y_max, x_half

    def _extract_hand_candidates(self, hand_result, frame_shape: tuple[int, int, int]) -> list[_HandCandidate]:
        """Erzeugt pro erkannter Hand einen Kandidaten fuer die Signalgewinnung.

        Je nach Modus wird als Signal verwendet:
        - `wrist`: Landmark 0
        - `palm`: Mittelwert aus Handflaechenpunkten (robuster)
        - `hybrid`: Mischung aus Palm + Wrist
        """
        if not hand_result.multi_hand_landmarks:
            return []
        h, w = frame_shape[:2]
        candidates: list[_HandCandidate] = []
        for hand in hand_result.multi_hand_landmarks:
            coords = np.array([(lm.x, lm.y) for lm in hand.landmark], dtype=np.float32)
            coords = np.clip(coords, 0.0, 1.0)
            palm = coords[_PALM_INDICES]
            palm_y = self._robust_mean(palm[:, 1])
            wrist = coords[0]
            wrist_y = float(wrist[1])

            if self.hand_signal_mode == "wrist":
                signal_y = wrist_y
                anchor_norm = (float(wrist[0]), float(wrist[1]))
                track_pts_norm = coords[[0, 5, 9]]
            elif self.hand_signal_mode == "hybrid":
                signal_y = float(0.7 * palm_y + 0.3 * wrist_y)
                anchor_all = np.vstack([palm, wrist[None, :]])
                anchor_norm = (float(np.mean(anchor_all[:, 0])), float(np.mean(anchor_all[:, 1])))
                track_pts_norm = palm
            else:
                signal_y = palm_y
                anchor_norm = (float(np.mean(palm[:, 0])), float(np.mean(palm[:, 1])))
                track_pts_norm = palm

            anchor_px = (int(anchor_norm[0] * w), int(anchor_norm[1] * h))
            track_pts_px = np.array(
                [[[float(pt[0] * w), float(pt[1] * h)]] for pt in track_pts_norm],
                dtype=np.float32,
            )
            candidates.append(
                _HandCandidate(
                    signal_y=signal_y,
                    anchor_px=anchor_px,
                    anchor_norm=anchor_norm,
                    track_points_px=track_pts_px,
                )
            )
        return candidates

    def _select_hand_candidate(self, candidates: list[_HandCandidate], pose_landmarks) -> Optional[_HandCandidate]:
        """Waehlt die plausibelste Hand fuer die CPR-Auswertung aus.

        Prioritaet:
        1. zeitliche Stabilitaet (naehe zum zuletzt verfolgten Signal)
        2. optional Pose-ROI im Brustbereich
        3. leichter Bildmittelpunkt-Bias als Fallback
        """
        if not candidates:
            return None
        chest_roi = self._pose_chest_roi(pose_landmarks) if self.enable_pose else None

        best_idx = 0
        best_score = float("inf")
        for idx, cand in enumerate(candidates):
            score = 0.0
            if self.tracked_signal_y is not None:
                score += 2.0 * abs(cand.signal_y - self.tracked_signal_y)

            if chest_roi is not None:
                chest_x, y_min, y_max, x_half = chest_roi
                x_norm, y_norm = cand.anchor_norm
                outside = not (y_min <= y_norm <= y_max and abs(x_norm - chest_x) <= x_half)
                if outside:
                    score += 1.0
                score += 0.5 * abs(x_norm - chest_x)
            else:
                score += 0.1 * abs(cand.anchor_norm[0] - 0.5)

            if score < best_score:
                best_score = score
                best_idx = idx
        return candidates[best_idx]

    def _estimate_flow_signal(
        self, gray: np.ndarray, frame_shape: tuple[int, int, int]
    ) -> tuple[float, Optional[tuple[int, int]], Optional[np.ndarray]]:
        """Schaetzt die Handbewegung ueber Optical Flow (LK) zwischen zwei Frames.

        Rueckgabe:
        - `flow_y`: normierte vertikale Signalposition (0..1) oder NaN
        - `flow_anchor_px`: Pixelpunkt fuer Visualisierung
        - `flow_points`: neue Trackingpunkte fuer den naechsten Frame
        """
        if self.prev_gray is None or self.prev_track_points is None:
            return np.nan, None, None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_track_points,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            self.prev_track_points = None
            return np.nan, None, None

        valid = status.reshape(-1) == 1
        if int(np.sum(valid)) < 2:
            self.prev_track_points = None
            return np.nan, None, None

        prev_valid = self.prev_track_points[valid][:, 0, :]
        next_valid = next_pts[valid][:, 0, :]
        h = max(frame_shape[0], 1)
        dy_norm = float(np.median((next_valid[:, 1] - prev_valid[:, 1]) / h))
        base_y = (
            self.tracked_signal_y
            if self.tracked_signal_y is not None
            else float(np.median(prev_valid[:, 1]) / h)
        )
        flow_y = float(np.clip(base_y + dy_norm, 0.0, 1.0))
        anchor = np.median(next_valid, axis=0)
        flow_anchor_px = (int(anchor[0]), int(anchor[1]))
        flow_points = next_valid.reshape(-1, 1, 2).astype(np.float32)
        return flow_y, flow_anchor_px, flow_points

    def _fuse_hand_signal(
        self,
        gray: np.ndarray,
        frame_shape: tuple[int, int, int],
        candidate: Optional[_HandCandidate],
    ) -> tuple[float, Optional[tuple[int, int]]]:
        """Fusioniert Landmark-Signal mit Optical Flow.

        Strategie:
        - Landmark hat Prioritaet (wenn vorhanden)
        - im `hybrid`-Modus wird Flow leicht beigemischt
        - bei Landmark-Ausfall wird nur Flow verwendet
        """
        flow_y, flow_anchor_px, flow_points = self._estimate_flow_signal(gray, frame_shape)

        if candidate is not None:
            fused_y = candidate.signal_y
            if not np.isnan(flow_y) and self.hand_signal_mode == "hybrid":
                fused_y = float(0.8 * candidate.signal_y + 0.2 * flow_y)
            fused_y = float(np.clip(fused_y, 0.0, 1.0))
            self.tracked_signal_y = fused_y
            self.prev_track_points = candidate.track_points_px
            return fused_y, candidate.anchor_px

        if not np.isnan(flow_y):
            self.tracked_signal_y = flow_y
            self.prev_track_points = flow_points
            return flow_y, flow_anchor_px

        self.tracked_signal_y = None
        self.prev_track_points = None
        return np.nan, None

    def _estimate_cpm_from_autocorr(
        self,
        signal: np.ndarray,
        sample_fps: float,
        min_interval_sec: float,
        max_interval_sec: float,
    ) -> Optional[float]:
        """Autokorrelations-basierte CPR-Frequenzschaetzung als Fallback/Second Opinion."""
        if len(signal) < 8 or sample_fps <= 0:
            return None
        centered = signal - np.mean(signal)
        if np.max(np.abs(centered)) < 1e-8:
            return None

        ac = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        lag_min = max(1, int(min_interval_sec * sample_fps))
        lag_max = min(len(ac) - 1, int(max_interval_sec * sample_fps))
        if lag_max < lag_min:
            return None
        window = ac[lag_min : lag_max + 1]
        if len(window) == 0:
            return None
        best_idx = int(np.argmax(window))
        best_lag = lag_min + best_idx
        if best_lag <= 0:
            return None
        return float(60.0 * sample_fps / best_lag)

    def process_frame(self, frame: np.ndarray, t_sec: float) -> tuple[np.ndarray, CPRMetrics]:
        """Analysiert ein einzelnes Frame und liefert annotiertes Bild + Metriken.

        `t_sec` muss die echte Zeitachse des Videos/Live-Streams repraesentieren.
        Dadurch bleibt die CPM-Schaetzung unabhaengig von UI-FPS oder Scheduler-Jitter.
        """
        # Vorverarbeitung fuer MediaPipe (RGB) und Optical Flow (Graustufen).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hand-Landmarks fuer Primaersignal.
        hand_result = self.hands.process(rgb)
        for hand in hand_result.multi_hand_landmarks or []:
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

        pose_landmarks = None
        if self.enable_pose:
            # Pose wird absichtlich seltener gerechnet, um Rechenzeit zu sparen.
            if self.frame_counter % self.pose_every_n == 0:
                pose_result = self.pose.process(rgb)
                self._last_pose_landmarks = pose_result.pose_landmarks
            pose_landmarks = self._last_pose_landmarks
            if pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Handsignal aus Landmarks + Optical Flow zusammensetzen.
        candidates = self._extract_hand_candidates(hand_result, frame.shape)
        selected = self._select_hand_candidate(candidates, pose_landmarks)
        signal_y, signal_anchor_px = self._fuse_hand_signal(gray, frame.shape, selected)

        # Historie begrenzen -> konstante Rechenlast auch bei langen Videos.
        self.y_positions.append(signal_y)
        self.timestamps.append(t_sec)
        if len(self.y_positions) > self.max_hist_samples:
            self.y_positions = self.y_positions[-self.max_hist_samples :]
            self.timestamps = self.timestamps[-self.max_hist_samples :]

        y_arr = np.array(self.y_positions, dtype=float)
        t_arr = np.array(self.timestamps, dtype=float)
        valid = ~np.isnan(y_arr)

        cpm: Optional[float] = None
        regularity: Optional[float] = None
        signal_value: Optional[float] = None
        is_peak = False
        sample_fps_est: Optional[float] = None
        valid_intervals_count = 0
        signal_confidence: Optional[float] = None

        valid_count = int(np.sum(valid))
        if valid_count >= 3:
            valid_times = t_arr[valid]
            valid_values = y_arr[valid]
            if (valid_times[-1] - valid_times[0]) >= 0.8:
                # Effektive Samplerate aus echten Zeitstempeln schaetzen (nicht aus UI-FPS).
                sample_fps = self._estimate_sample_fps(valid_times, self.fps)
                sample_fps_est = sample_fps
                # Zeitbasiertes Resampling auf gleichmaessige Zeitachse fuer Filter/Peaks.
                t_uniform = np.arange(valid_times[0], valid_times[-1] + (0.5 / sample_fps), 1.0 / sample_fps)
                if len(t_uniform) >= 3:
                    y_interp = np.interp(t_uniform, valid_times, valid_values)
                    # Vorzeichen drehen: Kompressionsrichtung soll "Peaks" ergeben.
                    signal = -y_interp
                    signal -= np.mean(signal)
                    signal = self._bandpass_filter(signal, sample_fps)
                    signal = self._smooth_signal_same_length(signal)
                    signal, t_uniform = self._sync_signal_and_time_lengths(signal, t_uniform)

                    if len(signal) >= 3 and len(t_uniform) >= 3:
                        # Nur aktuelles Zeitfenster fuer Live-Anzeige/Frequenz verwenden.
                        mask = t_uniform > (t_uniform[-1] - self.live_window_sec)
                        t_live = t_uniform[mask]
                        s_live = signal[mask]

                        if len(s_live) > 0:
                            signal_value = float(s_live[-1])

                        if len(t_live) >= 3 and len(s_live) >= 3:
                            # CPR-Intervallgrenzen werden direkt aus BPM-Limits abgeleitet.
                            min_interval_sec = 60.0 / self.max_valid_cpm
                            max_interval_sec = 60.0 / self.min_valid_cpm
                            min_dist = int(max(self.min_peak_distance_sec, min_interval_sec) * sample_fps)
                            peaks, _ = find_peaks(s_live, distance=max(min_dist, 1), prominence=self.prominence)

                            cpm_peak: Optional[float] = None
                            if len(peaks) >= 2:
                                # Peak-basierte Schaetzung aus echten Zeitabstaenden.
                                intervals = np.diff(t_live[peaks])
                                valid_intervals = intervals[
                                    (intervals >= min_interval_sec) & (intervals <= max_interval_sec)
                                ]
                                valid_intervals_count = int(len(valid_intervals))
                                if valid_intervals_count > 0:
                                    cpm_peak = float(60.0 / np.median(valid_intervals))
                                    if valid_intervals_count >= 2 and np.mean(valid_intervals) > 0:
                                        regularity = float(
                                            max(0.0, 1.0 - (np.std(valid_intervals) / np.mean(valid_intervals)))
                                        )

                                if peaks[-1] >= (len(s_live) - 2):
                                    is_peak = True

                                last_global = self.peak_times_global[-1] if self.peak_times_global else -1e9
                                for pt in t_live[peaks]:
                                    ptf = float(pt)
                                    if ptf > (last_global + 0.5 * min_interval_sec):
                                        self.peak_times_global.append(ptf)
                                        last_global = ptf

                            # Zweite, unabhaengige Schaetzung: Autokorrelation.
                            cpm_ac = self._estimate_cpm_from_autocorr(
                                s_live, sample_fps, min_interval_sec=min_interval_sec, max_interval_sec=max_interval_sec
                            )

                            # Kandidaten fusionieren und an letztem stabilen Wert orientieren.
                            candidates_cpm = [
                                c
                                for c in (cpm_peak, cpm_ac)
                                if c is not None and self.min_valid_cpm <= c <= self.max_valid_cpm
                            ]
                            if len(candidates_cpm) > 0:
                                if self.last_valid_cpm is not None:
                                    cpm_raw = min(candidates_cpm, key=lambda x: abs(x - self.last_valid_cpm))
                                elif cpm_peak is not None:
                                    cpm_raw = cpm_peak
                                else:
                                    cpm_raw = candidates_cpm[0]

                                if self._cpm_ema is None:
                                    self._cpm_ema = cpm_raw
                                else:
                                    self._cpm_ema = (1.0 - self._cpm_alpha) * self._cpm_ema + self._cpm_alpha * cpm_raw
                                cpm = float(self._cpm_ema)

                            # Einfache Konfidenz (heuristisch) fuer Debugging/Diagnose.
                            conf = 0.0
                            if cpm_peak is not None:
                                conf += 0.55
                            if cpm_ac is not None:
                                conf += 0.30
                            conf += min(0.15, 0.03 * valid_intervals_count)
                            signal_confidence = float(min(1.0, conf))

                            if self.enable_ventilation_detection and len(s_live) >= 3:
                                troughs, _ = find_peaks(-s_live, distance=max(min_dist, 1), prominence=self.prominence)
                                for tr in troughs:
                                    tt = float(t_live[tr])
                                    if not self.vent_times or tt - self.vent_times[-1] > 1.0:
                                        if self.peak_times_global and (
                                            tt - self.peak_times_global[-1] > self.ventilation_pause_sec
                                        ):
                                            self.vent_times.append(tt)

        cpm, regularity = self._apply_cpm_hold(cpm, regularity, t_sec)

        if self.enable_pose:
            posture_score, posture_label = self._posture_from_pose(pose_landmarks)
        else:
            posture_score, posture_label = None, "pose-off"

        compression_count = len(self.peak_times_global)
        ventilation_count = len(self.vent_times)

        in_ventilation_pause = False
        if self.enable_ventilation_detection and self.peak_times_global:
            in_ventilation_pause = (t_sec - self.peak_times_global[-1]) > self.ventilation_pause_sec
            if in_ventilation_pause:
                cpm = None
                regularity = None
                signal_value = None

        if is_peak and signal_anchor_px is not None:
            # Markiert den aktuell erkannten Peak im Bild.
            cv2.circle(frame, signal_anchor_px, 8, (0, 255, 0), -1)

        self.prev_gray = gray
        self.frame_counter += 1

        metrics = CPRMetrics(
            cpm=cpm,
            regularity=regularity,
            compression_count=compression_count,
            ventilation_count=ventilation_count,
            compression_quality=rate_quality_label(cpm),
            posture_score=posture_score,
            posture_label=posture_label,
            ratio_30_2_score=ratio_score_30_2(compression_count, ventilation_count),
            in_ventilation_pause=in_ventilation_pause,
            signal_value=signal_value,
            sample_fps_est=sample_fps_est,
            valid_intervals=valid_intervals_count,
            signal_confidence=signal_confidence,
            hold_age_ms=self.last_hold_age_ms,
        )
        return frame, metrics
