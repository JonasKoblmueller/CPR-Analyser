from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from cpr_analyser.metrics import rate_quality_label, ratio_score_30_2


@dataclass
class CPRMetrics:
    cpm: Optional[float] = None
    regularity: Optional[float] = None
    compression_count: int = 0
    ventilation_count: int = 0
    compression_quality: str = "unknown"
    posture_score: Optional[float] = None
    posture_label: str = "unknown"
    ratio_30_2_score: Optional[float] = None


class CPRAnalyzer:
    """Führt die CPR-Analyse auf Frame-Basis durch.

    Pipeline:
    1) Handgelenk-Tracking (MediaPipe Hands)
    2) Signalaufbereitung (Interpolation, Bandpass, Glättung)
    3) Peak-Detektion -> CPM + Kompressionszählung + Regelmäßigkeit
    4) Heuristische Beatmungszählung
    5) Pose-basierte Haltungsbewertung
    """
    def __init__(
        self,
        fps: float,
        live_window_sec: float = 8.0,
        min_peak_distance_sec: float = 0.35,
        prominence: float = 0.003,
        smoothing_window: int = 5,
        bp_low: float = 1.0,
        bp_high: float = 2.5,
        bp_order: int = 3,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.live_window_sec = live_window_sec
        self.min_peak_distance_sec = min_peak_distance_sec
        self.prominence = prominence
        self.smoothing_window = smoothing_window
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.bp_order = bp_order

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

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Bandpass im CPR-Bereich (typisch ~1.0 bis 2.5 Hz)."""
        nyq = 0.5 * self.fps
        low = max(self.bp_low / nyq, 1e-4)
        high = min(self.bp_high / nyq, 0.999)
        if low >= high:
            return signal
        b, a = butter(self.bp_order, [low, high], btype="band")
        if len(signal) < (3 * max(len(a), len(b))):
            return signal
        return filtfilt(b, a, signal)

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = a - b
        cb = c - b
        denom = (np.linalg.norm(ab) * np.linalg.norm(cb))
        if denom == 0:
            return 0.0
        cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    def _posture_from_pose(self, pose_landmarks) -> Tuple[Optional[float], str]:
        """Heuristische Haltungsauswertung (Arme gestreckt + Schultern über Händen)."""
        if pose_landmarks is None:
            return None, "no-pose"

        lm = pose_landmarks.landmark
        ls = np.array([lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                       lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        le = np.array([lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                       lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].y])
        lw = np.array([lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                       lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y])

        rs = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                       lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        re = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        rw = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y])

        left_angle = self._angle(ls, le, lw)
        right_angle = self._angle(rs, re, rw)
        arm_ext_score = np.clip(((left_angle + right_angle) / 2.0 - 140.0) / 35.0, 0.0, 1.0)

        shoulder_mid_x = (ls[0] + rs[0]) / 2.0
        wrist_mid_x = (lw[0] + rw[0]) / 2.0
        over_chest_score = np.clip(1.0 - abs(shoulder_mid_x - wrist_mid_x) * 4.0, 0.0, 1.0)

        score = 0.7 * arm_ext_score + 0.3 * over_chest_score
        label = "good" if score >= 0.65 else "improve"
        return float(score), label


    def process_frame(self, frame: np.ndarray, t_sec: float) -> tuple[np.ndarray, CPRMetrics]:
        """Verarbeitet ein einzelnes Frame und liefert annotiertes Frame + aktuelle Metriken."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = self.hands.process(rgb)
        pose_result = self.pose.process(rgb)

        wrist_y = np.nan
        wrist_px = None

        if hand_result.multi_hand_landmarks:
            ys = []
            pxs = []
            for hand in hand_result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                lm = hand.landmark[self.mp_hands.HandLandmark.WRIST]
                ys.append(lm.y)
                pxs.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))
            wrist_y = float(np.mean(ys))
            wrist_px = pxs[0]

        if pose_result.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        self.y_positions.append(wrist_y)
        self.timestamps.append(t_sec)

        y_arr = np.array(self.y_positions, dtype=float)
        t_arr = np.array(self.timestamps, dtype=float)
        valid = ~np.isnan(y_arr)

        cpm: Optional[float] = None
        regularity: Optional[float] = None
        is_peak = False

        if np.sum(valid) > self.fps:
            y_interp = np.interp(t_arr, t_arr[valid], y_arr[valid])
            signal = -y_interp
            signal -= np.mean(signal)
            signal = self._bandpass_filter(signal)
            signal = np.convolve(signal, np.ones(self.smoothing_window) / self.smoothing_window, mode="same")

            mask = t_arr > (t_arr[-1] - self.live_window_sec)
            t_live = t_arr[mask]
            s_live = signal[mask]

            min_dist = int(self.min_peak_distance_sec * self.fps)
            peaks, _ = find_peaks(s_live, distance=max(min_dist, 1), prominence=self.prominence)

            if len(peaks) >= 2:
                duration = t_live[peaks[-1]] - t_live[peaks[0]]
                if duration > 0:
                    cpm = float((len(peaks) - 1) / duration * 60.0)
                intervals = np.diff(t_live[peaks])
                if len(intervals) > 0 and np.mean(intervals) > 0:
                    regularity = float(max(0.0, 1.0 - (np.std(intervals) / np.mean(intervals))))

                if peaks[-1] == len(s_live) - 1:
                    is_peak = True

                for p in peaks:
                    pt = float(t_live[p])
                    if not self.peak_times_global or pt - self.peak_times_global[-1] > 0.2:
                        self.peak_times_global.append(pt)

            troughs, _ = find_peaks(-s_live, distance=max(min_dist, 1), prominence=self.prominence)
            for tr in troughs:
                tt = float(t_live[tr])
                if not self.vent_times or tt - self.vent_times[-1] > 1.0:
                    # Heuristik: längere Pause seit letzter Kompression als Beatmungskandidat
                    if self.peak_times_global and (tt - self.peak_times_global[-1] > 0.8):
                        self.vent_times.append(tt)

        posture_score, posture_label = self._posture_from_pose(pose_result.pose_landmarks)
        compression_count = len(self.peak_times_global)
        ventilation_count = len(self.vent_times)

        if is_peak and wrist_px is not None:
            cv2.circle(frame, wrist_px, 10, (0, 255, 0), -1)

        metrics = CPRMetrics(
            cpm=cpm,
            regularity=regularity,
            compression_count=compression_count,
            ventilation_count=ventilation_count,
            compression_quality=rate_quality_label(cpm),
            posture_score=posture_score,
            posture_label=posture_label,
            ratio_30_2_score=ratio_score_30_2(compression_count, ventilation_count),
        )
        return frame, metrics
