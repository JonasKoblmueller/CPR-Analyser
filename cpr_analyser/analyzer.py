from __future__ import annotations

"""Core CPR analysis logic."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from cpr_analyser.metrics import rate_quality_label, ratio_score_30_2


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


class CPRAnalyzer:
    """Frame-based CPR analyzer using MediaPipe Hands + Pose."""

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
        ventilation_pause_sec: float = 0.8,
        min_valid_cpm: float = 20.0,
        max_valid_cpm: float = 170.0,
        enable_ventilation_detection: bool = False,
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

        self.max_hist_sec = live_window_sec + 2.0
        self.max_hist_samples = max(1, int(self.max_hist_sec * self.fps))
        self.pose_every_n = 2
        self.frame_counter = 0

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_wrist_point: Optional[np.ndarray] = None
        self.tracked_wrist_y: Optional[float] = None

        self.last_valid_cpm: Optional[float] = None
        self.last_valid_regularity: Optional[float] = None
        self.last_valid_t: Optional[float] = None
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
        nyq = 0.5 * sample_fps
        low = max(self.bp_low / nyq, 1e-4)
        high = min(self.bp_high / nyq, 0.999)
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
            return cpm, regularity

        if self.last_valid_t is None:
            return None, None
        if (t_sec - self.last_valid_t) <= self.cpm_hold_sec:
            return self.last_valid_cpm, self.last_valid_regularity
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

    def _estimate_cpm_from_autocorr(
        self,
        signal: np.ndarray,
        sample_fps: float,
        min_interval_sec: float,
        max_interval_sec: float,
    ) -> Optional[float]:
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

    def _fuse_wrist_signal(
        self,
        gray: np.ndarray,
        frame_shape: tuple[int, int, int],
        wrist_y_landmark: float,
        wrist_px_landmark: Optional[tuple[int, int]],
    ) -> tuple[float, Optional[tuple[int, int]]]:
        flow_wrist_y = np.nan
        flow_wrist_px = None

        if self.prev_gray is not None and self.prev_wrist_point is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_wrist_point,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_pts is not None and status is not None and int(status[0][0]) == 1:
                prev_pt = self.prev_wrist_point[0][0]
                next_pt = next_pts[0][0]
                dy_norm = float((next_pt[1] - prev_pt[1]) / max(frame_shape[0], 1))
                base_y = self.tracked_wrist_y if self.tracked_wrist_y is not None else float(prev_pt[1] / frame_shape[0])
                flow_wrist_y = float(np.clip(base_y + dy_norm, 0.0, 1.0))
                flow_wrist_px = (int(next_pt[0]), int(next_pt[1]))
                self.prev_wrist_point = next_pts
            else:
                self.prev_wrist_point = None

        if not np.isnan(wrist_y_landmark):
            self.tracked_wrist_y = float(np.clip(wrist_y_landmark, 0.0, 1.0))
            if wrist_px_landmark is not None:
                self.prev_wrist_point = np.array(
                    [[[float(wrist_px_landmark[0]), float(wrist_px_landmark[1])]]], dtype=np.float32
                )
            return self.tracked_wrist_y, wrist_px_landmark

        if not np.isnan(flow_wrist_y):
            self.tracked_wrist_y = flow_wrist_y
            return flow_wrist_y, flow_wrist_px

        self.tracked_wrist_y = None
        return np.nan, None

    def process_frame(self, frame: np.ndarray, t_sec: float) -> tuple[np.ndarray, CPRMetrics]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hand_result = self.hands.process(rgb)
        if self.frame_counter % self.pose_every_n == 0:
            pose_result = self.pose.process(rgb)
            self._last_pose_landmarks = pose_result.pose_landmarks
        pose_landmarks = self._last_pose_landmarks

        wrist_y_landmark = np.nan
        wrist_px_landmark = None
        if hand_result.multi_hand_landmarks:
            ys = []
            pxs = []
            for hand in hand_result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                lm = hand.landmark[self.mp_hands.HandLandmark.WRIST]
                ys.append(lm.y)
                pxs.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))
            wrist_y_landmark = float(np.mean(ys))
            wrist_px_landmark = pxs[0]

        wrist_y, wrist_px = self._fuse_wrist_signal(gray, frame.shape, wrist_y_landmark, wrist_px_landmark)
        if pose_landmarks:
            self.mp_draw.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        self.y_positions.append(wrist_y)
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

        valid_count = int(np.sum(valid))
        if valid_count >= 3:
            valid_times = t_arr[valid]
            valid_values = y_arr[valid]
            if (valid_times[-1] - valid_times[0]) >= 0.8:
                sample_fps = self._estimate_sample_fps(valid_times, self.fps)
                t_uniform = np.arange(valid_times[0], valid_times[-1] + (0.5 / sample_fps), 1.0 / sample_fps)
                if len(t_uniform) >= 3:
                    y_interp = np.interp(t_uniform, valid_times, valid_values)
                    signal = -y_interp
                    signal -= np.mean(signal)
                    signal = self._bandpass_filter(signal, sample_fps)
                    signal = self._smooth_signal_same_length(signal)
                    signal, t_uniform = self._sync_signal_and_time_lengths(signal, t_uniform)

                    if len(signal) >= 3 and len(t_uniform) >= 3:
                        mask = t_uniform > (t_uniform[-1] - self.live_window_sec)
                        t_live = t_uniform[mask]
                        s_live = signal[mask]

                        if len(s_live) > 0:
                            signal_value = float(s_live[-1])

                        if len(t_live) >= 3 and len(s_live) >= 3:
                            min_interval_sec = 60.0 / self.max_valid_cpm
                            max_interval_sec = 60.0 / self.min_valid_cpm
                            min_dist = int(max(self.min_peak_distance_sec, min_interval_sec) * sample_fps)
                            peaks, _ = find_peaks(s_live, distance=max(min_dist, 1), prominence=self.prominence)

                            cpm_peak: Optional[float] = None
                            if len(peaks) >= 2:
                                intervals = np.diff(t_live[peaks])
                                valid_intervals = intervals[
                                    (intervals >= min_interval_sec) & (intervals <= max_interval_sec)
                                ]
                                if len(valid_intervals) > 0:
                                    cpm_peak = float(60.0 / np.median(valid_intervals))
                                    if len(valid_intervals) >= 2 and np.mean(valid_intervals) > 0:
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

                            cpm_ac = self._estimate_cpm_from_autocorr(
                                s_live, sample_fps, min_interval_sec=min_interval_sec, max_interval_sec=max_interval_sec
                            )

                            candidates = [
                                c for c in (cpm_peak, cpm_ac) if c is not None and self.min_valid_cpm <= c <= self.max_valid_cpm
                            ]
                            if len(candidates) > 0:
                                if self.last_valid_cpm is not None:
                                    cpm_raw = min(candidates, key=lambda x: abs(x - self.last_valid_cpm))
                                elif cpm_peak is not None:
                                    cpm_raw = cpm_peak
                                else:
                                    cpm_raw = candidates[0]

                                if self._cpm_ema is None:
                                    self._cpm_ema = cpm_raw
                                else:
                                    self._cpm_ema = (1.0 - self._cpm_alpha) * self._cpm_ema + self._cpm_alpha * cpm_raw
                                cpm = float(self._cpm_ema)

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

        posture_score, posture_label = self._posture_from_pose(pose_landmarks)
        compression_count = len(self.peak_times_global)
        ventilation_count = len(self.vent_times)

        in_ventilation_pause = False
        if self.enable_ventilation_detection and self.peak_times_global:
            in_ventilation_pause = (t_sec - self.peak_times_global[-1]) > self.ventilation_pause_sec
            if in_ventilation_pause:
                cpm = None
                regularity = None
                signal_value = None

        if is_peak and wrist_px is not None:
            cv2.circle(frame, wrist_px, 8, (0, 255, 0), -1)

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
        )
        return frame, metrics
