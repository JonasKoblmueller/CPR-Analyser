import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
from cpr_analyser.analyzer import CPRAnalyzer


def _analyzer_stub() -> CPRAnalyzer:
    analyzer = CPRAnalyzer.__new__(CPRAnalyzer)
    analyzer.smoothing_window = 5
    analyzer.last_valid_cpm = None
    analyzer.last_valid_regularity = None
    analyzer.last_valid_t = None
    analyzer.cpm_hold_sec = 1.5
    return analyzer


def test_smooth_signal_same_length_for_short_signal():
    analyzer = _analyzer_stub()
    signal = np.array([1.0, 2.0, 3.0], dtype=float)
    smoothed = analyzer._smooth_signal_same_length(signal)
    assert len(smoothed) == len(signal)


def test_sync_signal_and_time_lengths_trims_to_min_length():
    signal = np.array([1, 2, 3, 4, 5], dtype=float)
    timestamps = np.array([0.0, 0.1, 0.2], dtype=float)
    synced_signal, synced_t = CPRAnalyzer._sync_signal_and_time_lengths(signal, timestamps)
    assert len(synced_signal) == 3
    assert len(synced_t) == 3


def test_apply_cpm_hold_keeps_last_value_temporarily():
    analyzer = _analyzer_stub()

    cpm, reg = analyzer._apply_cpm_hold(108.0, 0.82, 10.0)
    assert cpm == 108.0
    assert reg == 0.82

    cpm, reg = analyzer._apply_cpm_hold(None, None, 11.0)
    assert cpm == 108.0
    assert reg == 0.82

    cpm, reg = analyzer._apply_cpm_hold(None, None, 11.7)
    assert cpm is None
    assert reg is None
