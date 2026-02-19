from __future__ import annotations

from typing import Optional


def rate_quality_label(cpm: Optional[float]) -> str:
    if cpm is None:
        return "unknown"
    if 100 <= cpm <= 120:
        return "target"
    if 90 <= cpm < 100 or 120 < cpm <= 130:
        return "close"
    return "off-target"


def ratio_score_30_2(compressions: int, ventilations: int) -> Optional[float]:
    if compressions < 10:
        return None
    expected_vent = compressions / 15.0
    if expected_vent <= 0:
        return None
    rel_error = abs(ventilations - expected_vent) / expected_vent
    return float(max(0.0, 1.0 - rel_error))
