from cpr_analyser.metrics import rate_quality_label, ratio_score_30_2


def test_ratio_score_near_30_2():
    score = ratio_score_30_2(60, 4)
    assert score is not None
    assert score > 0.95


def test_rate_quality_label():
    assert rate_quality_label(110) == "target"
    assert rate_quality_label(95) == "close"
    assert rate_quality_label(140) == "off-target"
