import math

from person_type_a.calibrator import (
    bucket_key, ece, fit_temperature, load_calibrator, save_calibrator,
)


def overconfident_samples(n=60):
    """60% 判对、40% 高置信判错：T=1 时过度自信，拟合应给出 T>1。

    温度缩放的前提是存在"自信但错误"的样本；若 gold 恒为最高分，
    拟合反而会选择 sharpening（T<1）。
    """
    out = []
    for i in range(n):
        correct = (i % 5) < 3  # 60% 判对
        gold = i % 2
        scores = [0.0, 0.0, 0.0]
        scores[gold] = 4.0 if correct else 0.5
        scores[1 - gold] = 0.5 if correct else 4.0  # 强干扰项
        out.append((scores, gold))
    return out


def test_fit_temperature_greater_than_one():
    t = fit_temperature(overconfident_samples())
    assert t > 1.0


def test_bucket_key():
    assert bucket_key("choice", 4) == "choice|4"
    assert bucket_key("binary", 4) == "binary|4"


def test_ece_perfect_calibration_is_zero():
    confs, hits = [0.8] * 10, [1.0] * 8 + [0.0] * 2
    assert ece(confs, hits) < 0.01


def test_ece_detects_overconfidence():
    confs, hits = [0.99] * 10, [0.5] * 10
    assert ece(confs, hits) > 0.4


def test_calibrator_roundtrip(tmp_path):
    path = tmp_path / "calibrator.json"
    save_calibrator(path, {"choice|4": 1.7, "binary|4": 2.1})
    assert load_calibrator(path)["choice|4"] == 1.7
