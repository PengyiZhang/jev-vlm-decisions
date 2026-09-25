from person_type_a.benchmark import summarize


def test_summarize_basic_stats():
    xs = [100.0, 120.0, 110.0, 130.0, 105.0]
    s = summarize(xs)
    assert s["min"] == 100.0 and s["max"] == 130.0
    assert 105.0 <= s["median"] <= 120.0
    assert s["p95"] >= s["median"] and s["mean"] > 0


def test_summarize_single_run():
    s = summarize([42.0])
    assert s["min"] == s["max"] == s["median"] == 42.0
