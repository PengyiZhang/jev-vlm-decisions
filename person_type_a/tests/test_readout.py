from person_type_a.readout import LOW_SCORE, decide, masked_softmax, read_slot


def test_masked_softmax_basic():
    assert masked_softmax([2.0, 1.0, 0.0])[0] > 0.5


def test_masked_softmax_temperature_flattens():
    sharp = masked_softmax([3.0, 0.0], temperature=1.0)
    flat = masked_softmax([3.0, 0.0], temperature=5.0)
    assert flat[0] < sharp[0] and abs(sum(flat) - 1.0) < 1e-9


def test_read_slot_picks_candidate_logits():
    vocab = [0.0] * 300
    vocab[ord("A")] = 5.0
    vocab[ord("B")] = 3.0
    scores = read_slot(vocab, [ord("A"), ord("B")])
    assert scores[0] > scores[1] and len(scores) == 2


def test_decide_gates():
    labels = ("airport ground staff", "flight attendant", "__insufficient_evidence__")
    r = decide([0.9, 0.08, 0.02], labels)
    assert r["top_label"] == "airport ground staff" and r["gate"] == "auto"
    assert decide([0.7, 0.25, 0.05], labels)["gate"] == "review"
    assert decide([0.5, 0.3, 0.2], labels)["gate"] == "human"


def test_decide_reports_abstain_mass():
    labels = ("a", "b", "__insufficient_evidence__")
    r = decide([0.2, 0.2, 0.6], labels)
    assert r["abstain_mass"] == 0.6 and r["gate"] == "human"


def test_read_slot_accepts_sparse_row():
    row = {ord("A"): 5.0, ord("B"): 3.0}
    scores = read_slot(row, [ord("A"), ord("B"), ord("C")])
    assert scores[0] == 5.0 and scores[1] == 3.0
    assert scores[2] == LOW_SCORE  # top-K 未覆盖的字母给地板分


def test_read_slot_sparse_floor_on_empty_row():
    assert read_slot({}, [7]) == [LOW_SCORE]


def test_sparse_floor_prob_becomes_zero():
    probs = masked_softmax([2.0, LOW_SCORE])
    assert probs[1] < 1e-9 and abs(sum(probs) - 1.0) < 1e-9
