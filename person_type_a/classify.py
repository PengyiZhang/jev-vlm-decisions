"""组装管线：任务 → 字母编码 → 单次打分 → 温度 Softmax → 门控结果。"""
from __future__ import annotations

from .calibrator import bucket_key
from .engine import ClassifyTask, Scorer, slot_letter_ids
from .encoding import assign_letters
from .readout import decide, masked_softmax, read_slot


def classify(
    scorer: Scorer,
    tokenizer,
    system: str,
    scene: str,
    evidence: tuple[str, ...],
    questions: tuple,
    image=None,
    temperatures: dict[str, float] | None = None,
) -> dict[str, dict]:
    temperatures = temperatures or {}
    task = ClassifyTask(system, scene, evidence, questions)
    letters_by_qid = {q.qid: assign_letters(q.effective_options) for q in questions}
    ids_by_qid = slot_letter_ids(tokenizer, task, letters_by_qid)

    logits_by_qid = scorer.slot_logits(task, image=image)

    results = {}
    for q in questions:
        scores = read_slot(logits_by_qid[q.qid], ids_by_qid[q.qid])
        t = temperatures.get(bucket_key(q.kind, len(q.effective_options)), 1.0)
        probs = masked_softmax(scores, t)
        labels = tuple(letters_by_qid[q.qid].keys())
        results[q.qid] = {"distribution": dict(zip(labels, (round(p, 4) for p in probs))),
                          **decide(probs, labels)}
    return results
