# Single-Forward VLM Decision Engine

**[中文](README.zh-CN.md)** | English

**Jev-style letter-slot decisions for vision-language models.** One forward pass turns any VLM into a millisecond-level image decision engine: no autoregressive decoding, no JSON parsing, calibrated probability distributions out of the box.

```text
[ptype] airport ground staff (0.85)  gate=auto  abstain=0.01
    airport ground staff (0.85)  flight attendant (0.10)  passenger (0.05)
```

### Why

Asking a VLM to "classify this image and reply in JSON" pays for a full autoregressive loop: dozens to hundreds of decode steps, format errors, and uncalibrated confidence. For closed-set decisions — routing, gating, attribute tagging — none of that is necessary.

This project follows the decision-model pattern popularized by [Jev](https://docs.typesafe.ai/) (TypeSafe's "System One" model) and its open ecosystem:

- **Zero decode steps.** Candidates are rendered as letter slots `(A)…(B)…` in the prompt; answer slots are left empty. A single prefill returns next-token logits at every position, and the letter distribution *is* the answer.
- **Structure instead of parsing.** The output is a masked softmax over candidate letter tokens — it can never be malformed.
- **Calibration as a first-class concern.** Raw softmax is over-confident (a recurring finding across Jev reproductions). Temperatures are fitted per (question type × candidate count) bucket on a held-out labeled set, in the spirit of RLCD-style calibrated training — but training-free.
- **Abstention built in.** Every question carries an explicit `__insufficient_evidence__` slot; when it wins, the result is routed to a human, never auto-executed.

The full design rationale, failure modes, and an 18-repository field study of the Jev ecosystem live in the companion essay [`understanding-jev`](https://github.com/PengyiZhang/understanding-jev).

### How it works

```text
system:  <your task instruction>            ← task injection: any decision task
         judging criteria / scene
user:    [image]                            ← injected by the chat template
         Question 1: person type?
         (A) airport ground staff: hi-vis vest, ground crew uniform
         (B) flight attendant: airline uniform, scarf
         …
         (G) __insufficient_evidence__
         Answer 1:                          ← slot left EMPTY
         Question 2: hi-vis vest? …  Answer 2:
         Question 3: visible luggage? …  Answer 3:

one forward pass
   └─ read next-token logits at each "Answer k:" position
   └─ mask to letter tokens → softmax(T_bucket) → distribution + gate + abstain mass
```

- **Task injection via the system prompt.** The engine is task-agnostic: your scenario JSON defines the instruction, judging criteria, and questions. Person-type classification is just the built-in demo; any closed-set visual decision works.
- **Canonical option order** (sorted + trailing abstain slot) removes prompt-order jitter; criteria live in the shared instruction area.
- **Three-level gating** on calibrated confidence: `auto ≥ 0.90`, `review 0.60–0.90`, `human < 0.60`; an abstaining top choice always routes to human.

### Install & quickstart

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# run with a local VLM (tested with Qwen3.8-27B and gemma-4-E4B-it)
uv run --with torch --with transformers --with pillow \
    python -m person_type_a.run_demo \
    --scenario person_type_a/scenarios/terminal.json \
    --model /path/to/your-vlm \
    --image crop1.jpg --image crop2.jpg \
    --engine transformers          # or vllm-perq / vllm-plogprob
```

Startup self-checks: every letter must be a single token, and every answer anchor must be locatable in the tokenized prompt — failures exit immediately instead of producing garbage.

### Defining a scenario

```jsonc
{
  "name": "terminal",
  "system": "You are a person-type classifier at an airport. Output only the option letter.",
  "scene": "terminal arrivals, cropped pedestrian images",
  "evidence": ["uniform style", "hi-vis vest", "badge", "luggage"],
  "questions": [
    { "qid": "ptype", "kind": "choice",
      "instructions": "Which type of person is shown?",
      "options": ["airport ground staff", "flight attendant", "passenger", "..."],
      "criteria": ["hi-vis vest, ground crew uniform", "airline uniform", "civilian clothing", "..."] },
    { "qid": "vest", "kind": "binary", "instructions": "Wearing a hi-vis vest?" },
    { "qid": "luggage", "kind": "binary", "instructions": "Carrying visible luggage?" }
  ]
}
```

- `choice`: up to 25 options + 1 auto-appended abstain slot (single-letter capacity).
- `binary`: fixed `no / unclear / yes` + abstain.
- The `system` field is the task-injection point — swap it (and the questions) to repurpose the engine for damage inspection, document triage, UI-state checks, etc.

### Engines

| | `transformers` | `vllm-perq` | `vllm-plogprob` |
| --- | --- | --- | --- |
| Requests per image | 1 | M (one per question) | 1 |
| Image prefill | once | once (needs MM prefix caching) | **once, guaranteed** |
| Decode steps | 0 | 0 | 0 |
| Slot distribution conditioned on | question texts, no answers | its own question only | question texts + neutral placeholder |
| Logits access | full vocab | top-K (sparse) | top-K (sparse) |
| Notes | chat-template assembly, max control | `enable_prefix_caching` on by default | read at placeholder positions via `prompt_logprobs` |

Measured runs of all three engines on a single crop (Qwen3.8-27B): [docs/runtime-en.md](docs/runtime-en.md).

GPU selection for vLLM engines is via `CUDA_VISIBLE_DEVICES` (there is no `device` flag). Image placeholders differ per model family (`<|image_pad|>` for Qwen-style, `<|image|>` for gemma-4); the transformers engine resolves this automatically, vLLM engines accept `--image-token`.

### Calibration

Zero-shot probabilities are normalized, not calibrated. The pipeline:

1. Collect labeled crops from your **production detector** (same resolution/composition distribution), split by person/video source — never by frame — into `train / calibration / test`.
2. Run the scorer at temperature 1.0, dump slot scores.
3. Fit one temperature per `(question kind × K)` bucket by NLL grid search (`calibrator.py`), save `calibrator.json`.
4. Serve with `--calibrator calibrator.json`; monitor ECE on the test split.

Until a calibrator is supplied, treat the confidence values as ranking-only.

### Tests

Pure-Python core; no torch/vLLM needed:

```bash
uv run --with pytest python -m pytest person_type_a/tests -q
```

### Project layout

| Module | Role |
| --- | --- |
| `schema.py` | scenario/question config, canonical ordering, abstain slot, K ≤ 26 guard |
| `encoding.py` | letter assignment, single-token checks, in-context token-id resolution |
| `prompt.py` | system/question text builders; raw-text layouts for vLLM |
| `readout.py` | masked softmax (sparse top-K aware), three-level gating |
| `calibrator.py` | per-bucket temperature fitting, ECE, `calibrator.json` I/O |
| `engine.py` | `ClassifyTask` + `Scorer` protocol + fakes for dependency-free tests |
| `transformers_scorer.py` | chat-template assembly, single forward, all slots at once |
| `vllm_scorers.py` | per-question fan-out and placeholder/prompt-logprobs strategies |
| `classify.py` | orchestration: task → letters → one scoring pass → calibrated results |

### Roadmap

- **Route B** — per-candidate yes-scoring fan-out (structurally immune to option interference; shares one image prefill).
- **Route C** — LoRA fine-tuning on letter-slot CE with order augmentation (the nimble/decider recipe) once ≥300 labels per class accumulate.
- Remote service scorers (OpenAI-compatible endpoints, shared vLLM server with cross-tenant continuous batching).

### Acknowledgments

- [TypeSafe's Jev](https://docs.typesafe.ai/) and Archer Hume's architecture write-ups for the decision-model pattern this project implements.
- The open reproduction community — `kev` (pointer-head + branch masks), `bespokelabsai/nimble` (open recipe + human-labeled benchmarks), `Mapika/decider-2b` (three primitives), `rlcd-modernbert-151m` (per-cardinality temperatures) — for empirically validating the design choices summarized here.

### License

MIT — see [LICENSE](LICENSE).
