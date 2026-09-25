# Measured Runs: All Three Engines on One Image

[中文版](runtime-zh.md)

> Numbers measured on server178 (single A100-80G) after the chat-template
> unification (v0.4+); the raw-text-era numbers are obsolete. Zero-shot,
> temperature=1, uncalibrated throughout.

## Setup

| Item | Value |
| --- | --- |
| Model | Qwen3.8-27B (local checkpoint at `~/LLMs/Qwen3.8-27B`) |
| GPU | single A100-80G |
| Scenario | `person_type_a/scenarios/terminal-sim.json`, 4 questions (person type / sex / hair color / age group) |
| Image | a single pedestrian crop |
| Calibration | zero-shot, temperature=1 (uncalibrated — confidence values are ranking-only) |

Note: single-image measurement including first-call overhead; treat the latency
numbers as order-of-magnitude, not a benchmark. The transformers path runs the
reference PyTorch implementation of Qwen3.8's hybrid-attention kernels
(flash-linear-attention / causal_conv1d not installed), inflating its latency
somewhat.

## CLI

```text
usage: run_demo.py [-h] --scenario SCENARIO --model MODEL --image IMAGE
                   [--engine {transformers,vllm-perq,vllm-plogprob}] [--topk TOPK]
                   [--calibrator CALIBRATOR] [--device DEVICE]

options:
  --scenario SCENARIO   path to the scenario JSON
  --model MODEL         VLM model id (e.g. Qwen/Qwen3.8-27B-Instruct)
  --image IMAGE         crop image path; repeatable
  --engine              transformers=single-forward readout; vllm-perq=one request per question;
                        vllm-plogprob=dummy-letter + prompt_logprobs
  --topk TOPK           vLLM logprobs K (must be >= the number of candidate letters)
  --calibrator          optional calibrator.json path
  --device              transformers engine only; vLLM engines use CUDA_VISIBLE_DEVICES
```

## transformers (single forward, all questions in one prompt, 1453 ms)

```text
[人员类型] airport ground staff (0.49)  gate=human  abstain=0.00
    airport ground staff (0.49)  flight attendant (0.26)  passenger (0.25)
[性别] yes (0.68)  gate=review  abstain=0.02
    no (0.27)  unclear (0.02)  yes (0.68)  __insufficient_evidence__ (0.02)
[头发颜色] gray (0.30)  gate=human  abstain=0.04
    black (0.11)  blue (0.07)  gray (0.30)  green (0.05)  other (0.14)  red (0.09)  white (0.20)  __insufficient_evidence__ (0.04)
[年龄段] 中年 (0.36)  gate=human  abstain=0.34
    中年 (0.36)  儿童 (0.12)  老年 (0.16)  青少年 (0.01)  __insufficient_evidence__ (0.34)
```

## vllm-perq (one independent request per question, 1143 ms)

```text
[人员类型] flight attendant (0.55)  gate=human  abstain=0.00
    airport ground staff (0.33)  flight attendant (0.55)  passenger (0.12)
[性别] yes (0.99)  gate=auto  abstain=0.00
    yes (0.99)
[头发颜色] black (1.00)  gate=auto  abstain=0.00
    black (1.00)
[年龄段] 中年 (0.99)  gate=auto  abstain=0.00
    中年 (0.99)
```

## vllm-plogprob (dummy-letter + prompt_logprobs, 1188 ms)

```text
[人员类型] airport ground staff (0.81)  gate=review  abstain=0.00
    airport ground staff (0.81)  flight attendant (0.09)  passenger (0.10)
[性别] yes (0.95)  gate=auto  abstain=0.00
    no (0.02)  unclear (0.03)  yes (0.95)
[头发颜色] other (0.55)  gate=human  abstain=0.00
    black (0.38)  blue (0.04)  other (0.55)  red (0.03)
[年龄段] 中年 (0.90)  gate=review  abstain=0.10
    中年 (0.90)  __insufficient_evidence__ (0.10)
```

## Interpretation

1. **Layout sensitivity flips even the top-1**: on the same image and weights,
   the multi-question layout (transformers) and the question-isolated layout
   (perq) disagree on the winner for person type (ground staff 0.49 vs flight
   attendant 0.55) — multi-question context makes the model conservative,
   isolation sharpens it. Stronger than "distributions differ": **calibrate
   with the same engine and layout you serve with**, or even the ranking is
   not transferable.
2. **Abstention drifts with layout too**: the age-group question abstains at
   0.34 / 0.10 / ≈0 across the three layouts on the same input. Abstention
   semantics need per-engine calibration like everything else.
3. **Latency order (this batch)**: perq 1143 ms < plogprob 1188 ms <
   transformers 1453 ms. perq submits M short requests that the engine batches
   internally and comes out fastest; transformers is dragged by the reference
   kernels.
4. **Uncalibrated confidence is not operational**: the 0.95+ values on binary
   questions are the softmax over-confidence repeatedly documented across Jev
   reproductions. Until `--calibrator` is wired up, treat every gate decision
   as a demo.

## Engineering findings: three chat-template pitfalls (measured on Qwen3.8)

Three in-context traps surfaced while unifying on chat templates; all are
fixed and locked by tests:

| Pitfall | Symptom | Fix |
| --- | --- | --- |
| Template inserts a `<think>` block by default | perq's generation position lands inside the reasoning block; letters never reach top-K (the tell-tale exactly-uniform distribution) | render with `enable_thinking=False`, fall back when unsupported |
| Empty-slot collapse | after an empty "Answer k:" the model emits `<\|im_end\|>` at ~100% or answers in prose ("male", "short hair"); letters drop out of top-K for slots 2+ | fill slots with a dummy letter beyond the candidate set |
| Neutral-symbol mimicry | a "？" placeholder gets imitated as an in-context format example; later slots follow with punctuation | same dummy-letter fix: anchors the "answers are letters" format without polluting candidates |

The diagnostic signature is worth remembering: **every option at an exactly
uniform probability (0.25 / 0.125 …) means no candidate letter made it into
top-K at all — they all hit the floor score** (`readout.LOW_SCORE`).

## gemma-4-E4B across three engines (letter-slot)

Same image and scenario, `~/LLMs/gemma-4-E4B-it`:

| Engine | Latency | Person type | Sex | Hair | Age |
| --- | --- | --- | --- | --- | --- |
| transformers | 1146 ms | passenger 0.68 | no 0.69 | black 0.80 | elderly 0.38 (child 0.30) |
| vllm-perq | 1083 ms | ground staff 0.68 | yes 1.00 | black 0.97 | middle-aged 0.95 |
| vllm-plogprob | 1245 ms | passenger 0.68 | yes 1.00 | other 0.91 | **abstain 0.97** |

Notes: gemma reads this image differently from Qwen (passenger vs ground staff —
a between-model difference), and the layout effect exists *within* a model too:
isolation (perq) systematically sharpens judgments. The 0.97 abstain on the age
question under plogprob is genuine abstention under the dummy-letter mechanism,
not a failure (the distribution is not uniform).

## JSON-generation baseline comparison

`json_baseline.py`: same scenario, same image, same chat-template assembly,
but the model autoregressively generates a JSON answer (4 questions, hard
labels only — no distributions, abstention, or gating):

| Model | Engine | Latency | Decode tokens | Parse | Answers |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | transformers (thinking on) | **26850 ms** | 256 (cap hit) | ❌ failed | reasoning text leaked, no JSON |
| Qwen3.8-27B | transformers (thinking off) | 6230 ms | 43 | ✅ | flight attendant / yes / black / middle-aged |
| Qwen3.8-27B | vLLM | 1690 ms | 43 | ✅ | flight attendant / yes / black / middle-aged |
| gemma-4-E4B | transformers | 4153 ms | 46 | ✅ | ground staff / yes / black / middle-aged |
| gemma-4-E4B | vLLM | **507 ms** | 46 | ✅ | ground staff / yes / black / middle-aged |

Takeaways (same-engine comparisons):

1. **Letter-slot wins across the board at 27B**: Qwen transformers 1453 ms vs
   JSON 6230 ms (**4.3×**); vLLM 1143/1188 ms vs 1690 ms (**1.5×**). The bigger
   the model, the more expensive each decode step, the larger the zero-decode
   advantage.
2. **On small models JSON can be faster**: gemma-E4B (small MoE, ~90 tok/s
   decode) does vLLM JSON in 507 ms, beating letter-slot (1083/1245 ms). perq
   pays 4 prefills; plogprob pays top-K logprobs for every prompt position;
   E4B's 46 decode steps are simply cheap. In this regime letter-slot's case
   is **structural**: distributions + abstention + three-level gating + no
   parsing and no thinking-handling — not raw latency.
3. **JSON fragility, measured**: with thinking left on, the 27B emits 26.8
   seconds of reasoning text and fails to parse — the canonical death of the
   naive JSON approach; and even on success you get hard labels with no
   confidence to calibrate.
4. **Answer agreement**: Qwen JSON matches the letter-slot perq top-1 exactly
   (flight attendant / yes / black / middle-aged) — mutual corroboration.
