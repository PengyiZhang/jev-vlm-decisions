# Measured Runs: All Three Engines on One Image

[中文版](runtime-zh.md)

## Setup

| Item | Value |
| --- | --- |
| Model | Qwen3.8-27B (local checkpoint at `LLMs/Qwen3.8-27B/`) |
| GPU | single card (`CUDA_VISIBLE_DEVICES=1`) |
| Scenario | `person_type_a/scenarios/terminal-sim.json`, 4 questions (person type / male? / hair color / age group) |
| Image | a single pedestrian crop (`fewshot/9.png`) |
| Calibration | zero-shot, temperature=1 (uncalibrated — confidence values are ranking-only) |

Note: single-image measurement including first-call overhead; treat the latency numbers as order-of-magnitude, not a benchmark.

## CLI

```text
usage: run_demo.py [-h] --scenario SCENARIO --model MODEL --image IMAGE
                   [--engine {transformers,vllm-perq,vllm-plogprob}] [--topk TOPK]

options:
  --scenario SCENARIO   path to the scenario JSON
  --model MODEL         VLM model id (e.g. Qwen/Qwen3.8-27B-Instruct)
  --image IMAGE         crop image path; repeatable
  --engine              transformers=single-forward readout; vllm-perq=one request per question;
                        vllm-plogprob=placeholder + prompt_logprobs
  --topk TOPK           vLLM logprobs K (must be >= the number of candidate letters)
  --calibrator          optional calibrator.json path
  --device              transformers engine only; vLLM engines use CUDA_VISIBLE_DEVICES
```

## transformers (single forward, all questions in one prompt)

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m person_type_a.run_demo \
  --scenario person_type_a/scenarios/terminal-sim.json \
  --model LLMs/Qwen3.8-27B/ --image fewshot/9.png --engine transformers
```

Original output (question ids in Chinese):

```text
== fewshot/9.png（1424 ms，single forward, 4 questions）==
[人员类型] airport ground staff (0.49)  gate=human  abstain=0.00
    airport ground staff (0.49)  flight attendant (0.26)  passenger (0.25)
[男性？] yes (0.68)  gate=review  abstain=0.02
    no (0.27)  unclear (0.02)  yes (0.68)  __insufficient_evidence__ (0.02)
[头发颜色] gray (0.30)  gate=human  abstain=0.04
    black (0.11)  blue (0.07)  gray (0.30)  green (0.05)  other (0.14)  red (0.09)  white (0.20)  __insufficient_evidence__ (0.04)
[年龄段] 中年 (0.36)  gate=human  abstain=0.34
    中年 (0.36)  儿童 (0.12)  老年 (0.16)  青少年 (0.01)  __insufficient_evidence__ (0.34)
```

## vllm-perq (one independent request per question)

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m person_type_a.run_demo \
  --scenario person_type_a/scenarios/terminal-sim.json \
  --model LLMs/Qwen3.8-27B/ --image fewshot/9.png --engine vllm-perq
```

```text
== fewshot/9.png（1536 ms）==
[人员类型] airport ground staff (0.53)  gate=human  abstain=0.00
    airport ground staff (0.53)  flight attendant (0.25)  passenger (0.22)
[男性？] yes (0.99)  gate=auto  abstain=0.00
    yes (0.99)
[头发颜色] black (1.00)  gate=auto  abstain=0.00
    black (1.00)
[年龄段] 中年 (0.99)  gate=auto  abstain=0.00
    中年 (0.99)
```

## vllm-plogprob (placeholder + prompt_logprobs)

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m person_type_a.run_demo \
  --scenario person_type_a/scenarios/terminal-sim.json \
  --model LLMs/Qwen3.8-27B/ --image fewshot/9.png --engine vllm-plogprob
```

```text
== fewshot/9.png（1248 ms）==
[人员类型] airport ground staff (0.62)  gate=review  abstain=0.00
    airport ground staff (0.62)  flight attendant (0.27)  passenger (0.11)
[男性？] yes (1.00)  gate=auto  abstain=0.00
    yes (1.00)
[头发颜色] black (0.88)  gate=review  abstain=0.00
    black (0.88)  blue (0.03)  gray (0.08)
[年龄段] 中年 (1.00)  gate=auto  abstain=0.00
    中年 (1.00)
```

## Interpretation

1. **Top-1 agrees across all engines** (ground staff / yes / black-or-gray / middle-aged), but the distribution shapes differ substantially. Layout sensitivity is real: in the transformers multi-question layout later questions see earlier question texts and the distributions come out conservative (male? = 0.68), while the question-isolated vLLM layouts give 0.99–1.00 for the same judgments. This matches the Jev ecosystem finding that slot distributions are conditioned on the layout — **calibrate with the same engine and layout you serve with**.
2. **Abstention mass drifts with layout too**: the age-group question carries 0.34 abstain mass in the transformers layout but ~0 in both vLLM layouts, on the same image and weights. Abstention semantics need per-engine calibration like everything else.
3. **Latency order**: plogprob 1248 ms < transformers 1424 ms < perq 1536 ms (single image, first-call overhead included). perq pays one round trip per question; plogprob is a single request; transformers is single-forward with Python-side assembly overhead. Proper batched steady-state numbers are future work.
4. **Uncalibrated confidence is not operational**: the 0.99+ values on binary questions in vLLM layouts are the softmax over-confidence repeatedly documented across Jev reproductions. Until `--calibrator` is wired up, treat every gate decision as a demo.
