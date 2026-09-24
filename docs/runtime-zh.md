# 实测记录：三引擎同图对比

[English version](runtime-en.md)

## 环境

| 项 | 值 |
| --- | --- |
| 模型 | Qwen3.8-27B（本地权重 `LLMs/Qwen3.8-27B/`） |
| GPU | 单卡（`CUDA_VISIBLE_DEVICES=1`） |
| 场景 | `person_type_a/scenarios/terminal-sim.json`，4 问（人员类型 / 男性？ / 头发颜色 / 年龄段） |
| 图像 | 单张行人 crop（`fewshot/9.png`） |
| 校准 | 零样本、temperature=1（未校准，置信度仅作排序参考） |

说明：单张图、含首轮开销的非严谨测量，延迟数字用于量级感知，不构成基准。

## CLI

```text
usage: run_demo.py [-h] --scenario SCENARIO --model MODEL --image IMAGE
                   [--engine {transformers,vllm-perq,vllm-plogprob}] [--topk TOPK]
                   [--image-token IMAGE_TOKEN] [--calibrator CALIBRATOR] [--device DEVICE]

options:
  --scenario SCENARIO   场景 JSON 路径
  --model MODEL         VLM 模型 id（如 Qwen/Qwen3.8-27B-Instruct）
  --image IMAGE         crop 图路径，可多次
  --engine              transformers=单前向直读；vllm-perq=每问独立请求；vllm-plogprob=占位符 prompt_logprobs
  --topk TOPK           vLLM logprobs 的 K（须 ≥ 候选字母数）
  --image-token         vLLM 引擎的图像占位符；缺省经 AutoProcessor 自动解析
  --calibrator          calibrator.json 路径（可选）
  --device              仅 transformers 引擎生效；vLLM 引擎用 CUDA_VISIBLE_DEVICES 选卡
```

## transformers（单前向，多问单发）

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m person_type_a.run_demo \
  --scenario person_type_a/scenarios/terminal-sim.json \
  --model LLMs/Qwen3.8-27B/ --image fewshot/9.png --engine transformers
```

```text
== fewshot/9.png（1424 ms，单次前向 4 问）==
[人员类型] airport ground staff (0.49)  gate=human  abstain=0.00
    airport ground staff (0.49)  flight attendant (0.26)  passenger (0.25)
[男性？] yes (0.68)  gate=review  abstain=0.02
    no (0.27)  unclear (0.02)  yes (0.68)  __insufficient_evidence__ (0.02)
[头发颜色] gray (0.30)  gate=human  abstain=0.04
    black (0.11)  blue (0.07)  gray (0.30)  green (0.05)  other (0.14)  red (0.09)  white (0.20)  __insufficient_evidence__ (0.04)
[年龄段] 中年 (0.36)  gate=human  abstain=0.34
    中年 (0.36)  儿童 (0.12)  老年 (0.16)  青少年 (0.01)  __insufficient_evidence__ (0.34)
```

## vllm-perq（每问独立请求）

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

## vllm-plogprob（占位符 + prompt_logprobs）

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

## 解读

1. **三引擎 Top-1 全部一致**（ground staff / yes / black 或 gray / 中年），但分布形态差异显著。布局敏感性是真实存在的：transformers 的多问单发布局里，后问能看到前问文本，分布更保守（男性 0.68）；vLLM 两引擎的问题隔离布局让同样的判断给出 0.99~1.00。这与 Jev 生态研究中"槽位分布条件于布局"的发现一致——**校准必须与推理使用同一引擎同一布局**。
2. **弃权质量也随布局漂移**：年龄段问题在 transformers 布局下弃权质量 0.34，在两个 vLLM 布局下接近 0。同图同权重，说明弃权槽的语义同样需要按引擎校准。
3. **延迟量级**：plogprob 1248ms < transformers 1424ms < perq 1536ms（单图、含首轮开销）。perq 为每问独立请求付出 M 次往返；plogprob 单请求一次前向，最快；transformers 单前向但有 Python 侧组装开销。批量与稳态数字待正式基准。
4. **未校准的置信度不可运营**：vllm 布局下 binary 问题普遍 0.99+，属于生态研究反复记录的 softmax 过度自信；接入 `--calibrator` 前所有 gate 判定仅作演示。
