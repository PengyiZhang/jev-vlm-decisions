# 实测记录：三引擎同图对比

[English version](runtime-en.md)

> 数据为 chat 模板统一后（v0.4+）在 server178（单卡 A100-80G）实测；旧版裸文本
> 时代的数据已废弃。全程零样本、temperature=1，未校准。

## 环境

| 项 | 值 |
| --- | --- |
| 模型 | Qwen3.8-27B（本地权重 `~/LLMs/Qwen3.8-27B`） |
| GPU | 单卡 A100-80G |
| 场景 | `person_type_a/scenarios/terminal-sim.json`，4 问（人员类型 / 性别 / 头发颜色 / 年龄段） |
| 图像 | 单张行人 crop |
| 校准 | 零样本、temperature=1（未校准，置信度仅作排序参考） |

说明：单张图、含首轮开销的非严谨测量，延迟数字用于量级感知，不构成基准。
transformers 路径因 Qwen3.8 混合注意力优化内核（flash-linear-attention /
causal_conv1d）未安装而走参考实现，延迟略有虚高。

## CLI

```text
usage: run_demo.py [-h] --scenario SCENARIO --model MODEL --image IMAGE
                   [--engine {transformers,vllm-perq,vllm-plogprob}] [--topk TOPK]
                   [--calibrator CALIBRATOR] [--device DEVICE]

options:
  --scenario SCENARIO   场景 JSON 路径
  --model MODEL         VLM 模型 id（如 Qwen/Qwen3.8-27B-Instruct）
  --image IMAGE         crop 图路径，可多次
  --engine              transformers=单前向直读；vllm-perq=每问独立请求；vllm-plogprob=哑字母 prompt_logprobs
  --topk TOPK           vLLM logprobs 的 K（须 ≥ 候选字母数）
  --calibrator          calibrator.json 路径（可选）
  --device              仅 transformers 引擎生效；vLLM 引擎用 CUDA_VISIBLE_DEVICES 选卡
```

## transformers（单前向，多问单发，1453 ms）

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

## vllm-perq（每问独立请求，1143 ms）

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

## vllm-plogprob（哑字母 + prompt_logprobs，1188 ms）

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

## 解读

1. **布局敏感性连 Top-1 都会翻转**：同一图同一权重，多问单发（transformers）与
   问题隔离（perq）在人员类型上给出不同的第一名（ground staff 0.49 vs flight
   attendant 0.55）——多问上下文让模型保守，问题隔离让判断更尖锐。这比"分布形态
   不同"更强：**校准与推理必须同引擎同布局**，否则连排序都不可迁移。
2. **弃权行为也随布局漂移**：年龄段问题在 transformers 布局下弃权 0.34、
   plogprob 下 0.10、perq 下 ≈0。弃权槽语义需要按引擎分别校准。
3. **延迟排序（本组数据）**：perq 1143ms < plogprob 1188ms < transformers 1453ms。
   perq 一次提交 M 个短请求由引擎合批，反而最快；transformers 受参考实现内核拖累。
4. **未校准的置信度不可运营**：perq/plogprob 下 binary 问题普遍 0.95+，属于生态
   研究反复记录的 softmax 过度自信；接入 `--calibrator` 前所有 gate 判定仅作演示。

## 工程发现：chat 模板化的三个坑（Qwen3.8 实测）

统一 chat 模板组装的过程中踩到三个 in-context 陷阱，全部已修复并有测试锁定：

| 坑 | 症状 | 修复 |
| --- | --- | --- |
| 模板默认插入 `<think>` 块 | perq 生成位置落在思考块内，字母进不了 top-K（全选项恰好均匀分布的指纹） | 渲染传 `enable_thinking=False`，模板不支持时回退 |
| 空槽位塌缩 | 空"答案 k："后模型以 ~100% 概率输出 `<\|im_end\|>` 或散文式作答（"男性/短发"），槽 2+ 字母掉出 top-K | 槽位填候选集外的哑字母 |
| 中性符号被学舌 | 占位符"？"被模型当成 in-context 格式示例，后续槽位跟着输出标点 | 同上：哑字母锚定"答案=字母"格式且不污染候选内容 |

诊断指纹值得记住：**所有选项恰好均匀分布（0.25/0.125…）= 候选字母全部不在
top-K、全部吃到地板分**——见 `readout.LOW_SCORE`。
