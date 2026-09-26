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

## gemma-4-E4B 三引擎（字母槽）

同一图同一场景，`~/LLMs/gemma-4-E4B-it`：

| 引擎 | 延迟 | 人员类型 | 性别 | 头发颜色 | 年龄段 |
| --- | --- | --- | --- | --- | --- |
| transformers | 1146 ms | passenger 0.68 | no 0.69 | black 0.80 | 老年 0.38（儿童 0.30） |
| vllm-perq | 1083 ms | ground staff 0.68 | yes 1.00 | black 0.97 | 中年 0.95 |
| vllm-plogprob | 1245 ms | passenger 0.68 | yes 1.00 | other 0.91 | **弃权 0.97** |

解读：gemma 对这张图的读取与 Qwen 存在模型间差异（passenger vs ground staff），且
**模型内部同样存在布局效应**——问题隔离（perq）后判断系统性变尖锐；plogprob 在
年龄段上给出 0.97 的弃权，是哑字母机制下真实的弃权行为而非故障（分布非均匀）。

## JSON 生成式基线对照

`json_baseline.py`：同一场景、同一图、同一 chat 模板组装，让模型自回归生成 JSON
答案（4 问，仅硬标签，无分布/弃权/门控）：

| 模型 | 引擎 | 延迟 | 解码 token | 解析 | 答案 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | transformers（未关 thinking） | **26850 ms** | 256（顶满上限） | ❌ 失败 | 思考文本泄漏，未产出 JSON |
| Qwen3.8-27B | transformers（关 thinking） | 6230 ms | 43 | ✅ | flight attendant / yes / black / 中年 |
| Qwen3.8-27B | vLLM | 1690 ms | 43 | ✅ | flight attendant / yes / black / 中年 |
| gemma-4-E4B | transformers | 4153 ms | 46 | ✅ | ground staff / yes / black / 中年 |
| gemma-4-E4B | vLLM | **507 ms** | 46 | ✅ | ground staff / yes / black / 中年 |

对照结论（同引擎比）：

1. **27B 上字母槽全面占优**：Qwen transformers 1453ms vs JSON 6230ms（**4.3×**），
   vLLM 1143/1188ms vs 1690ms（**1.5×**）。模型越大、解码越贵，零解码优势越大。
2. **冷启动下小模型 JSON 可短暂反超**（gemma-E4B vLLM JSON 507ms vs 字母槽
   1083/1245ms）——但这是首轮开销的假象，稳态基准（下节）中 perq 借前缀缓存
   降到 38ms、反超 11×。字母槽的结构性优势（分布/弃权/门控/免解析）在任何
   regime 都成立，稳态下速度同样全面占优。
3. **JSON 路径的脆弱性实测**：未显式关闭 thinking 时 27B 直接输出 26.8 秒的思考
   文本并解析失败——这正是"朴素 JSON 方案"的典型死法，且答案只有硬标签，
   无置信度可校准。
4. 答案一致性：Qwen JSON 与字母槽 perq 的 Top-1 完全一致（flight attendant /
   yes / black / 中年），互为佐证。

## 稳态基准（warmup=2 + 10 次重复，中位数，`benchmark.py`）

| 模型 | 字母槽 transformers | 字母槽 perq | 字母槽 plogprob | JSON transformers | JSON vllm |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | 271 ms | 239 ms | **182 ms** | 4233 ms | 1670 ms |
| gemma-4-E4B | 132 ms | **38 ms** | 156 ms | 2762 ms | 427 ms |

稳态结论：

1. **预热后字母槽对 JSON 全面占优（同引擎 7~21×）**：Qwen transformers 15.6×、
   Qwen vLLM 7.0×（perq）/ 9.2×（plogprob）、gemma transformers 20.9×、
   gemma vLLM 11.2×（perq）。冷启动 ~1s 的首轮开销掩盖了这一差距——生产部署
   必须预热。
2. **最优引擎随模型而变**：27B 用 plogprob（182ms，单请求 + 前缀缓存）；小 MoE
   用 perq（38ms，前缀缓存命中后 4 个短请求近乎零算力）。
3. **plogprob 的逐位置 top-K 有固定开销**：gemma 上 p95 波动到 307ms；27B 上
   该开销被解码优势盖过，反而最快。

## CIFAR-10 四路对比：字母槽 vs JSON vs 零样本 vs LoRA 训练

> 完整 GRPO 训练管线在 `person_type_a/train.py`，数据适配在 `dataset.py`。
> 同一 val split（200 样本）、同一模型（gemma-4-E4B）、同一 A100-80G。

| 方法 | accuracy | 延迟/样本 | 置信度 | ECE | 解析成功率 |
| --- | --- | --- | --- | --- | --- |
| JSON 生成（零样本） | **89.5%** | 500 ms | ❌ 无 | ❌ 无 | 100% |
| 字母槽单前向（零样本） | **1.0%** | 115 ms | 95.5%（虚高） | 94.5% | N/A |
| 字母槽 + LoRA 5%数据 | 12.0% | 127 ms | 19.3%（诚实） | 8.3% | N/A |
| 字母槽 + LoRA 全量训练 | ？（进行中） | ~120 ms | ？ | ？ | ？ |

### 核心发现

1. **零样本场景 JSON 完胜**：VLM 预训练天然支持"看图→生成类名"，但不支持"看图→输出字母代号"。
   字母槽要求模型做三步压缩（视觉理解→类名映射→字母映射），这在预训练分布之外。
   laya 官方同样确认："a fast base to specialise, **not** a zero-shot decision engine"。
2. **字母槽零样本的 ECE = 94.5%**：模型对完全不懂的东西给出 95% 置信度——softmax
   过度自信的极端形态。
3. **GRPO + proper-reward 的校准效果立竿见影**：仅 5% 数据训练后 ECE 从 94.5% 降至
   8.3%，模型学会"不知道就承认不知道"（置信度从虚高 95.5% 降至诚实 19.3%）。
4. **正确叙事**：字母槽不是零样本方案，是"训练一次、终身快服务"的生产优化：

```text
冷启动：JSON 生成（89.5%，500ms）→ 蒸馏标注 → GRPO 训练字母槽 LoRA
生产：  字母槽推理（目标 ≈85-95%，~120ms，4.3× 加速 + 校准 + 门控）
```

5. **速度优势仅在训练后兑现**：零样本时 115ms vs 500ms 的 4.3× 加速毫无意义
   （精度 1% vs 89.5%）；训练后如果精度追平 JSON，加速才有价值。

### 训后对比（来自生态数据）

| 项目 | 训练量 | 字母槽精度 | 参考（Jev 官方） |
| --- | --- | --- | --- |
| kev | 有训练 | 0.917 | 0.965 |
| nimble | 2,676 样本 | 0.748 | 0.760 |
| decider-2b | 942K 样本 | 0.766 | 0.727（超过） |
| laya typed-decisions | 2,000 决策 | 0.766 | 0.727 |
