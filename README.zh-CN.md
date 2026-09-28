# 单前向 VLM 决策引擎

中文 | **[English](README.md)**

**面向视觉语言模型的 Jev 式字母槽决策。** 一次前向把任意 VLM 变成毫秒级图像决策引擎：零自回归解码、零 JSON 解析、开箱即得概率分布。

```text
[ptype] airport ground staff (0.85)  gate=auto  abstain=0.01
    airport ground staff (0.85)  flight attendant (0.10)  passenger (0.05)
```

**CIFAR-10 实测验证**：标准 MCQ 格式 + RLCD 训练（GRPO + proper-reward，LoRA rank 8）达到
**97.0% 精度、ECE 2.65%、133ms**——4B 模型击败 27B JSON 生成（94.5%，846ms），加速 **6.4×**。
零样本标准 MCQ 即可达 86.5%。完整 11 条件实验矩阵见
[docs/cifar10-experiment-report.md](docs/cifar10-experiment-report.md)

## 为什么

让 VLM"分类这张图并用 JSON 回答"要付出完整自回归循环的代价：几十到上百步解码、格式错误、未校准的置信度。对闭集决策——路由、门控、属性打标——这些代价都可以免掉。

本项目沿用 [Jev](https://docs.typesafe.ai/)（TypeSafe 的 "System One" 模型）及其开源生态带火的决策模型模式：

- **零解码步。** 候选渲染成 prompt 里的字母槽，答案槽留空。一次 Prefill 返回每个位置的 next-token Logits，字母分布本身就是答案。
- **结构代替解析。** 输出是候选字母 token 上的掩码 Softmax——永远不会格式错误。
- **校准是一等公民。** 裸 Softmax 普遍过度自信（Jev 复现生态反复验证的结论）。温度按（问题类型 × 候选数）分桶、在留出标注集上拟合。
- **弃权内置。** 每个问题都带显式 `__insufficient_evidence__` 槽；弃权胜出时恒转人工，绝不自动执行。

完整设计依据、失败模式与 18 仓 Jev 生态实地研究见姊妹长文 [`understanding-jev`](https://github.com/PengyiZhang/understanding-jev)。两篇正文已随本仓库收录：[万字解读长文](docs/jev-starter.md)与 [18 仓生态实地研究](docs/jev-ecosystem-research.md)；另附 [laya 专项深读](docs/laya-deep-dive.md)。

## 工作原理

```text
system:  <你的任务指令>                     ← 任务注入：任意决策任务
         判定依据 / 场景
user:    [图像]                             ← 由 chat template 注入
         问题 1：人员类型？
         (A) airport ground staff：反光背心、地面作业制服
         (B) flight attendant：航司制服丝巾
         …
         (G) __insufficient_evidence__
         答案 1：                            ← 槽位留空
         问题 2：反光背心？ …  答案 2：
         问题 3：可见行李？ …  答案 3：

一次前向
   └─ 读每个"答案 k："位置的 next-token Logits
   └─ 掩码到字母 token → softmax(T_桶) → 分布 + 门控 + 弃权质量
```

- **任务经 system 提示词注入。** 引擎与任务无关：场景 JSON 定义指令、判定依据与问题。
- **规范化选项顺序**（字典序 + 末位弃权槽）消除 prompt 顺序抖动。
- **三级门控**：`auto ≥ 0.90`、`review 0.60~0.90`、`human < 0.60`。

## 安装与快速开始

需要 Python 3.10+ 与 [`uv`](https://docs.astral.sh/uv/)。

```bash
uv run --with torch --with transformers --with pillow \
    python -m person_type_a.run_demo \
    --scenario person_type_a/scenarios/terminal.json \
    --model /path/to/your-vlm \
    --image crop1.jpg --image crop2.jpg \
    --engine transformers          # 或 vllm-perq / vllm-plogprob
```

## 定义场景

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

- `choice`：最多 25 个选项 + 1 个自动追加的弃权槽。
- `binary`：固定 `no / unclear / yes` + 弃权。
- `system` 字段是任务注入点——换掉它（和 questions）即可改造为损毁检测、单据分诊等任务。

## 推理引擎

| | `transformers` | `vllm-perq` | `vllm-plogprob` |
| --- | --- | --- | --- |
| 每图请求数 | 1 | M（每问一个） | 1 |
| 图像 Prefill | 1 次 | 1 次（依赖多模态前缀缓存） | **必然 1 次** |
| 解码步 | 0 | 0 | 0 |
| Logits 形态 | 全词表 | top-K（稀疏） | top-K（稀疏） |

三引擎实测记录：[docs/runtime-zh.md](docs/runtime-zh.md)。

## 校准

零样本概率只保证归一，不保证校准。流程：采集标注 → 以 T=1 跑 scorer → 按（问题类型 × K）分桶 NLL 网格搜索拟合温度 → `--calibrator calibrator.json` 注入 → test 集 ECE 监控。

## 测试

核心纯标准库，无需 torch/vLLM：

```bash
uv run --with pytest python -m pytest person_type_a/tests -q
```

## 模块结构

### 推理核心（测试无 ML 依赖）

| 模块 | 职责 |
| --- | --- |
| `schema.py` | 场景/问题配置、字典序规范排序、弃权槽、K≤26 校验 |
| `encoding.py` | 字母分配、单 token 校验、上下文 token id 解析 |
| `prompt.py` | 系统/问题文本构建与共享 chat 消息组装 |
| `readout.py` | 掩码 Softmax（支持稀疏 top-K）、三级门控 |
| `calibrator.py` | 分桶温度拟合、ECE、`calibrator.json` 读写 |
| `engine.py` | `ClassifyTask` + `Scorer` 协议 + 无依赖测试假实现 |
| `classify.py` | 组装管线：任务 → 字母 → 一次打分 → 校准结果 |

### 推理引擎（惰性导入 torch/vLLM）

| 模块 | 职责 |
| --- | --- |
| `transformers_scorer.py` | chat template 组装、单前向读全部槽位 |
| `vllm_scorers.py` | 每问扇出（`vllm-perq`）+ 哑字母 `prompt_logprobs`（`vllm-plogprob`） |
| `run_demo.py` | CLI：场景 + 模型 + 图像 → 逐问题分布输出 |
| `benchmark.py` | 预热 + 重复推理速度基准（min/mean/median/p95） |

### 训练管线（CIFAR-10 验证）

| 模块 | 职责 |
| --- | --- |
| `proper_reward.py` | 严格 proper scoring rule（log + spherical），GRPO reward |
| `grpo_trainer.py` | GRPO 损失：噪声采样、组内 advantage、策略梯度 + CE 引导 |
| `dataset.py` | CIFAR-10 → 字母槽适配（顺序增广、四路切分） |
| `train.py` | 训练 CLI：定制格式（GRPO / `--pure-ce` 可切换） |
| `train_mcq.py` | 训练 CLI：**标准 MCQ 格式**（RLCD，突破性成果） |
| `pointer_head.py` | 交叉注意力指针头架构（kev 式） |
| `train_pointer.py` | 指针头训练 CLI（marker 检测待修复） |

### 评测与基线

| 模块 | 职责 |
| --- | --- |
| `evaluate.py` | 统一评测 CLI（`--mode letter/json`、`--lora`、ECE） |
| `eval_mcq.py` | 标准 MCQ 评测（`--model`、`--lora`，精度/ECE/延迟） |
| `json_baseline.py` | JSON 生成基线（速度对比） |
| `distill.py` | 蒸馏：JSON 伪标签 → GRPO 字母槽 LoRA |

### 测试（15 个文件，61+ 用例）

全部在 `person_type_a/tests/`。

### 文档

| 文档 | 内容 |
| --- | --- |
| [cifar10-experiment-report.md](docs/cifar10-experiment-report.md) | **完整 11 条件实验矩阵**（格式/训练/模型消融，RLCD 突破） |
| [runtime-zh.md](docs/runtime-zh.md) / [runtime-en.md](docs/runtime-en.md) | 三引擎实测、JSON 基线对比、稳态基准 |
| [jev-starter.md](docs/jev-starter.md) | 万字解读长文：机制 / 失败模式 / 设计模式 / 生产架构 |
| [jev-ecosystem-research.md](docs/jev-ecosystem-research.md) | 18 仓 Jev 生态实地研究 |
| [laya-deep-dive.md](docs/laya-deep-dive.md) | encoder 路线决策引擎：RAG 排序、RLCD 训练 |

## 开发时间线

| 日期 | 里程碑 |
| --- | --- |
| 2026-09-21 | 路线 A 实现：字母槽 + 三引擎 + 61 测试 |
| 2026-09-22 | vLLM 双策略统一 chat 模板；稳态基准（对 JSON 加速 7~21×） |
| 2026-09-23 | JSON 生成基线对比（27B 上 4.3× 加速）；gemma-4-E4B 跨引擎实测 |
| 2026-09-25 | GRPO 训练管线落地（proper_reward + dataset + grpo_trainer + train）；CIFAR-10 四路对比 |
| 2026-09-26 | GRPO 40K 训练 → **σ 退火陷阱定位**（收敛到"诚实均匀"）；纯 CE 对照 |
| 2026-09-27 | Rank-32 实验（容量不是瓶颈）；指针头架构；Qwen-27B 跨模型验证 |
| 2026-09-28 | **标准 MCQ 突破**：零样本 86.5%（格式才是瓶颈，不是模型能力） |
| 2026-09-28 | **RLCD 训练成果：97.0% 精度、ECE 2.65%、133ms**——4B 击败 27B JSON 基线 |
| 2026-09-29 | 纯 CE 对照（标准 MCQ）：96.5% / ECE 2.57%——好格式上两种目标都成功，RLCD 再挤 +1.0pp 精度 |

## 路线图

- **指针头修复**：交叉注意力选项打分（彻底绕过字母映射）
- **多问题 MCQ**：扩展单问 MCQ 到多属性（逐问独立 MCQ 块）
- **生产服务化**：vLLM server 模式 + continuous batching + OpenAI 兼容端点
- **领域适配**：在生产域数据上从 JSON 生成蒸馏

## 许可

MIT——见 [LICENSE](LICENSE)。
