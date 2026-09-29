# SafeGuard jev 化完整实验报告：格式微调模型的单步判定改造

> 实验周期：2026-09-28 ~ 2026-09-29
> 环境：server178（A100-80G）、Qwen3Guard-Gen-0.6B（HK 养老领域微调）、
> ElderDomain 24K train / 6K val（`hk_safeguard_*_subset1.jsonl`）
> 代码：`safeguard/`（模块速查见文末）
> 姊妹报告：CIFAR-10 通用 VLM 实验 [`cifar10-experiment-report.md`](cifar10-experiment-report.md)

## 一、任务与起点

SafeGuardElderlyDomain 微调出的 Qwen3Guard-Gen-0.6B 是一个**生成式**安全
判别模型：自回归输出三行文本（~16 解码步，~400ms）：

```text
Safety: Unsafe
Categories: Violent, Hidden Elder Crisis
Refusal: No
```

13 类 = Qwen3Guard 基础 9 类 + HK 养老域新增 4 类（HK Welfare & Financial
Scam / RCHE & Caregiver Malpractice / Medication & Health Misguidance /
Hidden Elder Crisis）。目标：改成 jev 范式——0 解码步、单次 prefill、
锚点读 next-token logits、带校准置信度的全链判定。

研究对象与 CIFAR-10 报告互为反面：那边是**通用** VLM 零样本做字母槽，
这边是一个**被输出格式深度微调过的小模型**——格式先验既是资产也是枷锁。

## 二、Phase 1：零样本三路线（6000 条 val）

| 路线 | 形态 | 前向 | safety acc | 类别 | refusal | ECE | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 生成基线 | 自回归三行 | 1+16 解码 | 97.07% | EM 84.27% | 96.99% | — | ~400 |
| **Route 1 原生锚定** | guard 模板 + `"Safety:"` 前缀 | **1** | **97.07%（无损）** | 结构性不可覆盖 | 同左 | 0.52% | **34.4** |
| Route 2a 每问 MCQ | 标准字母 MCQ ×14-15 问 | 15 | 19.05% | EM 46.17% | 65.57% | 59% | 159.6 |
| Route 2b 答题卡 | 字母槽卡 | 1 | 40%（debug 20 条） | 0 | 0% | — | ~38 |

关键发现：

1. **Route 1 与生成基线 acc/macro-F1 四位小数一致**——greedy 首 token 即
   锚点 argmax，同一分布。白送 ECE 0.0052 与三级门控：auto 段 91.9%
   覆盖 @ 99.35% 准确率，human 段（1.7%）准确率掉到 52%
2. **字母槽零样本全灭**，三种死法：2a 被训练格式劫持（`'Safety':0.18`
   挤进 top、字母 logits 平坦）；2b 哑字母被抄（`' E':0.97`）；空槽只想
   换行填卡（`' \n':0.999`）
3. **格式锁死效应**（CIFAR 结论的对偶）：通用 VLM 零样本标准 MCQ 86.5%，
   同一格式在此模型上 19%——领域微调把"看题选字母"的通用能力洗掉了。
   反向证据：2b 的 safety 槽 top 是原生标签词 `' Unsafe':0.99`——
   **顺着微调分布读原生 token，而不是逼模型写字母**

## 三、Phase 2：链式读取（50 条初测）

把训练格式当程序逐步执行：逐步 prefill
`"Safety: <标签>\nCategories: <已识别类别>，` 读下一位置概率。

| safety | 类别 P / R / F1 | refusal | 前向 / 延迟 |
| --- | --- | --- | --- |
| 90.0% | 0.094 / 0.726 / 0.166 | 100%（18/18） | **16 次 / 484ms** |

结论：链尾条件充分时 refusal 极准，但变长输出的前向次数、停止阈值
（0.05 拦不住多判，精确率崩）都不可控，延迟反而劣于生成基线。
**变长输出是万恶之源——需要固定槽位重写。**

## 四、Phase 3 设计：Native-Slot Answer Card

把三行输出重写为固定答案卡，全部槽位留空，一次前向读全部：

```text
Safety:                      ← 槽 0：Safe / Unsafe / Controversial（原生锚）
Violent:                     ← 槽 1-13：13 类各一行，读 P(Yes)/P(No)
...（13 类，含新增 4 类）
Refusal:                     ← 槽 14：Yes / No（仅 assistant 响应样本）
```

三个结构性要点：

1. **槽间互不干扰**：槽 k 的 next-token 分布只依赖锚 k 之前的 token——
   这是"全部留空、单前向并行读"成立的结构基础
2. **第一行保留原生锚**：safety 槽零样本即有 Route 1 起跑线
3. **训练/推理条件化严格一致**：训练 prompt 同样是空槽卡，loss 落在各
   锚末 token；槽 1+ 零样本读不到（微调分布里 "Safety:" 后面跟的是
   标签+换行），RLCD 训练把"空前缀槽位"写入分布

实现坑（真实踩过）：Qwen pretokenizer 把 `:` 与后续换行合并成单 token
`":\n"`（正则 ` ?[^\s\p{L}\p{N}]+[\r\n]*` 把标点+换行切成一个 pretoken），
不带换行的锚在卡内找不到——锚必须带尾换行，且需尾部 token 对齐校验。

## 五、RLCD 训练与全量结果

训练配置：24K 样本 1 epoch（24000 步），LoRA rank 8，多槽 GRPO +
proper-reward（G=4，σ 0.2→0.05 余弦退火，CE 引导 λ=1），损失为 15 槽
平均，94 分钟（A100 单卡）。

**全量 6000 条终测（LoRA 合并后）**：

| 指标 | 零样本卡 | **RLCD 训后** | 生成基线 |
| --- | --- | --- | --- |
| safety acc / ECE | 94.35% / 2.63% | **96.20% / 0.52%** | 97.07% / — |
| 类别 P / R / F1 | 0.160 | **0.900 / 0.788 / 0.841** | EM 84.27% |
| refusal acc | 93.68% | **96.69%**（2890/2989） | 96.99% |
| binary ECE | 2.36% | **0.41%**（80989 槽） | — |
| 前向 / 延迟 | 1 / 33ms | **1 / 33ms** | 1+16 解码 / ~400ms |

- safety / refusal 距生成基线各差 **0.87 / 0.30pp**，换来 **~12× 加速**
  （400→33ms）+ 全槽概率 + 顶级校准；6000 条 0 定位错误
- 合并 LoRA 后延迟与零样本持平（33ms）——推理期适配器开销归零
- 训练签名与 CIFAR RLCD 突破 run 一致：loss 围绕 0 波动（proper-reward
  持续正奖励）、safety 槽 CE 前 1K 步收敛（2.56→0.005）、24000 步仅
  1 样本跳过
- 类别词表审计（`check_categories.py`）：代码 13 类与 train/val 标签
  词表完全一致（新增 4 类训练标签量 4087/2646/1962/1683）；
  `Categories: None`（train 10957 / val 2786，Safe 样本）在卡内等价于
  13 行全 No，无漏标

## 六、温度校准：RLCD 后无事可做

按 person_type_a 流程分桶拟合（val 前 2000 拟合 / 后 4000 holdout，
NLL 网格 [0.5, 4.0]，拟合与报告数据严格分离）：

| 桶 | 拟合温度 | holdout ECE | 施加温度后 |
| --- | --- | --- | --- |
| safety（3 候选） | **T=1.0** | 0.57%（acc 96.07%） | 无变化 |
| binary（54018 槽） | **T=1.0** | 0.37% | 无变化 |

**双桶 T=1 全胜**：proper scoring rule 的期望最优策略就是输出真实概率，
RLCD 把校准内化进了权重——jev 生态对零样本模型的标配事后补救在此
无事可做。对照链：零样本 MCQ Qwen-27B ECE 17.7%（必须校准）→
本任务零样本卡 2.36% → RLCD 后 0.41% 且 T=1。

门控质量（holdout，safety 槽）：auto@0.90 段覆盖 86.9% @ 准确率
99.45%——置信度干净地分出可自动执行段。

## 七、与 CIFAR-10 结论的互证

| 结论 | CIFAR-10（通用 VLM） | SafeGuard（格式微调模型） |
| --- | --- | --- |
| 格式 >> 训练方法 | 定制格式训练全灭（10%），标准 MCQ 零样本 86.5% | 字母格式零样本全灭（19%），原生 token 锚零样本即 97% |
| 格式的两面性 | 用预训练熟悉的格式（MCQ）白嫖能力 | 格式微调收窄分布，通用 MCQ 能力被洗掉——**格式锁死** |
| 好基线上 RLCD | MCQ + RLCD：97.0%，σ 陷阱不出现 | 答案卡 + RLCD：F1 0.841，同签名（loss 围绕 0） |
| RLCD 内化校准 | ECE 8.9% → 2.65% | binary ECE 2.36% → 0.41%，事后 T=1 |

统一叙事：**决策模型的输出格式是第一设计变量**。对通用模型，格式决定
能否零样本起跑；对微调模型，格式决定你能读什么——顺分布读原生 token，
或者花一次 RLCD 把新格式写进分布。

## 八、生产建议

| 场景 | 推荐 | 预期 |
| --- | --- | --- |
| 只需 safety 三分类 | Route 1（零训练） | 97.07% / ECE 0.52% / 34ms |
| 全链判定（safety+13 类+refusal） | 答案卡 + RLCD（已交付） | 96.20% / F1 0.841 / ECE 0.41% / 33ms |
| 精度绝对优先 | 生成基线 | +0.87pp safety，代价 12× 延迟 |
| 域漂移后 | 重跑 `calibrate_card.py` 重拟合温度 | 监控 holdout ECE |

## 九、模块与复现

| 模块 | 职责 |
| --- | --- |
| `scenario.py` / `tokensurf.py` / `mcq.py` | Phase 1 基础设施（问题配置 / token 表层 / MCQ prompt） |
| `native.py` / `run_phase1.py` | Route 1 原生锚定 + 三路线评测 driver |
| `chain.py` / `run_phase2.py` | Phase 2 链式读取 |
| `card.py` | 答案卡核心：槽定义 / 构建 / 尾部对齐定位 / 单前向读取 |
| `train_card.py` | RLCD 训练（`--pure-ce` 对照开关） |
| `eval_card.py` | 评测（`--lora` / `--calibrator` / `--model` 全可换） |
| `merge_lora.py` / `calibrate_card.py` / `check_categories.py` | 合并 / 温度拟合 / 数据词表审计 |

```bash
# 178（feat/safeguards 分支）复现全流程
V=~/Projects/SafeGuardElderlyDomain/.venv/bin/python
CUDA_VISIBLE_DEVICES=1 $V -m safeguard.train_card            # 训练 94min
CUDA_VISIBLE_DEVICES=1 $V -m safeguard.merge_lora            # 合并
CUDA_VISIBLE_DEVICES=1 $V -m safeguard.eval_card --n 6000 \
    --model safeguard/output/card_rlcd/merged \
    --out-dir safeguard/output/card_rlcd_eval_full           # 终测
CUDA_VISIBLE_DEVICES=1 $V -m safeguard.calibrate_card        # 温度校准
```

## 开发时间线

| 日期 | 里程碑 |
| --- | --- |
| 2026-09-28 | Phase 1 三路线 6000 条：Route 1 无损 97.07%/34ms；字母槽零样本全灭（格式锁死） |
| 2026-09-28 | Phase 2 链式读取：全链可读但 16 前向/484ms，精确率崩——变长输出不可控 |
| 2026-09-29 | Phase 3 答案卡：15 槽单前向 33ms；零样本类别 F1 0.160 |
| 2026-09-29 | RLCD 训练（94min）：全量 96.20% / F1 0.841 / binary ECE 0.41%，12× 加速 |
| 2026-09-29 | 温度校准 T=1 双桶全胜（RLCD 内化校准）；类别词表审计无偏移 |
