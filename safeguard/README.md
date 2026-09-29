# SafeGuard jev 化：格式微调模型的单步判定改造

把 SafeGuardElderlyDomain 微调出的 Qwen3Guard-Gen-0.6B（HK 养老领域 13 类
安全判别，自回归生成三行文本）适配成 jev 范式判定模型：0 解码步、单次
prefill、固定锚点读 next-token logits、掩码 Softmax 出带置信度的离散判定。

本 demo 复用 `person_type_a` 的字母槽基础设施（readout/calibrator），
但研究对象相反：person_type_a 是通用 VLM 零样本做字母槽，这里是一个
**被输出格式深度微调过的小模型**——格式先验既是资产也是枷锁。

**完整阶段报告**（Phase 1 三路线 → Phase 2 链式 → Phase 3 答案卡 RLCD →
温度校准，全量 6000 条终测）：[`docs/safeguard-experiment-report.md`](../docs/safeguard-experiment-report.md)

## 任务到原语的映射

| 生成式输出 | jev 原语 | 槽位 |
| --- | --- | --- |
| Safety: Safe/Unsafe/Controversial | choice（字典序 + 弃权槽） | 1 |
| Categories: 13 类多标签 | 每类一个 noul（no/unclear/yes） | 13 |
| Refusal: Yes/No（仅 assistant 响应） | noul | 1 |

user_query 14 问，assistant_response 15 问。

## Phase 1：零样本三路线（`run_phase1.py`）

| 路线 | 形态 | 前向次数 | 零样本先验 |
| --- | --- | --- | --- |
| (a) 生成基线 | 自回归解码三行 | 1 prefill + N decode | 已有结果（safety acc 97.07%） |
| (b) Route 1 原生锚定 | guard 模板 + 强制 `Safety:` 前缀，读标签首 token | 1 | 复用微调分布，零格式偏移 |
| (c) Route 2a 每问 MCQ | 标准 MCQ（`A. option` + `Answer:` 锚在末尾） | 14-15（批量） | person_type_a 实证的零样本有效形态 |
| (d) Route 2b 答题卡 | 多问合一，assistant 骨架 + 槽位表，offset 定位读槽 | 1 | jev 终极形态；`--2b-fill empty` 为空槽消融 |

Route 1 只覆盖 safety 三分类：Categories 是变长逗号列表，单一列表位置读
不出 13 个独立概率；Refusal 在第三行，条件于前两行内容。结构性限制。

## Phase 1 结果（6000 条 val，详见 output/phase1/report.md）

| 系统 | 前向/样本 | safety acc | 类别 EM | refusal acc | ECE | ms/样本 |
| --- | --- | --- | --- | --- | --- | --- |
| 自回归生成 | 1 + ~16 解码 | 97.07% | 84.27% | 96.99% | — | ~400 |
| Route 1 原生锚定 | **1（0 解码）** | **97.07%（无损）** | 结构性不可覆盖 | 同左 | 0.52% | **34.4** |
| Route 2a 每问 MCQ | 15 | 19.05% | 46.17% | 65.57% | 59% | 159.6 |
| Route 2b 答题卡 | 1 | 40%（20 条 debug） | 0 | 0% | — | ~38 |

- **Route 1 与生成基线 acc/macro-F1 四位小数一致**（greedy 首 token 即锚点
  argmax，同一分布），白送 ECE 0.0052 和三级门控：auto 段 91.9% 覆盖 @
  99.35% 准确率，human 段（1.7%）准确率掉到 52%——置信度干净地分出了
  该转人工的样本
- **字母槽零样本全灭**，三种死法：2a 被训练格式劫持（`'Safety':0.18` 挤进
  top，字母 logits 平坦到 A=C 同分）；2b 哑字母被抄（`' E':0.97`）；空槽
  只想换行填卡（`' \n':0.999`）。与 person_type_a 的 CIFAR-10（27B 通用
  VLM 标准 MCQ 零样本 86.5%）对照：**格式微调收窄了输出分布，通用模型
  的看题选字母能力被领域微调洗掉**
- 反向证据：2b 的 safety 槽 top 是原生标签词 `' Unsafe':0.99`——顺着微调
  分布读原生 token 比逼模型写字母更接近可行

## 运行

```bash
# 需 torch/transformers/sklearn：用 SafeGuardElderlyDomain 的 venv（与训练环境一致）
cd <repo-root>
CUDA_VISIBLE_DEVICES=1 ~/Projects/SafeGuardElderlyDomain/.venv/bin/python \
    -m safeguard.run_phase1 \
    --out-dir safeguard/output/phase1 --gen-latency-n 150

# 空槽消融
CUDA_VISIBLE_DEVICES=1 ~/Projects/SafeGuardElderlyDomain/.venv/bin/python \
    -m safeguard.run_phase1 --routes 2b --2b-fill empty \
    --out-dir safeguard/output/phase1_2b_empty --skip-gen-latency
```

默认资产（模型 / val 集 / 生成基线）路径写死指向 SafeGuardElderlyDomain
仓库，均可用参数覆盖。启动自检：字母单 token、标签可解析、续写无合并、
2b 锚位递增，任一失败立即退出。

## 模块

| 模块 | 职责 |
| --- | --- |
| `scenario.py` | 13 类别清单、问题配置（choice/noul）、字典序 + 弃权槽 |
| `tokensurf.py` | 候选 token 表层变体（裸/空格字母）与 LSE 合并、续写自检 |
| `mcq.py` | Route 2 prompt（2a 每问独立 / 2b 答题卡）、offset 锚定位 |
| `native.py` | Route 1：guard 模板 + `Safety:` 锚 |
| `run_phase1.py` | driver：三路线评测 + 基线对齐 + 报告 |
| `chain.py` / `run_phase2.py` | Phase 2 链式读取（逐步 prefill 读全链） |
| `card.py` | Phase 3 答案卡：15 槽定义/构建/解析/尾部对齐定位/单前向读取 |
| `train_card.py` | Phase 3 RLCD 训练（多槽 GRPO + proper-reward，`--pure-ce` 对照） |
| `eval_card.py` | Phase 3 评测（`--lora` 可选；全槽指标 + 双 ECE + 延迟） |
| `merge_lora.py` | 合并 LoRA 到基座，消除推理期适配器开销 |
| `calibrate_card.py` | 分桶温度拟合 + holdout ECE 前后对比（RLCD 后 T=1） |

复用 `person_type_a`：`readout.masked_softmax`、三级门控阈值、`calibrator.ece`。

## Phase 2：链式读取（`chain.py` / `run_phase2.py`）

把模型的训练格式当程序逐步执行：共享对话前缀，逐步 prefill
"Safety: <标签>\nCategories: <已识别类别>，" 读下一位置概率。
免训练，但前向次数随类别数增长。

50 样本实测（`output/phase2_test/summary.json`）：

| 指标 | 值 | 备注 |
| --- | --- | --- |
| safety acc | 90.0% | 链条件化反而略降 |
| 类别 P / R / F1 | 0.094 / 0.726 / 0.166 | 高召回低精度：0.05 停止阈值拦不住多判 |
| refusal acc | 100%（18/18） | 链尾条件充分时极准 |
| 前向 / 延迟 | **16 次 / 484 ms** | 比生成基线还慢 |

结论：链式读取验证了"逐步执行训练格式"可读全链，但变长输出的
前向次数与精度都不可控——需要 Phase 3 的固定槽位重写。

## Phase 3：Native-Slot Answer Card（`card.py` / `train_card.py` / `eval_card.py`）

把变长三行输出重写为固定答案卡，全部槽位留空，**一次前向读全部**：

```text
Safety:                      ← 槽 0：Safe / Unsafe / Controversial（原生锚）
Violent:                     ← 槽 1-13：13 类各一行，读 P(Yes)/P(No)
...（13 类字典序）
Refusal:                     ← 槽 14：Yes / No（仅 assistant 响应样本）
```

设计要点：

- **槽间互不干扰**：槽 k 的 next-token 分布只依赖锚 k 之前的 token，
  所以空槽可以并行读——这是单前向成立的结构基础
- **第一行保留 Route 1 原生锚**：safety 槽零样本即有起跑线
- **槽 1+ 零样本读不到**：模型期望 "Safety: <标签>\nCategories:"，
  空 "Violent:" 行在微调分布之外——RLCD 训练把空前缀条件化写入分布
- **训练/推理条件化完全一致**：训练 prompt 同样是空槽卡，loss 落在各
  锚末 token（Qwen 把 ":+\n" 合并成单 token，锚需带尾换行）
- **多槽 RLCD**：每槽独立 proper reward（GRPO 组内归约 + CE 引导），
  损失为 15 槽平均；safety 3 候选 / 类别与 refusal 2 候选

零样本基线（200 样本，`output/card_zeroshot/summary.json`）：

| 指标 | 零样本卡 | 说明 |
| --- | --- | --- |
| safety acc / ECE | 95.0% / 4.2% | 略低于 Route 1（97.07%）："Safety:\n" 条件化不同 |
| 类别 F1 | 0.152 | 槽 1+ 在微调分布外，预期弱 |
| refusal acc | 94.7% | 意外地好 |
| **前向 / 延迟** | **1 次 / 36 ms** | 全链覆盖，速度天花板 |

目标（RLCD 训后）：safety ≥97%、类别 F1 >80%、refusal >95%、
双 ECE <3%，保持 1 前向 ~36ms。

### 训后结果（24K 样本 1 epoch，LoRA rank 8，94 分钟）

> **勘误（2026-09-30）**：本次训练名义为 GRPO+proper-reward，因
> `grpo_trainer` 的 detach 缺陷实际由**多槽 CE** 驱动；修复后真 RLCD
> 重训全面劣于此结果（ECE 0.52%→3.75%）。数字有效，归因见
> `docs/rlcd-ablation-report.md`。

**全量 val（6000 条，LoRA 已合并，`output/card_rlcd_eval_full/`）：**

| 指标 | 零样本卡 | **RLCD 合并** | 生成基线（Phase 1） |
| --- | --- | --- | --- |
| safety acc / ECE | 94.35% / 2.63% | **96.20% / 0.52%** | 97.07% / — |
| 类别 P / R / F1 | F1 0.160 | **0.900 / 0.788 / 0.841** | EM 84.27% |
| refusal acc | 93.68% | **96.69%**（2890/2989） | 96.99% |
| binary ECE（80989 槽） | 2.36% | **0.41%** | — |
| 前向 / 延迟 | 1 / 33ms | **1 / 33ms** | 1+16 解码 / ~400ms |

- 全槽单前向 + 合并 LoRA 后延迟与零样本持平（33ms），对比生成基线
  **~12× 加速**，6000 条 0 定位错误
- safety/refusal 距生成基线各差 0.87/0.30pp，换来全槽概率分布 +
  顶级校准（safety ECE 0.52%、二值槽 ECE 0.41%）——可直接上三级门控
- 200 条子集初测（`output/card_rlcd_eval/`，未合并 LoRA）：safety
  96.5% / 类别 F1 0.853 / refusal 96.8% / 60ms，与全量一致
- 训练签名与 CIFAR RLCD 突破 run 一致：loss 围绕 0 波动、safety 槽 CE
  前 1K 步即收敛（2.56→0.005）、σ 0.2→0.05 正常退火、24000 步仅 1 跳过

### 温度校准：RLCD 后无事可做（`calibrate_card.py`）

按 person_type_a 流程拟合分桶温度（val 前 2000 拟合 / 后 4000 holdout，
NLL 网格 [0.5, 4.0]），结果 **safety|3 与 binary|2 双桶 T=1.0 全胜**：

| 桶 | holdout ECE（T=1） | 施加温度后 |
| --- | --- | --- |
| safety（3 候选） | 0.57%（acc 96.07%） | 无变化 |
| binary（54018 槽） | 0.37% | 无变化 |

解读：**勘误后归因——CE + 任务本身已校准**（one-hot 标签下
proper-reward 与 CE 共享最优解；本次训练实际由 CE 驱动），事后温度
校准（jev 生态对零样本模型的标配补救）在此无事可做。对照：零样本
MCQ 的 Qwen-27B ECE 17.7% 必须校准；本任务零样本卡 binary ECE
2.36% → 训后 0.41%。

**门控质量**（holdout，safety 槽）：auto@0.90 段覆盖 86.9% 样本、
其中准确率 99.45%——置信度干净地分出了可自动执行段。

部署接口：`calibrator.json` 已生成；`eval_card --calibrator <path>`
按桶施加温度（argmax 不变，仅重塑置信度），供域漂移后重拟合用。
