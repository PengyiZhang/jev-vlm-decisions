# RLCD 消融报告：GRPO + proper-reward 相对 CE 的四臂对照

> 实验周期：2026-09-29 ~ 2026-09-30
> 环境：server178（A100-80G）
> 涉及模型：Qwen3Guard-Gen-Domain-0.6B、gemma-4-E4B-it、convaiinnovations/laya（421M）
> 本文档是"RLCD 训练增益"这一历史宣称的权威勘误与最终裁决。

## 一、一句话结论

**在标签监督的闭集判定任务上，GRPO + proper scoring reward 相对（软）交叉熵
无可测增益；one-hot 标签下还明显劣化校准。此前所有标注为 "RLCD 训练" 的
模型实际由 CE 训练（见二），其成果数字全部有效、归因需要改写。**

## 二、缺陷发现：策略梯度项从未生效

`grpo_trainer.py` 移植 laya 时的一个 `.detach()` 缺失：

```python
z = mu + eps                      # 我方（缺陷版）：z 挂着 mu 的计算图
log_pi = -((z - mu) ** 2) / 2σ²  # z − mu 全导数恒为 0 → PG 项梯度恒为 0

z = logits.detach() + eps         # laya 原版：采样点为常数，梯度走减数侧
```

数值验证：缺陷版 `GRPOLoss` 对 μ 的梯度与纯 CE **逐位相同**。因此此前
所有 "GRPO/RLCD" 训练（CIFAR 定制格式 4 组、CIFAR 标准 MCQ、safeguard
答案卡）的权重更新全部来自 CE 项；σ 退火仅影响 loss 数值与随机流
（dropout 掩码），不影响梯度方向。

修复：`z = mu.detach() + eps`（对齐 laya），并新增两个回归测试
（λ_ce=0 时梯度非零；总梯度 ≠ 纯 CE 梯度）。

## 三、四臂对照实验（同种子 42、同超参、唯一变量是目标函数）

### 实验 1：safeguard 答案卡（one-hot 标签，6000 条全量评测）

| 指标 | **多槽 CE**（= 现交付模型） | 真 GRPO+proper（修复后） |
| --- | --- | --- |
| safety acc / ECE | **96.20% / 0.52%** | 96.15% / 3.75% |
| 类别 P / R / F1 | **0.900** / 0.788 / **0.841** | 0.849 / 0.788 / 0.817 |
| refusal acc | **96.69%** | 96.45% |
| binary ECE（80989 槽） | **0.41%** | 3.12% |

召回完全相同，精确率与校准全面下降——活过来的策略梯度把 CE 已收敛的
解抖散（σ→0.05 时 1/σ²=400 的梯度噪声放大）。

### 实验 2：CIFAR 标准 MCQ（one-hot 标签，200 条）

| 指标 | 纯 CE | 真 GRPO+proper（修复后） |
| --- | --- | --- |
| accuracy | 96.5% | 96.5%（打平） |
| ECE | **2.57%** | 3.49% |

### 实验 3：laya typed-decisions（**软标签**，1200 决策，T=1）

在 laya 自己的模型（convaiinnovations/laya）+ 数据（LocalLLaMA/
typed-decisions，6000 训练 items）+ 超参（4 epoch，单卡等效 batch 64）
上复刻其训练脚本：

| 指标 | **纯软 CE** | GRPO+proper（laya 原版损失） | 直接可微 proper（无 GRPO） |
| --- | --- | --- | --- |
| hard acc | **80.50%** | 79.08% | 79.92% |
| ECE | 11.82% | **11.18%** | 12.04% |
| Brier | **0.0501** | 0.0543 | 0.0516 |
| TV | **0.1294** | 0.1336 | 0.1304 |
| KL | **0.0840** | 0.0916 | 0.0847 |

注：laya 官方 notebook 本身没有 pure-CE 对照臂。

## 四、理论解释：为什么必然如此

1. **恒等式**：软标签交叉熵 = log score 的期望形式——soft CE 本身就是
   严格 proper scoring rule，最优解 q = 标签分布，与 proper-reward 的
   期望最优**重合**。one-hot 下逐样本 log score 最优 = MLE = CE 最优。
   两种目标不存在"第二个最优点"。
2. **估计器**：GRPO 的 `E[eps·(r−r̄)]/σ²` 是对 `∇E[r]` 的进化策略式
   高方差有限差分估计；而 reward 对 logits 本可**直接精确求导**（实验 3
   第三臂）。GRPO 版 = 同方向 + 纯加方差。
3. **我们的任务太干净**：训练即收敛、泛化 gap≈0，CE 免费获得了渐近
   校准性质。这不构成"知道自己知道不重要"的反证（见六）。

## 五、历史结论勘误表

| 原表述 | 修正后 |
| --- | --- |
| "RLCD 训练成果：97.0% / ECE 2.65%"（CIFAR MCQ） | 数字有效；实际由标准 MCQ 格式 + CE 训练达成（GRPO 项未生效） |
| "RLCD 训后 96.20% / F1 0.841 / ECE 0.41%"（safeguard 卡） | 数字有效；实际为多槽 CE 训练。真 RLCD 重训：ECE 0.41%→3.12%（更差），**现交付模型即最优** |
| "RLCD 内化校准（T=1 双桶全胜）" | T=1 结果有效；归因改为 **CE + 任务本身已校准**（gold 是 one-hot 单次实现时 proper-reward 与 CE 同最优） |
| "纯 CE 对照 96.5% vs RLCD 97.5%，proper-reward +1.0pp 精度" | 两者实为同一目标 + 不同随机流；1pp 为 RNG 噪声。真 A/B：精度打平、CE 校准更优 |
| "σ 退火陷阱：后期策略梯度压过 CE 拉向均匀" | 该机制在缺陷实现中不可能发生（PG 无梯度）；loss 回升另有原因（数据序/dropout） |

**保持有效不受影响**：格式 >> 训练方法（零样本事实）、Route 1 无损、
答案卡架构与全部推理/校准/门控机制、12× 加速、所有模型精度/延迟数字。

## 六、RLCD 的重新设计空间（另开工作）

proper-reward 重新获得存在权的判据：**reward 与 CE 不共享最优解**。
四个方向：

1. 参照系换成结局（环境反馈/延迟真值/业务代价，不可微 → PG 才有必要）
2. 目标换成选择性行为效用（risk-coverage 覆盖×精度−成本，分段不可微，
   弃权槽语义升级为"行为"）
3. 认知不确定性 = 扰动一致性 reward（选项重排/改写下稳定且对才奖励置信）
4. 难例自挑战（confident-wrong 挖掘 + 非对称 proper penalty）

检验方法沿用本报告：任何新 reward 设计先过"对照 CE 看
Brier/ECE/coverage"这一关。

## 七、复现

```bash
# 178（understanding-jev 仓库 + SafeGuardElderlyDomain venv）
# 实验 1/2：修复后 GRPO vs CE
CUDA_VISIBLE_DEVICES=1 python -m safeguard.train_card --output-dir ...   # 真 RLCD
CUDA_VISIBLE_DEVICES=5 python -m safeguard.train_card --pure-ce ...
CUDA_VISIBLE_DEVICES=4 python -m person_type_a.train_mcq --model <gemma> ...

# 实验 3：~/Projects/laya_ab/（prep_laya_data.py / train_laya_ab.py / eval_laya_ab.py）
# 三臂：--mode rlcd | pure-ce | direct-proper
```

梯度回归测试：`person_type_a/tests/test_grpo_trainer.py`（6 用例）。
