# laya 深读：encoder 路线单步决策引擎的应用全景与 RLCD 训练机制

> 研究快照：上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) @ `4066d5d`（2026-09-25；`42626c3..4066d5d` 一次 +100,540 行的大更新）。
> 姊妹文档：[jev-ecosystem-research.md](jev-ecosystem-research.md) 中 laya 小节为旧版（~1,400 行时代）研究，本文以其为准。
> 所有 file:line 引用均相对上游仓库根。

## 一、形态升级：从 1,400 行小库到 100K 行工程

| 维度 | 旧版（09-21 快照） | 新版（09-25 快照） |
| --- | --- | --- |
| 代码规模 | ~1,400 行核心 + 脚本 | 273 文件变更，+100K 行 |
| CI | 基础 | ci / docker / docs / **evals（周度权重回归门）** / release / security 六条 workflow |
| 集成 | LlamaIndex | **LangChain（LlamaIndex 已移除）** + MCP server + pydantic/JSON Schema 直入 |
| 训练 | 仅推理侧代码 | Kaggle 2×T4 完整微调 notebook + 浏览器 agent 微调配方文档 |
| act_head | "System 2 升级信号" | **标准 RL 动作头**（act_costs 配套 reward） |
| 部署 | LRU 常驻 | Docker 多阶段（cpu/cu128/cu130）、compose 三态、Jev 兼容 `/v1/systemone` API |

底层路线不变：**双向 encoder（ModernBERT-large 421M / mmBERT-base 322M）+ [MASK] 标记打分**，一次前向出全部问题的离散分布——这决定了它没有 KV Cache 概念，也决定了下文所有能力边界的来源。

## 二、应用与拓展（含 RAG 场景）

### 2.1 五个内置预设（`laya/presets.py`）

| 预设 | 输入 | 问题构成 |
| --- | --- | --- |
| triage（presets.py:5） | `{"message": ...}` | intent(choice×6)、is_urgent/refund/churn(noul)、frustration(score×4) |
| email（:45） | `{"body": ...}` | category(choice，可自定义)、is_spam/is_phishing/needs_reply(noul)、urgency(score×3) |
| guard（:82） | `{"prompt": ...}` | jailbreak/injection/sensitive_data(noul)、harm_severity(score)、topic(choice) |
| moderation（:122） | `{"post": ...}` | toxic/harassment/threat/spam(noul)、severity(score) |
| router（:154） | `{"request": ...}` | difficulty(score×4)、domain(choice×6)、needs_tools/is_sensitive(noul) |

领域工具 `laya/email.py`：`clean_email_body`（:161）剥离引用历史/签名/免责声明——新版标记从英语扩展到英/葡/西三语（:16-105）。

### 2.2 RAG 与检索场景的四种用法（重点）

**用法一：passage 相关度过滤（noul 直出）**
RAG 管线里最直接的用法——检索回来的段落逐个问"与 query 相关吗"（noul），低于阈值的丢弃再进生成。基准：RAG passage relevance laya 0.625 / multilingual 0.657（BENCHMARKS.md:133，标注 in training）。32.8ms/次的延迟意味着过滤开销可以忽略。

**用法二：shortlist——embedding 粗排 + choice 精排（`laya/shortlist.py`）**
超过 20 个选项时的官方方案（README:894）：

```text
候选全集（如 Banking77 的 77 类）
  → 调用方 embed_fn 余弦相似度粗排 top-k（默认 20，shortlist.py:26）
  → laya choice 头一次前向精排
```

- 角色定位：**cross-encoder reranker 的轻量替代**——35ms 级 vs 生成式 rerank 数百 ms
- 两个变体：MCP 工具 `laya_shortlist`（README:810-815）可直接用 checkpoint 自身 encoder 的 mean-pool 做 embed_fn，**免部署第二个嵌入模型**
- 边界要诚实：Banking77 0.425 vs Jev 0.870（BENCHMARKS.md:149）——文档自己讲清这是 `head_max_len` 选项区 token 预算的**架构性上限**（encoder 一次要吞下全部选项文本），不是能力问题；Jev（decoder + 字母槽）天然支持 255 选项
- 与分层分类同题异构：我们的 decoder 路线用"树状剪枝"解高基数，laya 用"嵌入粗排"解——一个靠注意力分层，一个靠向量空间预聚集

**用法三：model routing——LLM 管线分流**
router 预设的 difficulty/domain/needs_tools 决定请求走小模型还是 frontier 模型（README:700-701）；LangChain 集成里 `LayaRouter` 直接做 **LangGraph 条件边**，带 confidence_threshold 回退（integrations/langchain.py，README:754-764）。这是 Jev 生态里"语义路由"模式的 encoder 实现。

**用法四：守门与升级（act_head）**
`act_head = Linear(d+4, 256) → GELU → Linear(256, n_act)`（common.py:170），输入 4 维特征：top-prob、top1−top2 margin、归一化熵、k/255（common.py:210-216；单选项时 margin 补 1.0）。**新版语义变化**：从"是否升级 System 2"的单一信号变成标准 RL 动作头——`n_act = len(act_costs)+1`（common.py:263），与 proper_reward/td_lambda 配套训练（见第三节）。在 RAG/Agent 管线里它回答"这一步该快判还是该请示"。

### 2.3 集成面

- **LangChain**：LayaRouter（条件边路由）+ LayaGuardrail（护栏 + 置信度回退）
- **MCP server**（laya/mcp/）：laya_predict / laya_route / laya_shortlist / laya_preset / laya_status 五个工具
- **structured 直入**：`Agent.decide`/`Router.decide`（agent.py:1112, router.py:639）接受 JSON Schema / pydantic
- **Jev 兼容 API**：`[serve]` extra 暴露 `/v1/systemone` + `/health`——直接替换 Jev 客户端的部署位
- 注意：**旧版研究提到的 LlamaIndex 集成已从代码中移除**（integrations/ 仅剩 langchain.py）

### 2.4 多语言路由与部署工程

- 三 checkpoint 同仓（english 421M/512tok、multilingual 322M/1024tok/100+ 语、typed-decisions 421M/1024tok），`subfolder` 按需下载
- 路由优先级链（router.py:470-548）：explicit model > task > workflow 匹配 > lang > lang_guess > 脚本检测 > default；路由判据是"英文 checkpoint 能否读"而非语言 id（:145）
- 新增 `route_batch`/`predict_batch`（:660,702）**异构批量**：先路由、按 checkpoint 分组、按 question schema 共享前向——多语言混流下的吞吐关键
- LRU 常驻（max_loaded=2，逐出时 gc + cuda empty_cache）；tokenizer 进程级缓存 keyed on (dir, mtime)
- `tests/test_tokenizer_concurrency.py` 守护一个真实坑：共享 Rust tokenizer 的 `enable_truncation` 突变在多线程触发 "Already Borrowed"，`encode_text` 串行化解决
- `tests/test_truncation_direction.py` 同样值得记：**截头还是截尾取决于 state 语义**——文档类 state 截尾保头部（指令在前），时间序对话 state 截头保尾部（最新意图在后）；`Agent.system_one` 对 list state 自动传 `truncate_left=True`（common.py:143-144）

### 2.5 基准数字速览（BENCHMARKS.md）

| 维度 | laya | Jev 1.13.0 | 备注 |
| --- | --- | --- | --- |
| typed-decisions | **0.766** | 0.727 | 软分布匹配 Jev 反超（0.580 vs 0.471） |
| AG News / Emotion | **0.953** / **0.600** | 0.910 / 0.480 | |
| ECE | 0.081 | 0.246 | laya 为温度后拟合值；raw 0.213 劣于 Jev 0.144 |
| 延迟（T4 单问） | **32.8 ms** | 236-276 ms | 快 6~7× |
| Banking77（77 选项） | 0.425 | **0.870** | encoder 选项区预算的架构性上限 |
| moderation | 0.530（macro-F1 0.400） | — | 诚实承认失败 |
| 基座无微调 | 0.362（低于多数类基线 0.461） | — | 能力全靠微调 |

另：周度 evals workflow 对英文 checkpoint 跑 MASSIVE 并对比 committed baseline（.github/workflows/evals.yml）——**校准漂移的 CI 门禁**，与我们生态研究里 jevcal 的思路同源；TileLang fast path 在 1 问场景再提 3.8~5.1×。

## 三、矫正训练：RLCD 在 laya 中的真实形态

README:1071 自述方法为 "RLCD (proper-scoring-rule rewards, GRPO-style policy gradient)"。训练代码在 `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`（916 行）+ `docs/finetune_browser_agent.md`。

### 3.1 proper_reward：奖励函数逐项拆解（common.py:278-304）

对预测分布 q ∈ Δ^K 与目标 y（one-hot 或软分布）：

```
r(q, y) = log_score(q, y) + w_sph · spherical(q, y) − w_rps · RPS(q, y) · 1[score 类型]
```

- **log score**（:293-294）：Σᵢ yᵢ·log qᵢ，q 截到 1e-12、log 夹到 −9.21 防 −inf（test_training.py:111 验证有限性）——即对数似然
- **spherical score**（:295）：(Σᵢ yᵢqᵢ)/‖q‖₂——对向量尺度不变，只奖励"方向对"，天然抗过度自信的锐化
- **RPS（Ranked Probability Score）**（:297-303）：Σᵢ (CDF_qᵢ − CDF_yᵢ)²·maskᵢ/(k−1)，仅序数 score 类型启用——利用等级的序关系，是序数输出上的严格 proper score
- **properness 论证**：log 与 spherical 均为严格 proper scoring rule（E_y[r] 在 q=y 处唯一最大），非负权重组合与负 RPS 保持严格 proper；mask 保证无效选项不进奖励（test_training.py:75-97 用数值网格验证 choice/score 两类 qtype 的 argmax 性质）

**这就是 "RLCD" 口号的落地方式**：用严格 proper score 做 reward，最优策略 = 输出真实概率分布——**校准成为目标函数的性质，而不是后处理**。与 Jev 官方 RLCD（未公开）的对比叙事即在于此；同时 README:894-896 诚实承认 raw ECE 0.213 仍劣于 Jev 0.144，0.081 是后拟合温度的结果。

### 3.2 GRPO 式训练管线（notebook）

- **数据**：`LocalLLaMA/typed-decisions`（四工作流 ~2,000 决策），gold 的 `probabilities` 归一化为**软目标**（教师分布蒸馏，非硬标签）
- **GRPO 机制**：① 采样 G=4 组零均值高斯噪声 ε·σ（σ 从 0.4 余弦退火到 0.1，投影到零和保证选项对称），z = logits + ε；② no_grad 下 proper_reward（w_sph=0.75, w_rps=1.0）算 reward，**组内标准化为 advantage**；③ loss = 高斯策略梯度（−Σ(z−logits)²·mask/2σ²，advantage 加权）+ 1.0× 软交叉熵引导项
- **超参**：4 epochs，有效 batch 64（2×T4 DDP + fp16 + GradScaler + 梯度裁剪 1.0），encoder lr 2.5e-5 / head 1e-4 分组，CosineAnnealing，双梯度检查点
- **校准隔离**：训练前固定种子抽 min(400, 10%) 作 calib 集，严格不进训练

### 3.3 td_lambda_targets：多轮轨迹的信用分配（common.py:307-322）

按 `ep_group`/`ep_step` 分组，从终局真值 y_T 反向 bootstrap：`G ← (1−λ)·p_next + λ·G`，逐步置 target 为 (1−G, G)。λ=1 退化为 Monte-Carlo（终局真值广播），λ=0 为一步 TD（用下一步模型概率）——Agent 多步决策里"这一步的选择对终局负责"的标准折中。

### 3.4 校准体系

- **桶结构**：`temp_bucket = "{qtype}:{2|3-5|6-10|11+}"`（common.py:367-369）——类型×选项数分桶，与我们 decoder 路线及 rlcd-modernbert 的 per-K 温度**三方独立收敛**
- **推理端桶级优先**：先查 `temperature_by_options[bucket]`，回退按类型（agent.py:768-771）；notebook 导出时特意 pop 旧桶值防遮蔽
- **拟合**：正式实现逐类型一个温度（notebook `fit_one_temp`，LBFGS on log_t，clamp [0.1,10]）；research 脚本另有逐桶网格版
- **TEMP_MIN=0.5 clamp**（common.py:372-388）：出厂 choice:11+ 桶 0.1006 的过锐温度会把 0.24 概率发布成 0.99——拒绝执行过锐温度，宁可欠调不过调
- **两种置信度二分**：`answer_confidence`（max p，温度作用对象）vs `confidence_from_probs`（1−H/log k，未校准）——前者做运营阈值，后者只做排序
- **出厂未校准的现状没变**：README:634 明说两 checkpoint 出厂过度自信、multilingual 完全没拟合温度
- **一个诚实的坑**（README:1090 自认）：notebook 的校准集从训练项里抽——温度在"见过分布"上拟合，**需要 held-out 复评**（与我们校准纪律一节的原则一致）

### 3.5 数据策展：浏览器 agent 微调配方（docs/finetune_browser_agent.md)

- 数据 = 爬取 421 页 + Qwen3-8B 反向生成目标 5,244 + 真实 DONE 落地页 700 + 中途负例 659 + Mind2Web 7,296 + DAgger 在线纠正 177
- 关键发现：**把候选元素放进 option 列表而非塞进 state**，top-1 从 0.44 → 0.51——与我们"判据放共享区、选项独立槽"的布局纪律同源
- 对齐 jev-ultrafast 的 `/v1/systemone` 格式，与 decoder 生态互通

## 三点五、决策生成机制：与 decoder 三引擎的根本差异

### laya 怎么"生成"决策：[MASK] 标记打分

laya 没有生成循环，也没有字母槽——它的决策读取发生在**序列内部的标记位置**：

```text
[CLS] <问题类型> instructions [SEP] [MASK] 选项0 [MASK] 选项1 ... [SEP] state [SEP]
                              ↑marker    ↑marker
一次 encoder 前向（双向注意力）
  → 每个选项前的 [MASK] 位置隐向量 h_i
  → scorer：2 层 Transformer head + MLP，把 h_i 映射为该选项的 logit
  → 全部选项 logit 一次 Softmax = 该问题的分布
```

三原语的读取方式（`laya/common.py`）：
- **choice**：argmax + 全分布（marker logit 直接归一）
- **score**：期望分 Σᵢ i·pᵢ（等级值加权），保留犹豫信息
- **noul**：固定 [false] [true] 两个 marker，P(true) 即答案

关键机制差异：**选项文本本身进入序列、通过 [MASK] 槽与全局上下文双向交互**。decoder 路线里"候选"最终坍缩成一个字母 token 的 logits；encoder 路线里候选始终保持为完整语义实体，由 marker 位置聚合表征。这带来两个结构性后果：
1. 选项容量受 token 预算限制（`head_max_len`，约 20 个选项）——选项文本必须全部塞进序列；
2. 选项间不存在"首 token 碰撞"与"顺序稀释"问题（无字母、双向注意力把所有选项置于对称位置），但代价是全注意力的选项间竞争依然存在（与 decoder 的干扰形式不同源）。

### 与本项目三引擎的机制对照

| 维度 | laya（encoder+marker） | transformers（锚位直读） | vllm-perq（生成位 top-K） | vllm-plogprob（哑字母+prompt_logprobs） |
| --- | --- | --- | --- | --- |
| 底座 | 双向 encoder（421M，纯文本） | decoder VLM（27B 级） | decoder VLM | decoder VLM |
| 候选编码 | 选项文本进序列，[MASK] 槽 | 字母槽（选项文本在 prompt） | 字母槽，每问独立 | 字母槽 + 候选集外哑字母 |
| 读取位置 | marker 位隐向量 → MLP logit | 锚位全词表 Logits 直读 | 生成位（assistant 头后）top-K | 哑字母位 prompt_logprobs top-K |
| 注意力方向 | **双向**（选项对称、全文互见） | 单向因果 | 单向因果，问题隔离 | 单向因果 |
| 槽位分布条件于 | 全部选项文本 + state | 各问题文本，无答案 | 仅本问文本 | 问题文本 + 哑字母常量 |
| 选项容量 | ~20（token 预算） | 25+弃权 | 26 | 26 |
| 视觉输入 | ❌（无视觉塔） | ✅ | ✅ | ✅ |
| 训练需求 | **必须微调**（基座 0.362 低于多数类基线） | 零训练 | 零训练 | 零训练 |
| 稳态延迟 | 32.8 ms（T4） | 271 ms（A100） | 239 ms（A100） | 182 ms（A100） |

三个值得记的对照结论：

1. **速度来源不同**：laya 快是因为模型小 65 倍（421M vs 27B）且无模板开销；我们 plogprob 的 182ms 是在 27B 上做到的——零解码在这个量级依然成立，但小模型上 encoder 路线的绝对延迟不可追赶。
2. **独立性语义三方不同**：perq 完全隔离（问题互不可见）、plogprob 条件于哑字母常量、laya 条件于全部选项的双向上下文——三者给出三种不同的"问题间条件分布"，校准口径都不可互换（再次印证同引擎同布局校准原则）。
3. **训练-免训练的分界**：laya 必须微调才有能力（基座低于多数类基线），我们的字母槽零训练即可用——这是 decoder 预训练分布（instruction-following + 选项作答）带来的免费午餐；反过来 laya 微调后 ECE 可达 0.081，是我们路线 C 的标杆。

## 四、对 decoder 路线（jev-vlm-decisions）的启示

1. **三条趋同演化**：per-K 桶温度（laya / rlcd-modernbert / 我们）、高基数的分层 vs 粗排两种解法、act_head 与我们三级门控的"升级信号"——不同底座独立收敛到同一组设计，说明这些是问题的本质结构而非风格
2. **encoder/decoder 分工边界清晰**：encoder 赢延迟（32.8ms）与多语言路由，decoder 赢选项容量（255 vs ~20）、零训练适配与视觉任务；RAG 短名单场景 laya 的"自嵌入粗排"值得 decoder 路线借鉴（视觉场景可换成检测器置信度粗排）
3. **proper-reward 训练把校准变成目标函数性质**——这是我们路线 C（LoRA）可借用的损失设计：字母槽交叉熵之外叠加 spherical/RPS 项
4. **周度 evals 门禁**（校准漂移的 CI 防护）在 laya 已是生产实践，验证了我们设计文档里 jevcal 思路的可行性
