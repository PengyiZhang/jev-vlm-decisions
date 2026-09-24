# repos 生态实测：Jev 复现与生态仓库的实地研究

> 研究基线：`docs/jev-starter.md` 生态地图与参考资料的仓名清单。
> 第一批克隆：2026-09-21（5 个 HuggingFace 仓库，LFS 权重跳过）。
> 第二批克隆：2026-09-21（用户补充的 13 个仓库，12 个 GitHub + 1 个 HF Space）。
> 全部位于 `repos/`，共 18 个仓库。

## 探测记录（含一次自我修正）

第一轮探测（9-21 上午）结论是"GitHub 18 仓全部不存在、HF 经代理确认 5 仓"。**当晚这个结论就过时了**：用户补充克隆的 13 个仓库里，12 个直接来自 GitHub（`AnotiaWang/awesome-jev`、`rorshopping/jev-on-a-laptop`、`anisselbd/jev-phishing-bench`、`browser-use/jev-ultrafast`、`droidrun/mobile-jev`、`DevMortimer/pi-warden`、`jomatsu/pi-jev-auto-mode`、`jaredpalmer/kev`、`NandhaKishorM/laya`、`TianyuCodings/NanoJev`、`bespokelabsai/nimble`、`TheoLeeCJ/SemIf`），1 个来自 HF Space（`multimodalart/jev-reproductions-tracker`）。`awesome-jev` 自述公开访问开启于 9-21 当天。这个生态以天为单位演化，存在性结论必须带时间戳，探测未命中不等于不存在。

网络环境：huggingface.co 直连不通，git 需走代理；GitHub 直连正常。

| repos/ 目录 | 来源 | 实际定位 | 校准方案 |
| --- | --- | --- | --- |
| `openjev` | HF AlexWortega/openjev | NLI 交叉编码器 + 训练评测代码 | 无 |
| `Qwen-2.5-1B-RLCD` | HF harshatheg | PCD 推理应用（Gradio + FastAPI） | 无（碰撞置信度系合成） |
| `rlcd-modernbert-151m` | HF heman10x | GLiClass 零样本分类器微调 | 全局 + 按候选数分桶温度 |
| `decider-2b` | HF Mapika | Qwen3.5-2B 全流程复现 | 单温度 1.3 + RL 阶段 |
| `system-one-mini` | HF DavidHatley | DistilBERT + 5 个固定决策头 | 逐头温度（两头撞上界） |
| `kev` | GitHub jaredpalmer | 指针头架构复现 + 训练 + 兼容服务 | 分组温度实验（T=2.0） |
| `laya` | GitHub NandhaKishorM | encoder-only 双向注意力路线 | 按类型×选项数桶的温度 |
| `NanoJev` | GitHub TianyuCodings | 0.6B 游戏决策复刻（集合注意力头） | 分区拟合但默认 T=1 |
| `SemIf` | GitHub TheoLeeCJ | 浏览器端单步判定（原 openjev.com） | 显式声明未校准 |
| `nimble` | GitHub bespokelabsai | 首个全开放 recipe + 13 子集人标基准 | 无（T=1，校准是自认短板） |
| `jev-on-a-laptop` | GitHub rorshopping | 73.8% 一致性评测与自我审计 | 无（只有测量与计划） |
| `jev-phishing-bench` | GitHub anisselbd | 2000 封钓鱼邮件双模型基准 | 不适用（评测仓） |
| `pi-warden` | GitHub DevMortimer | Pi 运行时守卫（正则地板 + Jev） | 每问题独立阈值（实测校准） |
| `pi-jev-auto-mode` | GitHub jomatsu | Pi 语义审批扩展（fail-closed） | 每规则双侧阈值带 |
| `jev-ultrafast` | GitHub browser-use | 浏览器 Agent（一次请求多问题协议） | 不适用（消费方） |
| `mobile-jev` | GitHub droidrun | Android Agent（云 API + 结构化状态） | 不适用（消费方） |
| `jev-reproductions-tracker` | HF Space multimodalart | 生态追踪静态页（热度指标） | 不适用 |
| `awesome-jev` | GitHub AnotiaWang | 生态索引（100+ 条目，双语） | 不适用 |

## 第一批：五个 HF 复现仓

### openjev：NLI 路线的完整实现

Qwen3.5（0.8B / 4B / 35B-A3B MoE）底座，`Premise/Hypothesis` 模板，`Qwen3_5ForSequenceClassification` 三分类头（0=矛盾、1=蕴含、2=中立），取蕴含概率 argmax 决策。与文档描述一致。

三个文档没写的发现：

- **共享前缀的实现细节**（`modeling_openjev.py:89-140`，`test_shared_prefix.py` 验证）：同一 premise 配多 hypothesis 时，先算 token 公共前缀、一次 prefill，再 `cache.reorder_cache` 把缓存物理复制到 batch 各分支。不用 4D tree mask 的原因是 **Qwen3.5 混有线性注意力（recurrent）层，其状态无法用注意力 mask 隔离分支**，只能复制缓存。
- **第二条打分路径**：`mlp_heads_35b/` 下 9 个任务目录（mmlu、gsm8k、chess 等），冻结 35B 骨干 + 每任务 d→512→1 的 MLP 头（`LatentMLPHead`）。README 一边喊"零任务训练"一边并列这条 per-task 路径。
- v2 checkpoint 带视觉塔，可直接从像素玩 Doom；Minecraft 场景实际用 backward-chaining 脚手架，jev 只负责校验陈述，并非全程单步。

### Qwen-2.5-1B-RLCD：名字叫 RLCD，实为 PCD

仓库实为 **Parallel Constrained Decoding** 项目，代码里 `run_rlcd_generation` 只是改名为 PCD 前的遗留别名，**没有任何 RLCD 算法实现**。底座 Qwen2.5-1.5B-Instruct（仓名里的 1B 与实际不符）。

核心机制（`core/schema.py:134-187`、`core/engine_torch.py:106-133`）：每个 schema 字段编译成 suffix（公共前缀 + 候选首 token ID）；prefill 一次拿 KV cache，`copy.deepcopy` 后 `batch_repeat_interleave(M)` 复制到 batch 维；一次 batched suffix 前向，取各字段末位 logits 的候选切片做局部 Softmax。免重新训练属实。

两个诚实警示：多 token 候选碰撞时，MLX 路径的置信度是**启发式合成的**，夹逼在 [0.75, 0.9999] 区间（`core/engine_mlx.py:416-418`）；temperature 恒为 1.0，无任何校准拟合。README 宣称的 5.6x 到 7.0x 加速为作者自报数据。

### rlcd-modernbert-151m：校准做得最认真的一家

底座是 **GLiClass v2 头 + ModernBERT-base 编码器**（不是裸 ModernBERT），151,378,177 参数与文档相符。25 个候选槽 = 24 个实选项 + 1 个显式弃权槽 `__insufficient_evidence__`（注入 prompt 的候选字符串，非特殊 token）。

`calibrator.json` 是全场最有研究价值的文件：

- 全局温度 T = 2.8039，L-BFGS 在 1753 条留出样本上最小化 NLL 拟合；
- **按候选数 K 分桶的独立温度**：K=2/3 给 5.01，K=5 给 3.06，K=9 给 1.67，K=25 给 1.51。候选越多温度越小，对应 Softmax 竞争加剧；K=11 的 3.39 是明显异常值；
- 声明"零拟合于 JevBench 评测任务"，防 benchmark 泄漏。

训练侧：CE + Brier 复合损失（λ=1），3 个 epoch 共 11 分钟（MPS），按 val_nll 选点（91.6% 准确率）而不是选准确率更高的 epoch 3（93%），校准优先的取舍言行一致。瑕疵：README 写 T=1.0716 与 calibrator.json 的 2.8039 自相矛盾，README 落后于最近一次 commit；延迟宣称小于 35ms，README 自己的表格是 p50 35.58ms。

### decider-2b：最完整的复现，也有明确的差距数字

Qwen3.5-2B-Base（1.9B，24 层混合注意力，线性注意力依赖 flash-linear-attention 内核）。v10 发布于 2026-09-19，是目前三原语齐备且给出与官方对比数字的复现。

- **三原语实现**（`decider/systemone.py`）：choice 支持 2 到 255 选项（≤10 用 (A)..(J) 渲染，>10 用字母宽表，每选项一个 label token）；score 的 `isolated_levels` 把 0 到 10 每个等级单独判一次 yes/no，剥离等级编号与邻居干扰；noul 即 yes 概率。支持 `abstain_below` 弃权阈值。
- **校准**：单温度 1.3（schema_first 布局 1.18），in-task 数据拟合。
- **训练两阶段**：95 个公开决策数据集混合 + CE（v1 到 v8）；384 步 calibration-aware RL（PPO outcome reward + belief log score + KL 保持项）得到 v10。"RLCD"同样与实际不符。
- **与官方的差距**：Bespoke 套件 v10 macro 0.704 vs Jev 1.13.0 的 0.760；JevBench easy/standard/hard 为 1.000/0.847/0.459 vs 官方 1.000/0.986/0.730。standard 差 14 点，hard 差 27 点。held-out 任务 ECE 0.108 明显差于 in-task 的 0.051。

### system-one-mini：单步反射的一份内部反证

实际是 **DistilBERT-base 微调 + 5 个固定分类头**（69,328,144 参数精确吻合文档的 69M），README 自己声明"这不是 Jev 或 RLCD 的复现"。五个头对应固定的 agent 运维决策，无动态提问能力。

价值恰在其诚实的失败记录：`next_action` 头最终准确率 62.8%、平均置信度 99.2%，温度缩放后 ECE 反而恶化；`temperatures.json` 逐头温度里两个头 T≈20.09 = e³ 撞上 `log_temperature_bounds` 上界，标量温度校准未收敛。**策略推理（多条件规则组合）不是单步分类能可靠承载的**，这是对"单步反射"论点最强的内部反证。训练仅 2 万条合成样本、RTX 3060、385 秒。

## 第二批：训练复现与架构路线

### kev：架构层面最深入的开源复现

jaredpalmer 出品，明确基于 Archer Hume 逆向博文的架构描述，用 Qwen3.5 底座训练出 0.8B/4B/9B 家族（rank-16 LoRA + 指针头），并提供 System One API 兼容的 FastAPI 服务。三个架构要点值得记录：

- **指针头读出**（`kev/model.py:123-129`）：不用候选 token Logits，而是 `softmax((W_k·h_opt) · (W_q·h_decide))`，每个选项的 `</opt>` 隐状态与 `<decide>` 隐状态做点积打分。绕开了首 token 截断与多 token 候选碰撞问题。
- **块因果分支掩码**（`model.py:70-101`）：state 共享、每题一个分支，token 只能看到 state 和本题，`allow = causal & same & valid_key`。并验证了打包式与分列式概率一致（差异 <4e-6）。
- **混合底座特例**：Qwen3.5 的 Gated DeltaNet 循环层无视注意力掩码，因此每题作为独立 batch 行、state 只算一次共享缓存（`model.py:104-118, 230-291`）。服务端有 LRU state 前缀缓存（`serve.py:53-78`），重复 772-token state 从 861ms 降到 242ms。

校准是研究性的：`scripts/temperature_groups.py` 实验"每（题型，选项数）一组温度"，Qwen3.5 家族拟合出 T=2.0，环境变量可选启用。与 Jev 的系统对照贯穿全仓（经 Vercel AI Gateway 调真实 Jev）；"unknowable" 题上高置信错误率 Kev-9B 5% vs Jev 9% vs Kev-8B 26%。

### laya：encoder-only 路线，彻底绕开 KV Cache

ConvAI Innovations 出品，Apache 2.0，`pip install laya`。底座是**双向 encoder**（ModernBERT-large 421M / mmBERT-base 322M），思路与所有 decoder 复现不同：

- **[MASK] 标记打分**（`laya/common.py:57-124`）：序列格式 `[CLS] 指令 [SEP] [MASK] 选项0 [MASK] 选项1 ... [SEP] state [SEP]`，每个选项前置一个 [MASK]，2 层 Transformer head + MLP 把每个 marker 位置映射成一个 logit，一次前向出全分布。双向注意力天然允许选项互相看见完整上下文，**没有 KV Cache 概念，也就没有前缀共享的需求**。
- **校准**：按（题型 × 选项数桶 2/3-5/6-10/11+）的温度缩放。出厂 raw ECE 0.213 劣于 Jev 的 0.144，温度拟合后才到 0.081——校准是后处理拟合出来的，不是训出来的。
- **诚实的高基数短板**：Banking77（77 选项）Jev 0.870 vs laya 0.425，受选项区 token 预算限制。tracker 显示其 3908 stars 是全场 GitHub 最高。
- **独有原语**：`act_head` 输出"是否升级到 System 2"的概率，是 Jev 没有的显式分层信号。

### NanoJev：0.6B 游戏复刻与候选集合注意力

Qwen3-0.6B-Base 底座，核心实现不到 400 行，但 scripts 目录有 162 个脚本。架构上有两个独特选择：

- **每候选独立序列 + 集合注意力头**（`scripts/train_toy_decisions.py:86-141`）：每个候选拼成独立序列 batch 前向，取每条序列最后真实 token 的隐状态作为候选向量，标量头打分后再经 `nn.MultiheadAttention` 做候选间交互（类似 Set Transformer），让"列表排序"显式建模而不是依赖 decoder 注意力的副作用。
- **显式承认没有前缀共享**：推理结果自报 `"prefix_sharing": False`，路线图未勾选项；公共前缀 token 被重复计算。这是一个诚实的工程缺口记录。

亮点是同条件对比：ViZDoom Basic 上 NanoJev 128/128 vs Jev 56/128（游戏状态分布外推上小模型反超）。校准分区设计严谨（五分区，温度只在 calibration 分区拟合，评测脚本硬性断言 `temperature_fitted is False` 防泄漏），但服务默认 T=1。`research/jevlike_repo_audit_zh.md` 还包含对另一个复刻仓的逐行审计。

### SemIf：浏览器端单步判定（原 openjev.com）

README 第一行就确认了身份：**"SemIf (formerly OpenJev)"**，即 starter 文档生态表里 "TheoLeeCJ/openjev (openjev.com)" 的正式名。

- 浏览器端形态：vendored wllama 3.6.1（llama.cpp 的 wasm 绑定）+ WebGPU，三档 GGUF 模型（Qwen3-0.6B 手机档 / **MiniCPM5-2B Q4_K_M 1.56GB 桌面档** / Qwen3.5-4B 高内存档），纯静态托管、无后端无遥测。文档说"MiniCPM5-2B"属实但只是中间档，基准主力是 Qwen3.5-4B。
- 判定实现：每个选项绑定单 token 字母标签（A/B/C...），一次前向读标签槽位 Logits 归一化；同时提供生成式 JSON 对照路径（direct 1.023s/0 token vs 生成 5.332s/111 token）。
- **校准声明是全场最诚实的**：所有后端输出统一标注 `"probability_status": "conditional option score; uncalibrated as decision confidence"`，METHOD.md 明确 softmax over allowed tokens 不等于可运营置信度。
- 服务端 `shared.py` 做共享 prefill 并行判定（777 判定 38.8s）；与 Jev 在 102 行 TypeSafe 公开子集上的一致率 0.845 vs Jev 自身复测 0.883。

## 第二批：基准与评测

### nimble：首个全开放 recipe 与 13 子集人标基准

Bespoke Labs 出品，自称 "Data, Model, Recipe for an open Jev"，明确声明**未从 Jev 蒸馏**。Bespoke-Nimble-9B = Qwen3.5-9B + rank-16 LoRA，训练目标是候选 logits 上的 hard 交叉熵，配套 2,676 条训练样本开放。

- **实现**：候选答案编码为单 token 字母码（A-Z，上限 26 选项），FP32 隐状态投影（`nimble/scoring/cuda_scorer.py:46-50`）；`parallel_scorer.py` 共享 context prefill 一次后并行给所有字段打分。
- **校准是其自认短板**：temperature 固定 1.0，"No temperature has been fitted"；13 子集中 11 个 Jev 的 ECE 更低。其对抗校准问题的手段在数据侧：对比式数据策展（仅改动一个 focus fact 即翻转标签的成对样本）。
- **13 子集公开人标基准**（3,880 条、种子 20260918、family-whole 采样、checksum 可逐字节重建）是目前最可信的第三方横评：Nimble-9B 74.8%/75.9% vs Jev 1.13.0 76.0%/77.3%（宏/微平均）。分原语：Noul Jev 明显领先（84.6 vs 80.2），Score Nimble 反超（54.6 vs 50.1）。该文档还指出 TypeSafe 官方数字全部来自私有基准，Jev 此前没有任何公开数据集成绩。
- 彩蛋：训练文档里 adapter 历史名 `openjeff-diverse9b-v2`，项目内部代号 "openjeff"。

### jev-on-a-laptop：73.8% 的审计报告

starter 文档里"裸 Qwen2.5-7B 一致性 73.8%、错误时置信度常高于 0.90"两个数字的出处。核验结论：**数字属实，但成色需要打折**。

- 73.8% vs 86.6% 可在 `evals/results/full-summary.json` 等多处复核，但它是"与 GPT-6 Astra + Claude Fable 5.1 双模型共识的一致性"，不是与真值的一致性；且只在 343 对严格公共子集上（全量 373 对是 73.2% vs 86.9%）。
- "错误时 >0.90"来自另一个自建的 72 字段集：7B 模型 20 个错误字段里 13 个（65%）置信度 >0.90，正确/错误字段平均置信度仅差 0.06。TypeSafe 评测本身没有做逐字段置信度分析。
- 仓库内**没有任何校准实现**（温度缩放只在 ROADMAP 里）；README 明确承认 confidence 是 "raw softmax over candidate logits — a proxy, not trained calibration"，引用的外部 ECE 0.094 来自作者另一个包。
- Addendum 自我审计很扎实：noul 题真实正例率仅 18.6%（常数 false 即得 92.7%）、choice 题 82% 选列表首项、3 题参考答案不在选项内。评测协议本身的坑比模型差距更有教育意义。
- 引擎代码与 Qwen-2.5-1B-RLCD 同源（同一套 `core/engine_mlx.py` 家族），同样存在碰撞分支置信度夹逼问题。

### jev-phishing-bench：钓鱼基准与正则基线的打脸

2000 封邮件（PhishNChips v5.2，1000 钓鱼 + 1000 合法，含 333 封跨域误报陷阱）。结果（`results/metrics.json`）：

| | Jev | Haiku 4.5 |
| --- | --- | --- |
| 准确率 | 62.6% | 81.3% |
| 钓鱼召回 | 43.2% | 76.4% |
| AUROC / ECE | 0.689 / 0.154 | 0.837 / 0.097 |

三个修正性发现：

1. **Haiku 是明确关闭 thinking 跑的**（temperature 0）。starter 文档"支持思维链推理的 Claude Haiku 优于 Jev"的归因不成立。
2. **数据集没有"多层转折与伪造身份"的构造维度**。欺骗信号主要在 URL 与发件人一致性（短链、免费托管、知名域名滥用），正文是合成的。Jev 的失败模式是"信任知名域名"（firebase 类钓鱼仅 45.5% 被检出、google_docs 类 1.5%），不是推理深度不足。
3. **非 AI 正则基线（短链/免费托管列表 + eTLD+1 比对）单规则 91.8%**，超过 Jev verdict 的 62.6% 达 29 个百分点；Haiku 同五问单规则 94.2%；Jev 五信号 + 逻辑回归 95.0% 与 Haiku 打平。作者结论（法文原文）："jev est mauvais pour juger, correct pour observer"——Jev 判断差、观察可以：把 Jev 当特征提取器接确定性规则，比让它直接下 verdict 强得多。

## 第二批：生产守卫与 Agent 集成

### pi-warden 与 pi-jev-auto-mode：两种守卫哲学

同为 Pi coding agent 的扩展、同样调用 `api.typesafe.ai`，设计哲学几乎相反：

| | pi-warden | pi-jev-auto-mode |
| --- | --- | --- |
| 架构 | 正则地板 + Jev 只能抬高地板 | 确定性分诊 + 只有被升级的调用才到 Jev |
| 失败默认 | **fail-open**：Jev 不可用放行并警告 | **fail-closed**：超时/无 key/响应畸形/取消全部阻断 |
| 写文件 | 从不阻断 write/edit（防半写文件），只 steer | 保护路径或 cwd 外的 write/edit 送 Jev |
| 阈值 | 每问题独立实测校准（`docs/guards.md` AUC 表，AUC 0.51 的问题降级为 trace-only） | 每规则双侧阈值带（p≥t 满足、p≤1-t 否决、中间带 uncertain），uncertain 默认 deny |
| 实盘数据 | 321 会话 17,160 次守护调用，20 次懊悔（2%）；单日 1,042 次判定：948 放行 / 87 警告 / 7 hold | 11 条普通命令实测 193-642ms/次 |

pi-warden 的两点工程亮点值得单独记录：一是 **arming 状态机**（编辑某 glob 文件后在时间窗内武装某命令模式），捕捉"单步无害、组合致命"的跨调用风险，这是单次判定模型覆盖不到的维度；二是模式层做了**数据文本剥离**（heredoc 正文、commit message 里出现的 `git push --force` 不触发，经 `sh -c` 执行的才保留命中），避免纯关键词黑名单的误报跑步机（其前代路径分类扩展一周约 2,000 次提示、87% 误报）。

pi-jev-auto-mode 的 `docs/calibration.md` 与 `questions.ts` 实码存在阈值/模式不同步的文档滞后，读时以代码为准。

### jev-ultrafast 与 mobile-jev：GUI Agent 集成

两个项目共享同一套协议形态，可作为 Jev API 消费端的参考实现：

- **一次请求多问题 + 投机性目标头**：主问题选操作（operation），同一请求里并发问 `click_target` / `type_text_target` 等投机性子问题（候选只限兼容元素索引），只消费被选中操作对应的答案。这正是 starter 文档"推测性扇出"在闭式 API 上的实例。
- **状态是结构化文本不是截图**：jev-ultrafast 用 CDP 一次读出可见控件 + WeakMap 节点身份 + 语义指纹（`snapshot.js`）；mobile-jev 把 a11y tree 压平为索引化元素列表（`device.mjs:88-189`），纯结构化、无视觉输入。
- **新鲜度防线**：两者都在动作下发前重读屏幕做语义比对（fingerprint / `assertFresh()`），过期即拒绝执行且不重放变更操作。
- **修正 starter 文档的说法**：mobile-jev **完全没有生成模型调用路径**（输入文本从 goal 里枚举 span 作为候选，Jev 只选或选 NONE）；jev-ultrafast 只在 TYPE_TEXT 时调文本模型生成**短字符串**（城市名级别，2000 字符硬上限），不是"长文本"。
- 延迟数据：jev-ultrafast 实测 Jev 单请求中位 178ms，完整 Google Flights 任务 7.07s；mobile-jev 演示任务 8.7s/5 次调用。两者都明确声明不构成成功率证据，DONE 不是成功的独立证明。

## 第二批：生态地图

### jev-reproductions-tracker：热度榜与演化时间线

HF Space 静态页（非 Gradio），数据是 `index.html` 底部手工维护的 JS 数组，PR 驱动更新；`refresh.js` 可选抓取 X/GitHub/HF 热度指标（2026-09-20 快照）。分类五档：decode（无新权重的推理技巧）、trained（真训练产物）、diffusion、prior（前作主张）、explain。

- 指标是社交热度不是性能，但描述里散落自报性能数字（Kev-8B 79.6% vs 85.7%、reflex ECE 0.039 vs Jev 0.031 等），由各项目自填，tracker 不做统一评测。
- **date 字段就是演化时间线**：9-16 第一波 decode 复现 → 9-17 校准质疑与解释文 → 9-18 Bespoke Nimble 首个真训练产物 → 9-19/20 成本对比与可自训家族 → 9-21 Archer Hume 承诺的 Solomon 交付。
- "Still not in the open" 区块明确列出生态缺口：TypeSafe 权重、RLCD 算法/损失/数据、一个真正达到 Jev 校准主张的开放模型，均不存在。

### awesome-jev：100+ 条目的生态索引

双语同步维护，收录标准明确（须真实使用 Jev/System One API），禁止星标徽章。值得记录的条目：

- **官方失败模式页**：Jev 1.13 jaggedness 文档（starter 文档"失败模式"章节可引用的权威来源）。
- **jevcal**：阈值锁定 + CI 回归检测，把"模型更新导致校准漂移"做成工程防护。
- **ASSAY-001**：预注册的第三方校准检验，含完整日志。
- **LitJev**：宣称零训练把任意 Qwen 变成同 wire-schema 决策模型，与所有需训练的复现路线形成对照。
- **jev-tree**：递归 Choice 突破 255 选项上限，与 decider 的字母宽表是同题两解。
- **sqlite3-jev / jevql**：把三原语做成 SQL 函数，判定能力进数据库。
- 索引内嵌诚实负结果："Haiku wins accuracy here"（phishing bench）、"Fusion wins; Jev alone does not beat embeddings"（rerank 评测）。

## 横向发现

### 校准光谱（更新）

18 个仓库排出来，校准实践构成一条完整光谱：

| 校准方案 | 仓库 |
| --- | --- |
| 显式声明未校准 | SemIf、nimble（T=1 自认短板）、jev-on-a-laptop |
| 合成/夹逼置信度 | Qwen-2.5-1B-RLCD（碰撞分支 [0.75, 0.9999]） |
| 分区拟合但默认不启用 | NanoJev（五分区防泄漏设计，服务 T=1） |
| 单温度 | decider-2b（1.3）、kev（实验值 2.0） |
| 按结构分桶 | rlcd-modernbert-151m（按候选数 K）、laya（按题型×选项数桶）、system-one-mini（逐头，两头撞界） |
| 每规则/每问题独立阈值 + 双侧不确定带 | pi-warden、pi-jev-auto-mode |

两个收敛信号：**按候选数分桶温度被两家独立发明**（modernbert 精确按 K、laya 按 2/3-5/6-10/11+ 桶），印证 starter 文档门控章节的讨论方向；**没有一家用 RL 直接优化校准**，RLCD 在所有仓库里的真实形态是 proper scoring loss（CE+Brier/RPS）+ 训练后温度缩放，decider 的 RL 阶段优化的也是任务结果而非 ECE。

### KV 前缀共享的路线图（更新）

结合 18 仓，前缀共享存在四条明确路线：

1. **物理复制缓存**：openjev（`reorder_cache`）、Qwen-2.5-1B-RLCD（`deepcopy + batch_repeat_interleave`）、nimble/SemIf（共享 prefill + 并行分支）。原因统一指向混合注意力底座的线性层状态无法用 tree mask 隔离。
2. **行式分支 + 共享 state 缓存**：kev 对 Gated DeltaNet 的适配（每题独立行，state KV 复用，`model.py:104-118`），服务端再叠 LRU 前缀缓存（861ms→242ms）。
3. **块因果分支掩码**：kev 在 attention-only 底座上的打包式实现（与分列式概率差 <4e-6），即 starter 文档扇出章节示意图的严格版本。
4. **encoder 路线无需缓存**：laya 的双向注意力一次前向出全分布，NanoJev 的每候选独立序列则相反（显式 `prefix_sharing: False`，公共前缀重复计算）。

### 守卫的两种哲学

pi-warden（fail-open + 正则地板 + 从不阻断写入）与 pi-jev-auto-mode（fail-closed + 全路径阻断 + uncertain 默认 deny）是同一 API 上的一组对照实验。选型逻辑清晰：warden 守的是"别把工作搞砸"（误报代价高，2,000 次/周提示会训练用户无脑点确认），auto-mode 守的是"别让 agent 乱来"（漏放代价高，宁可打断）。starter 文档网关章节的三形态（拦截器/路由器/守卫）可补充这第四个维度：失败时的默认动作。

### 与官方差距的可靠数字

第三方横评目前只有两家可信：Bespoke 13 子集人标（Nimble 74.8 vs Jev 76.0 宏平均，校准 Jev 明显更好）与 decider-2b 引用的 JevBench（standard 差 14 点、hard 差 27 点）。jev-on-a-laptop 的 86.6% 是共识一致性不是真值；tracker 里的各项目自报一致率无统一口径。开放复现目前没有任何一家的校准达到 Jev 水平（tracker "Still not in the open" 也确认这一点）。

### 候选槽位工程（更新）

四种方案并存：首 token 截断 + 碰撞消歧（PCD 家族，置信度失真的来源）、专用槽 token + 弃权槽（GLiClass 25 槽）、字母宽表（decider 255 上限、nimble A-Z 上限 26）、指针头（kev，隐状态点积，完全绕开 tokenization）。指针头工程上最干净，字母宽表次之。

## 文章失实与夸大清单（更新）

对照 `docs/jev-starter.md`：

| 文档声称 | 实测 |
| --- | --- |
| modernbert-151m 延迟小于 35ms | README 自测 p50 35.58ms |
| system-one-mini "自研紧凑网络" | DistilBERT-base 微调 + 5 个线性头 |
| decider-2b "数十万标注样本全参数微调" | 无直接证据；tracker 称 942k 样本但训练代码不在仓内 |
| "RLCD" 校准 | 所有仓库的真实形态是 proper scoring loss + 温度缩放，无 RL 优化校准 |
| phishing bench "思维链 Haiku 优于 Jev" | Haiku 明确关闭 thinking；差距来自 URL 判别而非推理深度 |
| phishing bench "多层转折与伪造身份邮件" | 数据集无此构造维度，欺骗信号在 URL/发件人 |
| （未提） | 正则基线 91.8% 超 Jev verdict 29 个百分点；Jev 当特征提取器接逻辑回归 95.0% 反而与 Haiku 打平 |
| pi-warden "替代关键词黑名单" | 实为正则地板 + Jev 分层；且默认 fail-open，与文中守卫语义相反 |
| jev-ultrafast / mobile-jev "需要生成长文本时调用生成模型" | mobile-jev 无生成模型；jev-ultrafast 只生成短字符串 |
| "TheoLeeCJ/openjev (openjev.com)" | 项目已改名 SemIf |
| openjev "基于 NLI 序列分类" | 属实，但漏了 per-task MLP 头路径与多模态能力 |
| 18 个 GitHub 仓库"不存在"（本文第一版结论） | 12 个真实存在，生态以天为单位演化，探测结论需带时间戳 |

## 后续深挖建议（更新）

1. `kev/kev/model.py`（298 行注释极密）是指针头 + 分支掩码 + hybrid 适配的最完整单文件教材。
2. `pi-warden/src/guard.ts`（1479 行）与 `docs/guards.md`：判定问题的阈值校准方法论（AUC 淘汰、trace-only 降级）可复用。
3. `nimble/evaluation/evaluate_public_jev.py`：Jev wire 协议的一手材料，配合 `mobile-jev/scripts/mobile-agent/policy.mjs` 可完整还原 API 契约。
4. starter 文档可回写的修正：phishing bench 的归因（Haiku 无思维链、正则基线数据）、pi-warden 的 fail-open 语义、SemIf 改名、门控章节补 per-cardinality 温度与"失败默认动作"维度。
5. 索引线索：LitJev（零训练）、ASSAY-001（预注册校准检验）、jevcal（CI 阈值锁定）、官方 jaggedness 页。
6. 拉取 LFS 权重验证（`git lfs pull` 走代理）：system-one-mini 参数量、modernbert 校准效果复测。
