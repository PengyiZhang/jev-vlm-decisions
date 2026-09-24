# 单前向 VLM 决策引擎

中文 | **[English](README.md)**

**面向视觉语言模型的 Jev 式字母槽决策。** 一次前向把任意 VLM 变成毫秒级图像决策引擎：零自回归解码、零 JSON 解析、开箱即得概率分布。

```text
[ptype] airport ground staff (0.85)  gate=auto  abstain=0.01
    airport ground staff (0.85)  flight attendant (0.10)  passenger (0.05)
```

### 为什么

让 VLM"分类这张图并用 JSON 回答"要付出完整自回归循环的代价：几十到上百步解码、格式错误、未校准的置信度。对闭集决策——路由、门控、属性打标——这些代价都可以免掉。

本项目沿用 [Jev](https://docs.typesafe.ai/)（TypeSafe 的 "System One" 模型）及其开源生态带火的决策模型模式：

- **零解码步。** 候选渲染成 prompt 里的字母槽 `(A)…(B)…`，答案槽留空。一次 Prefill 返回每个位置的 next-token Logits，字母分布本身就是答案。
- **结构代替解析。** 输出是候选字母 token 上的掩码 Softmax——永远不会格式错误。
- **校准是一等公民。** 裸 Softmax 普遍过度自信（Jev 复现生态反复验证的结论）。温度按（问题类型 × 候选数）分桶、在留出标注集上拟合——RLCD 式校准训练的精神，但零训练。
- **弃权内置。** 每个问题都带显式 `__insufficient_evidence__` 槽；弃权胜出时恒转人工，绝不自动执行。

完整设计依据、失败模式与 18 仓 Jev 生态实地研究见姊妹长文 [`understanding-jev`](https://github.com/PengyiZhang/understanding-jev)。两篇正文已随本仓库收录：[万字解读长文](docs/jev-starter.md)（机制拆解 / 失败模式 / 设计模式 / 生产架构）与 [18 仓生态实地研究](docs/jev-ecosystem-research.md)（复现路线 / 校准光谱 / 失败实证）。

### 工作原理

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

- **任务经 system 提示词注入。** 引擎与任务无关：场景 JSON 定义指令、判定依据与问题。人员类型分类只是内置示例，任意闭集视觉决策都适用。
- **规范化选项顺序**（字典序 + 末位弃权槽）消除 prompt 顺序抖动；判据放共享指令区。
- **三级门控**基于校准置信度：`auto ≥ 0.90`、`review 0.60~0.90`、`human < 0.60`；弃权胜出恒转人工。

### 安装与快速开始

需要 Python 3.10+ 与 [`uv`](https://docs.astral.sh/uv/)。

```bash
# 用本地 VLM 跑（已在 Qwen3.8-27B 与 gemma-4-E4B-it 上实测）
uv run --with torch --with transformers --with pillow \
    python -m person_type_a.run_demo \
    --scenario person_type_a/scenarios/terminal.json \
    --model /path/to/your-vlm \
    --image crop1.jpg --image crop2.jpg \
    --engine transformers          # 或 vllm-perq / vllm-plogprob
```

启动自检：字母必须全部单 token、答案锚必须可在分词后的 prompt 中定位——失败立即退出，不产出垃圾结果。

### 定义场景

```jsonc
{
  "name": "terminal",
  "system": "你是机场的人员类型分类器。只输出选项字母。",
  "scene": "航站楼到港层，行人 cropped 图像",
  "evidence": ["制服样式", "反光背心", "工牌", "行李特征"],
  "questions": [
    { "qid": "ptype", "kind": "choice",
      "instructions": "图中人员属于哪种类型？",
      "options": ["airport ground staff", "flight attendant", "passenger", "..."],
      "criteria": ["反光背心、地面作业制服", "航司制服", "普通旅客装束", "..."] },
    { "qid": "vest", "kind": "binary", "instructions": "是否穿着反光背心？" },
    { "qid": "luggage", "kind": "binary", "instructions": "是否携带可见行李？" }
  ]
}
```

- `choice`：最多 25 个选项 + 自动追加的弃权槽（单字母容量）。
- `binary`：固定 `no / unclear / yes` + 弃权。
- `system` 字段是任务注入点——换掉它（和 questions）即可改造为损毁检测、单据分诊、UI 状态判断等任务。

### 推理引擎

| | `transformers` | `vllm-perq` | `vllm-plogprob` |
| --- | --- | --- | --- |
| 每图请求数 | 1 | M（每问一个） | 1 |
| 图像 Prefill | 1 次 | 1 次（依赖多模态前缀缓存） | **必然 1 次** |
| 解码步 | 0 | 0 | 0 |
| 槽位分布条件于 | 各问题文本，无答案 | 仅本问文本 | 问题文本 + 中性占位符 |
| Logits 形态 | 全词表 | top-K（稀疏） | top-K（稀疏） |
| 说明 | chat template 组装，控制力最强 | 默认开 `enable_prefix_caching` | 经 `prompt_logprobs` 读占位符位置 |

三引擎同图实测记录（Qwen3.8-27B）：[docs/runtime-zh.md](docs/runtime-zh.md)。

vLLM 引擎选卡用 `CUDA_VISIBLE_DEVICES`（没有 `device` 参数）。图像占位符因模型家族而异（Qwen 系 `<|image_pad|>`、gemma-4 `<|image|>`）；transformers 引擎自动解析，vLLM 引擎可用 `--image-token` 指定。

### 校准

零样本概率只保证归一，不保证校准。流程：

1. 从**生产检测器**采集标注 crop（分辨率/构图分布一致），按人/视频源切分——严禁按帧切——分为 `train / calibration / test`。
2. 以 temperature=1 跑 scorer，落盘槽位分数。
3. 按（问题类型 × K）分桶做 NLL 网格搜索拟合温度（`calibrator.py`），保存 `calibrator.json`。
4. 服务时 `--calibrator calibrator.json` 注入；在 test 上监控 ECE。

接入校准器之前，置信度只作排序参考。

### 测试

核心纯标准库，无需 torch/vLLM：

```bash
uv run --with pytest python -m pytest person_type_a/tests -q
```

### 模块结构

| 模块 | 职责 |
| --- | --- |
| `schema.py` | 场景/问题配置、规范化排序、弃权槽、K ≤ 26 校验 |
| `encoding.py` | 字母分配、单 token 校验、上下文 token id 解析 |
| `prompt.py` | 系统/问题文本构建；vLLM 裸文本布局 |
| `readout.py` | 掩码 Softmax（支持稀疏 top-K）、三级门控 |
| `calibrator.py` | 分桶温度拟合、ECE、`calibrator.json` 读写 |
| `engine.py` | `ClassifyTask` + `Scorer` 协议 + 无依赖测试用的假实现 |
| `transformers_scorer.py` | chat template 组装、单前向读全部槽位 |
| `vllm_scorers.py` | 每问扇出与占位符/prompt_logprobs 两策略 |
| `classify.py` | 组装管线：任务 → 字母 → 一次打分 → 校准结果 |

### 路线图

- **路线 B**——逐候选 yes 打分扇出（结构性免疫选项干扰；共享一次图像 Prefill）。
- **路线 C**——字母槽交叉熵 LoRA 微调 + 顺序增广（nimble/decider 的配方），每类 ≥300 标注后启动。
- 远程服务化 scorer（OpenAI 兼容端点、共享 vLLM 服务 + 跨租户连续批处理）。

### 致谢

- [TypeSafe 的 Jev](https://docs.typesafe.ai/) 与 Archer Hume 的架构逆向文章——本项目实现的决策模型模式来源。
- 开源复现社区——`kev`（指针头 + 分支掩码）、`bespokelabsai/nimble`（开放配方 + 人标基准）、`Mapika/decider-2b`（三原语）、`rlcd-modernbert-151m`（按候选数分桶温度）——为本文总结的设计选择提供了实证。

### 许可

MIT——见 [LICENSE](LICENSE)。
