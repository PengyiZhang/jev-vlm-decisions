# 路线 A Demo：字母槽单次前向人员类型判定

设计文档：`docs/plans/2026-09-21-vlm-person-type-design.md`（路线 A 章节）。

对 cropped 行人图像输出人员类型概率分布与多属性判定，例如：

```
[ptype] airport ground staff (0.85)  gate=auto  abstain=0.01
    airport ground staff (0.85)  flight attendant (0.10)  passenger (0.05)
```

## 机制

一次 Prefill 读所有"答案 k："槽位的 next-token Logits，取字母 Token 做掩码 Softmax。
N 个问题一次前向出全部结果；判定依据放共享指令区；弃权槽恒在末位。

## 结构

| 模块 | 职责 |
| --- | --- |
| `schema.py` | 场景/问题配置、字典序规范排序、弃权槽、K≤26 校验 |
| `encoding.py` | 字母槽分配与单 token 校验 |
| `prompt.py` | 答案槽留空的多问题布局 |
| `readout.py` | 候选掩码 Softmax、三级门控（0.90/0.60，弃权胜出恒走人工） |
| `calibrator.py` | 按（类型×K）分桶的温度拟合（NLL 网格）、ECE、calibrator.json |
| `engine.py` | ClassifyTask + Scorer 协议 + FakeScorer（纯逻辑测试） |
| `transformers_scorer.py` | transformers 直连：chat template 组装（`apply_chat_template` + image content），一次前向读全部槽位（惰性导入） |
| `vllm_scorers.py` | vLLM 双策略：每问独立请求（策略一）/ 占位符 prompt_logprobs（策略二b） |
| `classify.py` | 组装管线 |

## 测试（无 torch 依赖）

```bash
uv run --with pytest python -m pytest demos/person_type_a/tests -q
```

## 跑真实模型

```bash
uv run --with torch --with transformers --with pillow python -m demos.person_type_a.run_demo \
    --scenario demos/person_type_a/scenarios/terminal.json \
    --model Qwen/Qwen3.8-27B-Instruct --image crop1.jpg \
    --engine transformers   # 或 vllm-perq / vllm-plogprob
```

启动自检：字母单 token 校验 + 锚定位校验（transformers 引擎），任一失败立即退出。

## 三引擎对比

| | transformers | vllm-perq（策略一） | vllm-plogprob（策略二b） |
| --- | --- | --- | --- |
| 请求数/图 | 1 | M（每问独立） | 1 |
| 图像 Prefill | 1 次 | 1 次（需多模态前缀缓存）/ M 次 | **必然 1 次** |
| 解码步 | 0 | 0 | 0 |
| 槽位条件于 | 问题文本，无答案 | 仅本问文本（完全独立） | 问题文本 + 中性占位符 |
| Logits 形态 | 全词表 | top-K 稀疏 | top-K 稀疏 |
| 依赖 | torch/transformers | vLLM + `--enable-prefix-caching` | vLLM `prompt_logprobs` |

vLLM 部署提示：`--topk` 必须 ≥ 场景最大候选数（CLI 已校验）；采样 temperature 恒 1.0，
温度校准由 calibrator.json 按桶施加；高吞吐离线批量用 `vllm.LLM` 引擎直连（本 scorer 即此形态）。

## 校准

在标注集上按（问题类型 × 候选数 K）分桶拟合温度（`calibrator.fit_temperature`），
`save_calibrator` 落盘后经 `--calibrator` 注入。未校准时概率只保证归一。

## 已知边界

- **图像占位符随模型而异**：Qwen 系为 `<|image_pad|>`（自动包 `<|vision_start|>/<|vision_end|>`），gemma-4 为 `<|image|>`。transformers 引擎在 load 时经 `resolve_image_token` 自动解析（processor 属性优先、词表探测兜底）；vLLM 引擎用 `--image-token` 或构造参数显式传入，占位符不符会报 `tokens: 0, features: N` 或 `<|image|> tokens ... 0`
- 字母单 token 与锚定位依赖词表：BPE 边界效应可能造成偏移，启动自检拦截
- 多 token 类名不参与打分（字母槽方案的结构优势，无首 token 碰撞问题）
- 候选项间存在注意力交互（生态研究已实证）；选项描述保持简短，评测时做顺序扰动量化
- vLLM 前缀缓存模式与路线 B（逐候选打分）/ 路线 C（LoRA）为后续任务
