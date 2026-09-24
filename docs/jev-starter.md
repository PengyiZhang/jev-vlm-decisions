# Jev Starter：毫秒级判定原理与工程应用边界

> 来源：谷粒《万字长文解读 Jev 模型：毫秒级判定原理与工程应用边界》（让知识开源 · 科学季 2026）
> 原文体验：<https://jev.kuhung.me/>
> 整理日期：2026-09-21

一句话总结：Jev 是面向状态评估的单步判别模型。零自回归生成，输入按 Token 计费，输出免费，端到端延迟 70 到 500 毫秒。适合路由、预筛、门控，不适合生成与多步推理。

## 目录

- [为什么关注 Jev](#为什么关注-jev)
- [Jev 是什么](#jev-是什么)
- [拆底层](#拆底层)
- [二十行代码实测](#二十行代码实测)
- [压测数据分析](#压测数据分析)
- [失败模式](#失败模式)
- [四种设计模式](#四种设计模式)
- [生产架构](#生产架构)
- [生态地图](#生态地图)
- [该不该用](#该不该用)
- [参考资料](#参考资料)

---

## 为什么关注 Jev

在过去的项目里，我为了优化接口延迟花过不少功夫。用户对响应速度其实高度敏感。系统一旦超过一两秒没动静，看起来就像坏了一样。现实情况是，很多大模型为了做推理思考，响应动辄两三秒甚至更久。大家虽然尝试了各种工程手段去优化，但大模型逐字做自回归生成的本质摆在那里，延迟很难压缩到极致。

很多时候，我们并不需要一个能写长篇大论的推理模型。在做状态判断或者枚举分类时，响应速度远比生成能力更重要。这就是我想聊聊 Jev 的原因。

为什么这个模型最近能掀起波澜？通用大模型的方向在业界已经基本固定，新进的产品团队必须寻找差异化的切入点。Jev 号称闭门研发了两年，主打的就是极速响应。它去掉了自回归生成循环，输入按 Token 计费，输出 Token 则是零元。这很合理，它本质上是一个判别器模型，就像没有人会为随机森林的输出结果按 Token 买单一样。

## Jev 是什么

Jev 本质上是一个面向状态评估的统计分类器。大模型擅长生成，Jev 专注于判别。它把传统分类任务用大模型处理时既慢又贵的痛点，做到了低成本和高速度。

产品发布初期，不少开发者以为是恶搞，因为创始人 Diogo Almeida 的名字和 Anthropic 首席执行官 Dario Amodei 读音相近。直到社区看到用 Jev 毫秒级操控经典游戏 DOOM 的录屏演示，技术圈才开始认真讨论它到底能用来做什么。

### 名字由来

它的名字取自经济学中的杰文斯悖论（Jevons Paradox）。当某种资源的利用效率大幅提升时，其总消耗量往往不会缩减，反而会成倍增长。一旦单次模型决策的成本下降上百倍，系统里原本写死 if-else 的地方，可能都会考虑接入模型来做动态判断。

### 核心定位

Jev 的核心定位是机器对机器的状态评估。它不回答用户提问，也不生成自然语言。它的职责是根据输入上下文，在几十毫秒内返回一个预定义的离散标签，将上游请求分流到下游分支。这类似于物流分拣中的分流模型。在早期的大模型工程落地中，用轻量小模型做路由分流本来就是常见做法。

### 运行指标

官方公布的运行指标：

| 指标 | 数值 |
| --- | --- |
| 输入计费 | 每百万 Token 0.042 美元 |
| 输出计费 | 免费（0 元） |
| 端到端请求延迟 | 70 到 500 毫秒 |

不管是底座更小，还是单步前向推导，服务成本都被压得很低。

### 双系统叙事

官方在宣传时借用了认知科学的"双系统"概念：让昂贵的前沿大模型负责慢速的深度推演（System 2），让 Jev 充当快速廉价的反射弧（System 1）。在需要做前置过滤或路由分流时，用小判别模型就能搞定，不需要让生成模型全程陪跑。

### 三种输出原语

为了追求速度，Jev 舍弃了文本生成，只保留三种输出原语：

| 原语 | 含义 | 示例 |
| --- | --- | --- |
| `choice` | 在给定的枚举列表中完成多选一路由 | allow / block / escalate |
| `score` | 在有序离散区间内打分 | 0 到 10 的数值评分 |
| `noul` | 输出三值布尔概率 | true / false / unknown |

它的输出附带了置信度概率。很多开发者吃过大模型生成 JSON 时幻觉的苦头，大模型即使输出错误结果，也常常给出虚高的概率。Jev 宣称对置信度进行了校准，下游系统可以依据置信度设置阈值，一旦置信度偏低就由硬规则或人工接管。关于这个置信度是如何校准的，见[拆底层](#拆底层)一章。

### 调用方式

在工程调用上，TypeSafe AI 接入了 Cloudflare Workers AI 与 Vercel AI Gateway。走 Vercel AI SDK 时，用 `experimental_evaluate` 声明判断问题即可：

```ts
import { experimental_evaluate as evaluate } from 'ai';

const result = await evaluate({
  model: 'typesafe-ai/jev',
  state: contextText,
  questions: {
    action: {
      type: 'choice',
      instructions: 'How should this request be handled?',
      criteria: {
        allow: 'Safe to proceed',
        block: 'Should be blocked',
        escalate: 'Needs human review',
      },
    },
  },
});
```

### 能力边界

没必要把 Jev 捧上神坛。从技术本质来说，它没有超越统计学的理论创新，能力边界也很窄：它无法完成长文本创作、代码编写、多轮对话记忆，更解不了长步骤数学推导。但它提醒我们，不是所有任务都必须调用高价的前沿模型，专长于速度的判别模型同样有其工程价值。

## 拆底层

Jev 目前尚未完全开源，但结合官方技术报告与开源社区的逆向复现，它的实现路径已经相当明确。

### 零生成步数

传统大模型处理分类或决策，必须启动自回归解码循环，逐个生成 Token。整个过程通常耗时 2 到 5 秒，下游还得小心处理 JSON 字符串是否闭合。Jev 的生成步数为零。输入提示词与候选动作后，底层网络只执行单次前向传播，输出在数学上只是一次矩阵乘法的副产物。

### 两条社区复现路径

开源社区在发布两小时内就跑通了两条复现路径，都不需要重写注意力机制，直接在开源小模型上就能运行。

**路径一：候选词 Logits 掩码投影**

社区项目 `harshatheg/Qwen-2.5-1B-RLCD` 验证了这种思路：模型在单次前向推导后，直接提取序列最后一个位置的 Logits，仅筛选出候选标签对应的 Token ID，做局部 Softmax 归一化计算置信度。这也是过往做约束输出和意图路由时常用的方法。

```python
# harshatheg/Qwen-2.5-1B-RLCD logits projection
logits = model(input_ids).logits[:, -1, :]
candidate_logits = logits[:, candidate_token_ids]
probs = torch.softmax(candidate_logits, dim=-1)
```

**路径二：基于 NLI 的交叉编码器架构**

项目 `AlexWortega/openjev` 选用 Qwen 系列作为底座，把输入上下文作为前提（Premise），候选动作作为假设（Hypothesis）。分类头直接输出蕴含、中立与矛盾三分类得分，取蕴含概率做决策。

```python
# AlexWortega/openjev cross-encoder scoring
inputs = tokenizer(premise=state, hypothesis=action, return_tensors="pt")
logits = cross_encoder(**inputs).logits
entailment_score = logits[:, ENTAILMENT_IDX]
```

### KV Cache 前缀共享

如果单次前向每次只能评估一个选项，面对高并发多选项时依然不够快。Jev 依靠 Decoder-only 架构的 KV Cache 前缀缓存共享来化解瓶颈。面对长文本上下文，模型在 Prefill 阶段只计算一次并存入显存，后续几十个候选选项共享同一份前缀缓存指针，只并行计算各自少量 Token 的注意力。这使得评估几十个维度的总耗时几乎等同于单次评估。

### RLCD 置信度校准

普通小模型为什么不能直接套用这套逻辑？熟悉机器学习的朋友清楚，Softmax 输出的分数只是指数归一化的相对值，不等于真实概率。未经专门校准的模型普遍存在严重的过度自信，即使预测完全错误，Softmax 也可能给出 0.99 的高分。工业级安全拦截系统不能直接依赖这种裸露数值。

Jev 引入了面向校准决策的强化学习（RLCD）。该训练方式不再关注生成句子的语言文采，而是针对预期校准误差（ECE）进行优化。简单来说，就是通过样本校验与惩罚，确保模型输出 0.8 的置信度时，在统计上真实对应约 80% 的准确率。

### 残留的结构特性

即便经过概率校准，也无法完全消除底层注意力的结构特性：

**无关选项的干扰。** Archer Hume 的实验显示，在既有的四个有效选项后追加一个无关的"天气"选项，原有最优选项的优势 log-odds 会明显下滑。这说明候选项之间依然存在注意力交互，模型内部本质上是在做列表排序，选项并不是互相独立的。

**选项的物理顺序敏感性。** 由于自回归模型的单向注意力机制，排在后面的 Token 能看到前面的所有上下文。如果参考依据出现的位置发生变动，或者选项顺序调整，判断结果就会产生波动。只有把依据放在共享的 Prompt 上下文（State）中，注意力分配才会更加均衡。

### 与 BERT 分类器的区别

有人会问，这和 2018 年的 BERT 分类器有什么区别？区别在于底座的常识与上下文容量。BERT 的上下文窗口通常只有 512 Tokens，词表较小，无法理解长篇代码、系统日志与长篇业务文档。Jev 建立在经历海量预训练的现代因果 Transformer 底座上，具备现代语义理解能力，只是摘掉了自回归生成的环节。

## 二十行代码实测

脱去概念包装，零 Token 生成机制在本地用 20 行 Python 脚本就能跑通。

### 本地脚本

我们加载参数量只有 5 亿的开源小模型 Qwen2.5-0.5B-Instruct，给它一段带有候选标签的客服分类 Prompt。运行时绕过自回归生成循环，仅执行一次前向推导，提取末位对应候选标签的输出并做归一化：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

prompt = "判断用户意图：'我想退货'。选项：[投诉, 咨询, 售后]"
candidate_words = ["投诉", "咨询", "售后"]
# 演示假定每个标签对应单个 Token
candidate_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in candidate_words]

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    logits = model(inputs.input_ids).logits[0, -1, candidate_ids]
    probs = torch.softmax(logits, dim=-1)

for word, prob in zip(candidate_words, probs):
    print(f"{word}: {prob.item():.4f}")
```

如果让同一个 0.5B 模型按常规方式生成 JSON，需要自回归吐出数十个 Token，耗时通常在一两秒左右，且需要额外做 JSON 解析容错。直接提取末位 Logits 的耗时则可以压到 30 毫秒以内。

### 云端调用

在边缘计算场景中，通过 Cloudflare Workers 或 REST 接口调用 `typesafe/jev`，形态同样很精简：

```js
const result = await env.AI.run("typesafe/jev", {
  state: ticketContent,
  questions: {
    is_urgent: {
      type: "noul",
      instructions: "Does this ticket require urgent attention?",
    },
    department: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: {
        billing: "Payments and refunds",
        technical: "Bugs and outages",
        sales: "Pricing and accounts",
      },
    },
    frustration: {
      type: "score",
      instructions: "How frustrated is the customer?",
      criteria: ["Calm", "Frustrated", "Very angry"],
    },
  },
});
```

### 本地与云端的差异

跑完本地脚本与云端调用，技术层面的差异便清晰显现出来。

截取本地小模型的末位 Logits 虽然能拿到相对分数，但未校准的 Softmax 分数存在虚假自信问题。当模型遇到不确定的样本时，由于 Softmax 的指数放大效应，微小的 Logits 差距也会被拉大为 95% 以上的高置信度。

商业服务的主要工作，正是通过 RLCD 训练或大规模样本校准，将置信度与真实准确率拉齐。如果只是为了追求运行速度，本地部署小模型截取 Logits 已经足够可用；但如果要依赖置信度数值去做自动化拦截与放行，就需要自己在领域数据集上做温度缩放（Temperature Scaling）或后验校准，避免盲目放行错误分类。

## 压测数据分析

脱离真实环境的宣传指标往往不够客观。Archer Hume 针对 Jev 官方接口发起了上万次测试，记录了其在长文本与高并发场景下的实际表现。

### 长文本扩展测试

测试脚本向同一个长文本上下文投喂单个问题，记录端到端耗时：

| 输入上下文（Tokens） | 中位数延迟 | 极速延迟 |
| ---: | ---: | ---: |
| 360 | 57.5 ms | 44.0 ms |
| 9,796 | 89.0 ms | 81.0 ms |
| 29,835 | 218.0 ms | 211.0 ms |

当输入从 360 Tokens 增加到近 30,000 Tokens 时，中位数耗时从 57.5ms 增加到 218ms。文本量增长了约 80 倍，延迟仅增加约 160ms。这符合非自回归模型的特征：没有长文本逐字解码的累加耗时，主要耗时都在一次性的 Prefill 阶段。

### 高并发问答测试

固定短文本上下文，将并发问题数量从 1 逐步拉升至 1,500 个：

| 并发问题数量 | 中位数延迟 | 表现特征 |
| ---: | ---: | --- |
| 1 - 100 | 70.0 - 100.0 ms | 耗时相对平稳 |
| 500 | ~240.0 ms | 延迟温和上升 |
| 1,500 | 610.0 ms | 出现排队与积压 |

从 1 个问题增加到 100 个问题，延迟基本保持在 70ms 到 100ms 区间。当并发激增至 1,500 个时，耗时上升到 610ms。数据反映出底层复用了 KV Cache 共享前缀，不需要为公共上下文重复做计算。

### 判断质量对比

速度快并不代表判断质量完美。Richard Bäcker 在开源项目 `rorshopping/jev-on-a-laptop` 中，使用 Qwen2.5-7B 裸 Logits 与 TypeSafe 公布的工作流题目进行了对比：

TypeSafe 评测里，Jev 与前沿模型共识的一致性是 86.6%。同一批题目上，本地 Qwen2.5-7B 只有 73.8%。实测也证实了未校准模型的盲目自信问题：当 Qwen 预测错误时，输出的 Softmax 置信度依然经常高于 0.90。

```python
# https://github.com/rorshopping/jev-on-a-laptop
probs = torch.softmax(qwen_logits[:, target_token_ids], dim=-1)
pred_idx = torch.argmax(probs, dim=-1)
confidence = probs.max(dim=-1).values
# 预测错误时 confidence 仍常高于 0.90
```

### 输出计费口径

关于输出 Token 的计费方式：Jev 官方接口沿用了行业通用的"输出 Token"计费口径。但从技术机制来看，底层只执行了单次前向计算，并不存在自回归逐字生成。最终返回的几个标签字节被折算成虚拟 Token，本质上是厂商为了迎合开发者熟悉的计费习惯，按次计算服务成本即可。

## 失败模式

单步分类模型在特定场景下有清晰的局限性。

### 缺少思维链

单步前向推导无法完成多层因果逻辑推理。在钓鱼邮件基准测试 `anisselbd/jev-phishing-bench` 中，面对包含多层转折与伪造身份的诱骗邮件，支持思维链推理的 Claude Haiku 判定准确率明显优于 Jev。Jev 出现了较多的漏判。

原因在于计算路径被压缩为一次矩阵乘法。大模型依赖自回归逐步推导来构建中间逻辑支撑，而单步打分必须一次性输出结论。一旦语义欺骗嵌套超过两层，单步模型往往力不从心。

```python
# 单步评估缺乏中间推理步骤，多步逻辑推导容易失准
decision = jev.evaluate(email_body, primitive={"type": "noul", "question": "Is this phishing?"})
```

### 选项顺序偏差

候选项的排列顺序会影响最终判定。Archer Hume 在提示词扰动测试中发现，当参考依据放在候选项之后时，正确率维持在较高水平；一旦将依据调整到开头或中间，正确率便会出现明显下滑。

这是因为自回归模型的单向注意力机制决定了靠后的 Token 能完整聚合前序信息。如果线索出现在序列最前方，其注意力权重在长文本中容易被稀释。因此在设计 Prompt 时，依据不要塞进选项列表的开头或中间。能放进共享 State 最好；只能放进选项时，放在列表末尾。

### 无关项干扰

在多选一测试中，向枚举列表中加入一个完全无关的选项（如"天气"），原有业务选项之间的相对对数几率比（log-odds）会出现下滑。

这说明候选项之间并非完全独立打分。模型底层的全量注意力计算会让候选池内部产生竞争，无关选项同样会参与注意力权重分配。在对统计分布要求严苛的场景下，需要对候选集做严格清洗。

### 未校准风险

如果直接从自建开源小模型中提取未校准的裸 Logits 作为置信度，存在明显的安全隐患。本地 Qwen2.5-7B 裸读出的 Logits 与 TypeSafe 工作流参考答案的一致性仅约 73.8%。

如果业务网关依据高置信度直接自动放行，只在低置信度时转人工复核，未校准模型的虚高置信度就会导致错误分类直接穿透网关，自动化防护机制形同虚设。

### 无文本生成

Jev 没有自回归解码通道，无法生成连贯的自然语言句子。它不能承担异常日志总结、代码编写或客服回复等生成任务。它的定位是流水线上的判别员。如果系统在拦截请求后需要向用户解释拦截原因，需要把上下文交给后续的生成模型。

### 单步局限

这些局限的共同根源在于单步前向计算的固定容量。单个前向网络只能在固定深度的矩阵变换中处理特征。如果业务需要多步逻辑推导、长文本生成或严格的独立概率评估，单步模型无法替代自回归大模型。

## 四种设计模式

在实际工程落地中，单步判别模型通常需要与规则引擎及大模型组合使用。以下是四种常见的落地模式。

### 推测性扇出

面对数千字的用户工单或长文档，系统往往需要同时提取多个维度的标签。如果用传统大模型挨个提问，耗时与费用都会线性增加。

推测性扇出模式利用了前缀缓存共享机制：长文本上下文在显存中只做一次 Prefill 计算，后续十几个离散问题并发挂载在同一份缓存上，各自执行轻量前向计算。多维度的提取耗时被压缩到接近单次评估水平。

```js
// 共享前置长文本 KV Cache，一次网络往返并发求值多维标签
const profile = await jev.evaluateBatch({
  context: rawTicketContent,
  questions: { is_urgent: "noul", sentiment: "score", dept: { type: "choice", options: DEPTS } },
});
```

约束先行：并发提问的子问题在语义上必须相对独立，问题 B 依赖问题 A 的结果时无法放入同一批次。这条约束不是 API 的武断规定，而是注意力结构的直接推论，推导见[为什么分支必须相互独立](#为什么分支必须相互独立)。下面把这套机制拆开讲清。

#### 补课：KV Cache 与前缀共享

Decoder-only Transformer 的每个 Token 在前向时会投影出三组向量：Query、Key、Value。因果注意力的单向性决定了 Token 只能看到自己及之前的 Token。由此有两个直接推论：

- **解码端**：自回归生成时，历史部分的 K/V 不随新 Token 变化。把它们缓存进显存（即 KV Cache），每生成一个新 Token 只需计算它自己的 Q/K/V，再去查询缓存，不必重算全文。
- **判定端（Jev 的场景）**：单步判定没有解码循环，但长上下文的 Prefill（把全文 K/V 算出来）依然是最重的一步。一旦多个请求共享同一段前缀，这段前缀的 K/V 只需计算一次，后续请求直接挂载同一份缓存。

推测性扇出利用的正是第二点。序列布局长这样：

```text
共享上下文 State（L 个 Token）
┌────────────────────────────────┐
│ t₁ t₂ t₃ … t_L                 │  Prefill 一次，K/V 写入显存，
│                                │  之后只读不写
└──────┬─────────┬─────────┬─────┘
       │读前缀K/V │读前缀K/V │读前缀K/V
   ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
   │ Q₁后缀│ │ Q₂后缀│ │ Q₃后缀│   各自十几个 Token，并行前向，
   │  … → ■│ │  … → ■│ │  … → ■│   分支之间互不可见
   └───────┘ └───────┘ └───────┘
    末位Logits 末位Logits 末位Logits
    →候选Softmax→候选Softmax→候选Softmax
```

每个分支的注意力覆盖范围 = 前缀 + 自己的后缀；分支与分支之间没有注意力边。 ■ 是末位 Token，单步判定只取它的 Logits。

#### 成本模型：究竟省在哪里

设公共上下文长 L 个 Token，每个问题的后缀长 m 个 Token，共 N 个问题。以 Token 前向量近似计算量主项（FFN 占大头）：

| 方案 | Prefill 量 | 后缀量 | 合计 |
| --- | ---: | ---: | ---: |
| 逐题全量编码（无共享） | N × L | N × m | N·(L + m) |
| 前缀共享扇出 | L | N × m | L + N·m |

代入接近[压测数据分析](#压测数据分析)的量级：L = 8,000、m = 20、N = 20。

- 无共享：20 × 8,020 = 160,400 个 Token 前向；
- 共享：8,000 + 20 × 20 = 8,400 个 Token 前向。

差距约 19 倍。注意力里还有 O(L²) 的二次项，无共享时要付 N 次，共享后只付 1 次，实际差距只会更大。

更关键的是延迟结构：无共享要么串行跑 20 次 Prefill（延迟乘 20），要么占 20 份算力硬并行；共享版只有一次重 Prefill，其余 20 个轻量后向塞进同一个 batch，墙钟时间接近"一次 Prefill + 一次小前向"。这正是压测里并发从 1 拉到 100 个问题、延迟始终停在 70 到 100ms 的底层解释：公共上下文根本没有被重复计算。

#### Python 示意：一次 Prefill，多路探测

用 transformers 复现这套机制。对照组逐题重新编码全文；实验组只 Prefill 一次，每个问题在拷贝出的前缀缓存上只前向后缀：

```python
import copy
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
).eval()

SHARED_CONTEXT = "客户工单正文。" * 1500  # 撑出约 6k Token 的长上下文
QUESTIONS = {
    "is_urgent":  ("问：该工单是否紧急？选项：",   ["紧急", "一般", "不急"]),
    "sentiment":  ("问：客户当前情绪如何？选项：", ["平静", "不满", "愤怒"]),
    "department": ("问：应由哪个团队跟进？选项：", ["账单", "技术", "销售"]),
    "refund":     ("问：是否涉及退款？选项：",     ["是", "否", "不确定"]),
}

def classify(logits, candidate_words):
    # 与「二十行代码实测」同一约定：演示假定每个标签对应单个 Token
    cand_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in candidate_words]
    probs = torch.softmax(logits[cand_ids], dim=-1)
    return dict(zip(candidate_words, [round(p, 4) for p in probs.tolist()]))

def forward(ids, past_kv=None):
    with torch.no_grad():
        out = model(ids, past_key_values=past_kv, use_cache=True)
    return out.logits[0, -1, :]  # 单步判定：只取末位 Logits

# ---- 对照组：逐题全量编码，公共上下文被 Prefill 了 N 次 ----
def naive():
    t0 = time.perf_counter()
    for q_text, cands in QUESTIONS.items():
        ids = tokenizer(SHARED_CONTEXT + q_text, return_tensors="pt").to(model.device).input_ids
        classify(forward(ids), cands)
    return time.perf_counter() - t0

# ---- 实验组：Prefill 一次，N 路后缀共享同一份前缀 K/V ----
def fanout():
    t0 = time.perf_counter()
    ctx_ids = tokenizer(SHARED_CONTEXT, return_tensors="pt").to(model.device).input_ids
    with torch.no_grad():
        # 第 1 步：公共上下文只 Prefill 一次，产出可复用的 KV 前缀
        prefix_kv = model(ctx_ids, use_cache=True).past_key_values

    results = {}
    for name, (q_text, cands) in QUESTIONS.items():
        # 后缀前向会向传入的缓存追加新 K/V，污染前缀。
        # demo 用深拷贝隔离每个分支；生产引擎按块共享，见下文
        branch_kv = copy.deepcopy(prefix_kv)
        # 第 2 步：每个问题只前向自己的后缀 Token，读同一份前缀 K/V
        suf_ids = tokenizer(q_text, return_tensors="pt").to(model.device).input_ids
        results[name] = classify(forward(suf_ids, past_kv=branch_kv), cands)
    return time.perf_counter() - t0, results

t_naive = naive()
t_fanout, results = fanout()
print(f"逐题全量编码: {t_naive * 1000:7.1f} ms")
print(f"前缀共享扇出: {t_fanout * 1000:7.1f} ms")
print(results)
```

跑通后通常能看到实验组比对照组快数倍，且分支数量增加时，实验组只线性增加轻量后缀的开销，对照组线性叠加的是整段 Prefill。

三个诚实的注脚：

1. `copy.deepcopy(prefix_kv)` 会把整段前缀 K/V 在显存里复制一份。Qwen2.5-0.5B 用 GQA，fp16 下每 Token 的 K/V 约 12 KB（24 层 × 2(K,V) × 2 个 KV 头 × 64 维 × 2 字节），6k Token 约 72 MB。demo 只有 4 路才扛得住，分支上百路时显存直接翻倍爆炸。
2. 真正的"并发"应把 N 个后缀填充成一个 batch 一次前向。手写 HF 的胶水代码较繁琐（前缀 K/V 要沿 batch 维复制、后缀要左填充对齐），生产环境直接交给推理引擎，不必手写。
3. 候选词被切成多 Token 时的处理与[二十行代码实测](#二十行代码实测)相同，取首 Token 仅作示意。

#### 为什么分支必须相互独立

回到开头的约束。因果注意力里，后缀 Q₂ 的可见范围是"前缀 + Q₂ 自己"，它永远看不到 Q₁ 的后缀。换句话说，一轮扇出内的分支是平行世界，问题 B 想引用问题 A 的答案，在注意力结构上就不可能成立。

依赖只能靠"再来一轮"解决：第一轮扇出出结果，把需要传递的结论拼进新的 State，再发起第二轮扇出。轮内并行、轮间串行。这也是"推测性"三个字的来历：先赌各维度独立，把所有探测一次性并行发出；赌错了就多付一轮往返的延迟。

```python
# 示意：两轮扇出，第二轮依赖第一轮的结论
round1 = fanout(ctx, questions={"is_urgent": ..., "department": ...})
if round1["is_urgent"]["紧急"] > 0.6:
    state2 = ctx + f"\n已知该工单紧急，归属{max(round1['department'])}团队。"
    round2 = fanout(state2, questions={"sla_minutes": ..., "escalate_to": ...})
```

#### 生产引擎与 evaluateBatch

生产系统不会用 deepcopy，那会把前缀 K/V 复制 N 份。推理引擎在块（block）粒度做共享，分支之间只差一层指针表：

- **vLLM**：开 `--enable-prefix-caching`。PagedAttention 以默认 16 Token 一个 block 管理 K/V，多个请求命中相同前缀时共享 block，写时复制。
- **SGLang**：RadixAttention 用基数树组织历史前缀，命中即复用。
- 托管服务同理。Jev 的 `evaluateBatch` 本质就是把"一次 Prefill + N 路后缀并发前向"封装成一次网络往返。

在自建网关（ReflexGate 方案）里，挂一块开了前缀缓存的 vLLM，就能把本节 demo 换成生产可用的形态：长上下文 Prefill 一次常驻，后续每条请求只付后缀前向的成本。

### 置信度门控

门控模式的做法是：让预测结果决定业务动作，让置信度决定自动化等级。

- 置信度高于 0.90 的请求直接自动执行；
- 置信度在 0.60 到 0.90 之间的请求暂缓执行，异步记录日志并触发抽检；
- 低于 0.60 时判定为不确定区域，直接转交人工或更强模型处理。

```js
// 动作由 answer 指引，执行权限由 confidence 分级裁决
if (result.confidence >= 0.90) return autoExecute(result.answer);
if (result.confidence >= 0.60) return logAndAlertAsyncReview(result.answer);
return circuitBreakToHuman(requestId);
```

高并发内容审核是典型场景。绝大多数确信合规或确信违规的内容可以瞬间处理，只有处于模糊地带的请求才进入人工复核池。前提是模型具备可靠的概率校准，阈值需要根据误杀与漏放的容忍度定期调整。

#### 置信度从哪来，凭什么能当闸门

单步判定的置信度，就是候选 Logits 掩码 Softmax 后的归一化值。它衡量的只是模型内部对几个候选的相对偏好，天然不等于真实正确率。这在[拆底层](#拆底层)已经说过：未经校准的 Softmax 普遍过度自信，Qwen2.5-7B 判错时照样输出 0.90 以上的置信度。

门控模式的合法性完全建立在校准之上。校准良好的 0.90，含义是长期统计上十次里约九次正确，闸门才有意义。官方靠 RLCD 在训练层面拉齐置信度与准确率；自建模型时，最常用的等效手段是温度缩放，几行代码就能把虚高的尖峰压平。

两条配套铁律：

- 未校准的裸 Logits 不能当闸门。虚高置信度会让错误分类直接穿透网关，防护形同虚设（见[失败模式](#失败模式)的未校准风险）。
- `noul` 的 unknown 与门控互补。unknown 是模型主动弃权，低置信度是系统被动拦截，两个逃生门都留着。

#### 温度缩放：本地模型的校准起点

温度缩放只动输出端的一个标量 T：用 `softmax(logits / T)` 替代 `softmax(logits)`。T 大于 1 会压平过度自信的尖峰。T 在带标注的验证集上拟合，目标是让负对数似然（NLL）最小：

```python
import torch
import torch.nn.functional as F

# val_logits: 验证集每个样本的候选 Logits，形状 [N, K]
# val_labels: 人工标注的正确类别下标，形状 [N]

def fit_temperature(logits, labels):
    """网格搜索让 NLL 最小的温度。T > 1 意味着原始输出过度自信。"""
    return min(
        [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        key=lambda T: F.cross_entropy(logits / T, labels).item(),
    )

def ece(probs, labels, bins=10):
    """期望校准误差：按置信度分桶，加权累计 |实际准确率 - 平均置信度|。"""
    conf, pred = probs.max(dim=-1)
    hit = (pred == labels).float()
    total, out = len(labels), 0.0
    for b in range(bins):
        mask = (conf > b / bins) & (conf <= (b + 1) / bins)
        if mask.any():
            out += mask.sum().item() / total * abs(hit[mask].mean() - conf[mask].mean()).item()
    return out

T = fit_temperature(val_logits, val_labels)
before = ece(torch.softmax(val_logits, dim=-1), val_labels)
after = ece(torch.softmax(val_logits / T, dim=-1), val_labels)
print(f"T = {T:.2f}，ECE {before:.3f} → {after:.3f}")
```

RLCD 在训练损失层面优化 ECE，温度缩放在输出层面做同样的事。这也是[该不该用](#该不该用)选型四问里强调基准标注数据的原因：没有带标注的验证集，T 无从拟合，校准无从验证。

#### 阈值不是拍的：按误杀与漏放的代价选

0.90 与 0.60 不是通用常数，是特定代价结构下的产物。把它写成一个可以重算的决策，而不是抄来的魔法数字：

```python
def best_auto_threshold(conf, hit, cost_wrong_auto=10.0, cost_review=1.0):
    """在验证集上扫描切点。高置信自动执行但判错付 cost_wrong_auto，
    转人工每条付 cost_review，取期望总代价最低的切点。
    conf/hit 为张量：置信度，以及是否判对（bool）。"""
    best_t, best_cost = 0.0, float("inf")
    for t in torch.unique(conf):
        auto = conf >= t
        cost = cost_wrong_auto * (auto & ~hit).sum().item() + cost_review * (~auto).sum().item()
        if cost < best_cost:
            best_t, best_cost = t.item(), cost
    return best_t, best_cost
```

误杀与漏放代价不对称的场景（资金支付、账号封禁）把 `cost_wrong_auto` 调大，最优切点自然右移，更多流量进人工池。第二条 0.60 阈值同理，控制的是抽检带的宽窄：带越宽，人工抽检越多，回流标注也越多。

#### 上线之后：置信度会漂

校准是训练与验证分布上的快照，不是永久属性。线上输入分布漂移后，同一组 Logits 的语义会悄悄变化。

抽检带在这里派上第二用场：它产出的带标注样本就是免费的监控集。盯两个指标：滚动窗口的 ECE（涨了说明概率失真）与自动执行占比（塌了说明分布漂移或阈值过时）。异常时用最新回流数据重跑温度拟合与代价扫描。

```python
# 抽检带回流样本 → 免费的校准监控集
recent = reviewed_samples(window="7d")
if ece(torch.softmax(recent.logits / T, dim=-1), recent.labels) > 0.05:
    alert("置信度失真：重标定温度或阈值")
```

### 复合评分

很多团队尝试让模型直接输出一个百分制的综合质量分，这种做法往往不稳定，细微的 Prompt 扰动就会导致综合分数剧烈漂移。

复合评分的做法是让模型只对单一维度给出 0 到 10 的离散分，加权公式与安全硬规则全部交由后端确定性代码执行。调整业务策略时只需修改本地权重配置，不需要重新调整提示词。

```python
# 模型仅提供单项原子分，加权公式与安全硬规则由确定性代码执行
scores = {k: jev.score(text, metric=k) for k in ["clarity", "depth", "factuality"]}
composite_score = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
is_qualified = composite_score >= 80 and scores["factuality"] >= 6
```

#### 为什么直出综合分不稳定

一次前向是一次模式匹配。让模型输出"85 分"，它并不能在一次矩阵乘法里先算清晰度、再算深度、再做加权，只是在模仿训练语料里评分文本的语气分布。"85"作为答案 Token，没有可拆解的组合语义。

直出综合分还有两宗罪：

- 单一维度的波动直接污染总分。Prompt 里一处措辞变化，整个分布跟着漂，总分跟着震。
- 不可审计。总分不合格时，说不清是清晰度塌了还是事实性塌了，无从改进。

拆成原子分后，clarity 掉了能定位，权重调整变成确定性代码的配置变更，可回滚、可单测、不用碰提示词。这正是原式 `jev.score(text, metric=k)` 每次只问一个维度的原因。

#### 本地近似：分值 Token 上的期望分

沿用[二十行代码实测](#二十行代码实测)加载的 model 与 tokenizer，score 原语可以这样落地：把 0 到 10 的分值当候选 Token 做掩码 Softmax，再取期望：

```python
def local_score(text, ask, levels=list(range(11))):
    """score 原语的本地近似：分值 Token 做候选，掩码 Softmax 后取期望分。
    演示假定每个分值对应单个 Token（"10" 可能被切成多 Token，同前文约定）。"""
    prompt = f"{text}\n{ask}"
    ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    cand_ids = [tokenizer.encode(str(v), add_special_tokens=False)[0] for v in levels]
    probs = torch.softmax(logits[cand_ids], dim=-1)
    w = torch.tensor(levels, dtype=torch.float32, device=probs.device)
    return float((probs * w).sum())
```

取期望而不是 argmax 是有讲究的：模型在 7 分上给 0.7、6 分上给 0.3 时，argmax 只回 7，期望回 6.7。期望保留了犹豫信息，对下游加权更平滑，也与置信度门控的语义对得上。

完整的复合评分闭环：

```python
RUBRIC = {
    "clarity":    "Evaluate writing clarity. Score 0-10:",
    "depth":      "Evaluate technical depth. Score 0-10:",
    "factuality": "Evaluate factual accuracy. Score 0-10:",
}
WEIGHTS = {"clarity": 0.2, "depth": 0.3, "factuality": 0.5}

def evaluate_quality(text):
    scores = {k: local_score(text, ask) for k, ask in RUBRIC.items()}
    composite = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS) * 10  # 归一到百分制
    return {
        "scores": scores,      # 单项分可审计、可单独告警
        "composite": composite,
        "qualified": composite >= 80 and scores["factuality"] >= 6,  # 硬规则一票否决
    }
```

#### 锚点、权重与硬规则

三个让分数站得住的细节：

- **锚点**：给分值区间挂语义。云端 API 里 score 原语的 criteria（如 `["Calm", "Frustrated", "Very angry"]`）就是干这个的：两端与中间各有一个可参照的描述，同一文本跨次打分的漂移会明显变小。本地近似同样建议在 ask 里写清 0 分与 10 分各代表什么。
- **权重可拟合**：有标注数据时不必手拍。最小二乘就能拟合出一组与人工总分最贴近的权重：

```python
import numpy as np
# X: [N 个样本, M 个维度] 的单项分矩阵；y: [N] 人工总分
w, *_ = np.linalg.lstsq(X, y, rcond=None)
WEIGHTS = dict(zip(RUBRIC, w / w.sum()))
```

- **硬规则与权重分层**：安全相关维度不参与加权调和。factuality 走 `>= 6` 的一票否决，权重只表达品质偏好，硬规则守住底线。加权平均会被高分维度拉高，违规内容可能靠"写得很清晰"骗过总分，一票否决堵死这条路。

最后一个陷阱：维度相关性。clarity 与 depth 往往正相关，权重直接相加等于重复计数。选维度时先保证近似正交，或者在拟合权重前对 X 做去相关处理。

### 分层分类

当分类标签数量达到数百甚至数千个时，一次性把全部选项塞进枚举列表会导致注意力稀释。

分层分类参考了树状检索思路：先在顶层大类中选出 Top-K 分支，再沿着胜出的大类细化到二级子类，逐级剪枝收窄。

```python
# 顶层先粗筛出 Top-2 大类，随后沿活跃分支并发检索细分类
top_roots = jev.top_k(product_text, candidates=ROOT_CATEGORIES, k=2)
sub_candidates = [leaf for root in top_roots for leaf in TAXONOMY[root]]
final_leaf = jev.evaluate(product_text, candidates=sub_candidates)
```

例如电商商品归类，先判定属于数码还是户外，再细化到具体类目。实现时建议每层保留二到三个候选分支，避免早期误剪枝导致后续全部走偏。

#### 平铺枚举为什么会稀释

候选列表扩大到百千级，三个问题叠加出现：

1. **注意力稀释**。候选越多，每个候选分到的注意力权重越薄。[失败模式](#失败模式)里加一个无关"天气"选项都能让最优选项的 log-odds 下滑，千级候选下这种竞争是乘性放大的。
2. **输入成本**。1000 个选项连同各自的描述文本，本身就要吃掉几千输入 Token，而且每次请求都要重付一遍。
3. **校准退化**。Softmax 的分母被大量陪跑候选摊薄，Top-1 概率的绝对值系统性变低。置信度门控的阈值语义随之漂移，原来 0.90 的闸门不再对应九成准确率。

#### 树状剪枝的本地实现

沿用前面加载的 model 与 tokenizer。核心是两个动作：每层只在少量候选里比较，以及保枝后做一次全局决赛：

```python
TAXONOMY = {
    "数码": ["手机", "笔记本电脑", "相机", "智能手表"],
    "户外": ["帐篷", "登山鞋", "登山包", "睡袋"],
    "家居": ["沙发", "台灯", "收纳盒", "床垫"],
    "其他": [],  # 兜底分支：胜出则路由给生成模型或人工
}
ROOTS = list(TAXONOMY)

def probe(text, candidates, top_k=1):
    """与「二十行代码实测」同一机制：末位候选 Logits → 掩码 Softmax → Top-K。"""
    prompt = f"商品描述：{text}\n它属于以下哪个类别？选项：[{'、'.join(candidates)}]"
    ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    cand_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in candidates]
    probs = torch.softmax(logits[cand_ids], dim=-1)
    top = probs.topk(min(top_k, len(candidates)))
    return [(candidates[i], p.item()) for i, p in zip(top.indices, top.values)]

def hierarchical_classify(text, keep=2):
    # 第一层：只在大类里比，候选数量级是个位数
    roots = probe(text, ROOTS, top_k=keep)
    # 第二层：把存活分支的叶子合并成一个候选池，一次全局决赛。
    # 不在单枝里钻到底，第一层的小失误由此兜住
    leaves = [leaf for root, _ in roots for leaf in TAXONOMY[root]]
    return {"roots": roots, "leaf": probe(text, leaves, top_k=1)}

print(hierarchical_classify("登山时用的轻量化双层帐篷，铝杆，防雨指数 3000mm"))
```

两个设计决策值得强调：

- **保枝，不钻单枝**。误剪枝的代价不可逆，第一层错一次，后面全链路陪葬。保留 Top-2 再把两枝的叶子合并决赛，把"正确大类差一点排第二"的情况兜回来。
- **与推测性扇出组合**。保留下来的各枝子类枚举互相独立，正好并发挂载在共享前缀缓存上，轮内并行、轮间串行，树检索的延迟被进一步压平。

#### 成本与误差账

| 方案 | 枚举规模 | 输入 Token 量级 | 主要风险 |
| --- | --- | --- | --- |
| 平铺 1000 叶 | 1 轮 × 1000 候选 | 数千 | 注意力稀释、校准退化 |
| 两层树（保 2 枝） | 1 轮 × 4 根（含兜底）+ 1 轮 × ≤8 叶 | 数百 | 层间误差累积 |

- **误差累积**：设每层 Top-1 准确率为 p，两层串联的粗上界是 p²。保枝后上界放宽为"正确叶子所在大类进 Top-2"的概率，决赛又是全局比较，实际衰减远好于 p²。
- **经验边界**：候选二三十个以内，直接平铺一轮最省事也最快；上百再上树；每层分支因子控制在 8 到 16，保 2 到 3 枝。
- **兜底分支**：根层永远留一个"其他"选项。out-of-taxonomy 的输入让它胜出，路由给生成模型或人工，别硬塞进错误叶子。

## 生产架构

在生产架构中，将低时延的判别模型作为前置网关，可以拦截大量不必要的生成请求。

### 混合路由

生产系统的请求在穿过 API 网关后，首先进入单步分类器。路由依据分类标签与置信度将流量分流：常规查询命中预计算或缓存，安全违规直接拦截，只有多步推理任务才唤醒后端的生成式大模型。

### 成本与延迟测算

以日均 10 万次请求的在线服务为例做测算（假设输入均长 800 Token，输出均长 400 Token）：

| 架构方案 | 日均 LLM 调用量 | 日均推理费用 (USD) | 月度总支出 (USD) | 端到端 P50 延迟 | 端到端 P99 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全量 LLM 直连（Claude 3/15 计价） | 100,000 次 | 840.00 | 25,200.00 | 1,800 ms | 4,200 ms |
| 判别模型前置 + LLM 混合路由 | 20,000 次 | 171.36 | 5,140.80 | 70 ms | 2,100 ms |
| 优化幅度 | -80.0% | -79.6% | -79.6% | -96.1% | -50.0% |

如果约 80% 的常规请求能在前置判别层直接分流或命中缓存，透传至大模型的调用量将大幅缩减，既降低了开销，也避免了用户经历无谓的等待。

### 三种网关形态

- **拦截器形态**：在流量入口对恶意输入进行毫秒级预筛，直接切断注入攻击。
- **路由器形态**：识别用户意图，将高难度任务导向重型推理集群，简单闲聊或固定问答分发给本地轻量模型。
- **守卫形态**：在 Agent 执行 Shell 命令或修改敏感配置前进行动态鉴权，对越权操作即刻阻断。

```js
const verdict = await jev.evaluate({
  state: `Command: ${command}; WorkingDir: ${cwd}`,
  primitive: { type: "choice", options: ["allow", "block", "escalate"] }
});
if (verdict.choice !== "allow") throw new SecurityViolationError(verdict);
```

### 生产落地案例

社区中已有一些将该思路落地的开源项目：

- `pi-warden`：拦截 Agent 运行时的文件覆写与终端命令，替代传统的关键词黑名单。
- `pi-jev-auto-mode`：用 Jev 语义审批 Pi 的 bash、write、edit 调用，无法判定时默认阻断。
- `jev-router` / `jev-codex-router`：在入口处解析任务难度，把简单请求分给快模型。

### ReflexGate 方案

把网关决策沉淀为标准化中间件，通常称为 ReflexGate 方案。接入层挂载 1B 到 3B 参数量的本地轻量模型（如 Qwen2.5-1.5B），关闭自回归生成，只执行确定性的单步前向推理。单卡吞吐量大幅提高，网关可以在 10 到 30 毫秒内完成安全检查与意图打分。

```python
with torch.no_grad():
    logits = model(inputs.input_ids).logits[0, -1, candidate_token_ids]
    calibrated_probs = torch.softmax(logits / temperature, dim=-1)
```

### 影子流量校验

分流模型上线前，建议采用影子流量进行验证。将生产流量旁路复制给新分类器，与现有主流程决策做对比。当统计差异在容限范围内且置信度表现稳定后，再切入主链路。

## 生态地图

Jev 发布后，开源社区迅速展开了多维度的复现与改造。从几千万参数的轻量网络到数百亿参数的模型，均有团队进行单步决策适配。

### 复现项目

| 仓库 / 项目 | 底座与架构 | 核心特性 |
| --- | --- | --- |
| harshatheg/Qwen-2.5-1B-RLCD | Qwen2.5-1.5B | 并行约束解码（PCD），免重新训练，单步输出离散分布 |
| AlexWortega/openjev | Qwen3.5-4B / 35B MoE | 基于 NLI 序列分类，提供 Qwen3.5-35B-A3B MoE 配套方案 |
| heman10x/rlcd-modernbert-151m | ModernBERT 151M | 支持 25 个候选槽位，延迟小于 35ms，适配边缘推理 |
| Mapika/decider-2b | Qwen3.5-2B-Base | 基于数十万标注样本完成全参数微调 |
| TheoLeeCJ/openjev ([openjev.com](http://openjev.com)) | MiniCPM5-2B-GGUF | 浏览器端纯本地 WebGPU 运行，无需后端服务 |
| vinnylarouge/jevlike | 微型字节编码器 | 采用交叉注意力解耦选项间干扰 |
| DavidHatley/system-one-mini | 自研紧凑网络 | 约 69M 参数量，验证低算力下的单步反射能力 |

### 应用方向

在具体应用场景上，社区主要集中在以下方向：

- **GUI 自动化**：如 `browser-use/jev-ultrafast` 与 `droidrun/mobile-jev`，将屏幕状态判定与动作选择交给单步决策，仅在需要生成长文本时调用生成模型。
- **安全网关**：如 `pi-warden`，在 Agent 调用命令前做权限判定。
- **语义路由**：如 `jev-router`，在入口甄别任务难度并做模型分流。
- **游戏微操**：如 `lukaske/jev-doom-agent`，利用毫秒级响应接管 DOOM 的实时走位。

## 该不该用

在做架构设计时，首先要明确业务究竟需要的是快速的直觉判断，还是深度的因果推演。很多系统为了做一次枚举分类，让数百亿参数的模型去逐字生成，本质上是算力错配。

### 选型四问

1. **请求是否高度可枚举？** 如果绝大多数请求都在预设类别内，直接用单步判别模型更经济；如果是开放长尾输入，则需要大模型。
2. **标签体系是否稳定？** 频繁变更的分类标签需要动态传入候选集；长期稳定的标签更适合离线轻量化微调。
3. **分类错误的容错成本多高？** 如果容错率低（如涉及资金支付），建议配规则引擎兜底或人工复核。
4. **是否有基准标注数据？** 缺少基准数据时很难检验分类与校准效果，贸然自建模型容易过拟合。

### 三种范式权衡

| 评估维度 | 传统 BERT 分类器 | 生成式大模型 (LLM) | Jev / 单步判别模型 |
| --- | --- | --- | --- |
| 运行时灵活性 | 较低（标签固定，增删需重训） | 极高（提示词自由定义输出） | 较高（运行时动态传入候选选项） |
| 通用常识储备 | 较弱（依赖垂直域监督信号） | 极丰富（海量通用常识） | 丰富（继承大模型预训练常识） |
| 上下文长度 | 短（通常 512 Tokens） | 极长（可达数十万 Tokens） | 中长（依赖底座长文本缓存） |
| 推理延迟 | 极快（通常小于 20ms） | 较慢（数百毫秒至数秒） | 快（20ms 至 200ms） |
| 输出形式 | 固定分类概率分布 | 生成文本（易产生格式波动） | 结构化离散概率直出 |
| 置信度质量 | 需额外标定 | 原始 logits 虚高较普遍 | 经校准训练后较为可信 |
| 深度推理能力 | 无 | 极强（支持多步思维链） | 弱（仅限单步模式匹配） |

### 适用场景

适合使用单步判别模型的场景：

- 入口处的意图识别与路由分发
- 高并发内容合规预筛
- 敏感 API 与终端命令权限审查
- 流程中固定的枚举状态流转

### 不适用场景

不适合使用的场景：

- 需要向用户直接输出自然语言解释
- 依赖两步以上因果推理的任务
- 涉及多层伪装的安全风控
- 缺乏明确候选选项的开放式生成

### 两点工程建议

第一，如果分类准确率要求极高且依赖多步推导，建议直接使用具备思维链的大模型。单步前向推导缺少中间思考过程，遇到多层转折容易失准。

第二，冷启动阶段如果缺少标注数据，可以先用通用大模型的 Few-shot 提示词跑通业务流程，顺带沉淀真实访问日志。等数据积累到一定规模并完成标注后，再考虑用单步判别模型做替换或蒸馏。

## 参考资料

### 官方与文章

- **Introducing System One Models and Jev**：TypeSafe 官方发布说明，含定价、延迟与 RLCD
- **Jev's Architecture Unmasked**（Archer Hume）：对官方接口的延迟、选项顺序与无关项压测
- 原文体验：<https://jev.kuhung.me/>

### 仓库列表（已克隆至 repos/，实测研究笔记见 [jev-ecosystem-research.md](jev-ecosystem-research.md)）

以下描述均经本地克隆核验，与上文生态地图的原文名录相比已修正若干失实之处。

**模型与训练复现**

- **AlexWortega/openjev**（HF）：Qwen3.5 NLI 交叉编码器，0.8B / 4B / 35B MoE，含共享前缀推理与游戏实测
- **harshatheg/Qwen-2.5-1B-RLCD**（HF）：并行约束解码（PCD）推理应用；实为 PCD 项目改名遗留，无 RLCD 算法
- **heman10x/rlcd-modernbert-151m**（HF）：GLiClass v2 + ModernBERT 151M，24 实选项 + 1 弃权槽，按候选数分桶温度校准
- **Mapika/decider-2b**（HF）：Qwen3.5-2B 全流程复现，三原语齐备，含 calibration-aware RL 阶段与 JevBench 对比
- **DavidHatley/system-one-mini**（HF）：DistilBERT + 5 个固定决策头（69M），README 自认非 Jev 复现
- **jaredpalmer/kev**（GitHub）：指针头 + 块因果分支掩码的架构复现，0.8B/4B/9B 家族，System One API 兼容服务
- **NandhaKishorM/laya**（GitHub）：encoder-only 双向注意力路线（ModernBERT-large 421M），无 KV Cache 概念
- **TianyuCodings/NanoJev**（GitHub）：Qwen3-0.6B 游戏决策复刻，候选集合注意力头；ViZDoom 上反超 Jev
- **TheoLeeCJ/SemIf**（GitHub）：浏览器端单步判定引擎（wllama + WebGPU），即原 openjev.com 项目改名
- **bespokelabsai/nimble**（GitHub）：Bespoke 的全开放 recipe（Qwen3.5-9B + LoRA），未从 Jev 蒸馏

**评测与基准**

- **rorshopping/jev-on-a-laptop**（GitHub）：裸 Qwen2.5-7B 与双前沿模型共识一致性 73.8% vs 86.6% 的评测与切片审计
- **anisselbd/jev-phishing-bench**（GitHub）：2000 封钓鱼邮件双模型基准；Haiku 关闭思维链仍胜 Jev，正则基线 91.8% 超 Jev verdict

**生产守卫与 Agent 集成**

- **DevMortimer/pi-warden**（GitHub）：Pi 运行时守卫，正则地板 + Jev 分层判定，默认 fail-open
- **jomatsu/pi-jev-auto-mode**（GitHub）：Pi 语义审批扩展，每规则双侧阈值带，默认 fail-closed
- **browser-use/jev-ultrafast**（GitHub）：浏览器 Agent，一次请求多问题 + 投机性目标头协议的参考实现
- **droidrun/mobile-jev**（GitHub）：Android Agent，云 API + a11y 结构化状态，无生成模型路径

**生态地图**

- **multimodalart/jev-reproductions-tracker**（HF Space）：社区复现追踪榜，热度指标 + 自报性能，含生态演化时间线
- **AnotiaWang/awesome-jev**（GitHub）：100+ 条目双语生态索引，含官方 jaggedness 失败模式页与 ASSAY-001 校准检验
