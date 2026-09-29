"""guard 模型 jev 化 case：把微调后的 Qwen3Guard（HK 养老领域）从自回归
生成改造成单次前向判定，复用 person_type_a 的字母槽基础设施。

Phase 1（零样本）对比四套系统：
  (a) 自回归生成基线（SafeGuardElderlyDomain 仓库已有结果）
  (b) Route 1 原生锚定：guard 模板 + "Safety:" 强制前缀，读标签首 token（native.py）
  (c) Route 2a 每问独立标准 MCQ，锚在 prompt 末尾（mcq.py）
  (d) Route 2b 多问合一答题卡式单次前向，哑字母槽 + offset 锚定位（mcq.py）
"""
