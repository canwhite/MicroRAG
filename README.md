# MicroRAG

A minimal Retrieval-Augmented Generation (RAG) system implemented from scratch in pure Python, without any external ML libraries.

**纯Python从零实现的最小可运行RAG系统，无外部机器学习依赖。**

---

## 核心特性 / Core Features

| 特性 / Feature | 说明 / Description |
|----------------|-------------------|
| **Retriever** | 基于对比学习（InfoNCE）的向量检索 / Cosine-similarity retrieval with contrastive learning |
| **Generator** | 单层 GPT，字符级 tokenizer / Single-layer GPT with character-level tokenizer |
| **无外部依赖 / No deps** | 仅用 Python 标准库实现自动微分 / Pure Python autograd with stdlib only |
| **可运行 / Runnable** | 训练 + 推理完整流程，开箱即用 / Full train + inference pipeline, ready to run |

---

## 架构 / Architecture

```
用户问题 → Retriever(检索) → 拼接上下文 → Generator(GPT生成) → 答案
Query      retrieve        concat context   generate           answer
```

### Retriever

- Query 和 Document 分别通过不同的嵌入矩阵映射到 32 维向量
- InfoNCE 对比学习训练：正样本相似度应比负样本高 0.5
- 推理阶段使用 cosine 相似度检索

### Generator

- 单层自注意力（causal mask，无 sliding window）
- 字符级 tokenizer（词表仅几十个 token）
- RMSNorm + MLP + LM Head
- 训练：next-token prediction

---

## 运行 / Run

```bash
uv run main.py
```

输出示例：
```
vocab_size=57, TOTAL_VOCAB=60, EMBED_DIM=32
chars:  ,。0-9,...A...Z...a...z...
num params: 23888
step    0 | loss 0.7368 | sp=0.112 sn=0.431
step  200 | loss 0.0000 | sp=0.999 sn=0.000
step  400 | loss 0.0000 | sp=0.999 sn=0.000
step  600 | loss 0.0000 | sp=0.999 sn=0.000
step  800 | loss 0.0000 | sp=1.000 sn=0.000

--- 训练 Generator ---
gpt step    0 | loss 4.4424
gpt step  500 | loss 0.1389
gpt step 1000 | loss 0.0152
gpt step 1500 | loss 0.0005

--- RAG 完整流程测试 ---
问: 太阳有多热？
答: 太阳表面约6000度，核心约1500万度。
```

---

## 超参数 / Hyperparameters

| 参数 / Param | 值 / Value | 含义 / Meaning |
|-------------|------------|----------------|
| `EMBED_DIM` | 32 | 嵌入维度 / Embedding dimension |
| `BLOCK_SIZE` | 64 | 位置编码表大小（位置 > 64 复用 wpe[63]）|
| `lr` (Retriever) | 0.01 | 检索器学习率 |
| `gpt_lr` | 0.05 | 生成器学习率 |
| `retriever_steps` | 10 | 检索器训练步数 |
| `gpt_steps` | 2000 | 生成器训练步数 |

---

## 概念区分 / Concept Clarification

- **Causal mask**: ✅ 有 / Present — 不看未来位置 / No future tokens
- **Attention window**: ❌ 无 / None — 不限制历史长度 / No historical limit
- **Position embedding truncation**: ✅ 有 / Present — 位置 ≥ 64 复用编码 / Positions ≥ 64 reuse last encoding

详见 [docs/attention_mask_vs_truncation.md](docs/attention_mask_vs_truncation.md)

---

## 局限性 / Limitations

- 字符级 tokenizer，序列极短（几十个 token）
- 单层 transformer，表达能力有限
- 知识库固定为 8 条 QA
- 无真实 attention window 截断（适合短文本）

This is a **learning/demonstration** implementation, not a production system.
