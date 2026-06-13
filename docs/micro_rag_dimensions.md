# micro_rag.py 维度变化详解

## 核心超参数

| 参数 | 值 | 含义 |
|------|-----|-----|
| `EMBED_DIM` | 32 | 每个 token 嵌入向量的维度 |
| `BLOCK_SIZE` | 64 | 模型能处理的最大上下文长度（token 数） |
| `VOCAB_SIZE` | ~20+ | 字符级词表大小（从 knowledge_base 去重得到） |
| `TOTAL_VOCAB` | VOCAB_SIZE + 3 | 含特殊 token（SEP, PAD）的总词表大小 |

---
## 一、输入 → Embedding

```
输入文本: "太阳有多热？"
         ↓ encode()
token ids: [12, 5, 23, 41, ...]  (每个元素是 0~VOCAB_SIZE 的整数)

序列长度 seq_len = 6 ，这个就是token数量啦
```

**Embedding 层**（`wte`）将整个序列映射为矩阵：

```
token_ids [seq_len]     wte [EMBED_DIM, TOTAL_VOCAB]
      ↓                           ↓
X = embed(token_ids)  →  [seq_len, EMBED_DIM]  矩阵
```

查表操作：`wte` 形状为 [EMBED_DIM, TOTAL_VOCAB]，对每个 token id，取 `wte[e][token_id] for e in 0..31` 作为该位置的 embedding [32]。

加上**位置编码**（`wpe`）后，维度不变：

```
X[pos] [EMBED_DIM] + pe[pos] [EMBED_DIM] = [EMBED_DIM]
```

整个序列：`X [seq_len, EMBED_DIM] + P [seq_len, EMBED_DIM] = [seq_len, EMBED_DIM]`

---

## 二、单层自注意力中的维度变化

### 2.1 Q, K, V 投影

```
对每个 token：x_i [32] @ Wq^T [32, 32] = q_i [32]
全序列：X [seq_len, 32] → Q [seq_len, 32]（逐 token 矩阵乘法）
```

即代码中 `linear(x, sd['wq'])` 对每个位置独立做矩阵乘法。

### 2.2 注意力分数计算

代码是**逐位置**计算每个位置 i 与历史位置 t ≤ i 的分数：

```
对位置 i：scores_i[t] = Q[i] · K[t] / sqrt(32),  for t = 0..i
         len(scores_i) = i + 1
```

形式上等价于 `Q @ K^T / sqrt(32)` 加 causal mask，但代码存储为下三角结构。

### 2.3 Softmax（按位置）

对每个位置 i：`weights_i = softmax(scores_i)`，长度 = i + 1，和为 1。

### 2.4 注意力输出

```
对位置 i：out_i = Σ_{t=0}^{i} weights_i[t] * V[t]  → [32]
其中 V[t] 是位置 t 的 value 向量 [32]
全序列：attn = [out_0, out_1, ..., out_{L-1}]  → [seq_len, 32]
```

形式上等价于 `weights @ V`，但代码是逐位置计算。

### 2.5 完整的注意力过程

代码 main.py:270-275 是**逐位置**计算的（数学上等价于矩阵乘法）：

```python
Q = linear(x, sd['wq'])  # [seq_len, 32]，对每个token独立做矩阵乘法
K = linear(x, sd['wk'])  # [seq_len, 32]
V = linear(x, sd['wv'])  # [seq_len, 32]

attn = []
for i in range(seq_len):
    # 当前位置 i 与历史位置 0..i 的注意力分数
    scores_i = [sum(Q[i][k] * K[t][k] for k in range(32)) / sqrt(32)
                for t in range(i + 1)]
    weights_i = softmax(scores_i)  # [i+1]，归一化权重

    # 加权求和 v[t]
    out_i = [sum(weights_i[t] * V[t][j] for t in range(i + 1))
             for j in range(EMBED_DIM)]  # [32]
    attn.append(out_i)

attn = linear(attn, sd['wo'])  # [seq_len, 32]
x = [x[i] + attn[i] for i in range(seq_len)]  # 残差连接
```

每个位置 i 的输出 `out_i` 是 [32]，与 Q/K/V 的维度一致。

---

## 三、MLP 中的维度变化

MLP 将维度扩展再收缩，实现"宽宽窄窄"的结构：

```
# 对单个位置 pos：
h_pre  = x[pos] [32]  @ fc1^T [32, 128]  = [128]
h_act  = relu(h_pre)                   = [128]
h_post = h_act    @ fc2^T [128, 32]     = [32]
x[pos] = x[pos] + h_post               = [32]  (残差)
```

对全序列循环执行上述过程，最终 `x` 为 [seq_len, 32]。

其中 `fc1 [128, 32]` 表示权重矩阵形状（nout=128, nin=32），`fc1^T [32, 128]` 是其转置。

---

## 四、LM Head（解码器）

```
logits = x [seq_len, 32]  @  lm [TOTAL_VOCAB, 32]^T
       = [seq_len, TOTAL_VOCAB]
```

取**最后一个位置**的 logits 预测下一个 token：

| 位置 | 能看到的上下文 | logits 预测的是 |
|------|-------------|---------------|
| 位置 0 | 只能看位置 0 | 下一个 token 是位置1 的概率 |
| 位置 1 | 只能看位置 0, 1 | 下一个 token 是位置2 的概率 |
| ... | ... | ... |
| **位置 n（最后一个）** | **能看到位置 0..n** | **下一个 token 是位置 n+1 的概率** |

这是由 **causal mask** 决定的：每个位置只能看到自己和之前的位置，不能看到未来的位置。位置 n 的视野最大（包含全部历史），因此取 `logits[-1]`。

```
logits[-1] [TOTAL_VOCAB]  →  softmax  →  采样 next_token_id
```

---

## 五、生成过程中的维度流转

以 `generate(input_ids, max_new_tokens=50)` 为例：

```
输入: [12, 5, 23]  (3个token, seq_len=3)
         ↓
┌─────────────────────────────────────┐
│  gpt_forward([12, 5, 23])           │
├─────────────────────────────────────┤
│ X = embed + pos  → [3, 32]          │
│ Q,K,V        → [3, 32]              │
│ scores       → 下三角变长             │
│ attn_output  → [3, 32]              │
│ MLP          → [3, 32]              │
│ logits_all   → [3, TOTAL_VOCAB]     │
│ 取 logits[-1]: [TOTAL_VOCAB]        │
│ softmax + sample → next_token_id    │
│  ↓                                  │
│ tokens = [12, 5, 23, 42]  (+1)      │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  gpt_forward([12, 5, 23, 42])       │
├─────────────────────────────────────┤
│ 重复上述过程，seq_len=4              │
│  ↓                                  │
│ tokens = [12, 5, 23, 42, 17]  (+1)  │
└─────────────────────────────────────┘
         ↓
       ... (重复直到 EOS 或达到 max_new_tokens)
```

---

## 六、维度变化总结

| 阶段 | 输入维度 | 输出维度 | 说明 |
|------|----------|----------|------|
| Token Embedding | `[seq_len]` | `[seq_len, 32]` | 查表 |
| 位置编码相加 | `[seq_len, 32]` | `[seq_len, 32]` | 逐元素相加 |
| RMSNorm | `[seq_len, 32]` | `[seq_len, 32]` | 归一化 |
| Q/K/V 投影 | `[seq_len, 32]` | `[seq_len, 32]` | 矩阵乘法 |
| 注意力分数 | `[seq_len, 32]` | `下三角变长` | Q @ K^T / sqrt(32)，len(scores_i)=i+1 |
| Softmax | `下三角变长` | `下三角变长` | 每组 len=i+1，和为1 |
| 注意力输出 | `下三角变长` | `[seq_len, 32]` | Σ weights_i[t] * V[t] |
| Linear(WO) | `[seq_len, 32]` | `[seq_len, 32]` | 矩阵乘法 |
| 残差连接 | `[seq_len, 32]` | `[seq_len, 32]` | 逐元素相加 |
| MLP FC1 | `[seq_len, 32]` | `[seq_len, 128]` | x @ fc1^T，扩展4倍 |
| MLP FC2 | `[seq_len, 128]` | `[seq_len, 32]` | x @ fc2^T，压缩回原维度 |
| LM Head | `[seq_len, 32]` | `[seq_len, TOTAL_VOCAB]` | 映射到词表 |

---

## 七、BLOCK_SIZE = 64 的限制

`BLOCK_SIZE` 限制了位置编码表的大小，同时也限制了自注意力计算的范围：

```python
sd['wpe'] = matrix(BLOCK_SIZE, EMBED_DIM)  # [64, 32] 位置编码表
```

当输入序列长度超过 64 时：
- 位置 `≥ 64` 的 token 会使用 `wpe[63]`（最大位置）的编码
- 自注意力仍然计算所有位置的加权平均（只是位置信息不够精确）

这是 **context window** 限制，是 micro 版本的简化处理。真实 GPT 的 BLOCK_SIZE 可以达到 1024、2048 甚至 4096。
