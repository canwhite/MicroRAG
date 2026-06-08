# micro_rag.py 维度变化详解

## 核心超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `EMBED_DIM` | 32 | 每个 token 嵌入向量的维度 |
| `BLOCK_SIZE` | 64 | 模型能处理的最大上下文长度（token 数） |

---

## 一、输入 → Embedding

```
输入文本: "太阳有多热？"
         ↓ encode()
token ids: [12, 5, 23, 41, ...]  (每个元素是 0~VOCAB_SIZE 的整数)
```

**Embedding 层**（`wte`）将每个 token id 映射为 32 维向量：

```
token_id (标量)
     ↓ 查表 wte[*, token_id]
[32] 向量
```

加上**位置编码**（`wpe`）后，维度不变：

```
token_embedding [32] + position_embedding [32] = [32]
```

---

## 二、单层自注意力中的维度变化

### 2.1 Q, K, V 投影

```
x[i] [32]  ──linear──→  q[i] [32]
x[i] [32]  ──linear──→  k[i] [32]
x[i] [32]  ──linear──→  v[i] [32]
```

维度不变：`[32] → [32]`

### 2.2 注意力分数计算

对于位置 `i`，计算与所有历史位置 `t ≤ i` 的注意力分数：

```
q[i] [32] · k[t] [32] = scalar  (点积)
     ↓ 除以 sqrt(32)
score(i,t): scalar
```

结果：`scores_i: list[scalar]`，长度为 `i+1`

### 2.3 Softmax 权重

```
softmax([score(i,0), score(i,1), ..., score(i,i)]) → [0.1, 0.3, ..., 0.6]
```

长度仍是 `i+1`，和为 1。

### 2.4 注意力输出

```python
out_i[j] = sum(weights[t] * v[t][j] for t in 0..i)
```

这是对所有历史位置的 value 向量加权求和：

```
[32] ← sum over t of (scalar × [32])
```

**关键**：输出维度仍然是 `[32]`，与输入一致。

### 2.5 完整的注意力过程（位置 i=2 的例子）

```
假设 i=2，当前要计算位置2的输出
历史位置: t=0, t=1, t=2

Step 1: 计算分数
  q[2]·k[0] = scalar
  q[2]·k[1] = scalar
  q[2]·k[2] = scalar

Step 2: Softmax
  weights = [0.2, 0.3, 0.5]

Step 3: 加权求和
  out[2] = 0.2 * v[0] + 0.3 * v[1] + 0.5 * v[2]
         = [32]

Step 4: 线性变换 + 残差连接
  attn_out = linear(out[2], wo)  → [32]
  x[2] = x[2] + attn_out  → [32]  (残差)
```

---

## 三、MLP 中的维度变化

MLP 将维度扩展再收缩，实现"宽宽窄窄"的结构：

```
x[pos]      [32]
    ↓ linear(wfc1): [32] → [128]  (EMBED_DIM → EMBED_DIM*4)
    ↓ relu
    ↓ linear(wfc2): [128] → [32]  (EMBED_DIM*4 → EMBED_DIM)
    ↓
x[pos] + h_post → [32]  (残差)
```

---

## 四、LM Head（解码器）

```
x[pos] [32]  ──linear──→  logits [VOCAB_SIZE]
```

这是最后一步，将 32 维隐藏状态映射到词表大小，用于预测下一个 token。

---

## 五、生成过程中的维度流转

以 `generate(input_ids, max_new_tokens=50)` 为例：

```
输入: [12, 5, 23]  (3个token)
         ↓
┌─────────────────────────────────────┐
│  第一次生成 (自回归第1步)              │
├─────────────────────────────────────┤
│ tokens = [12, 5, 23]                │
│  ↓ gpt_forward                      │
│ 每个token: 标量 → [32]              │
│ 自注意力: [32] → [32]               │
│ MLP: [32] → [32]                    │
│ LM Head: [32] → [VOCAB_SIZE]        │
│ 取最后一个位置的logits: [VOCAB_SIZE]  │
│ softmax + sample → next_token_id    │
│  ↓                                  │
│ tokens = [12, 5, 23, 42]  (+1)      │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  第二次生成 (自回归第2步)              │
├─────────────────────────────────────┤
│ tokens = [12, 5, 23, 42]            │
│  ↓ gpt_forward                      │
│ 每个token: 标量 → [32]              │
│ 自注意力: [32] → [32]               │
│ MLP: [32] → [32]                    │
│ LM Head: [32] → [VOCAB_SIZE]        │
│ 取最后一个位置的logits: [VOCAB_SIZE]  │
│ softmax + sample → next_token_id    │
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
| Token Embedding | `标量(id)` | `[32]` | 查表 |
| 位置编码相加 | `[32] + [32]` | `[32]` | 相加，维度不变 |
| RMSNorm | `[32]` | `[32]` | 归一化，维度不变 |
| Q/K/V 投影 | `[32]` | `[32]` | 线性变换，维度不变 |
| 注意力分数 | `[32]·[32]` | `标量` | 点积产生标量 |
| Softmax | `list[scalar]` | `list[scalar]` | 概率分布，长度=i+1 |
| 注意力输出 | `sum(scalar×[32])` | `[32]` | 加权求和 |
| Linear(WO) | `[32]` | `[32]` | 线性变换 |
| 残差连接 | `[32]+[32]` | `[32]` | 相加 |
| MLP FC1 | `[32]` | `[128]` | 扩展4倍 |
| MLP FC2 | `[128]` | `[32]` | 压缩回原维度 |
| LM Head | `[32]` | `[VOCAB_SIZE]` | 映射到词表 |

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
