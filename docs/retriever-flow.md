## `retrieve` 函数详解

### 一层：入口

```python
def retrieve(question_embed: Sequence[float], top_k: int = 1) -> list[tuple[float, int]]:
```

**输入**：问题的 32 维嵌入向量（纯 float，推理阶段）
**输出**：top_k 个 `(相似度分数, 知识库索引)` 元组

---

### 二层：逐条遍历知识库

```python
for i, (q, a) in enumerate(knowledge_base):
    doc_tokens = encode(a)                        # 答案文本 → token ids
    doc_emb = mean_pool(sd['ret_k'], doc_tokens)  # key 矩阵查表 + mean pooling → [32]
    doc_emb_f = [v.data for v in doc_emb]         # Value → float
    score = cosine_sim_f(question_embed, doc_emb_f)  # cosine 相似度
    scores.append((score, i))
```

核心三步：

1. **encode(a)**：字符级分词，`"水的沸点是多少？" → [token_id_1, token_id_2, ...]`
2. **mean_pool(sd['ret_k'], doc_tokens)**：`[EMBED_DIM=32, TOTAL_VOCAB]` 的 key 矩阵，查表取每个 token 的嵌入向量，然后平均池化成一个 32 维向量
3. **cosine_sim_f**：计算问题向量和答案向量的 cosine 相似度

---

### 三层：mean_pool 的本质

```python
def mean_pool(embed_matrix, token_ids):
    result = [Value(0) for _ in range(EMBED_DIM)]
    for t in token_ids:
        for e in range(EMBED_DIM):
            result[e] = result[e] + embed_matrix[e][t]
    n = Value(len(token_ids))
    return [r / n for r in result]
```

等效于矩阵查表 + 平均：

```
doc_emb = (1/k) * Σ_{i=1}^{k} W_key[:, token_i]
```

即每个 token id 对应嵌入矩阵的一列（one-hot 查表），最后对所有 token 向量取平均。

---

### 四层：与 Generator 的关系

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────┐
│  rag_answer(question)                        │
│                                              │
│  1. encode(question) → token ids             │
│  2. mean_pool(ret_q, token_ids) → [32]      │  ← Retriever: 将问题映射到语义空间
│  3. retrieve(q_emb) → 找到最相关答案索引     │
│  4. 拼接: [question] + [SEP] + [检索到的答案] │
│  5. generate(input_ids) → 自回归生成          │  ← Generator: GPT 自回归生成
└──────────────────────────────────────────────┘
    │
    ▼
生成答案
```

**数据流**：
1. Retriever 用 `ret_q` 将问题嵌入，用 `ret_k` 将答案嵌入，在语义空间做 cosine 相似度检索
2. 检索到的答案拼到 GPT 输入序列的上下文中（格式：`问题 + SEP + 检索答案`）
3. Generator 负责在给定上下文下自回归生成下一个 token

**两者独立训练**：
- Retriever：InfoNCE 对比学习（10 步），让问题和对应答案在向量空间靠近
- Generator：GPT next-token prediction（2000 步），学习语言建模

核心设计：Retriever 负责"找到答案"，Generator 负责"组织语言"，分工明确。

---

## 各函数作用详解

### 1. `encode(text: str) -> list[int]`

**作用**：字符级分词器，将文本转为 token id 列表

**实现**：
```python
def encode(text: str) -> list[int]:
    return [char_to_id.get(c, 0) for c in text]
```

**原理**：
- 每个字符（汉字、英文字母、标点）对应一个唯一的 id
- `"太阳"` → `[char_to_id['太'], char_to_id['阳']]`
- 未出现的字符默认映射到 id=0（vocab 中不存在的字符会被忽略）

**为什么用字符级**：
- microRAG 追求极简，词汇表仅约 60 个字符（所有问答中出现的唯一字符）
- 相比词级分词，字符级无需处理 OOV 问题

---

### 2. `mean_pool(embed_matrix, token_ids) -> list[Value]`

**作用**：将变长 token 序列聚合成一个固定维度的向量

**实现**：见上方第三层详解

**数学本质**：
```
h = (1/k) * Σ W[:, token_i]  其中 k = len(token_ids)
```

**输入输出**：
- 输入：`["太", "阳", "表", "面", "约"]` → 5 个 token id
- 输出：5 个 32 维嵌入向量的平均值 → 一个 32 维向量

**为什么用 mean_pool**：
- 序列长度短（平均 10-20 字符），简单平均即可
- 无需 attention 之类的复杂聚合

---

### 3. `cosine_sim_f(a: Sequence[float], b: Sequence[float]) -> float`

**作用**：计算两个向量的 cosine 相似度（纯 float 版本，用于推理）

**实现**：
```python
def cosine_sim_f(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum((ai * bi for ai, bi in zip(a, b)), 0.0)
    na = sum((ai * ai for ai in a), 0.0) ** 0.5
    nb = sum((bi * bi for bi in b), 0.0) ** 0.5
    return dot / (na * nb + 1e-8)
```

**数学公式**：
```
cosine(a, b) = (a · b) / (||a|| * ||b||)
```

**范围**：[-1, 1]，值越大表示越相似

**训练版本 vs 推理版本**：
- `cosine_sim(Value)`：保留计算图，用于训练时反向传播
- `cosine_sim_f(float)`：纯计算，无梯度，用于推理加速

---

### 4. `rag_answer(question: str) -> str`

**作用**：串联 Retriever 和 Generator 的入口函数

**实现**：
```python
def rag_answer(question: str) -> str:
    # 1. 检索
    q_ids = encode(question)
    q_emb = mean_pool(sd['ret_q'], q_ids)
    q_f = [v.data for v in q_emb]
    _, best_idx = retrieve(q_f, top_k=1)[0]
    retrieved_a = knowledge_base[best_idx][1]

    # 2. 拼接输入
    input_ids = encode(question) + [SEP] + encode(retrieved_a + " ")

    # 3. 生成
    output_ids = generate(input_ids, max_new_tokens=60, temperature=0.8)
    return decode(output_ids[len(input_ids):])
```

**流程**：
1. 问题编码 → 嵌入 → 检索得到最相关答案
2. 拼接 `[问题 + SEP + 检索答案]` 作为 GPT 输入
3. 自回归生成新 token，只返回"新生成"的部分（去掉输入本身）

---

### 5. `generate(input_ids: list[int], ...) -> list[int]`

**作用**：GPT 自回归生成

**实现**：
```python
def generate(input_ids, max_new_tokens=50, temperature=1.0):
    ids = input_ids.copy()
    for _ in range(max_new_tokens):
        logits_all = gpt_forward(ids)
        logits = logits_all[-1]  # 只取最后一个位置的 logits
        if temperature == 0:
            next_id = argmax(logits)
        else:
            probs = softmax([v / temperature for v in logits])
            next_id = sample_from_probs(probs)
        if next_id == PAD or next_id >= TOTAL_VOCAB:
            break
        ids.append(next_id)
    return ids
```

**核心逻辑**：
- 自回归：每次生成一个 token，加到序列末尾
- 停止条件：遇到 PAD token 或超出 vocab 范围
- temperature 控制随机性：0 = 贪婪取最大，>1 = 更高随机性

---

### 6. `gpt_forward(tokens: list[int]) -> list[list[Value]]`

**作用**：GPT 的前向传播，返回每个位置的 logits

**实现**：
```python
def gpt_forward(tokens):
    L = len(tokens)

    # 1. 嵌入 + 位置编码
    x = []
    for i, tid in enumerate(tokens):
        emb = [sd['wte'][e][tid] for e in range(EMBED_DIM)]
        pos = sd['wpe'][min(i, BLOCK_SIZE - 1)]
        x.append([e + p for e, p in zip(emb, pos)])
    x = [rmsnorm(xi) for xi in x]

    # 2. 自注意力（causal mask）
    q = linear(x, sd['wq'])
    k = linear(x, sd['wk'])
    v = linear(x, sd['wv'])

    attn = []
    for i in range(L):
        scores_i = [sum(q[i][j] * k[t][j] ... ) / sqrt(d) for t in range(i+1)]
        weights = softmax(scores_i)
        out_i = [sum(weights[t] * v[t][j] for t in range(i+1)) for j in range(EMBED_DIM)]
        attn.append(out_i)

    attn = [linear_single(a, sd['wo']) for a in attn]
    x = [x[i] + attn[i] for i in range(L)]
    x = [rmsnorm(xi) for xi in x]

    # 3. MLP
    logits_list = []
    for pos in range(L):
        h_pre = linear_single(x[pos], sd['fc1'])
        h_act = [hi.relu() for hi in h_pre]
        h_post = linear_single(h_act, sd['fc2'])
        x_pos = [x[pos][j] + h_post[j] for j in range(EMBED_DIM)]
        logits_list.append(linear_single(x_pos, sd['lm']))
    return logits_list
```

**各模块作用**：

| 模块 | 作用 |
|------|------|
| `wte` | token 嵌入表，将 token id 转为 EMBED_DIM 维向量 |
| `wpe` | 位置嵌入表，位置 i 的向量与内容向量相加 |
| `rmsnorm` | RMSNorm 归一化，稳定训练 |
| `wq, wk, wv, wo` | Q/K/V/O 投影矩阵，构成自注意力 |
| `fc1, fc2` | MLP 的两层全连接 |
| `lm` | 语言模型头，将隐向量映射到 vocab 维度的 logits |

---

### 7. `softmax(logits: list[Value]) -> list[Value]`

**作用**：将 logits 转为概率分布

```python
def softmax(logits):
    max_v = max(v.data for v in logits)
    exps = [(v - max_v).exp() for v in logits]
    total = sum(exps, Value(0))
    return [e / total for e in exps]
```

**为什么要减 max**：数值稳定性，防止 exp 大数溢出

---

### 8. `sample_from_probs(probs) -> int`

**作用**：按概率分布采样一个 token id

```python
def sample_from_probs(probs):
    ps = [p.data for p in probs]
    ps = [max(p, 0) for p in ps]
    total = sum(ps)
    if total == 0:
        return PAD
    ps = [p / total for p in ps]
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(ps):
        cumsum += p
        if cumsum >= r:
            return i
    return PAD
```

---

## Retriever 与 Generator 的完整数据流

```
用户输入: "太阳有多热？"
              │
              ▼ encode()
         [token_id_1, token_id_2, ...]
              │
              ▼ mean_pool(ret_q)
           [32维向量 q_emb]
              │
              ▼ cosine_sim_f(遍历知识库)
        找到最高分答案索引 → best_idx
              │
              ▼
检索结果: knowledge_base[best_idx][1] = "太阳表面约6000度，核心约1500万度。"
              │
              ▼ 拼接输入
input_ids = encode("太阳有多热？") + [SEP] + encode("太阳表面约6000度，核心约1500万度。 ")
              │
              ▼ generate() → gpt_forward() 自回归生成
output_ids = [..., 新生成的token_ids]
              │
              ▼ decode()
         "太阳表面约6000度，核心约1500万度。"（或更完整的回答）
```

---

## 设计决策总结

| 决策 | 理由 |
|------|------|
| Q/K 分离 | 问题和答案语义角色不同，允许独立优化 |
| 字符级 tokenizer | 极简实现，vocab 仅 60 字符，无 OOV |
| mean_pool | 短序列直接平均足够，无需复杂 attention |
| cosine 相似度 | 与训练 loss（margin-based）一致，直接优化检索效果 |
| 推理用 float | 无梯度计算，开销更小 |
