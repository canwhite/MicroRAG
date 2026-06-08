# 注意力掩码 vs 注意力截断 vs 位置编码截断

这三个概念经常被混淆，本文厘清它们的区别。

---

## 1. Causal Attention Mask（因果注意力掩码）

**作用**：保证自回归生成时，每个位置只能看到当前及之前的位置，**不能看未来**。

**代码实现**（main.py line 263-264）：
```python
for i in range(L):
    for t in range(i + 1):  # t ∈ [0, i]，只看过去和现在
        score = q[i] · k[t]
```

**形象比喻**：你在写句子时，只能参考已经写完的词，不能提前看到还没写的词。

**常见误解**：不要把 causal mask 和"截断"混为一谈。mask 只管"方向"（过去 vs 未来），不管"距离"（近还是远）。

---

## 2. Attention Window / Sliding Window（注意力截断 / 滑动窗口）

**作用**：限制每个位置只能看到**最近的 N 个 token**，忽略更早的历史。

**代码实现**（真实 GPT 的 sliding window attention）：
```python
window_size = BLOCK_SIZE  # e.g., 64
for i in range(L):
    start = max(0, i - window_size + 1)
    for t in range(start, i + 1):  # t >= i - 63
        score = q[i] · k[t]
```

**效果**：token 64 能看到 token 1~64，看不到 token 0。

**main.py**：**没有实现这个**。当前是 `range(i + 1)`，即 `start = 0`，永远能看到所有历史。原因是字符级生成序列极短（几十个 token），截断反而有害。

---

## 3. Position Embedding Truncation（位置编码截断）

**作用**：当位置超过预定义的最大值时，复用最后一个位置向量。

**代码实现**（main.py line 252）：
```python
pos = sd['wpe'][min(i, BLOCK_SIZE - 1)]
# 位置 0~63: 用 wpe[0]~wpe[63]
# 位置 64+:  都用 wpe[63]
```

**效果**：位置 64、100、1000 的 token 共享完全相同的位置编码。**位置信息变得模糊**，但注意力计算范围不受影响。

**注意**：这和 attention window 是完全独立的机制：
- position embedding truncation 影响的是"位置 i 在模型眼里看起来多靠后"
- attention window 影响的是"位置 i 能在注意力计算中看到哪些其他位置"

---

## 三者对比

| 机制 | 控制什么 | main.py 是否有 |
|------|---------|---------------|
| Causal mask | 不能看未来 | ✅ 有 |
| Attention window | 不能看太远的过去 | ❌ 没有 |
| Position embedding truncation | 位置 > MAX 时复用 | ✅ 有 |

---

## 常见混淆

**混淆 1**：以为 position embedding truncation 会限制注意力范围。

实际上，位置 100 的 token 用了 `wpe[63]` 的编码（位置信息模糊），但在注意力计算中它**仍然能看到** token 0~99，范围没有变小。

**混淆 2**：把 causal mask 称为"截断"。

causal mask 是单向的（只看过去），不是距离上的截断（只看最近 N 个）。`range(i + 1)` 覆盖的是**全部过去**，不是"最近 64 个"。

---

## 总结

```
Causal mask:        t ∈ [0, i]                (方向限制)
Attention window:   t ∈ [max(0, i-K+1), i]    (距离限制)
Position trunc:      pos = wpe[min(i, K-1)]     (编码复用)
```

main.py 有 ① 和 ③，没有 ②。这是简化设计，不是 bug。
