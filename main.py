"""
microRAG: 用纯Python实现的最简化RAG（检索增强生成）
Retriever + Generator，完整的两步流程
"""

import math
import random
from typing import Sequence

random.seed(42)

# ============================================================
# 1. 知识库 + 字符级Tokenizer
# ============================================================
knowledge_base = [
    ("太阳有多热？", "太阳表面约6000度，核心约1500万度。"),
    ("地球直径多少？", "地球直径约12742公里。"),
    ("水的化学式是什么？", "水的化学式是H2O。"),
    ("光速是多少？", "光速约每秒30万公里。"),
    ("谁发明了电话？", "电话是贝尔发明的。"),
    ("地球到月亮多远？", "地球到月亮约38万公里。"),
    ("氧气占空气多少？", "氧气约占空气的21%。"),
    ("水的沸点是多少？", "水在标准大气压下沸点是100摄氏度。"),
]

# 构建字符表
all_chars: set[str] = set()
for q, a in knowledge_base:
    all_chars.update(q)
    all_chars.update(a)
uchars = sorted(all_chars)
VOCAB_SIZE = len(uchars)
char_to_id = {c: i for i, c in enumerate(uchars)}
id_to_char = {i: c for c, i in char_to_id.items()}
# 这两个如何理解呢？
EMBED_DIM = 32
BLOCK_SIZE = 64

# 特殊 token
SEP = VOCAB_SIZE + 1      # 分隔符
PAD = VOCAB_SIZE + 2      # padding
TOTAL_VOCAB = VOCAB_SIZE + 3

print(f"vocab_size={VOCAB_SIZE}, TOTAL_VOCAB={TOTAL_VOCAB}, EMBED_DIM={EMBED_DIM}")
print(f"chars: {''.join(uchars[:20])}...")

def encode(text: str) -> list[int]:
    return [char_to_id.get(c, 0) for c in text]

# ============================================================
# 2. 自动微分引擎
# ============================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data / other.data, (self, other), (1/other.data, -self.data/other.data**2))

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        # 迭代拓扑排序，避免递归深度超限
        topo: list[Value] = []
        visited: set[Value] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]

        while stack:
            v, processed = stack.pop()
            if v in visited:
                continue
            if processed:
                visited.add(v)
                topo.append(v)
            else:
                stack.append((v, True))
                for child in v._children:
                    if child not in visited:
                        stack.append((child, False))

        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# ============================================================
# 3. 模型参数
# ============================================================
def matrix(nout: int, nin: int, std: float = 0.1) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

sd: dict[str, list[list[Value]]] = {}

# Retriever: query 和 key 都是 [TOTAL_VOCAB -> EMBED_DIM] 的嵌入
sd['ret_q'] = matrix(EMBED_DIM, TOTAL_VOCAB)  # [EMBED_DIM, TOTAL_VOCAB]
sd['ret_k'] = matrix(EMBED_DIM, TOTAL_VOCAB)  # [EMBED_DIM, TOTAL_VOCAB]

# Generator GPT
sd['wte']   = matrix(EMBED_DIM, TOTAL_VOCAB)  # [EMBED_DIM, TOTAL_VOCAB]
sd['wpe']   = matrix(BLOCK_SIZE, EMBED_DIM)    # [BLOCK_SIZE, EMBED_DIM]
sd['lm']    = matrix(TOTAL_VOCAB, EMBED_DIM)   # [TOTAL_VOCAB, EMBED_DIM]

sd['wq'] = matrix(EMBED_DIM, EMBED_DIM)
sd['wk'] = matrix(EMBED_DIM, EMBED_DIM)
sd['wv'] = matrix(EMBED_DIM, EMBED_DIM)
sd['wo'] = matrix(EMBED_DIM, EMBED_DIM)

sd['fc1'] = matrix(EMBED_DIM * 4, EMBED_DIM)
sd['fc2'] = matrix(EMBED_DIM, EMBED_DIM * 4)

params: list[Value] = [p for mat in sd.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ============================================================
# 4. 工具函数
# ============================================================
def linear(x: list[list[Value]], w: list[list[Value]]) -> list[list[Value]]:
    """x: [L, d] list of token embeddings, w: [d_out, d_in] weight matrix"""
    return [linear_single(xi, w) for xi in x]

def linear_single(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """x: [d_in] single vector, w: [d_out, d_in] weight matrix, returns [d_out]"""
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def softmax(logits: list[Value]) -> list[Value]:
    max_v = max(v.data for v in logits)
    exps = [(v - max_v).exp() for v in logits]  
    total = sum(exps, Value(0))
    return [e / total for e in exps]

def sample_from_probs(probs: list[Value]) -> int:
    """按概率采样一个 token id"""
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

def generate(input_ids: list[int], max_new_tokens: int = 50, temperature: float = 1.0) -> list[int]:
    """自回归生成。temperature=0 时用 argmax，否则用 softmax 采样。"""
    ids = input_ids.copy()
    for _ in range(max_new_tokens):
        logits_all = gpt_forward(ids)
        #生成的时候用最后一层，作为LM_Head，因为最后一层跟所有前置的都勾兑过
        logits = logits_all[-1]
        if temperature == 0:
            next_id = max(range(len(logits)), key=lambda i: logits[i].data)
        else:
            probs = softmax([v / temperature for v in logits])
            next_id = sample_from_probs(probs)
        if next_id == PAD or next_id >= TOTAL_VOCAB:
            break
        ids.append(next_id)
    return ids

def decode(ids: list[int]) -> str:
    return ''.join(id_to_char.get(i, '') for i in ids)

def rmsnorm(x: list[Value]) -> list[Value]:
    ms = sum((xi * xi for xi in x), Value(0)) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def cosine_sim(a: list[Value], b: list[Value]) -> Value:
    """输入: 两个 [EMBED_DIM] Value 列表，输出: scalar Value"""
    dot = sum((ai * bi for ai, bi in zip(a, b)), Value(0))
    na = sum((ai * ai for ai in a), Value(0)) ** 0.5
    nb = sum((bi * bi for bi in b), Value(0)) ** 0.5
    return dot / (na * nb + 1e-8)

def cosine_sim_f(a: Sequence[float], b: Sequence[float]) -> float:
    """cosine_sim 的纯 float 版本，用于推理阶段（无梯度）"""
    dot = sum((ai * bi for ai, bi in zip(a, b)), 0.0)
    na = sum((ai * ai for ai in a), 0.0) ** 0.5
    nb = sum((bi * bi for bi in b), 0.0) ** 0.5
    return dot / (na * nb + 1e-8)

def mean_pool(embed_matrix: list[list[Value]], token_ids: list[int]) -> list[Value]:
    """token_ids 的 list，做 mean pooling 得到 [EMBED_DIM] 向量"""
    if not token_ids:
        return [Value(0) for _ in range(EMBED_DIM)]
    result: list[Value] = [Value(0) for _ in range(EMBED_DIM)]
    for t in token_ids:
        for e in range(EMBED_DIM):
            result[e] = result[e] + embed_matrix[e][t]
    n = Value(len(token_ids))
    return [r / n for r in result]

# ============================================================
# 5. Retriever
# ============================================================
def retrieve(question_embed: Sequence[float], top_k: int = 1) -> list[tuple[float, int]]:
    """用 cosine 相似度检索知识库（推理阶段，纯 float）"""
    scores: list[tuple[float, int]] = []
    for i, (q, a) in enumerate(knowledge_base):
        doc_tokens = encode(a)
        doc_emb = mean_pool(sd['ret_k'], doc_tokens)  # [EMBED_DIM] Value list
        doc_emb_f = [v.data for v in doc_emb]
        score = cosine_sim_f(question_embed, doc_emb_f)
        scores.append((score, i))
    scores.sort(reverse=True)
    return scores[:top_k]

# ============================================================
# 6. Generator（GPT）
# ============================================================
def gpt_forward(tokens: list[int]) -> list[list[Value]]:
    """tokens: list of token ids, returns logits for ALL positions"""
    L = len(tokens)
    # embedding + 位置编码: x[i] = [EMBED_DIM]
    x: list[list[Value]] = []
    for i, tid in enumerate(tokens):
        emb = [sd['wte'][e][tid] for e in range(EMBED_DIM)]
        pos = sd['wpe'][min(i, BLOCK_SIZE - 1)]
        x.append([e + p for e, p in zip(emb, pos)])
    x = [rmsnorm(xi) for xi in x]

    # 自注意力（单层，单头，causal mask）
    q: list[list[Value]] = linear(x, sd['wq'])
    k: list[list[Value]] = linear(x, sd['wk'])
    v: list[list[Value]] = linear(x, sd['wv'])

    attn: list[list[Value]] = []
    for i in range(L):
        scores_i: list[Value] = [sum((q[i][j] * k[t][j] for j in range(EMBED_DIM)), Value(0)) / (EMBED_DIM ** 0.5)
                    for t in range(i + 1)]
        weights: list[Value] = softmax(scores_i)
        out_i: list[Value] = [sum((weights[t] * v[t][j] for t in range(i + 1)), Value(0)) for j in range(EMBED_DIM)]
        attn.append(out_i)

    attn = [linear_single(a, sd['wo']) for a in attn]
    x = [x[i] + attn[i] for i in range(L)]
    x = [rmsnorm(xi) for xi in x]

    # MLP：对所有位置做
    logits_list: list[list[Value]] = []
    for pos in range(L):
        h_pre: list[Value] = linear_single(x[pos], sd['fc1'])
        h_act: list[Value] = [hi.relu() for hi in h_pre]
        h_post: list[Value] = linear_single(h_act, sd['fc2'])
        x_pos = [x[pos][j] + h_post[j] for j in range(EMBED_DIM)]
        logits_list.append(linear_single(x_pos, sd['lm']))
    return logits_list

# ============================================================
# 7. 训练 Retriever（InfoNCE 对比学习）
# ============================================================
lr = 0.01
beta1, beta2, eps = 0.9, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

for step in range(10):
    # 随机选一个正样本
    gt_idx = random.randint(0, len(knowledge_base) - 1)
    gt_q, gt_a = knowledge_base[gt_idx]

    q_ids = encode(gt_q)
    pos_ids = encode(gt_a)

    # 负样本
    neg_idx = random.randint(0, len(knowledge_base) - 1)
    while neg_idx == gt_idx:
        neg_idx = random.randint(0, len(knowledge_base) - 1)
    neg_ids = encode(knowledge_base[neg_idx][1])

    # query, pos_doc, neg_doc 的 embedding
    q_emb = mean_pool(sd['ret_q'], q_ids)       # [EMBED_DIM] Value
    pos_emb = mean_pool(sd['ret_k'], pos_ids)    # [EMBED_DIM] Value
    neg_emb = mean_pool(sd['ret_k'], neg_ids)   # [EMBED_DIM] Value

    # cosine sim 直接在 Value 上算，保持计算图
    sp = cosine_sim(q_emb, pos_emb)
    sn = cosine_sim(q_emb, neg_emb)

    # InfoNCE margin loss：正样本要比负样本高出 0.5
    loss = (sn - sp + 0.5).relu()

    loss.backward()

    lr_t = lr * (1 - step / 1000)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        mh = m[i] / (1 - beta1 ** (step + 1))
        vh = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * mh / (vh ** 0.5 + eps)
        p.grad = 0

    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.data:.4f} | sp={sp.data:.3f} sn={sn.data:.3f}")

# ============================================================
# 7b. 训练 Generator（GPT 自回归语言模型）
# ============================================================
print("\n--- 训练 Generator ---")
gpt_params: list[Value] = [p for mat in ['wte', 'wpe', 'lm', 'wq', 'wk', 'wv', 'wo', 'fc1', 'fc2'] for row in sd[mat] for p in row]
gpt_m = [0.0] * len(gpt_params)
gpt_v = [0.0] * len(gpt_params)
gpt_lr = 0.05

for step in range(300):
    # 随机选一个 QA 对
    gt_idx = random.randint(0, len(knowledge_base) - 1)
    gt_q, gt_a = knowledge_base[gt_idx]

    # 拼接: question + SEP + answer，用 next-token prediction 训练
    seq = encode(gt_q) + [SEP] + encode(gt_a)
    if len(seq) < 2:
        continue

    # 取所有位置的 logits
    logits_all: list[list[Value]] = gpt_forward(seq[:-1])

    # 交叉熵 loss
    loss: Value = Value(0)
    for i, tid in enumerate(seq[1:]):
        logits_i = logits_all[i]
        # softmax -> log softmax -> nll
        max_v = max(v.data for v in logits_i)
        exps = [(v - max_v).exp() for v in logits_i]
        total = sum(exps, Value(0))
        probs_i = [e / total for e in exps]
        # target token 的 log prob
        p_target = probs_i[tid]
        nll = -p_target.log()
        loss = loss + nll
    loss = loss / len(seq[1:])

    loss.backward()

    lr_t = gpt_lr * (1 - step / 300)
    for i, p in enumerate(gpt_params):
        gpt_m[i] = beta1 * gpt_m[i] + (1 - beta1) * p.grad
        gpt_v[i] = beta2 * gpt_v[i] + (1 - beta2) * p.grad ** 2
        mh = gpt_m[i] / (1 - beta1 ** (step + 1))
        vh = gpt_v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * mh / (vh ** 0.5 + eps)
        p.grad = 0

    if step % 100 == 0:
        print(f"gpt step {step:4d} | loss {loss.data:.4f}")

# ============================================================
# 8. RAG 完整流程测试，从这里开始看
# ============================================================
def rag_answer(question: str) -> str:
    """完整 RAG: 检索 -> 拼上下文 -> 生成答案"""
    # 1. 检索相关答案
    q_ids = encode(question)
    q_emb = mean_pool(sd['ret_q'], q_ids)
    q_f = [v.data for v in q_emb]
    _, best_idx = retrieve(q_f, top_k=1)[0]
    retrieved_a = knowledge_base[best_idx][1]

    # 2. 拼接输入: question + SEP + retrieved_answer（格式与训练一致）
    input_ids = encode(question) + [SEP] + encode(retrieved_a + " ")

    # 3. 生成
    output_ids = generate(input_ids, max_new_tokens=60, temperature=0.8)
    return decode(output_ids[len(input_ids):])

print("\n--- RAG 完整流程测试 ---")
test_questions = [
    "太阳有多热？",
    "水的沸点是多少？",
    "谁发明了电话？",
    "光速是多少？",
]
for q in test_questions:
    answer = rag_answer(q)
    print(f"\n问: {q}")
    print(f"答: {answer}")
