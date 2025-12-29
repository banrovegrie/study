# From RNNs to DeepSeek v3.2: A Comprehensive Guide to Modern Language Model Architecture

> **A Research-Level Treatment of Sequence Modeling Evolution**
>
> This document provides a rigorous yet accessible exploration of language model architectures, from foundational recurrent networks to state-of-the-art systems like DeepSeek v3.2. We emphasize mathematical formulations, architectural intuitions, and the reasoning behind design choices.

---

## Table of Contents

1. [Introduction: The Sequence Modeling Problem](#1-introduction-the-sequence-modeling-problem)
2. [Tokenization: From Text to Numbers](#2-tokenization-from-text-to-numbers)
3. [The Recurrent Era: RNNs, LSTMs, and GRUs](#3-the-recurrent-era-rnns-lstms-and-grus)
4. [Gated Linear Units: A Bridge to Modern Architectures](#4-gated-linear-units-a-bridge-to-modern-architectures)
5. [The Attention Mechanism: A Paradigm Shift](#5-the-attention-mechanism-a-paradigm-shift)
6. [The Transformer Architecture](#6-the-transformer-architecture)
7. [Scaling and Pre-Training](#7-scaling-and-pre-training)
8. [Post-Training: RLHF, GRPO, and RLVR](#8-post-training-rlhf-grpo-and-rlvr)
9. [DeepSeek v3: State-of-the-Art Architecture](#9-deepseek-v3-state-of-the-art-architecture)
10. [DeepSeek v3.2: Sparse Attention and Beyond](#10-deepseek-v32-sparse-attention-and-beyond)
11. [Alternative Architectures: SSMs, Mamba, and RWKV](#11-alternative-architectures-ssms-mamba-and-rwkv)
12. [Recursive Reasoning: HRM and TRM](#12-recursive-reasoning-hrm-and-trm)
13. [Diffusion Language Models](#13-diffusion-language-models)
14. [Memory and Continual Learning](#14-memory-and-continual-learning)
15. [Conclusion: The Path Forward](#15-conclusion-the-path-forward)
16. [References](#16-references)

---

## 1. Introduction: The Sequence Modeling Problem

### 1.1 Problem Definition

Language modeling is fundamentally a **sequence prediction** problem. Given a sequence of tokens $(x_1, x_2, \ldots, x_{t-1})$, we aim to model the probability distribution over the next token:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

The full sequence probability factorizes autoregressively:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

### 1.2 The Core Challenges

| Challenge | Description |
|-----------|-------------|
| **Long-range dependencies** | Token at position 1 may be crucial for predicting token at position 1000 |
| **Variable-length sequences** | Must handle inputs from 1 to millions of tokens |
| **Computational efficiency** | Training on trillions of tokens requires parallelism |
| **Memory constraints** | Must fit in GPU memory during training and inference |

### 1.3 Historical Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1990   1997   2014    2017     2018      2020      2023       2024   2025  │
│    │      │      │       │        │         │         │          │      │   │
│   RNN   LSTM  Attn   Transformer GPT/BERT  GPT-3   Mamba/    DeepSeek  DSA  │
│                                           Scaling   RWKV        v3     v3.2 │
│                                            Laws                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tokenization: From Text to Numbers

### 2.1 The Fundamental Problem

Neural networks operate on numerical vectors, not text. **Tokenization** is the process of converting text into discrete tokens that can be embedded as vectors.

### 2.2 Tokenization Strategies

#### 2.2.1 Character-Level Tokenization

```
"hello" → ['h', 'e', 'l', 'l', 'o'] → [104, 101, 108, 108, 111]
```

**Advantages:**
- Small vocabulary (≈256 for ASCII, ≈65K for Unicode)
- No out-of-vocabulary tokens

**Disadvantages:**
- Very long sequences (5× longer than word-level)
- Model must learn spelling from scratch

#### 2.2.2 Word-Level Tokenization

```
"hello world" → ['hello', 'world'] → [1234, 5678]
```

**Advantages:**
- Semantically meaningful units
- Short sequences

**Disadvantages:**
- Massive vocabulary (millions of words)
- Cannot handle typos, neologisms, or rare words
- "ChatGPT" would be `<UNK>` (unknown token)

#### 2.2.3 Subword Tokenization (BPE)

**Byte Pair Encoding (BPE)** [Sennrich et al., 2016] provides the best of both worlds:

```
Algorithm: BPE Training
─────────────────────────────────────────────────────────────
Input: Corpus C, target vocabulary size V
Output: Vocabulary of subword tokens

1. Initialize vocabulary with all individual characters in C
2. Repeat until |vocabulary| = V:
   a. Count frequency of all adjacent token pairs
   b. Find most frequent pair (a, b)
   c. Merge (a, b) → new token "ab"
   d. Replace all occurrences of (a, b) with "ab" in C
   e. Add "ab" to vocabulary
3. Return vocabulary
─────────────────────────────────────────────────────────────
```

**Example Trace:**

```
Corpus: "low lower lowest"
Initial: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', ...]

Step 1: Most frequent pair = ('l', 'o') → merge to 'lo'
        Corpus: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', ...]

Step 2: Most frequent pair = ('lo', 'w') → merge to 'low'
        Corpus: ['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't']

Step 3: Most frequent pair = ('low', 'e') → merge to 'lowe'
        ...
```

**Result:** Common words become single tokens; rare words decompose into known subwords.

### 2.3 Byte-Level BPE

GPT-2 and modern LLMs use **byte-level BPE**:

```
Text → UTF-8 bytes → BPE merges → Token IDs

"café" → [99, 97, 102, 195, 169] → BPE → [token_id_for_café]
```

**Key insight:** By operating on bytes, any valid UTF-8 text can be tokenized—no unknown tokens ever.

### 2.4 SentencePiece

[Google's SentencePiece](https://github.com/google/sentencepiece) operates directly on raw text without pre-tokenization:

- Language-agnostic (works for Japanese, Chinese, etc.)
- Supports both BPE and Unigram algorithms
- Used by LLaMA, Mistral, DeepSeek

### 2.5 Tokenization's Impact on Model Behavior

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Tokenization of "123 + 456"                                               │
├───────────────────────────────────────────────────────────────────────────┤
│ Tokenizer A: ["123", "+", "456"]        → 3 tokens (good for arithmetic)  │
│ Tokenizer B: ["12", "3", "+", "45", "6"] → 5 tokens (harder for model)    │
│ Tokenizer C: ["1", "2", "3", "+", "4", "5", "6"] → 7 tokens (hardest)     │
└───────────────────────────────────────────────────────────────────────────┘
```

> **"Tokenization is the bane of LLMs"** — Andrej Karpathy
>
> Many LLM failures (arithmetic, spelling, certain languages) trace back to tokenization.

---

## 3. The Recurrent Era: RNNs, LSTMs, and GRUs

### 3.1 Recurrent Neural Networks (RNNs)

#### 3.1.1 Architecture

The fundamental idea: maintain a **hidden state** that summarizes all previous inputs.

```
                    ┌─────┐     ┌─────┐     ┌─────┐
      x₁ ──────────►│     │     │     │     │     │
                    │ RNN │────►│ RNN │────►│ RNN │────► ...
      h₀ ──────────►│     │     │     │     │     │
                    └──┬──┘     └──┬──┘     └──┬──┘
                       │           │           │
                       ▼           ▼           ▼
                      h₁          h₂          h₃
```

#### 3.1.2 Mathematical Formulation

At each timestep $t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

Where:
- $x_t \in \mathbb{R}^d$ is the input embedding at time $t$
- $h_t \in \mathbb{R}^n$ is the hidden state
- $W_{hh} \in \mathbb{R}^{n \times n}$ is the recurrent weight matrix
- $W_{xh} \in \mathbb{R}^{n \times d}$ is the input weight matrix
- $\tanh$ provides nonlinearity and bounds outputs to $[-1, 1]$

### 3.2 The Vanishing Gradient Problem

#### 3.2.1 The Mathematics of Gradient Flow

During backpropagation through time (BPTT), the gradient of the loss with respect to an early hidden state involves a product of Jacobians:

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Each Jacobian term is:

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(z_t)) \cdot W_{hh}$$

Where $\tanh'(z) = 1 - \tanh^2(z) \leq 1$.

#### 3.2.2 Why Gradients Vanish

The spectral norm governs gradient behavior:

$$\left\| \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} \right\| \leq \prod_{t=2}^{T} \left\| \frac{\partial h_t}{\partial h_{t-1}} \right\| \approx \gamma^{T-1}$$

Where $\gamma = \|\text{diag}(\tanh') \cdot W_{hh}\|$.

- If $\gamma < 1$: gradients **vanish** exponentially as $T$ grows
- If $\gamma > 1$: gradients **explode** exponentially

**Numerical example:**
```
γ = 0.9, T = 100
γ^99 = 0.9^99 ≈ 0.00003

→ Gradient from token 100 to token 1 is essentially zero
→ Model cannot learn long-range dependencies
```

#### 3.2.3 Visualization of Gradient Flow

```
                 Gradient magnitude over time
     │
   1 │ ████
     │ ██████
     │ █████████
     │ ██████████████
     │ ████████████████████████
     │ ████████████████████████████████████████████
     └──────────────────────────────────────────────────►
     t=1                                            t=100

     ↑ Gradient nearly zero        ↑ Gradient strong (near loss)
```

### 3.3 Long Short-Term Memory (LSTM)

[Hochreiter & Schmidhuber, 1997] introduced LSTMs to solve the vanishing gradient problem.

#### 3.3.1 Key Insight: The Cell State Highway

Instead of transforming the hidden state through a nonlinearity at each step, LSTMs maintain a **cell state** $C_t$ that can flow unchanged through time:

```
        Cell State C_{t-1} ─────────────────────────────────► C_t
                              │           │           │
                            [×]─────────[+]─────────[×]
                              ↑           ↑           ↓
                           Forget       Input      Output
                            Gate        Gate        Gate
```

#### 3.3.2 LSTM Equations

**Forget Gate** (what to erase from memory):
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** (what new information to store):
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Values** (new information to potentially add):
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update** (the key innovation):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate** (what to output):
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State**:
$$h_t = o_t \odot \tanh(C_t)$$

Where:
- $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$
- $\odot$ denotes element-wise (Hadamard) product
- $[\cdot, \cdot]$ denotes concatenation

#### 3.3.3 Why LSTMs Solve Vanishing Gradients

The gradient through the cell state:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

When $f_t \approx 1$ (forget gate open), gradients flow **unchanged**:

$$\frac{\partial L}{\partial C_1} = \frac{\partial L}{\partial C_T} \cdot \prod_{t=2}^{T} f_t \approx \frac{\partial L}{\partial C_T}$$

The network **learns** when to remember ($f \approx 1$) vs. forget ($f \approx 0$).

#### 3.3.4 LSTM Architectural Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                         Cell State (C)                                     │
│    C_{t-1} ──────────────────[×]──────────[+]──────────────────► C_t       │
│                               │            │                               │
│                               │            │                               │
│         ┌─────────────────────┘            └─────────────┐                 │
│         │                                                │                 │
│         │  f_t                              i_t    C̃_t   │                 │
│         │   │                                │      │    │                 │
│         └───┴────────────────────────────────┴──────┴────┘                 │
│                                                                            │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│    │  σ      │  │  σ      │  │  tanh   │  │  σ      │                      │
│    │ Forget  │  │ Input   │  │ Candidate│  │ Output │                      │
│    │  Gate   │  │  Gate   │  │  Values │  │  Gate  │                      │
│    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                      │
│         │            │            │            │                           │
│         └────────────┴────────────┴────────────┘                           │
│                          │                                                 │
│                    ┌─────┴─────┐                                           │
│    h_{t-1} ───────►│  Concat   │◄─────── x_t                               │
│                    └───────────┘                                           │
│                                                                            │
│    Hidden State:  h_t = o_t ⊙ tanh(C_t)                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Gated Recurrent Unit (GRU)

[Cho et al., 2014] simplified LSTM with fewer parameters:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$ (update gate)
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$ (reset gate)
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$ (candidate)
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$ (interpolation)

GRU merges the cell state and hidden state, reducing complexity while maintaining performance.

### 3.5 Fundamental Limitations of Recurrent Models

Despite solving vanishing gradients, recurrent models have inherent limitations:

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Sequential computation** | Must process $x_t$ before $x_{t+1}$ | Cannot parallelize on GPU |
| **Fixed-size bottleneck** | All history compressed into $h_t \in \mathbb{R}^n$ | Information loss inevitable |
| **O(T) path length** | Information from $x_1$ must flow through $T-1$ steps to reach $x_T$ | Signal degradation |

---

## 4. Gated Linear Units: A Bridge to Modern Architectures

### 4.1 The GLU Family

[Dauphin et al., 2017] introduced **Gated Linear Units (GLU)** for convolutional language models:

$$\text{GLU}(x) = (xW + b) \odot \sigma(xV + c)$$

Where:
- $(xW + b)$: the "content" pathway (what information)
- $\sigma(xV + c)$: the "gate" pathway (how much to let through)
- $\odot$: element-wise product

### 4.2 GLU Variants

[Shazeer, 2020] explored variants in the paper "GLU Variants Improve Transformer":

| Variant | Activation | Formula |
|---------|------------|---------|
| **GLU** | Sigmoid | $\sigma(xW) \odot (xV)$ |
| **ReGLU** | ReLU | $\text{ReLU}(xW) \odot (xV)$ |
| **GEGLU** | GELU | $\text{GELU}(xW) \odot (xV)$ |
| **SwiGLU** | Swish | $\text{Swish}(xW) \odot (xV)$ |

Where $\text{Swish}(x) = x \cdot \sigma(\beta x)$ (typically $\beta = 1$).

### 4.3 SwiGLU in Modern LLMs

**SwiGLU** became the standard FFN in LLaMA, PaLM, and DeepSeek:

```
Standard FFN:
    FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
    Parameters: d × d_ff + d_ff × d = 2 × d × d_ff

SwiGLU FFN:
    FFN(x) = (Swish(xW_gate) ⊙ (xW_up)) W_down
    Parameters: d × d_ff + d × d_ff + d_ff × d = 3 × d × d_ff
```

To match parameter count, $d_{ff}$ is reduced by factor of $\frac{2}{3}$:

$$d_{ff}^{\text{SwiGLU}} = \frac{2}{3} d_{ff}^{\text{standard}}$$

### 4.4 Why Gating Works

The gating mechanism provides:

1. **Adaptive information flow**: Network learns what to pass through
2. **Gradient highways**: When gate ≈ 1, gradients flow unimpeded
3. **Feature selection**: Gate acts as learned attention over features

> **"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."**
> — Noam Shazeer, "GLU Variants Improve Transformer"

---

## 5. The Attention Mechanism: A Paradigm Shift

### 5.1 The Bottleneck Problem

In sequence-to-sequence models with RNNs:

```
Encoder: "The cat sat on the mat" → [final hidden state h]
                                            ↓
Decoder: h → "Le chat s'est assis sur le tapis"
```

**Problem**: All source information must pass through a single fixed-size vector $h$.

### 5.2 Attention as Dynamic Memory Access

[Bahdanau et al., 2014] introduced attention: instead of one vector, let the decoder access **all** encoder states:

```
Encoder States:  [h₁]  [h₂]  [h₃]  [h₄]  [h₅]  [h₆]
                  ↑     ↑     ↑     ↑     ↑     ↑
                 The   cat   sat   on   the   mat
                  │     │     │     │     │     │
Attention        0.1   0.5   0.1   0.1   0.1   0.1
Weights:               ↑
                 (decoder attending to "cat")
                       │
Context:         Σ αᵢhᵢ = weighted sum → decoder
```

### 5.3 Mathematical Formulation

#### 5.3.1 Attention Scores

Given:
- Query $q \in \mathbb{R}^{d_k}$: what we're looking for
- Keys $K \in \mathbb{R}^{n \times d_k}$: what each position contains
- Values $V \in \mathbb{R}^{n \times d_v}$: information at each position

**Score function** (dot-product attention):

$$\text{score}(q, k_i) = \frac{q \cdot k_i}{\sqrt{d_k}}$$

The $\sqrt{d_k}$ scaling prevents scores from becoming too large (which would make softmax saturate).

#### 5.3.2 Attention Weights

Apply softmax to get a probability distribution:

$$\alpha_i = \text{softmax}(\text{scores})_i = \frac{\exp(\text{score}(q, k_i))}{\sum_j \exp(\text{score}(q, k_j))}$$

#### 5.3.3 Context Vector

Weighted sum of values:

$$\text{context} = \sum_i \alpha_i v_i = \text{softmax}\left(\frac{qK^T}{\sqrt{d_k}}\right) V$$

#### 5.3.4 Compact Matrix Form

For a batch of queries $Q \in \mathbb{R}^{m \times d_k}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 5.4 Self-Attention

The breakthrough of the Transformer: apply attention **within a single sequence**, with Q, K, V all derived from the same input:

```python
# X: (seq_len, d_model) - input embeddings

Q = X @ W_Q  # Project to queries
K = X @ W_K  # Project to keys
V = X @ W_V  # Project to values

# Each position can attend to every other position
attention_output = softmax(Q @ K.T / sqrt(d_k)) @ V
```

### 5.5 Multi-Head Attention

One attention pattern isn't enough. **Multi-head attention** runs $h$ parallel attention operations:

```
┌──────────────────────────────────────────────────────────────┐
│                     Multi-Head Attention                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Input X                                                    │
│      │                                                       │
│      ├────────┬────────┬────────┬────────┐                   │
│      ↓        ↓        ↓        ↓        ↓                   │
│   [Head₁] [Head₂] [Head₃] ... [Head_h]                       │
│      │        │        │        │        │                   │
│      │    Q₁,K₁,V₁  Q₂,K₂,V₂           │                     │
│      │        │        │        │        │                   │
│      ↓        ↓        ↓        ↓        ↓                   │
│   [Attn₁] [Attn₂] [Attn₃] ... [Attn_h]                       │
│      │        │        │        │        │                   │
│      └────────┴────────┴────────┴────────┘                   │
│                        │                                     │
│                    Concat                                    │
│                        │                                     │
│                       W_O                                    │
│                        │                                     │
│                     Output                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Intuition**: Different heads can learn different relationship types:
- Head 1: Subject-verb agreement
- Head 2: Positional relationships
- Head 3: Coreference (pronouns)
- Head 4: Semantic similarity

### 5.6 Computational Complexity

For sequence length $n$ and dimension $d$:

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Computing $QK^T$ | $O(n^2 d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Multiplication with $V$ | $O(n^2 d)$ | $O(nd)$ |
| **Total** | $O(n^2 d)$ | $O(n^2)$ |

**This quadratic complexity is the key limitation** that motivates efficient attention mechanisms.

---

## 6. The Transformer Architecture

### 6.1 Overview

The Transformer [Vaswani et al., 2017] consists of stacked identical layers:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TRANSFORMER BLOCK                                   │
│                                                                              │
│    Input                                                                     │
│      │                                                                       │
│      ▼                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                    Multi-Head Self-Attention                         │   │
│   └────────────────────────────────┬─────────────────────────────────────┘   │
│                                    │                                         │
│      ┌─────────────────────────────┤                                         │
│      │                             ▼                                         │
│      │                         ┌───────┐                                     │
│      └────────────────────────►│  Add  │◄── Residual Connection              │
│                                └───┬───┘                                     │
│                                    ▼                                         │
│                              ┌───────────┐                                   │
│                              │ LayerNorm │                                   │
│                              └─────┬─────┘                                   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │              Feed-Forward Network (position-wise)                    │   │
│   │                   FFN(x) = GELU(xW₁ + b₁)W₂ + b₂                     │   │
│   └────────────────────────────────┬─────────────────────────────────────┘   │
│                                    │                                         │
│      ┌─────────────────────────────┤                                         │
│      │                             ▼                                         │
│      │                         ┌───────┐                                     │
│      └────────────────────────►│  Add  │◄── Residual Connection              │
│                                └───┬───┘                                     │
│                                    ▼                                         │
│                              ┌───────────┐                                   │
│                              │ LayerNorm │                                   │
│                              └─────┬─────┘                                   │
│                                    │                                         │
│                                 Output                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

                                    × N layers
```

### 6.2 Positional Encoding

Self-attention is **permutation-equivariant**: it treats positions identically. We must inject position information.

#### 6.2.1 Sinusoidal Positional Encoding (Original)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Different dimensions encode position at different frequencies (wavelengths from $2\pi$ to $10000 \cdot 2\pi$).

#### 6.2.2 Rotary Position Embedding (RoPE)

[Su et al., 2021] introduced **RoPE**, now standard in LLaMA, Mistral, DeepSeek:

**Key idea**: Encode position as a **rotation** in 2D subspaces:

For a query/key vector, split into pairs of dimensions $(q_{2i}, q_{2i+1})$ and apply rotation by angle $\theta_i \cdot m$ where $m$ is the position:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

Where $\theta_i = 10000^{-2i/d}$.

**Why this works**: When computing dot product $q \cdot k$, the rotation angles **subtract**, giving **relative position**:

$$q'_m \cdot k'_n = f(q, k, m - n)$$

```
Position Encoding Comparison
────────────────────────────────────────────────────────────────────
    Method          │ Relative Position │ Extrapolation │ Parameters
────────────────────┼───────────────────┼───────────────┼────────────
    Learned         │       No          │     Poor      │    O(n·d)
    Sinusoidal      │       No          │     Good      │     0
    RoPE            │       Yes         │     Good      │     0
────────────────────────────────────────────────────────────────────
```

### 6.3 Normalization

#### 6.3.1 Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

Where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$.

#### 6.3.2 RMSNorm

[Zhang & Sennrich, 2019] simplified by dropping mean centering:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}$$

Where $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$.

**Used by**: LLaMA, Mistral, DeepSeek, and most modern LLMs.

#### 6.3.3 Pre-Norm vs Post-Norm

```
Post-Norm (Original):                Pre-Norm (Modern):
    x → Attention → Add → Norm          x → Norm → Attention → Add

Gradient path for Post-Norm:         Gradient path for Pre-Norm:
    Must pass through Norm               Direct residual path
    → More difficult optimization        → Easier to train deep models
```

**Pre-Norm** allows gradients to flow directly through the residual connection, enabling stable training of very deep models (100+ layers).

### 6.4 Causal Masking (Decoder-Only Models)

For autoregressive generation, position $i$ can only attend to positions $\leq i$:

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Applied before softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

```
Causal Attention Mask (4 tokens):

    Queries →    t₁    t₂    t₃    t₄
Keys ↓      ┌────────────────────────┐
    t₁      │  ✓     ✗     ✗     ✗   │
    t₂      │  ✓     ✓     ✗     ✗   │
    t₃      │  ✓     ✓     ✓     ✗   │
    t₄      │  ✓     ✓     ✓     ✓   │
            └────────────────────────┘

    ✓ = can attend, ✗ = masked (−∞)
```

### 6.5 KV Cache: Efficient Inference

During autoregressive generation, we generate one token at a time. Naively, this requires recomputing attention over all previous tokens at each step.

**KV Cache** stores computed keys and values:

```
Step 1: Input "The"
    Compute K₁, V₁
    Store in cache: [(K₁, V₁)]
    Generate: "quick"

Step 2: Input "quick"
    Compute K₂, V₂
    Append to cache: [(K₁, V₁), (K₂, V₂)]
    Query attends to cached K₁K₂, V₁V₂
    Generate: "brown"

Step 3: Input "brown"
    Compute K₃, V₃
    Append to cache: [(K₁, V₁), (K₂, V₂), (K₃, V₃)]
    Query attends to all cached KVs
    Generate: "fox"

...continues...
```

**Complexity reduction:**
- Without cache: $O(n^2)$ per token, $O(n^3)$ total for $n$ tokens
- With cache: $O(n)$ per token, $O(n^2)$ total for $n$ tokens

**Memory cost:** For a 7B parameter model with FP16:
$$\text{KV cache} = 2 \times \text{layers} \times \text{seq\_len} \times d_{\text{model}} \times 2 \text{ bytes}$$

For LLaMA-7B (32 layers, 4096 dim, 4096 context):
$$= 2 \times 32 \times 4096 \times 4096 \times 2 \approx 2 \text{ GB}$$

### 6.6 GPT-2: The Foundation

GPT-2 [Radford et al., 2019] established the modern decoder-only architecture:

```
GPT-2 Configuration (1.5B version):
─────────────────────────────────────────────────
    Parameter         │   Value
─────────────────────────────────────────────────
    Layers            │   48
    Hidden dimension  │   1600
    Attention heads   │   25
    Head dimension    │   64
    FFN dimension     │   6400 (4× hidden)
    Vocabulary size   │   50,257
    Context length    │   1024
    Parameters        │   1.5B
─────────────────────────────────────────────────
```

**Key innovations:**
- Decoder-only (no encoder)
- Learned positional embeddings
- Byte-level BPE tokenization
- Pre-normalization variant

---

## 7. Scaling and Pre-Training

### 7.1 The Scaling Laws

[Kaplan et al., 2020] discovered that loss follows power laws:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

Where $N$ = parameters, $D$ = data tokens, $C$ = compute (FLOPs).

### 7.2 Chinchilla Optimal Scaling

[Hoffmann et al., 2022] refined these laws. For compute-optimal training:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

**Rule of thumb**: Parameters and tokens should scale equally:
$$\frac{D}{N} \approx 20$$

A 10B parameter model should train on ~200B tokens.

### 7.3 Pre-Training Data

Modern LLMs train on diverse, filtered web data:

```
Typical Pre-Training Data Mix
─────────────────────────────────────────────────────────────
    Source              │  Tokens (approx)  │  Quality Filter
─────────────────────────────────────────────────────────────
    Common Crawl        │     8-10T         │  Perplexity, dedup
    Books               │     0.5T          │  Copyright filter
    Wikipedia           │     20B           │  High quality
    Code (GitHub)       │     0.5-1T        │  License filter
    arXiv/Papers        │     50B           │  High quality
    Curated/Synthetic   │     Varies        │  Human review
─────────────────────────────────────────────────────────────

DeepSeek-V3: 14.8T tokens total
```

### 7.4 Training Objective

**Next Token Prediction (Causal Language Modeling)**:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

**Fill-in-Middle (FIM)** for code models:

```
Original:  "def foo(): x = 1; return x"
FIM:       "<PRE> def foo(): <SUF> return x <MID> x = 1; "
```

This teaches the model to complete code given surrounding context.

### 7.5 Training Infrastructure

Training a 671B model requires massive parallelism:

```
Parallelism Strategies
────────────────────────────────────────────────────────────────────
    Strategy          │ What's Split    │  When to Use
────────────────────────────────────────────────────────────────────
    Data Parallel     │ Batch           │  Always (baseline)
    Tensor Parallel   │ Weight matrices │  Large hidden dims
    Pipeline Parallel │ Layers          │  Many layers
    Expert Parallel   │ MoE experts     │  MoE models
    Sequence Parallel │ Sequence length │  Very long contexts
────────────────────────────────────────────────────────────────────
```

**DeepSeek-V3 training setup:**
- 2,048 H800 GPUs
- 16-way pipeline parallelism
- 64-way expert parallelism
- FP8 mixed precision
- Total: 2.788M GPU-hours

---

## 8. Post-Training: RLHF, GRPO, and RLVR

### 8.1 Why Post-Training?

Pre-training optimizes for next-token prediction, not:
- Following instructions
- Being helpful
- Avoiding harmful content
- Refusing appropriately

**Post-training aligns** the model with human values and preferences.

### 8.2 Supervised Fine-Tuning (SFT)

Train on high-quality instruction-response pairs:

```
Dataset Example:
─────────────────────────────────────────────────────
Instruction: Explain photosynthesis in simple terms.
Response: Photosynthesis is how plants make food. They
          use sunlight, water, and carbon dioxide to
          create sugar for energy and release oxygen
          as a byproduct...
─────────────────────────────────────────────────────
```

SFT teaches the **format** of helpful responses.

### 8.3 RLHF: Reinforcement Learning from Human Feedback

#### 8.3.1 Step 1: Train Reward Model

Collect human preferences:
```
Prompt: "What's the capital of France?"
Response A: "The capital of France is Paris."
Response B: "Paris is the capital."

Human preference: A > B
```

Train reward model $R_\phi$ to predict preferences:
$$\mathcal{L}_{\text{RM}} = -\log \sigma(R_\phi(A) - R_\phi(B))$$

#### 8.3.2 Step 2: Policy Optimization (PPO)

**PPO Objective:**

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the probability ratio
- $A_t$ is the advantage (how much better than baseline)
- KL term keeps policy close to reference model

**PPO requires 4 models:**
1. Policy model (being trained)
2. Reference model (original SFT model)
3. Reward model
4. Value model (critic, for advantage estimation)

### 8.4 GRPO: Group Relative Policy Optimization

[DeepSeek, 2024] simplified RLHF by eliminating the value model:

#### 8.4.1 Algorithm

```
Algorithm: GRPO
─────────────────────────────────────────────────────────────────────
Input: Prompt x, policy π_θ, reward function R, group size G

1. Generate G responses: {y₁, y₂, ..., y_G} ~ π_θ(·|x)
2. Compute rewards: {r₁, r₂, ..., r_G} where rᵢ = R(x, yᵢ)
3. Normalize (group-relative baseline):
   μ = mean({r₁, ..., r_G})
   σ = std({r₁, ..., r_G})
   Aᵢ = (rᵢ - μ) / σ  ← Advantage for response i
4. Update policy:
   L = Σᵢ [ min(ρᵢ · Aᵢ, clip(ρᵢ) · Aᵢ) ] - β · KL(π_θ || π_ref)
   where ρᵢ = π_θ(yᵢ|x) / π_old(yᵢ|x)
5. θ ← θ + ∇L

Output: Updated policy π_θ
─────────────────────────────────────────────────────────────────────
```

#### 8.4.2 Key Insight

**No separate value model needed.** The group mean serves as the baseline:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r}$$

**Benefits:**
- 50% memory reduction (2 models instead of 4)
- Simpler implementation
- Comparable or better performance

### 8.5 RLVR: Reinforcement Learning from Verifiable Rewards

For tasks with verifiable correctness (math, code), we don't need human preferences:

```
Prompt: "What is 17 × 23?"

Response: "Let me calculate: 17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391"

Verification: 17 × 23 == 391? → True → Reward = 1.0
```

#### 8.5.1 DeepSeek-R1 Training

DeepSeek-R1 used RLVR with GRPO directly on the base model (no SFT!):

```python
def compute_reward(response, ground_truth):
    # Extract answer using regex
    answer = extract_boxed_answer(response)

    # Binary reward - no neural network!
    if answer == ground_truth:
        return 1.0
    else:
        return 0.0
```

**Remarkable finding:** The model spontaneously developed chain-of-thought reasoning without being explicitly trained to do so.

#### 8.5.2 Comparison

```
Training Paradigm Comparison
──────────────────────────────────────────────────────────────────────────────
    Method   │ Reward Source       │ Best For              │ Models Needed
──────────────────────────────────────────────────────────────────────────────
    RLHF     │ Learned from humans │ Style, safety         │ 4 (policy, ref,
             │                     │                       │    reward, value)
──────────────────────────────────────────────────────────────────────────────
    GRPO     │ Learned/verifiable  │ General tasks         │ 2 (policy, ref)
──────────────────────────────────────────────────────────────────────────────
    RLVR     │ Symbolic verifier   │ Math, code, logic     │ 2 (policy, ref)
──────────────────────────────────────────────────────────────────────────────
```

---

## 9. DeepSeek v3: State-of-the-Art Architecture

### 9.1 Overview

[DeepSeek-V3](https://arxiv.org/abs/2412.19437) represents the current open-source SOTA:

```
DeepSeek-V3 Specifications
───────────────────────────────────────────────────────────
    Metric              │   Value
───────────────────────────────────────────────────────────
    Total Parameters    │   671B
    Active Parameters   │   37B (per token)
    Layers              │   61
    Hidden Dimension    │   7168
    Attention Heads     │   128
    KV Compressed Dim   │   512
    Routed Experts      │   256
    Active Experts      │   8 + 1 shared
    Training Tokens     │   14.8T
    Training Cost       │   ~$5.5M (2.788M H800 GPU-hours)
───────────────────────────────────────────────────────────
```

### 9.2 Multi-Head Latent Attention (MLA)

**Problem:** KV cache grows linearly with sequence length and model depth, consuming massive memory.

**Solution:** Compress keys and values into a lower-dimensional latent space.

#### 9.2.1 Standard Multi-Head Attention (MHA)

```
Q = X @ W_Q  →  (seq_len, num_heads × head_dim)
K = X @ W_K  →  (seq_len, num_heads × head_dim)  ← Store this
V = X @ W_V  →  (seq_len, num_heads × head_dim)  ← Store this

KV Cache size: 2 × seq_len × num_heads × head_dim
```

#### 9.2.2 MLA: Low-Rank Compression

```
Step 1: Compress into latent
    c_KV = X @ W_compress  →  (seq_len, latent_dim)  ← Store this instead!

    Latent dim << num_heads × head_dim
    (512 vs 16384 in DeepSeek-V3)

Step 2: At attention time, decompress
    K = c_KV @ W_K_decompress
    V = c_KV @ W_V_decompress
```

#### 9.2.3 Mathematical Formulation

**Compression:**
$$c^{KV}_t = W^{DKV} h_t$$

Where $W^{DKV} \in \mathbb{R}^{d \times d_c}$ and $d_c \ll d \cdot n_h$.

**Decompression for keys:**
$$k^C_t = W^{UK} c^{KV}_t$$

**RoPE component** (handled separately for position encoding):
$$k^R_t = \text{RoPE}(W^{KR} h_t)$$

**Final key:**
$$k_t = [k^C_t; k^R_t]$$

**DeepSeek-V3 config:**
- 128 heads × 128 dim = 16,384 total KV dimensions
- Compressed to 512 dimensions
- **32× reduction in KV cache!**

#### 9.2.4 Architectural Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      Multi-Head Latent Attention (MLA)                      │
│                                                                            │
│    Input h_t                                                               │
│        │                                                                   │
│        ├──────────────────┬───────────────────────────────────┐            │
│        │                  │                                   │            │
│        ▼                  ▼                                   ▼            │
│   ┌─────────┐       ┌──────────────┐                   ┌───────────┐       │
│   │ W^Q     │       │ W^{DKV}      │                   │ W^{KR}    │       │
│   │ (query) │       │ (compress)   │                   │ (RoPE key)│       │
│   └────┬────┘       └──────┬───────┘                   └─────┬─────┘       │
│        │                   │                                 │             │
│        ▼                   ▼                                 ▼             │
│       q_t             c^{KV}_t   ← Cache this!          k^R_t (RoPE)       │
│        │                   │                                 │             │
│        │           ┌───────┴───────┐                         │             │
│        │           ▼               ▼                         │             │
│        │      ┌────────┐     ┌────────┐                      │             │
│        │      │ W^{UK} │     │ W^{UV} │                      │             │
│        │      └───┬────┘     └───┬────┘                      │             │
│        │          │              │                           │             │
│        │          ▼              ▼                           │             │
│        │        k^C_t          v_t                           │             │
│        │          │                                          │             │
│        │          └────────────┬─────────────────────────────┘             │
│        │                       │                                           │
│        │                       ▼                                           │
│        │                 k_t = [k^C_t ; k^R_t]                              │
│        │                       │                                           │
│        ▼                       ▼                                           │
│   ┌──────────────────────────────────────────┐                             │
│   │        Scaled Dot-Product Attention       │                             │
│   │   Attention(q_t, K, V) = softmax(qK^T/√d)V                             │
│   └────────────────────────────────────────────┘                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Mixture of Experts (MoE)

Instead of one large FFN, use many smaller "expert" networks:

#### 9.3.1 DeepSeekMoE Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            DeepSeekMoE Layer                               │
│                                                                            │
│    Input Token x_t                                                         │
│         │                                                                  │
│         ▼                                                                  │
│    ┌─────────────────┐                                                     │
│    │  Router Network │                                                     │
│    │  g = softmax(Wx)│                                                     │
│    └────────┬────────┘                                                     │
│             │                                                              │
│             ▼                                                              │
│    Routing scores for 256 experts: [g₁, g₂, ..., g₂₅₆]                     │
│             │                                                              │
│             ▼                                                              │
│    Select Top-8: experts {e₁, e₂, ..., e₈}                                 │
│             │                                                              │
│    ┌────────┼────────┬────────┬────────┬───────────────────┐               │
│    │        │        │        │        │                   │               │
│    ▼        ▼        ▼        ▼        ▼                   ▼               │
│  ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐             ┌───────┐           │
│  │E_i₁│  │E_i₂│  │E_i₃│  │E_i₄│  │...│             │Shared │           │
│  └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘             │Expert │           │
│    │        │        │        │        │               └───┬───┘           │
│    │        │        │        │        │                   │               │
│    ▼        ▼        ▼        ▼        ▼                   ▼               │
│   ×g_{i₁}  ×g_{i₂}  ×g_{i₃}  ×g_{i₄}    ...               ×1               │
│    │        │        │        │        │                   │               │
│    └────────┴────────┴────────┴────────┴───────────────────┘               │
│                               │                                            │
│                               ▼                                            │
│                           Σ (sum)                                          │
│                               │                                            │
│                           Output                                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Output = Shared_Expert(x) + Σᵢ gᵢ · Expertᵢ(x)  for top-8 experts
```

#### 9.3.2 Efficiency Analysis

```
Parameter Comparison
─────────────────────────────────────────────────────────────────────────
    Configuration        │ Total Params │ Active Params │ Ratio
─────────────────────────────────────────────────────────────────────────
    Dense 37B            │     37B      │      37B      │  1.0×
    Dense 671B           │    671B      │     671B      │  1.0×
    DeepSeek-V3 MoE      │    671B      │      37B      │ 18.1×
─────────────────────────────────────────────────────────────────────────

→ 18× more parameters with same compute cost!
```

### 9.4 Auxiliary-Loss-Free Load Balancing

#### 9.4.1 The Load Balancing Problem

Without intervention, all tokens route to a few "popular" experts → **routing collapse**.

**Traditional solution**: Auxiliary loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{balance}}$$

**Problem**: The auxiliary loss degrades language modeling performance.

#### 9.4.2 DeepSeek's Solution: Bias-Based Balancing

```python
# No auxiliary loss! Instead:

for each training step:
    for each expert i:
        load_i = count(tokens routed to expert i)

        if load_i > average_load × (1 + δ):  # Overloaded
            bias_i -= γ  # Decrease routing score
        elif load_i < average_load × (1 - δ):  # Underloaded
            bias_i += γ  # Increase routing score

# During routing:
scores = Router(x)
adjusted_scores = scores + bias  # Apply learned bias
top_k = argmax(adjusted_scores, k=8)
```

This achieves balanced routing without degrading the main objective.

### 9.5 Multi-Token Prediction (MTP)

#### 9.5.1 Motivation

Standard training: predict only the next token
$$\mathcal{L} = -\log P(x_{t+1} | x_{\leq t})$$

**MTP**: predict multiple future tokens
$$\mathcal{L}_{\text{MTP}} = -\sum_{k=1}^{D} \lambda_k \log P(x_{t+k} | x_{\leq t}, \hat{x}_{t+1}, \ldots, \hat{x}_{t+k-1})$$

#### 9.5.2 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Token Prediction                              │
│                                                                            │
│    Main Transformer                                                        │
│         │                                                                  │
│         ▼                                                                  │
│    Hidden State h_t                                                        │
│         │                                                                  │
│         ├─────────────────────┬─────────────────────┐                      │
│         │                     │                     │                      │
│         ▼                     ▼                     ▼                      │
│    ┌─────────┐          ┌──────────┐          ┌──────────┐                 │
│    │ Main    │          │ MTP      │          │ MTP      │                 │
│    │ Head    │          │ Module 1 │          │ Module 2 │                 │
│    └────┬────┘          └────┬─────┘          └────┬─────┘                 │
│         │                    │                     │                       │
│         ▼                    ▼                     ▼                       │
│    P(x_{t+1}|x_≤t)     P(x_{t+2}|...)        P(x_{t+3}|...)               │
│         │                    │                     │                       │
│         ▼                    ▼                     ▼                       │
│    Token t+1            Token t+2             Token t+3                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

MTP Module:
    Input: h_t, embedding(predicted_{t+k-1})
    Output: hidden state for predicting t+k

    Uses same output head as main model (shared vocabulary projection)
```

#### 9.5.3 Benefits

1. **Denser training signal**: More gradients per forward pass
2. **Planning**: Model must "think ahead" in representations
3. **Speculative decoding**: At inference, predict 2 tokens, verify, accept if correct
   - DeepSeek reports 85-90% acceptance rate
   - 1.8× inference speedup

### 9.6 FP8 Mixed Precision Training

#### 9.6.1 Numerical Formats

```
Format Comparison
─────────────────────────────────────────────────────────────────────────
    Format     │ Sign │ Exponent │ Mantissa │ Range        │ Precision
─────────────────────────────────────────────────────────────────────────
    FP32       │  1   │    8     │    23    │ ±3.4×10³⁸    │ 7 digits
    FP16       │  1   │    5     │    10    │ ±65,504      │ 3 digits
    BF16       │  1   │    8     │    7     │ ±3.4×10³⁸    │ 2 digits
    FP8 E4M3   │  1   │    4     │    3     │ ±448         │ 1 digit
    FP8 E5M2   │  1   │    5     │    2     │ ±57,344      │ 0.5 digits
─────────────────────────────────────────────────────────────────────────
```

#### 9.6.2 DeepSeek's FP8 Strategy

**Problem**: FP8 has very limited precision → numerical instability

**Solution**: Fine-grained quantization

```
Standard: Per-tensor quantization
    scale = max(|tensor|) / 448
    → One outlier ruins everything

DeepSeek: Per-tile quantization
    Activations: 1×128 tiles (per token, per 128 channels)
    Weights: 128×128 blocks
    → Outliers isolated to their tile
```

**Accumulation precision:**
```python
# Accumulate in FP32 every 128 elements
acc = 0.0  # FP32
for i in range(0, N, 128):
    partial = matmul_fp8(A[i:i+128], B[i:i+128])  # FP8 compute
    acc += partial.to_fp32()  # Accumulate in FP32
```

**Result**: 2× memory reduction, 1.5× training speedup, minimal quality loss.

---

## 10. DeepSeek v3.2: Sparse Attention and Beyond

### 10.1 Evolution Path

```
DeepSeek-V3 (Dec 2024)
    │
    ▼
DeepSeek-V3.1 (Unified chat + reasoning)
    │
    ▼
DeepSeek-V3.2-Exp (Sep 2025) ← Sparse attention research
    │
    ▼
DeepSeek-V3.2 (Dec 2025) ← Production sparse attention
```

### 10.2 DeepSeek Sparse Attention (DSA)

#### 10.2.1 The Problem

Standard attention: $O(L^2)$ for sequence length $L$

For 128K context: $128000^2 = 16.4$ billion attention computations per layer!

#### 10.2.2 DSA Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     DeepSeek Sparse Attention (DSA)                        │
│                                                                            │
│    Query q_t (current position)                                            │
│         │                                                                  │
│         ▼                                                                  │
│    ┌──────────────────────────────────────────────────────────┐            │
│    │              Lightning Indexer                           │            │
│    │                                                          │            │
│    │    For all positions i:                                  │            │
│    │        score_i = ReLU(q_t · k_i / √d)                    │            │
│    │                                                          │            │
│    │    This is CHEAP: no softmax, no value computation       │            │
│    └────────────────────────────────────────────────────────────┘           │
│                               │                                            │
│                               ▼                                            │
│    ┌──────────────────────────────────────────────────────────┐            │
│    │              Token Selector                              │            │
│    │                                                          │            │
│    │    selected = argtop_k(scores, k=2048)                   │            │
│    │                                                          │            │
│    │    Keep only top-2048 most relevant positions            │            │
│    └────────────────────────────────────────────────────────────┘           │
│                               │                                            │
│                               ▼                                            │
│    ┌──────────────────────────────────────────────────────────┐            │
│    │              Full Attention (on selected subset)         │            │
│    │                                                          │            │
│    │    K_sparse = K[selected]   # (2048, d)                  │            │
│    │    V_sparse = V[selected]   # (2048, d)                  │            │
│    │                                                          │            │
│    │    output = softmax(q · K_sparse^T / √d) · V_sparse      │            │
│    └────────────────────────────────────────────────────────────┘           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 10.2.3 Complexity Analysis

```
Standard Attention:     O(L²) = O(128000²) ≈ 16.4B operations
DSA (k=2048):          O(L·k) = O(128000 × 2048) ≈ 262M operations

Reduction: 62× fewer operations!

Cost: $2.40 → $0.70 per million tokens (128K context)
```

#### 10.2.4 Why It Works

The key insight: **most attention weights are near zero**.

In practice, each position strongly attends to only a small subset of other positions. DSA learns to identify this subset efficiently.

### 10.3 Thinking in Tool-Use

Previous limitation: When calling tools, models "lost their train of thought."

```
Old behavior:
    User: Solve x² - 5x + 6 = 0
    Model: Let me factor this... (thinking)
    Model: [CALL calculator(x^2 - 5x + 6)]
    Tool: Returns result
    Model: The answer is... (lost reasoning context)

V3.2 behavior:
    User: Solve x² - 5x + 6 = 0
    Model: Let me factor this... looking for factors of 6 that sum to -5
    Model: [CALL calculator(verify: (x-2)(x-3))]
    Tool: Returns True
    Model: So (x-2)(x-3) = 0, giving x = 2 or x = 3 ✓

    → Reasoning preserved across tool calls
```

### 10.4 Enhanced RLVR

V3.2 uses a refined reward function:

$$R = R_{\text{outcome}} + \lambda_1 R_{\text{length}} + \lambda_2 R_{\text{consistency}}$$

Where:
- $R_{\text{outcome}}$: Binary correctness (1 or 0)
- $R_{\text{length}}$: Penalty for unnecessarily long responses
- $R_{\text{consistency}}$: Reward for consistent language use

### 10.5 V3.2-Speciale

Extended-thinking variant:
- Reduced length penalties (allow longer reasoning)
- Trained exclusively on reasoning data
- Gold medal on 2025 IMO and IOI

---

## 11. Alternative Architectures: SSMs, Mamba, and RWKV

### 11.1 The Quadratic Attention Problem

Transformers scale as $O(L^2)$ in sequence length. For very long sequences, this becomes prohibitive.

**Alternative approaches** aim for $O(L)$ or $O(L \log L)$ complexity.

### 11.2 State Space Models (SSMs)

#### 11.2.1 Continuous-Time Formulation

A linear state space model:

$$\dot{h}(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

Where:
- $x(t) \in \mathbb{R}$ is the input signal
- $h(t) \in \mathbb{R}^N$ is the hidden state
- $y(t) \in \mathbb{R}$ is the output
- $A \in \mathbb{R}^{N \times N}$ is the state matrix

#### 11.2.2 Discretization

For sequence modeling, discretize with step size $\Delta$:

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

Recurrence:
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t + D x_t$$

#### 11.2.3 S4: Structured State Spaces

[Gu et al., 2022] introduced S4 with key insights:
- Initialize $A$ as a **HiPPO matrix** (captures long-range dependencies)
- Diagonalize for efficient computation
- Can compute as convolution OR recurrence

### 11.3 Mamba: Selective State Spaces

#### 11.3.1 Key Innovation: Input-Dependent Parameters

Standard SSM: $A$, $B$, $C$ are **fixed** matrices

Mamba: Parameters become **functions of input**:

$$B_t = \text{Linear}_B(x_t)$$
$$C_t = \text{Linear}_C(x_t)$$
$$\Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))$$

This enables **content-based reasoning** (like attention) while maintaining linear complexity.

#### 11.3.2 Mamba Block

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            Mamba Block                                     │
│                                                                            │
│    Input x                                                                 │
│       │                                                                    │
│       ├──────────────────────────────────┐                                 │
│       │                                  │                                 │
│       ▼                                  ▼                                 │
│   ┌───────────┐                    ┌───────────┐                           │
│   │ Linear    │                    │ Linear    │                           │
│   │ (expand)  │                    │ (expand)  │                           │
│   └─────┬─────┘                    └─────┬─────┘                           │
│         │                                │                                 │
│         ▼                                ▼                                 │
│   ┌───────────┐                    ┌───────────┐                           │
│   │   Conv1D  │                    │   SiLU    │                           │
│   └─────┬─────┘                    └─────┬─────┘                           │
│         │                                │                                 │
│         ▼                                │                                 │
│   ┌───────────┐                          │                                 │
│   │   SiLU    │                          │                                 │
│   └─────┬─────┘                          │                                 │
│         │                                │                                 │
│         ▼                                │                                 │
│   ┌─────────────────────────────────┐    │                                 │
│   │  Selective SSM                  │    │                                 │
│   │  (input-dependent B, C, Δ)      │    │                                 │
│   └───────────────┬─────────────────┘    │                                 │
│                   │                      │                                 │
│                   ▼                      │                                 │
│              ┌─────────┐                 │                                 │
│              │  × gate │◄────────────────┘                                 │
│              └────┬────┘                                                   │
│                   │                                                        │
│                   ▼                                                        │
│             ┌───────────┐                                                  │
│             │  Linear   │                                                  │
│             │ (project) │                                                  │
│             └─────┬─────┘                                                  │
│                   │                                                        │
│               Output                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 11.3.3 Complexity Comparison

```
Complexity Analysis
───────────────────────────────────────────────────────────────────────
    Model          │ Training      │ Inference    │ Memory
───────────────────────────────────────────────────────────────────────
    Transformer    │ O(L²d)        │ O(Ld)        │ O(L²) or O(Ld) w/KV$
    Mamba          │ O(Ld)         │ O(d)         │ O(d) (constant!)
───────────────────────────────────────────────────────────────────────

L = sequence length, d = hidden dimension
```

#### 11.3.4 Mamba Performance

- Mamba-3B matches Transformer-7B quality
- 5× higher inference throughput
- Scales to million-length sequences

### 11.4 RWKV: Reinventing RNNs

#### 11.4.1 Core Idea

[Peng et al., 2023] created RWKV (Receptance Weighted Key Value):
- **Trainable like Transformer** (parallelizable)
- **Runs like RNN** (constant memory inference)

#### 11.4.2 RWKV Attention Mechanism

$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} v_i + e^{u+k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}}$$

Key features:
- Exponential decay $e^{-(t-i)w}$ for past tokens
- No softmax over full sequence
- Can be computed recurrently

#### 11.4.3 RWKV-7 "Goose" (2025)

Latest version introduces:
- **Generalized Delta Rule**: Breaks TC0 computational limitation
- 14B parameters (largest dense RNN ever)
- Matches Transformer quality
- Infinite context length (theoretically)
- No KV cache needed

```
RWKV-7 Features
───────────────────────────────────────────────────────────────────────
    Feature                  │ RWKV-7         │ Transformer
───────────────────────────────────────────────────────────────────────
    Training parallelism     │ Yes            │ Yes
    Constant-time inference  │ Yes            │ No (O(L))
    Constant memory         │ Yes            │ No (KV cache)
    Max context length      │ ∞ (theoretic)  │ Limited by memory
───────────────────────────────────────────────────────────────────────
```

### 11.5 Hybrid Architectures

**Jamba** (AI21): Combines Transformer and Mamba layers
- Attention for complex reasoning
- Mamba for efficient long-context

**Zamba** (Zyphra): Similar hybrid approach

**Observation**: Hybrids often outperform pure architectures, suggesting complementary strengths.

---

## 12. Recursive Reasoning: HRM and TRM

### 12.1 The Recursion Problem

Standard Transformers are **feedforward**: input → fixed layers → output

Some tasks require **iteration**:
- "Apply this rule until no changes"
- "Find the fixed point"
- Complex multi-step reasoning

### 12.2 Hierarchical Reasoning Model (HRM)

[Sapient Intelligence, 2025]

#### 12.2.1 Biological Inspiration

The human brain processes information at multiple timescales:
- Fast: Sensory processing, reflexes
- Slow: Abstract planning, reasoning

HRM mimics this with two recurrent modules:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Hierarchical Reasoning Model (HRM)                      │
│                                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │               High-Level Module (Slow)                             │   │
│   │                                                                    │   │
│   │   Updates every M steps                                            │   │
│   │   Abstract planning, goal representation                           │   │
│   │                                                                    │   │
│   │      h^H_{t} ────────────────────────────────────────► h^H_{t+M}   │   │
│   │                            │                                       │   │
│   └────────────────────────────┼───────────────────────────────────────┘   │
│                                │                                           │
│                                ▼ (guidance)                                │
│                                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │               Low-Level Module (Fast)                              │   │
│   │                                                                    │   │
│   │   Updates every step                                               │   │
│   │   Detailed computations, token-level operations                    │   │
│   │                                                                    │   │
│   │      h^L_t → h^L_{t+1} → h^L_{t+2} → ... → h^L_{t+M}               │   │
│   │                                                                    │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   With just 27M parameters, HRM solves:                                    │
│   - Extreme Sudoku puzzles                                                 │
│   - Large maze pathfinding (30×30)                                         │
│   - ARC-AGI tasks                                                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 12.2.2 Key Innovations

1. **Multi-timescale processing**: Different update frequencies
2. **No BPTT**: One-step gradient approximation (avoids vanishing gradients)
3. **Sequential reasoning in single forward pass**: No autoregressive loop needed

### 12.3 Tiny Recursive Model (TRM)

[Samsung SAIT, 2025]

#### 12.3.1 Architecture

TRM is remarkably simple: **just 2 layers, 7M parameters**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Tiny Recursive Model (TRM)                           │
│                                                                            │
│    Initial State (from input)                                              │
│         │                                                                  │
│         ▼                                                                  │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │                    2-Layer Network                               │    │
│    │    (Mixer layer + FFN, with RMSNorm and residual)                │    │
│    └───────────────────────────────┬──────────────────────────────────┘    │
│                                    │                                       │
│                                    ▼                                       │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │                  Latent Consistency Check                        │    │
│    │                                                                  │    │
│    │    Is current solution consistent with input?                    │    │
│    │    If not → revise and loop back                                 │    │
│    │    If yes → output solution                                      │    │
│    └───────────────────────────────┬──────────────────────────────────┘    │
│                                    │                                       │
│                    ┌───────────────┼───────────────┐                       │
│                    │ Loop back     │ Converged     │                       │
│                    ▼               ▼               │                       │
│                ┌───────┐      ┌─────────┐          │                       │
│                │ Refine│      │ Output  │          │                       │
│                └───┬───┘      │ Answer  │          │                       │
│                    │          └─────────┘          │                       │
│                    └───────────────────────────────┘                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 12.3.2 "Decision-then-Revision" Strategy

```
Standard autoregressive:
    Generate token 1 → token 2 → token 3 → ... (no going back)

TRM approach:
    Draft complete solution → Check consistency → Revise if needed → Repeat
```

This **avoids exposure bias**: errors in early tokens don't cascade.

#### 12.3.3 Performance (7M Parameters!)

```
ARC-AGI Benchmark (2 attempts)
───────────────────────────────────────────────────────────────
    Model                │ Parameters │ ARC-AGI-1 │ ARC-AGI-2
───────────────────────────────────────────────────────────────
    TRM-Attention        │    7M      │   44.6%   │   7.8%
    HRM                  │   27M      │   40.3%   │   5.0%
    DeepSeek-R1          │  671B      │   14.0%   │   4.0%
    o3-mini              │    ?       │   34.0%   │   6.0%
    Gemini 2.5 Pro       │    ?       │   27.0%   │   5.0%
───────────────────────────────────────────────────────────────

→ 7M parameter model beats 671B model on reasoning benchmarks!
```

### 12.4 Implications

1. **Scale isn't everything**: Architecture matters for reasoning
2. **Recursion is powerful**: Iterative refinement beats one-shot prediction
3. **Specialized modules**: Future LLMs may incorporate HRM/TRM as reasoning components

---

## 13. Diffusion Language Models

### 13.1 The Autoregressive Paradigm

Standard LLMs generate left-to-right:
$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_{<t})$$

**Limitations:**
- Cannot revise earlier tokens
- Sequential generation (one token at a time)
- Exposure bias (train on ground truth, generate from own predictions)

### 13.2 Diffusion for Language

#### 13.2.1 Core Idea

Inspired by image diffusion (Stable Diffusion, DALL-E):

```
Image diffusion:
    Noise → Denoise → Denoise → ... → Image

Language diffusion:
    [MASK MASK MASK MASK] → [The MASK brown MASK] → [The quick brown fox]
```

#### 13.2.2 Masked Diffusion (LLaDA)

[Large Language Diffusion with mAsking](https://arxiv.org/abs/2502.09992)

**Forward process** (add noise by masking):
$$q(x_t | x_0) = \text{mask } x_0 \text{ with probability } \beta_t$$

**Reverse process** (denoise by predicting masks):
$$p_\theta(x_{t-1} | x_t) = \text{predict masked tokens}$$

```
Training:
    Input: "The quick brown fox"
    Masked: "The [M] brown [M]"
    Target: Predict "quick" and "fox" at [M] positions

Inference:
    t=T: "[M] [M] [M] [M]" (fully masked)
    t=3: "[M] quick [M] fox"
    t=2: "The quick brown fox"
    t=1: "The quick brown fox" (converged)
```

#### 13.2.3 Architectural Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Masked Diffusion Language Model                         │
│                                                                            │
│   Forward Process (Training)                                               │
│   ─────────────────────────                                                │
│                                                                            │
│   "The quick brown fox" ──────────────────────────────────►                │
│                                                                            │
│   t=0: The   quick  brown  fox                                             │
│         ↓      ↓      ↓      ↓                                             │
│   t=1: The   [M]   brown  fox      (β₁ masking)                            │
│         ↓      ↓      ↓      ↓                                             │
│   t=2: [M]   [M]   brown  [M]      (β₂ masking)                            │
│         ↓      ↓      ↓      ↓                                             │
│   t=3: [M]   [M]    [M]   [M]      (fully masked)                          │
│                                                                            │
│   Reverse Process (Inference)                                              │
│   ───────────────────────────                                              │
│                                                                            │
│   t=3: [M]   [M]    [M]   [M]                                              │
│         ↓      ↓      ↓      ↓                                             │
│      ┌───────────────────────────────┐                                     │
│      │   Transformer (bidirectional) │                                     │
│      │   Predicts all positions      │                                     │
│      └───────────────────────────────┘                                     │
│         ↓      ↓      ↓      ↓                                             │
│   t=2: The  [M]    [M]   fox       (unmask high-confidence)                │
│         ↓      ↓      ↓      ↓                                             │
│   t=1: The  quick brown  fox                                               │
│         ↓      ↓      ↓      ↓                                             │
│   t=0: The  quick brown  fox       (final output)                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 13.3 LLaDA Results

```
LLaDA-8B Performance
───────────────────────────────────────────────────────────────────────
    Benchmark           │ LLaDA-8B    │ LLaMA3-8B   │ Winner
───────────────────────────────────────────────────────────────────────
    MMLU                │   65.0%     │   66.6%     │ LLaMA3
    GSM8K               │   71.2%     │   75.0%     │ LLaMA3
    HumanEval           │   48.2%     │   45.1%     │ LLaDA
    Reversal Poem       │   85.0%     │   10.0%     │ LLaDA (!!!)
───────────────────────────────────────────────────────────────────────
```

**Reversal Curse Solved**: LLaDA dramatically outperforms on tasks requiring bidirectional reasoning.

### 13.4 Gemini Diffusion

Google's production diffusion model (2025):

- **1,479 tokens/second** (5× faster than autoregressive)
- Competitive on coding benchmarks
- Gap on complex reasoning (40% vs 56% on GPQA)

### 13.5 Trade-offs

```
Autoregressive vs Diffusion
───────────────────────────────────────────────────────────────────────
    Aspect              │ Autoregressive │ Diffusion
───────────────────────────────────────────────────────────────────────
    Generation speed    │ Sequential     │ Parallel (faster)
    Reasoning depth     │ Strong         │ Weaker (for now)
    Bidirectional ctx   │ No             │ Yes
    Revision capability │ No             │ Yes (iterative)
    Controllability     │ Limited        │ Strong (guide denoising)
    Maturity            │ High           │ Early
───────────────────────────────────────────────────────────────────────
```

---

## 14. Memory and Continual Learning

### 14.1 The Static Model Problem

Current LLMs are **frozen after training**:
- Cannot learn from deployment experience
- Cannot incorporate new knowledge
- Catastrophic forgetting if fine-tuned

This mirrors **anterograde amnesia**: can access old memories but cannot form new ones.

### 14.2 Nested Learning

[Behrouz et al., NeurIPS 2025]

#### 14.2.1 Core Insight

**Training itself is a form of associative memory.**

Standard gradient descent with momentum:
$$v_t = \beta v_{t-1} + g_t$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

**Nested Learning view**: This is a **two-level optimization**:
- Outer: Update parameters $\theta$
- Inner: The momentum $v_t$ is itself learning to compress gradient history

#### 14.2.2 Deep Momentum Gradient Descent (DMGD)

Replace linear momentum accumulation with a **neural network**:

$$v_t = f_\phi(v_{t-1}, g_t)$$

Where $f_\phi$ is a small neural network that learns **how to accumulate gradients**.

This enables:
- Non-linear gradient patterns
- Task-adaptive learning
- Improved continual learning

#### 14.2.3 HOPE Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        HOPE Architecture                                   │
│                 (Hierarchical Optimization for Persistent Experience)      │
│                                                                            │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                    Continuum Memory System                           │ │
│   │                                                                      │ │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │ │
│   │   │   Fast       │  │   Medium     │  │   Slow       │              │ │
│   │   │   Memory     │  │   Memory     │  │   Memory     │              │ │
│   │   │              │  │              │  │              │              │ │
│   │   │ Updates:     │  │ Updates:     │  │ Updates:     │              │ │
│   │   │ Every token  │  │ Every 10     │  │ Every 100    │              │ │
│   │   │              │  │ tokens       │  │ tokens       │              │ │
│   │   │ Stores:      │  │ Stores:      │  │ Stores:      │              │ │
│   │   │ Immediate    │  │ Recent       │  │ Long-term    │              │ │
│   │   │ context      │  │ patterns     │  │ knowledge    │              │ │
│   │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │ │
│   │          │                 │                 │                       │ │
│   │          └─────────────────┴─────────────────┘                       │ │
│   │                            │                                         │ │
│   │                            ▼                                         │ │
│   │                   ┌──────────────────┐                               │ │
│   │                   │   Integration    │                               │ │
│   │                   │   Module         │                               │ │
│   │                   └────────┬─────────┘                               │ │
│   │                            │                                         │ │
│   └────────────────────────────┼────────────────────────────────────────┘ │
│                                │                                          │
│                                ▼                                          │
│                   ┌──────────────────────┐                                │
│                   │   Self-Modifying     │                                │
│                   │   Recurrent Core     │                                │
│                   └──────────────────────┘                                │
│                                                                           │
│   Key Property: Model can update its own weights during inference!        │
│                                                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 14.2.4 Results

HOPE achieves:
- Lower perplexity than Transformers
- Better than modern recurrent models (Mamba, RWKV)
- **Continual learning** without catastrophic forgetting

### 14.3 Transformer² (Sakana AI)

#### 14.3.1 Self-Adaptive Weights

[Sakana AI, 2025] enables **runtime weight adaptation**:

```
Standard LLM:
    Weights fixed → Apply to all tasks identically

Transformer²:
    1. Detect task type (from prompt or few-shot examples)
    2. Adapt weights for that specific task
    3. Run inference with adapted weights
```

#### 14.3.2 Singular Value Fine-tuning (SVF)

**Key insight**: Weight matrices can be decomposed via SVD:
$$W = U \Sigma V^T$$

Where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)$ contains singular values.

**SVF**: Learn task-specific **scaling factors** for singular values:
$$\Sigma_{\text{task}} = \text{diag}(s_1 \sigma_1, s_2 \sigma_2, \ldots, s_r \sigma_r)$$
$$W_{\text{task}} = U \Sigma_{\text{task}} V^T$$

**Benefits:**
- Very few parameters per task (just the scalars $s_i$)
- No weight storage per task
- Fast adaptation at inference

#### 14.3.3 Task Detection Strategies

```
Transformer² Inference Strategies
───────────────────────────────────────────────────────────────────────
    Strategy            │ How It Works              │ Accuracy
───────────────────────────────────────────────────────────────────────
    Prompt-based        │ Keywords in prompt        │ Moderate
    Few-shot classify   │ Classify from examples    │ High
    Learned embedding   │ Dense task vector         │ Highest
───────────────────────────────────────────────────────────────────────
```

### 14.4 AB-MCTS and TreeQuest

**Multi-LLM Collaboration** [Sakana AI, 2025]

Multiple LLMs work together using Monte Carlo Tree Search:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          AB-MCTS Framework                                 │
│                                                                            │
│    Problem Statement                                                       │
│         │                                                                  │
│         ▼                                                                  │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │                    Search Tree                                     │ │
│    │                                                                    │ │
│    │                       Root                                         │ │
│    │                    (Problem)                                       │ │
│    │                    /      \                                        │ │
│    │                   /        \                                       │ │
│    │         ┌────────┐          ┌────────┐                             │ │
│    │         │ LLM A  │          │ LLM B  │    ← Breadth (new ideas)    │ │
│    │         │ idea 1 │          │ idea 2 │                             │ │
│    │         └───┬────┘          └───┬────┘                             │ │
│    │             │                   │                                  │ │
│    │      ┌──────┴──────┐     ┌──────┴──────┐                           │ │
│    │      │ LLM C       │     │ LLM A       │  ← Depth (refine ideas)   │ │
│    │      │ refine 1a   │     │ refine 2a   │                           │ │
│    │      └─────────────┘     └─────────────┘                           │ │
│    │                                                                    │ │
│    │    Adaptive branching: Switch between                              │ │
│    │    - Depth (improve existing solution)                             │ │
│    │    - Breadth (try new approach)                                    │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**TreeQuest**: Open-source framework implementing AB-MCTS.

---

## 15. Conclusion: The Path Forward

### 15.1 Evolution Summary

```
Historical Arc of Language Models
═══════════════════════════════════════════════════════════════════════════════

1990s    RNN                 Sequential, vanishing gradients
   │
   ▼
1997     LSTM                Gates solve vanishing gradients
   │
   ▼
2014     Attention           Dynamic memory access, still sequential
   │
   ▼
2017     Transformer         Parallel attention, O(L²) complexity
   │
   ├──────────────────┬──────────────────┐
   ▼                  ▼                  ▼
2018     GPT          BERT              Encoder-Decoder
         (Decoder)    (Encoder)         (T5, BART)
   │
   ▼
2020     GPT-3        Scaling laws      Emergent abilities
   │
   ▼
2022     InstructGPT  RLHF              Alignment
   │
   ├──────────────────┬──────────────────┐
   ▼                  ▼                  ▼
2023     GPT-4        Mamba             RWKV
         (Scaling)    (SSM)             (Linear RNN)
   │
   ▼
2024     DeepSeek-V3  MoE + MLA         Efficient at scale
   │
   ├──────────────────┬──────────────────┬──────────────────┐
   ▼                  ▼                  ▼                  ▼
2025     RLVR         Sparse Attention  Diffusion LLM      HRM/TRM
         (Verifiable  (O(L·k))          (Parallel gen)     (Recursion)
          rewards)
   │
   ▼
Future   Continual Learning + Memory + Self-Modification

═══════════════════════════════════════════════════════════════════════════════
```

### 15.2 Current State of the Art (Late 2025)

| Capability | SOTA Approach | Key Model |
|------------|---------------|-----------|
| General chat | Dense Transformer + RLHF | GPT-4, Claude |
| Efficient scale | MoE + MLA | DeepSeek-V3 |
| Long context | Sparse Attention | DeepSeek-V3.2 |
| Reasoning | RLVR + Long thinking | DeepSeek-R1 |
| Linear complexity | Mamba / RWKV | Mamba-3B, RWKV-7 |
| Fast generation | Diffusion | Gemini Diffusion |
| Recursive reasoning | HRM / TRM | TRM-7M |
| Continual learning | Nested Learning | HOPE |

### 15.3 Open Problems

1. **True continual learning**: Learning new knowledge without forgetting
2. **Grounded reasoning**: Connecting to physical world, not just text
3. **Efficiency at scale**: Linear attention that matches quadratic quality
4. **Multimodal integration**: Unified architecture for all modalities
5. **Interpretability**: Understanding what models actually learn
6. **Safety**: Robust alignment that doesn't break under pressure

### 15.4 Convergence Hypothesis

The field appears to be converging on:

- **Hybrid architectures**: Combining attention (for complex reasoning) with linear models (for efficiency)
- **Multi-scale processing**: Fast and slow systems (like HRM)
- **Verifiable training**: RLVR where possible, RLHF where necessary
- **Adaptive models**: Self-modifying weights (Transformer², HOPE)
- **Recursive refinement**: Draft-then-revise instead of one-shot generation

---

## 16. References

### Core Papers

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
   https://arxiv.org/abs/1706.03762

2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*.

3. DeepSeek-AI. (2024). "DeepSeek-V3 Technical Report."
   https://arxiv.org/abs/2412.19437

4. Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
   https://arxiv.org/abs/2312.00752

5. Peng, B., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
   https://arxiv.org/abs/2305.13048

### Architecture Innovations

6. Shazeer, N. (2020). "GLU Variants Improve Transformer."
   https://arxiv.org/abs/2002.05202

7. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
   https://arxiv.org/abs/2104.09864

8. Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
   https://arxiv.org/abs/1910.07467

### Post-Training

9. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback."
   https://arxiv.org/abs/2203.02155

10. DeepSeek-AI. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning."
    https://arxiv.org/abs/2402.03300

### Diffusion and Memory

11. Nie, S., et al. (2025). "Large Language Diffusion Models."
    https://arxiv.org/abs/2502.09992

12. Behrouz, A., et al. (2025). "Nested Learning: The Illusion of Deep Learning Architectures."
    *NeurIPS 2025*.

13. Sakana AI. (2025). "Transformer²: Self-Adaptive LLMs."
    https://sakana.ai/transformer-squared/

### Recursive Reasoning

14. Sapient Intelligence. (2025). "Hierarchical Reasoning Model."
    https://arxiv.org/abs/2506.21734

15. Samsung SAIT. (2025). "Less is More: Recursive Reasoning with Tiny Networks."
    https://arxiv.org/abs/2510.04871

### Tutorials and Surveys

16. Raschka, S. (2025). "From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates."
    https://sebastianraschka.com/blog/2025/technical-deepseek.html

17. Raschka, S. (2025). "The State of Reinforcement Learning for LLM Reasoning."
    https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html

18. The Gradient. (2023). "Mamba Explained."
    https://thegradient.pub/mamba-explained/

### Resources

19. Karpathy, A. (2024). "minbpe: Minimal BPE tokenizer."
    https://github.com/karpathy/minbpe

20. Google. "SentencePiece: Unsupervised text tokenizer."
    https://github.com/google/sentencepiece

21. RWKV. "RWKV-LM: RWKV Language Model."
    https://github.com/BlinkDL/RWKV-LM

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Attention** | Mechanism for computing weighted sums over sequence elements based on relevance |
| **BPE** | Byte Pair Encoding; subword tokenization algorithm |
| **Causal Mask** | Prevents attending to future tokens in autoregressive generation |
| **FFN** | Feed-Forward Network; position-wise MLP in transformers |
| **GRPO** | Group Relative Policy Optimization; simplified RL for LLMs |
| **KV Cache** | Stored key-value pairs for efficient autoregressive inference |
| **MLA** | Multi-head Latent Attention; DeepSeek's compressed attention |
| **MoE** | Mixture of Experts; sparse activation of specialized sub-networks |
| **MTP** | Multi-Token Prediction; predicting multiple future tokens |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **RLVR** | Reinforcement Learning from Verifiable Rewards |
| **RMSNorm** | Root Mean Square Normalization; simplified LayerNorm |
| **RoPE** | Rotary Position Embedding; rotation-based positional encoding |
| **SFT** | Supervised Fine-Tuning |
| **SSM** | State Space Model; linear recurrence for sequences |
| **SwiGLU** | Swish-Gated Linear Unit; gated FFN activation |

---

## Appendix B: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $x_t$ | Input token at position $t$ |
| $h_t$ | Hidden state at position $t$ |
| $W$ | Weight matrix |
| $\odot$ | Element-wise (Hadamard) product |
| $\sigma$ | Sigmoid function: $\sigma(x) = 1/(1+e^{-x})$ |
| $\text{softmax}$ | $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$ |
| $Q, K, V$ | Query, Key, Value matrices in attention |
| $d_k$ | Dimension of key vectors |
| $n$ | Sequence length |
| $d$ | Model hidden dimension |
| $L$ | Loss function |
| $\theta$ | Model parameters |
| $\nabla$ | Gradient operator |

---

*Last updated: December 2025*

*This document is intended as a research reference. For implementation details, consult the original papers and official codebases.*
