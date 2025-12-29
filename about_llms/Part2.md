# From GPT-2 to DeepSeek-V3: A Complete Guide to State-of-the-Art LLM Architecture

## Table of Contents

1. [The Foundation: Vanilla Transformer & GPT-2](#1-the-foundation-vanilla-transformer--gpt-2)
2. [Key Innovations Since GPT-2](#2-key-innovations-since-gpt-2)
3. [DeepSeek-V3 Architecture Overview](#3-deepseek-v3-architecture-overview)
4. [Multi-Head Latent Attention (MLA)](#4-multi-head-latent-attention-mla)
5. [Mixture of Experts (DeepSeekMoE)](#5-mixture-of-experts-deepseekmoe)
6. [Multi-Token Prediction (MTP)](#6-multi-token-prediction-mtp)
7. [Pre-Training: Data & Process](#7-pre-training-data--process)
8. [Post-Training: SFT & Reinforcement Learning](#8-post-training-sft--reinforcement-learning)
9. [Infrastructure Innovations](#9-infrastructure-innovations)
10. [Putting It All Together](#10-putting-it-all-together)

---

## 1. The Foundation: Vanilla Transformer & GPT-2

### 1.1 What is a Language Model?

A language model predicts the probability of the next word (or "token") given previous words. Think of autocomplete on your phone - that's a simple language model.

**Mathematical formulation:**

```
P(next_token | previous_tokens)
```

### 1.2 The Transformer Architecture (2017)

The Transformer, introduced in "Attention Is All You Need," revolutionized NLP. Here's the core idea:

**The Self-Attention Mechanism:**

Instead of processing text sequentially (like reading word by word), the Transformer looks at ALL words simultaneously and learns which words are important for understanding each other word.

```
Input: "The cat sat on the mat because it was tired"

For the word "it", attention helps the model focus on "cat"
(since "it" refers to "cat")
```

**The Math Behind Attention:**

Given an input sequence, we create three vectors for each position:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I carry?"

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- QK^T computes similarity between all pairs of positions
- √d_k is a scaling factor (d_k = dimension of keys)
- softmax converts scores to probabilities
- Multiplication with V gives weighted combination
```

**Example with numbers:**

```
Suppose d_k = 64 (dimension per head)
Q, K, V each have shape: [sequence_length, 64]

For a sentence of 10 tokens:
QK^T has shape: [10, 10]  (each token's attention to every other token)
After softmax: still [10, 10], but rows sum to 1
Output: [10, 64] (weighted values)
```

### 1.3 Multi-Head Attention

Instead of one attention operation, we run multiple "heads" in parallel, each learning different patterns:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

**Why multiple heads?**

- Head 1 might learn syntactic relationships
- Head 2 might learn semantic similarity
- Head 3 might learn positional patterns
- etc.

### 1.4 The GPT-2 Architecture

GPT-2 is a **decoder-only** Transformer (it only generates text, doesn't encode input separately).

**Structure:**

```
┌─────────────────────────────┐
│     Token + Position        │  ← Input embeddings
│        Embedding            │
├─────────────────────────────┤
│                             │
│    ┌───────────────────┐    │
│    │  Masked Multi-Head│    │
│    │    Attention      │    │  ← Can only see previous tokens
│    ├───────────────────┤    │
│    │   Add & LayerNorm │    │
│    ├───────────────────┤    │
│    │   Feed-Forward    │    │  ← MLP: expand then contract
│    │     Network       │    │
│    ├───────────────────┤    │
│    │   Add & LayerNorm │    │
│    └───────────────────┘    │
│           × 48              │  ← Repeat 48 times (for GPT-2 XL)
│                             │
├─────────────────────────────┤
│     Linear + Softmax        │  ← Output probabilities
└─────────────────────────────┘
```

**Key GPT-2 Parameters:**
| Model | Parameters | Layers | d_model | Heads |
|-------|-----------|--------|---------|-------|
| GPT-2 Small | 117M | 12 | 768 | 12 |
| GPT-2 Medium | 345M | 24 | 1024 | 16 |
| GPT-2 Large | 762M | 36 | 1280 | 20 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 |

**The Feed-Forward Network (FFN):**

```
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2

Typically: d_ff = 4 × d_model
```

This expands the dimension (e.g., 1600 → 6400) then contracts it back.

### 1.5 The KV Cache Problem

During text generation (inference), we generate one token at a time:

```
Step 1: "The" → predict "cat"
Step 2: "The cat" → predict "sat"
Step 3: "The cat sat" → predict "on"
...
```

**Naive approach:** Recompute attention for ALL previous tokens every step. Very slow!

**KV Cache solution:** Store the Key and Value vectors for previous tokens.

```
Memory per layer = 2 × sequence_length × num_heads × head_dim × bytes_per_value

For GPT-2 XL with 2048 tokens:
= 2 × 2048 × 25 × 64 × 2 bytes ≈ 13 MB per layer
× 48 layers ≈ 624 MB just for KV cache!
```

This becomes a huge problem for:

- Longer contexts (128K tokens)
- Larger models (hundreds of billions of parameters)
- Batch inference (multiple users)

**This is one of the key problems DeepSeek-V3 solves!**

---

## 2. Key Innovations Since GPT-2

Before diving into DeepSeek-V3, let's understand the evolutionary improvements:

### 2.1 Rotary Position Embeddings (RoPE)

**Problem:** GPT-2 uses learned absolute position embeddings, which don't generalize to longer sequences.

**Solution (RoPE):** Encode position through rotation in the embedding space.

```
Rotate Q and K by angle θ based on position:
  Q_rotated = Q × R(θ × position)
  K_rotated = K × R(θ × position)

Where R(θ) is a rotation matrix
```

**Benefits:**

- Relative positions are naturally encoded
- Better length generalization
- Can extend to longer contexts

### 2.2 RMSNorm (vs LayerNorm)

**LayerNorm (original):**

```
LayerNorm(x) = (x - mean(x)) / std(x) × γ + β
```

**RMSNorm (simpler, faster):**

```
RMSNorm(x) = x / RMS(x) × γ

where RMS(x) = √(mean(x²))
```

Removes the mean centering - empirically works just as well, but ~10-15% faster.

### 2.3 SwiGLU Activation (vs GELU)

**Original FFN:**

```
FFN(x) = GELU(xW_1) × W_2
```

**SwiGLU FFN:**

```
FFN(x) = (Swish(xW_gate) ⊙ (xW_up)) × W_down

where Swish(x) = x × sigmoid(x)
      ⊙ = element-wise multiplication
```

This "gated" structure gives the network more expressivity.

### 2.4 Grouped-Query Attention (GQA)

**Problem:** Standard Multi-Head Attention stores separate K, V for each head. Huge memory!

**GQA Solution:** Share K, V across groups of query heads.

```
Standard MHA (32 heads): 32 Q heads, 32 K heads, 32 V heads
GQA (8 KV groups):       32 Q heads, 8 K heads, 8 V heads
Multi-Query (MQA):       32 Q heads, 1 K head, 1 V head
```

Reduces KV cache by 4x (or more) with minimal quality loss.

**DeepSeek goes further with MLA - we'll see how soon!**

---

## 3. DeepSeek-V3 Architecture Overview

DeepSeek-V3 is a **671 billion parameter** model, but only **37 billion parameters** are activated for each token. This is achieved through Mixture of Experts (MoE).

### 3.1 High-Level Stats

| Property                   | Value                          |
| -------------------------- | ------------------------------ |
| Total Parameters           | 671B                           |
| Activated Parameters/Token | 37B                            |
| Training Tokens            | 14.8 trillion                  |
| Context Length             | 128K tokens                    |
| Number of Layers           | 61                             |
| Hidden Dimension           | 7168                           |
| Number of Attention Heads  | 128                            |
| Total Experts              | 257 (256 routed + 1 shared)    |
| Activated Experts/Token    | 9 (8 routed + 1 shared)        |
| Training Cost              | 2.788M H800 GPU hours (~$5.5M) |

### 3.2 The Basic Block Structure

```
┌─────────────────────────────────────────┐
│           Input Embedding               │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐  │
│  │    RMSNorm                        │  │
│  ├───────────────────────────────────┤  │
│  │    Multi-Head Latent Attention    │  │  ← Novel! Compressed KV
│  │           (MLA)                   │  │
│  ├───────────────────────────────────┤  │
│  │    Residual Connection            │  │
│  ├───────────────────────────────────┤  │
│  │    RMSNorm                        │  │
│  ├───────────────────────────────────┤  │
│  │    DeepSeekMoE                    │  │  ← 256 Experts, pick 8
│  │    (Mixture of Experts FFN)       │  │
│  ├───────────────────────────────────┤  │
│  │    Residual Connection            │  │
│  └───────────────────────────────────┘  │
│              × 61 layers                │
│                                         │
├─────────────────────────────────────────┤
│     Output: Next Token Probabilities    │
└─────────────────────────────────────────┘
```

---

## 4. Multi-Head Latent Attention (MLA)

This is DeepSeek's solution to the KV cache memory problem - and it's brilliant.

### 4.1 The Problem with Standard Attention

In standard attention, for each layer, we store:

```
K cache: [batch, seq_len, num_heads, head_dim]
V cache: [batch, seq_len, num_heads, head_dim]
```

For DeepSeek-V3 with 128K context:

```
Per layer: 128K × 128 heads × 128 dim × 2 (K+V) × 2 bytes = ~8.4 GB
× 61 layers = ~512 GB just for KV cache!
```

### 4.2 MLA's Key Insight: Compress First, Decompress Later

**Core idea:** Instead of storing full K and V, store a compressed "latent" vector.

```
Standard:
  h → W_K → K (high dimensional)
  h → W_V → V (high dimensional)
  Store K, V in cache

MLA:
  h → W_DKV → c_KV (LOW dimensional latent)  ← Compress!
  Store only c_KV in cache

  At attention time:
  c_KV → W_UK → K (decompress)
  c_KV → W_UV → V (decompress)
```

### 4.3 Mathematical Formulation

**Step 1: Create compressed latent vector**

```
c_KV = h × W_DKV

where:
  h ∈ R^d (hidden state, d = 7168 for DeepSeek-V3)
  W_DKV ∈ R^(d × d_c)
  c_KV ∈ R^d_c (compressed, d_c << d)

  d_c might be ~512, while full KV would be ~16384
```

**Step 2: Decompress when needed**

```
K = c_KV × W_UK + RoPE(c_KV × W_KR)
V = c_KV × W_UV

where:
  W_UK ∈ R^(d_c × n_h × d_h)
  W_UV ∈ R^(d_c × n_h × d_h)
  W_KR handles RoPE (rotary position embedding)
```

**Step 3: Standard attention computation**

```
Q = h × W_Q  (queries computed normally)
Attention = softmax(Q × K^T / √d) × V
```

### 4.4 Memory Savings

```
Standard MHA:
  Cache size per token = 2 × n_heads × d_head = 2 × 128 × 128 = 32,768 values

MLA:
  Cache size per token = d_c ≈ 512 values

Compression ratio: ~64x smaller!
```

### 4.5 Visual Comparison

```
Standard Multi-Head Attention:
┌─────┐    ┌─────────────────────────────┐
│  h  │───→│  W_K  │───→ K (store in cache)
│     │    ├───────┤
│     │───→│  W_V  │───→ V (store in cache)
│     │    ├───────┤
│     │───→│  W_Q  │───→ Q
└─────┘    └───────┴─────────────────────┘

Multi-Head Latent Attention:
┌─────┐    ┌──────────────────────────────────────┐
│  h  │───→│  W_DKV  │───→ c_KV (SMALL, store this)
│     │    ├─────────┤              │
│     │    │         │    ┌─────────┴─────────┐
│     │    │         │    │ W_UK → K (compute) │
│     │    │         │    │ W_UV → V (compute) │
│     │    ├─────────┤    └───────────────────┘
│     │───→│  W_Q    │───→ Q
└─────┘    └─────────┴────────────────────────────┘
```

### 4.6 Why This Works

The key insight is that K and V are **redundant** across heads. By projecting to a low-dimensional latent space, we capture the essential information while discarding redundancy. The learned projection matrices figure out what to keep.

---

## 5. Mixture of Experts (DeepSeekMoE)

### 5.1 The Basic MoE Idea

**Problem:** Making models bigger improves quality, but:

- More parameters = more computation
- Linear scaling: 2x params = 2x compute

**MoE Solution:** Have many "expert" sub-networks, but only activate a few per token.

```
Standard FFN:
  Every token uses the same FFN with ALL parameters

MoE:
  Token 1: Uses Experts 3, 7, 12 (relevant to this token)
  Token 2: Uses Experts 1, 5, 9 (different experts!)
  Token 3: Uses Experts 2, 7, 14
  ...
```

### 5.2 DeepSeekMoE Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Token                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐                                            │
│  │ Router  │─────────────────────────────────────┐      │
│  │ Network │  Computes affinity scores for all   │      │
│  └────┬────┘  256 experts                        │      │
│       │                                          │      │
│       ▼                                          ▼      │
│  ┌────────────────────────────────────────────────────┐ │
│  │   Expert Selection (Top-8 + 1 Shared)              │ │
│  └────────────────────────────────────────────────────┘ │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │                                                 │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐ ┌──────┐  │    │
│  │  │ E_1 │ │ E_2 │ │ E_3 │ ... │ E_8 │ │Shared│  │    │
│  │  └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘ └──┬───┘  │    │
│  │     │       │       │           │       │      │    │
│  │     └───────┴───────┴─────┬─────┴───────┘      │    │
│  │                           ▼                    │    │
│  │              Weighted Sum of Outputs           │    │
│  └─────────────────────────────────────────────────┘    │
│                           │                             │
│                           ▼                             │
│                    Output Token                         │
└─────────────────────────────────────────────────────────┘
```

### 5.3 The Router Mechanism

The router decides which experts process each token:

```python
# Simplified router logic
def route(token_embedding):
    # Compute affinity with all 256 experts
    scores = token_embedding @ expert_embeddings.T  # [256]

    # Apply sigmoid (not softmax!) - allows independent selection
    affinities = sigmoid(scores)

    # Select top-8 experts
    top_8_indices = topk(affinities, k=8)

    # Compute gating weights (how much each expert contributes)
    gates = affinities[top_8_indices]
    gates = gates / sum(gates)  # Normalize

    return top_8_indices, gates
```

### 5.4 Fine-Grained Experts

DeepSeek uses **more, smaller experts** than typical MoE:

```
Typical MoE (e.g., Mixtral):
  8 experts total, pick 2
  Each expert: Full FFN size (large)

DeepSeekMoE:
  256 experts total, pick 8
  Each expert: 1/32 of FFN size (small)
```

**Why fine-grained?**

- Better knowledge decomposition
- More specialized experts
- Smoother routing (8 small experts vs 2 large ones)

### 5.5 Shared Expert

One expert is **always active** for every token:

```
Output = Shared_Expert(x) + Σ gate_i × Expert_i(x)
```

**Purpose:** Capture common knowledge that applies to ALL tokens, freeing routed experts to specialize.

Think of it like:

- Shared expert = General practitioner (handles common cases)
- Routed experts = Specialists (handle specific cases)

### 5.6 The Load Balancing Problem

**Problem:** Without intervention, some experts become "popular" and others unused:

```
Bad routing:
  Expert 1: 50% of tokens (overloaded!)
  Expert 2: 30% of tokens
  Expert 3-256: 0.08% each (wasted capacity!)
```

**Traditional solution:** Add auxiliary loss to encourage balance:

```
L_total = L_language + λ × L_balance
```

**Problem with auxiliary loss:** Hurts model quality! You're optimizing for balance, not for good predictions.

### 5.7 Auxiliary-Loss-Free Load Balancing (DeepSeek Innovation!)

**Key insight:** Use a dynamic bias to nudge routing, without affecting the loss function.

```python
def route_with_balance(token, expert_biases):
    # Compute base scores
    base_scores = compute_affinity(token)

    # Add bias for routing decision ONLY
    biased_scores = base_scores + expert_biases

    # Select experts using biased scores
    selected = topk(biased_scores, k=8)

    # BUT use original scores for gating weights!
    gates = sigmoid(base_scores[selected])

    # Update biases based on load
    for expert in all_experts:
        if expert.load > average_load:
            expert.bias -= γ  # Make less attractive
        else:
            expert.bias += γ  # Make more attractive

    return selected, gates
```

**Why this works:**

- The bias affects routing decisions (which experts are chosen)
- But the actual gating weights use unbiased scores
- So gradients during training are "clean" - no auxiliary loss needed!

---

## 6. Multi-Token Prediction (MTP)

### 6.1 Standard Next-Token Prediction

```
Input:  "The cat sat on the"
Target: "mat"

Loss = -log P("mat" | "The cat sat on the")
```

You only get one gradient signal per position.

### 6.2 Multi-Token Prediction

```
Input:  "The cat sat on the"
Predict simultaneously:
  Position +1: "mat"
  Position +2: "because"

Loss = -log P("mat") - log P("because" | "mat")
```

### 6.3 DeepSeek's MTP Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Main Transformer                     │
│                    (61 layers)                        │
│                        │                              │
│                        ▼                              │
│              Hidden States h_t                        │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌───────────────┐   ┌───────────────┐               │
│  │   MTP Head 1  │   │   MTP Head 2  │               │
│  │  (predict +1) │   │  (predict +2) │               │
│  └───────┬───────┘   └───────┬───────┘               │
│          │                   │                        │
│          ▼                   ▼                        │
│     Token t+1           Token t+2                     │
│    (main output)    (auxiliary output)                │
└──────────────────────────────────────────────────────┘
```

**Important:** MTP maintains causality by predicting sequentially:

```
h_t → predict token_{t+1}
h_t + embed(token_{t+1}) → predict token_{t+2}
```

### 6.4 Benefits of MTP

1. **Denser training signal:** More gradients per forward pass
2. **Better planning:** Model learns to "think ahead"
3. **Speculative decoding:** Can use during inference for speedup

### 6.5 MTP at Inference Time (Speculative Decoding)

```
Without MTP:
  Generate: token1, token2, token3, token4 (4 forward passes)

With MTP (speculative):
  Step 1: Predict token1, token2 (1 forward pass)
  Step 2: Verify both are correct
    - If yes: accept both, continue
    - If no: fall back to single-token generation
```

DeepSeek reports **85-90% acceptance rate** for the second token prediction.

---

## 7. Pre-Training: Data & Process

### 7.1 Training Data

```
Total tokens: 14.8 trillion

Data sources:
├── Web pages (majority)
├── E-books and documents
├── Code repositories
├── Mathematical content
├── Scientific papers
└── Multilingual content
```

**Data quality pipeline:**

1. **Deduplication:** Remove near-duplicate documents
2. **Quality filtering:** Remove low-quality, spam, toxic content
3. **Content extraction:** Clean HTML, extract main text
4. **Language filtering:** Ensure quality across languages
5. **Sensitive content filtering:** Remove harmful content

### 7.2 Tokenization

DeepSeek uses a Byte-level BPE tokenizer with vocabulary size ~100K.

```
Example:
"DeepSeek-V3 is amazing!" → [34521, 12, 789, 456, 23]
```

### 7.3 Training Hyperparameters

```
Optimizer: AdamW
  β1 = 0.9
  β2 = 0.95
  weight_decay = 0.1

Learning rate schedule:
  - Warmup: 2000 steps
  - Peak LR: 2.2 × 10^-4
  - Cosine decay to 10% of peak

Batch size: ~60M tokens (varies with context)
Training precision: FP8 (novel!)
```

### 7.4 Context Length Extension

Training happens in stages:

```
Stage 1: Pre-train with 4K context
         (most efficient for initial learning)

Stage 2: Extend to 32K context
         (lower learning rate, RoPE adjustment)

Stage 3: Extend to 128K context
         (further RoPE adjustment, careful tuning)
```

### 7.5 Training Stability

DeepSeek reports **zero irrecoverable loss spikes** during the entire training. This is remarkable and achieved through:

- Careful initialization
- Gradient clipping
- Monitoring and early intervention
- FP8 training with proper scaling

---

## 8. Post-Training: SFT & Reinforcement Learning

After pre-training, we have a model that can complete text. Post-training makes it actually _helpful_.

### 8.1 The Post-Training Pipeline

```
┌─────────────────┐
│  Pre-trained    │
│  Base Model     │
│  (DeepSeek-V3   │
│      Base)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Supervised    │
│   Fine-Tuning   │
│     (SFT)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reinforcement  │
│    Learning     │
│     (GRPO)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DeepSeek-V3   │
│     (Chat)      │
└─────────────────┘
```

### 8.2 Supervised Fine-Tuning (SFT)

**Goal:** Teach the model the format of helpful responses.

**Data format:**

```
{"instruction": "Write a poem about spring",
 "response": "Petals dance on gentle breeze..."}
```

**DeepSeek's SFT Data (~1.5M examples):**

1. **Reasoning Data** (math, code, logic):

   - Generated by DeepSeek-R1 (their reasoning model)
   - Includes chain-of-thought reasoning
   - Verified for correctness

2. **Non-Reasoning Data** (creative, general):
   - Generated by DeepSeek-V2.5
   - Verified by human annotators
   - Covers diverse topics

**Key technique - Sample Masking:**

```
Multiple samples are packed into one sequence for efficiency:
[Sample 1][Sample 2][Sample 3]

But samples are "invisible" to each other through masking:
  Sample 1 can only see Sample 1
  Sample 2 can only see Sample 2
  etc.
```

### 8.3 Reinforcement Learning with GRPO

**Standard RLHF (Reinforcement Learning from Human Feedback):**

```
1. Generate responses
2. Get reward from reward model (trained on human preferences)
3. Update policy using PPO algorithm
4. Repeat
```

**Problem with PPO:** Requires a "critic" model (same size as policy) → expensive!

**GRPO (Group Relative Policy Optimization):**

DeepSeek's innovation that eliminates the critic model.

```python
# Simplified GRPO
def grpo_update(prompts, policy_model, reward_model):
    for prompt in prompts:
        # Generate GROUP of responses
        responses = [policy_model.generate(prompt) for _ in range(G)]

        # Get rewards for all responses
        rewards = [reward_model.score(prompt, r) for r in responses]

        # Normalize within group (this replaces the critic!)
        mean_reward = mean(rewards)
        std_reward = std(rewards)
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        # Update policy to increase probability of better responses
        for response, advantage in zip(responses, advantages):
            if advantage > 0:
                increase_probability(response)
            else:
                decrease_probability(response)
```

**Key insight:** Instead of learning a value function (critic), use the group of samples as a baseline!

### 8.4 Two Types of Reward Models

**1. Rule-Based Rewards (for verifiable tasks):**

```python
def math_reward(response):
    answer = extract_answer(response)
    if answer == correct_answer:
        return 1.0
    else:
        return 0.0

def code_reward(response):
    code = extract_code(response)
    result = execute(code)
    if result == expected_output:
        return 1.0
    else:
        return 0.0
```

**Why rule-based?**

- No "reward hacking" possible
- Perfect accuracy
- Scales infinitely

**2. Model-Based Rewards (for open-ended tasks):**

```
Prompt: "Write a creative story about a robot"
Response: [Generated story]

Reward Model:
  - Trained on human preference data
  - Takes (prompt, response) → scalar reward
  - Captures subjective quality
```

### 8.5 Knowledge Distillation from DeepSeek-R1

A unique aspect: DeepSeek-V3 learns reasoning from their DeepSeek-R1 model.

```
DeepSeek-R1 (reasoning specialist)
         │
         │ Generate reasoning traces
         ▼
   SFT Data for V3
         │
         │ Fine-tune
         ▼
DeepSeek-V3 (general + reasoning)
```

**But there's a cycle:**

- R1 was trained on V3-Base
- V3-Chat learns from R1
- Final V3 used to improve R1
- Improved R1 used for next V3 version

---

## 9. Infrastructure Innovations

### 9.1 FP8 Training (First at Scale!)

**Standard training:** BF16 (16-bit brain floating point)
**DeepSeek innovation:** FP8 (8-bit floating point)

```
BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
FP8:  1 sign bit, 4 exponent bits, 3 mantissa bits
```

**Benefits:**

- 2x memory savings
- 2x faster matrix multiplications
- Effectively doubles available compute!

**Challenges solved:**

- Proper scaling to prevent overflow/underflow
- Mixed precision strategy (not everything in FP8)
- Accumulation in higher precision for numerical stability

### 9.2 DualPipe: Computation-Communication Overlap

**The MoE communication problem:**

In MoE, tokens need to be routed to different GPUs based on which experts they need. This creates **all-to-all communication** between GPUs.

```
GPU 1 has token A, needs Expert 5 (on GPU 2)
GPU 2 has token B, needs Expert 1 (on GPU 1)
GPU 3 has token C, needs Expert 5 (on GPU 2)
...

Without optimization:
  1. Compute (wait for communication)
  2. Communicate (all GPUs waiting)
  3. Compute (wait for communication)
  ...
```

**DualPipe solution:**

```
Time: ───────────────────────────────────────────►
GPU:
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Compute │    │ Compute │    │ Compute │
      │ Batch 1 │    │ Batch 2 │    │ Batch 3 │
      └────┬────┘    └────┬────┘    └────┬────┘
           │              │              │
           ▼              ▼              ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │  Comm   │    │  Comm   │    │  Comm   │
      │ Batch 1 │    │ Batch 2 │    │ Batch 3 │
      └─────────┘    └─────────┘    └─────────┘

Overlapped:
      ┌─────────┬─────────┬─────────┐
      │Compute 1│Compute 2│Compute 3│  (on GPU compute units)
      └─────────┴─────────┴─────────┘
      ┌─────────┬─────────┬─────────┐
      │ Comm 1  │ Comm 2  │ Comm 3  │  (on network, simultaneously)
      └─────────┴─────────┴─────────┘
```

Achieves near **100% GPU utilization** by overlapping compute and communication.

### 9.3 Hardware: 2048 H800 GPUs

```
Cluster setup:
├── 2048 NVIDIA H800 GPUs
├── Connected via InfiniBand (800 Gbps)
├── NVLink within nodes (8 GPUs per node)
└── 256 nodes total

Training time:
├── Pre-training: 2.664M GPU hours
├── Post-training: 0.1M GPU hours
└── Total: ~2.788M GPU hours
└── Cost: ~$5.5M (at ~$2/GPU-hour)
```

---

## 10. Putting It All Together

### 10.1 The Complete DeepSeek-V3 Picture

```
┌────────────────────────────────────────────────────────────┐
│                    DEEPSEEK-V3 ARCHITECTURE                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INPUT: Token sequence                                     │
│           │                                                │
│           ▼                                                │
│  ┌────────────────────┐                                    │
│  │  Token Embedding   │  + Rotary Position Embedding       │
│  └─────────┬──────────┘                                    │
│            │                                               │
│            ▼                                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │                   LAYER 1-61                       │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │              RMSNorm                         │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                     │                              │    │
│  │                     ▼                              │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │     Multi-Head Latent Attention (MLA)        │  │    │
│  │  │  • Compress K,V to low-dim latent (64x save) │  │    │
│  │  │  • 128 attention heads                       │  │    │
│  │  │  • RoPE for position encoding                │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                     │                              │    │
│  │              ┌──────┴──────┐ (residual)            │    │
│  │              ▼             │                       │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │              RMSNorm                         │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                     │                              │    │
│  │                     ▼                              │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │         DeepSeekMoE (FFN Layer)              │  │    │
│  │  │  ┌────────────────────────────────────────┐  │  │    │
│  │  │  │ Router: sigmoid scores, pick Top-8     │  │  │    │
│  │  │  │ + Auxiliary-loss-free load balancing   │  │  │    │
│  │  │  └────────────────────────────────────────┘  │  │    │
│  │  │           │                                  │  │    │
│  │  │           ▼                                  │  │    │
│  │  │  ┌─────┬─────┬─────┬───┬─────┬────────┐     │  │    │
│  │  │  │ E1  │ E2  │ E3  │...│ E8  │ Shared │     │  │    │
│  │  │  │     │     │     │   │     │ Expert │     │  │    │
│  │  │  └──┬──┴──┬──┴──┬──┴───┴──┬──┴────┬───┘     │  │    │
│  │  │     └─────┴─────┴─────────┴───────┘         │  │    │
│  │  │              Weighted sum                    │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                     │                              │    │
│  │              ┌──────┴──────┐ (residual)            │    │
│  │              ▼             │                       │    │
│  │              +◄────────────┘                       │    │
│  └────────────────────────────────────────────────────┘    │
│            × 61 layers                                     │
│                     │                                      │
│                     ▼                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Final RMSNorm                         │    │
│  └────────────────────────────────────────────────────┘    │
│                     │                                      │
│         ┌───────────┴───────────┐                          │
│         ▼                       ▼                          │
│  ┌─────────────┐        ┌─────────────┐                    │
│  │ Main Output │        │ MTP Output  │                    │
│  │   Head      │        │   Heads     │                    │
│  │ (next token)│        │(+2,+3 tokens)│                   │
│  └─────────────┘        └─────────────┘                    │
│                                                            │
└────────────────────────────────────────────────────────────┘

TRAINING PIPELINE:
┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Pre-training │────▶│    SFT      │────▶│    GRPO    │
│ 14.8T tokens │     │ 1.5M samples│     │    (RL)    │
│   FP8        │     │             │     │            │
└──────────────┘     └─────────────┘     └────────────┘
```

### 10.2 Key Takeaways: Why DeepSeek-V3 is SOTA

| Innovation                    | Benefit                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| MLA                           | 64x smaller KV cache, enabling 128K context                    |
| DeepSeekMoE                   | 671B total, 37B active → quality of large model, cost of small |
| Auxiliary-loss-free balancing | Better performance than alternatives                           |
| MTP                           | Denser training signal, speculative decoding                   |
| FP8 training                  | 2x compute efficiency                                          |
| DualPipe                      | Near-perfect GPU utilization                                   |
| GRPO                          | Efficient RL without critic model                              |
| R1 distillation               | Reasoning capabilities in general model                        |

### 10.3 Evolution Summary: GPT-2 → DeepSeek-V3

```
GPT-2 (2019)                          DeepSeek-V3 (2024)
├── 1.5B parameters                   ├── 671B params (37B active)
├── Absolute positions                ├── RoPE
├── LayerNorm                         ├── RMSNorm
├── GELU activation                   ├── SwiGLU
├── Standard attention                ├── MLA (compressed KV)
├── Dense FFN                         ├── MoE (256 experts)
├── Next-token only                   ├── Multi-token prediction
├── FP32/FP16 training                ├── FP8 training
├── SFT only                          ├── SFT + GRPO
└── 40GB text                         └── 14.8T tokens
```

### 10.4 The Path Forward

DeepSeek-V3 represents the current state-of-the-art, but the field continues to evolve:

- **Longer contexts:** Beyond 128K tokens (some models reaching 1M+)
- **Better reasoning:** R1-style thinking integrated into base models
- **Multimodal:** Vision, audio, video understanding
- **Efficiency:** Even more aggressive sparsity, quantization
- **Agents:** Tool use, code execution, real-world interaction

The innovations in DeepSeek-V3 - particularly MLA, auxiliary-loss-free MoE, and GRPO - are already being adopted by other models (Kimi K2, Mistral 3, etc.), cementing their place in the LLM toolkit.

---

## Appendix A: Glossary

| Term          | Definition                                                   |
| ------------- | ------------------------------------------------------------ |
| **Attention** | Mechanism for tokens to "look at" other tokens               |
| **BPE**       | Byte Pair Encoding - tokenization algorithm                  |
| **FFN**       | Feed-Forward Network - MLP in each transformer layer         |
| **GRPO**      | Group Relative Policy Optimization - DeepSeek's RL algorithm |
| **KV Cache**  | Stored keys and values for efficient generation              |
| **MLA**       | Multi-Head Latent Attention - compressed attention           |
| **MoE**       | Mixture of Experts - sparse activation                       |
| **MTP**       | Multi-Token Prediction - predict multiple future tokens      |
| **PPO**       | Proximal Policy Optimization - standard RL algorithm         |
| **RLHF**      | Reinforcement Learning from Human Feedback                   |
| **RMSNorm**   | Root Mean Square Normalization                               |
| **RoPE**      | Rotary Position Embedding                                    |
| **SFT**       | Supervised Fine-Tuning                                       |
| **SwiGLU**    | Swish-Gated Linear Unit - activation function                |

## Appendix B: Further Reading

1. **Original Transformer:** "Attention Is All You Need" (Vaswani et al., 2017)
2. **GPT-2:** "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
3. **RoPE:** "RoFormer" (Su et al., 2021)
4. **MoE basics:** "Outrageously Large Neural Networks" (Shazeer et al., 2017)
5. **DeepSeek-V2:** Technical Report (DeepSeek-AI, 2024)
6. **DeepSeek-V3:** Technical Report (DeepSeek-AI, 2024)
7. **DeepSeek-R1:** Technical Report (DeepSeek-AI, 2025)
8. **GRPO:** "DeepSeekMath" (DeepSeek-AI, 2024)
