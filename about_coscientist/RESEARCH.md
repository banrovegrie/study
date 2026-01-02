# Comprehensive Research Report: LLM-Based Theorem Proving and the Case for Building on Lean

**Prepared for:** Formal Verification / Theorem Proving System
**Date:** December 8, 2025
**Focus:** Architectural patterns for scaling LLM-based proof systems with verification cascade

---

## Executive Summary

This report synthesizes cutting-edge research (2024-2025) on scaling automated theorem proving with LLMs and addresses a critical strategic question: **should we create a new semi-formal language, or build tooling on existing infrastructure?**

### The Strategic Answer

**No new language. Build better tooling for Lean + markdown + LaTeX.**

The reasoning:
1. **Lean already has `sorry`** — it IS the semi-formal language
2. **Blueprint proves the workflow works** — Terry Tao formalized a 33-page proof in 3 weeks with 25 contributors
3. **History warns against new languages** — Knuth's WEB failed despite his influence; 40 years, no adoption
4. **210K theorems in Mathlib** — you can't rebuild this library in a new language
5. **The gap is tooling, not syntax** — we need tactics libraries and LLM integration

### The Research Synthesis

Traditional synthesis approaches (CEGIS, enumerative) fail because they cannot handle the combinatorial explosion and semantic richness of tactic languages, while pure neural approaches lack the guarantees needed for formal verification. The state-of-the-art is converging on hybrid architectures that combine:

1. **Neural-guided proof search** with **symbolic verification**
2. **Hierarchical reasoning** (informal sketch → formal proof)
3. **Retrieval-augmented generation** from large lemma libraries
4. **Test-time compute scaling** via reinforcement learning
5. **Self-improving systems** through expert iteration and synthetic data generation

**Key Finding:** The "verification cascade" idea aligns perfectly with emerging research showing that **fast heuristic checks before expensive formal verification dramatically improves efficiency**. Systems like DeepSeek-Prover-V2, AlphaProof, and Aristotle demonstrate that this multi-stage approach is not just theoretical—it's the path to IMO gold-medal performance.

---

# Part I: Strategic Decision — Build on Lean, Not Replace It

## 1. The Question

Can/should we create a new "semi-formal compilable language" for spec-based math and code? Something like literate programming but with formal verification?

## 2. The Answer

**No, and here's why:**

1. **Lean already has `sorry`** — it IS the semi-formal language
2. **Blueprint proves the workflow works** — Terry Tao formalized a 33-page proof in 3 weeks with 25 contributors
3. **History warns against new languages** — Knuth's WEB failed despite his influence; 40 years, no adoption
4. **210K theorems in Mathlib** — you can't rebuild this library in a new language
5. **The gap is tooling, not syntax** — we need tactics libraries and LLM integration

---

## 3. What Already Exists

### 3.1 Lean's `sorry` = Semi-Formal

`sorry` magically produces a proof of anything. It's unsound, Lean warns you, but it lets you:

```lean
theorem sqrt2_irrational : Irrational (Real.sqrt 2) := by
  intro ⟨p, q, hq, h⟩
  have hp : Even (p^2) := by sorry  -- TODO: fill later
  have hp' : Even p := by sorry
  sorry  -- finish
```

This IS semi-formal. Lean checks types and structure even with sorry's. You iterate: write skeleton → Lean verifies types → fill sorry's → iterate.

**The language already exists. The workflow needs tooling.**

### 3.2 Blueprint = The Workflow

Terry Tao + Patrick Massot built [Blueprint](https://terrytao.wordpress.com/2023/11/18/formalizing-the-proof-of-pfr-in-lean4-using-blueprint-a-short-tour/): a LaTeX/Lean hybrid where:

- Nodes = definitions, lemmas, theorems
- Colors = status (green=proven, blue=stated, black=not-yet-writable)
- Graph shows dependencies
- Human-readable proof links to Lean proof

25 contributors. 3 weeks. No new language needed.

**Blueprint IS the "literate Lean" you'd design if you built a new language.**

### 3.3 Jupyter/Org-mode = Literate Computation

[Jupyter notebooks](https://jupyter.org/) provide literate programming for computation:
- Prose + code + output
- Millions of users
- LIGO gravitational wave discovery documented in Jupyter

But they have [problems](https://scicomp.aalto.fi/scicomp/jupyter-pitfalls/):
- Out-of-order execution breaks reproducibility
- No modularity
- Hard to version control
- "Linear nature becomes a limitation"

[Org-mode babel](https://orgmode.org/worg/org-contrib/babel/intro.html) does literate programming better:
- Multi-language support
- Tangling (export executable code)
- Better structure

**Neither required a new language. They built tooling on existing languages.**

---

## 4. What Would A New Language Even Look Like?

Let's steelman the idea. A hypothetical "semi-formal language" would need:

```
SEMIFORMAL-LANG
├── Markdown-like prose
├── LaTeX for math notation
├── Partial formal verification (checkable but incomplete)
├── Citations/references
└── Executable/testable code
```

**This is exactly what Lean + Blueprint + sorry already provides.**

The only thing a new language could add:
1. **Better syntax** — marginal gain, huge porting cost
2. **Unified parser** — nice-to-have, not essential
3. **Custom IDE** — can build on VS Code extensions
4. **Specialized semantics** — what would they be?

**The cost-benefit doesn't work.** You'd spend years designing, implementing, and building tooling for a language that's marginally better than what exists.

Meanwhile, [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) (DeepMind, 2024):
- Uses Lean directly, no new language
- Solved 4/6 IMO 2024 problems (silver medal level)
- Solved P6 (hardest problem, only 5/609 humans solved it)
- 100% verifiable — every proof checked by Lean
- Published in [Nature](https://www.nature.com/articles/s41586-025-09833-y) (2025)

**Google chose to build ON Lean, not replace it.** The tooling around Lean is where the value is.

---

## 5. Why New Languages Fail

### 5.1 Knuth's WEB: A Cautionary Tale

[WEB](https://en.wikipedia.org/wiki/Web_(programming_system)) (1984): Donald Knuth's literate programming system.
- TeX + Pascal + macros
- Write documentation with embedded code
- Tangle → extract code
- Weave → produce documentation

**Result:** TeX and Metafont were written in WEB. Almost nothing else.

[Problems](http://akkartik.name/post/literate-programming):
- "When programming, it's not uncommon to write a function that's 'good enough for now' and revise it later. This is impossible to adequately do in literate programming."
- No unit testing support
- Doesn't work with agile methods
- "Programmers do not view themselves as essayists while programming"

**CWEB exists. 40 years later, nobody uses it.** Even at Stanford, Knuth's own institution.

### 5.2 Dijkstra's Formal Methods: Partially Adopted

[Dijkstra](https://en.wikipedia.org/wiki/On_the_Cruelty_of_Really_Teaching_Computer_Science) (1988) argued programming should be a branch of mathematics with formal provability as the criterion.

**What happened:**
- Formal methods used in safety-critical systems (avionics, medical)
- Not used in typical software development
- "Most companies do not consider it cost-effective"

Dijkstra influenced Knuth's literate programming ("the top-down style of exposition"). But even Dijkstra's own students don't use pure formal methods.

**Formal rigor is valuable. Mandating it for all code is not practical.**

---

## 6. The Meta-Lesson: Don't Build Languages, Build Tools

From [Stack Overflow](https://softwareengineering.stackexchange.com/questions/62727/when-is-it-reasonable-to-create-my-own-programming-language):

> "Problems with creating your own language include: missing out on all the libraries and frameworks built for existing languages, spending a lot of time designing and implementing the new language instead of working on the real programming task."

Even Swift, Rust, and Clang use LLVM rather than building their own backend.

**The right abstraction level:** Build on Lean, not beside it.

### 6.1 Proof Assistants Already Exist

[Lean vs Coq vs Isabelle](https://proofassistants.stackexchange.com/questions/43/proof-assistants-for-beginners-a-comparison):

| System | Foundation | Automation | Use |
|--------|-----------|------------|-----|
| Lean | Dependent types | Growing | Math formalization |
| Coq | CIC | Moderate | Verified software |
| Isabelle | HOL | Strong (sledgehammer) | Verification |

Lean 4 has 210K+ theorems in Mathlib. Building a new proof assistant = rebuilding this library.

### 6.2 TLA+ Shows What Works

[TLA+](https://en.wikipedia.org/wiki/TLA+) (Leslie Lamport):
- Specification language for distributed systems
- Used by Amazon (DynamoDB, S3, EBS), Microsoft (Xbox, Azure Cosmos DB)
- Built on existing math (set theory, temporal logic)
- Didn't invent new foundations

**TLA+ succeeded because it solved a real problem without reinventing logic.**

---

## 7. Counter-Arguments Considered

### "But Lean's syntax is ugly"

True. Lean syntax isn't natural for mathematicians:

```lean
theorem foo : ∀ n : ℕ, n + 0 = n := by
  intro n
  rfl
```

vs. how a mathematician writes:

```
Theorem: For all n ∈ ℕ, n + 0 = n.
Proof: By reflexivity. □
```

**Counter:** Blueprint already provides the translation layer. The LaTeX document IS the human-readable version. The Lean code is the machine-checkable version. No new language needed — just better integration.

### "We could make something better"

Maybe. But consider:
- Lean took 10+ years to reach current state
- Mathlib has 100K+ definitions, 210K+ theorems
- Community of 2500+ contributors

You'd need to rebuild ALL of this. The opportunity cost is massive.

### "What about domain-specific needs?"

For Putnam problems specifically, maybe we want:
- Competition-math notation
- Specific problem templates
- Custom automation

**Counter:** This is EXACTLY what skills/tools/commands are for. Build domain-specific TOOLING on Lean, not a domain-specific LANGUAGE.

### "Dijkstra wanted pure formal methods"

Dijkstra argued programming should be mathematics. But:
- Pure formal methods never caught on
- Cost-benefit doesn't work for most software
- Even safety-critical systems use targeted formal methods, not full formalization

**Our approach:** Semi-formal as default, formal when needed. This is pragmatic Dijkstra.

---

# Part II: Research Synthesis — Scaling LLM-Based Theorem Proving

## 8. Program Synthesis Scaling: Why Traditional Approaches Fail

### 8.1 The Failure Modes of CEGIS and Enumerative Synthesis

**Counterexample-Guided Inductive Synthesis (CEGIS)** has fundamental limitations for theorem proving:

#### CEGIS Limitations:
1. **Non-termination on unsatisfiable specifications**: CEGIS explores the solution space incrementally. If no proof exists, it may never terminate—it must explore the entire space before concluding unsatisfiability.

2. **Universal quantifier explosion**: CEGIS relies on SMT solvers to verify candidates. Theorem proving requires reasoning about universally quantified formulas over infinite domains, which SMT solvers cannot encode efficiently.

3. **Tactic language expressiveness**: Lean/Coq tactic languages include higher-order functions, dependent types, and computational effects. These are **too powerful for SMT encoding**—you cannot reduce tactic synthesis to a finite-domain SAT problem.

4. **Known failure cases**: CEGIS works by testing candidates on finite example sets. For problems like "find C such that x ≠ y + C for all x,y", any finite set has a solution, but no universal solution exists. The synthesizer never learns this is impossible.

**Enumerative Synthesis** scales even worse:

1. **Exponential growth**: Time to enumerate all expressions up to size N grows exponentially with N.

2. **No semantic guidance**: Pure enumeration has no notion of "getting closer" to a solution—it's blind search.

3. **Cannot handle loops/recursion**: Most neural enumerative approaches cannot synthesize iterative loops or higher-order functions, limiting them to trivial programs.

### 8.2 Neural-Guided and Hybrid Approaches (State of the Art)

The solution: **combine neural models for semantic guidance with symbolic verifiers for correctness**.

#### DeepSeek-Prover-V2 (2025)
- **Architecture**: Prompts a frontier LLM (DeepSeek-V3) to decompose theorems into high-level proof sketches, then formalizes them in Lean 4 as subgoal sequences.
- **Key innovation**: Recursive proof search where a 7B model handles subgoal proofs, synthesizing successful traces into chain-of-thought training data.
- **Performance**: 88.9% on MiniF2F-test, 49/658 on PutnamBench, 6/15 on recent AIME problems.
- **Why it works**: Bridges informal mathematical reasoning (where LLMs excel) with formal verification (where symbolic systems excel).

#### AlphaProof (2024 - Nature 2025)
- **Architecture**: AlphaZero-inspired RL agent trained on millions of auto-formalized problems.
- **Test-Time RL**: Generates millions of problem variants at inference time for deep, problem-specific adaptation.
- **Training scale**: 100M formal mathematics problems; 50 epochs over 12 trillion tokens for the encoder-decoder transformer.
- **Performance**: IMO 2024 silver medal (3/5 non-geometry problems, including the hardest).
- **Key insight**: Operates entirely within Lean—proofs are verifiable "tactics," enabling RL reward computation without human verification.

#### Aristotle (2025)
- **Achievement**: IMO 2025 gold-medal performance.
- **Method**: Integrates informal, human-like reasoning with formal Lean 4 verification. Uses sophisticated proof search algorithm with high-level lemma generation and formalization.

#### DeepSeekMath-V2 (2025)
- **Performance**: IMO 2025 gold (5/6 problems), CMO 2024 gold, 118/120 on Putnam 2024.
- **Scaling**: Achieved with scaled test-time compute (not just model size).

#### Llemma (ICLR 2024)
- **Training**: Continued pretraining of Code Llama on Proof-Pile-2 (scientific papers, math web data, mathematical code).
- **Performance**: First open base model to demonstrate in-context theorem proving, surpassing GPT-4 approaches on miniF2F.
- **Innovation**: Includes 1.5B tokens of formal math (Lean/Isabelle proof states) in training data.

#### COPRA (2024)
- **Method**: Retrieval-augmented GPT-4 for formal theorem proving.
- **Features**:
  - Uses error messages as part of LLM prompts
  - Maintains a memory of incorrect predictions to avoid repetition
  - Takes informal proofs as optional input for zero-shot proof-step prediction
  - No best-first search; samples from GPT-4 up to 60 times

### 8.3 What Makes These Approaches Work

**Common architectural patterns:**

1. **Hierarchical decomposition**: Break complex proofs into smaller subgoals that are easier to synthesize and verify.

2. **Learned value functions**: Train models to estimate proof state quality, guiding search toward promising branches.

3. **Symbolic verification in the loop**: Use Lean/Coq kernel to filter out incorrect attempts, providing perfect supervision signal.

4. **Massive synthetic data generation**: Create training data by auto-formalization, proof mining, and variation generation.

5. **Test-time compute scaling**: Spend more computation on hard problems through tree search, RL, or iterative refinement.

---

## 9. Metaprogramming with LLMs: Tactic Generation and Strategy Learning

### 9.1 Tactic Prediction and Proof Step Generation

**State of the art in tactic-level interaction:**

#### LeanProgress (2025)
- **Innovation**: Predicts remaining proof steps through learned progress signals—going beyond log-probability or manual heuristics.
- **Performance**: +3.8% improvement on Mathlib4 (baseline 41.2% → 45.0%).
- **Method**: Constructs balanced dataset of 80K proof trajectories, selects shortest path as ground truth via tree search.

#### llmstep (2024)
- **Purpose**: Tactic suggestions in Lean 4 for human users.
- **Model**: Pythia 2.8B fine-tuned on (state, next-tactic) pairs from LeanDojo Benchmark.
- **Integration**: VSCode extension providing real-time suggestions.

#### Lean-STaR (2024)
- **Innovation**: Generates informal thoughts before each proof step, boosting reasoning.
- **Training**: ~50K thought-augmented examples using retrospective ground-truth tactics from Mathlib.
- **Performance**: State-of-the-art on miniF2F-test.
- **Key idea**: Chain-of-thought for theorem proving—make reasoning explicit before action.

### 9.2 Meta-Tactic Learning in Lean 4

**Lean 4 Metaprogramming Architecture:**

Lean 4's metaprogramming system operates through three core monads:

1. **MetaM**: Meta-level operations (expression manipulation, type checking, unification)
2. **TermElabM**: Term elaboration (building expressions from syntax)
3. **TacticM**: Tactic elaboration (modifying proof state)

**Hierarchy**: `TacticM = ReaderT Context $ StateRefT State TermElabM`

**Key capabilities for meta-tactics:**

- **Syntax extension**: Define custom tactic syntax via `elab` macro.
- **Elaboration**: Implement tactic behavior by manipulating goals/hypotheses.
- **Error recovery**: `MonadExcept` instance allows backtracking on failure.
- **Composition**: Tactics compose naturally through monadic sequencing.

**Implications for LLM systems:**

- LLMs can generate Lean 4 tactics as code, not just apply fixed tactics.
- Meta-tactics can be learned from successful proof patterns and stored as reusable skills.
- The distinction between "using a tactic" and "generating a new tactic" blurs—both are code generation.

### 9.3 Self-Improving Proof Systems

#### Meta-Rewarding Language Models (2024)
- **Innovation**: LLM judges its own judgments to improve meta-cognitive abilities.
- **Results**: Llama-3-8B-Instruct improved from 22.9% → 39.4% on AlpacaEval 2.
- **Application to theorem proving**: Model can evaluate proof step quality and refine its own evaluation criteria.

#### Recursive Meta-Prompting
- **Concept**: LLMs autonomously generate and refine prompts in a recursive manner.
- **Analogy to metaprogramming**: Similar to how Lean metaprograms generate Lean code.
- **For theorem proving**: Could generate tactic prompts, evaluate success, and refine prompting strategy.

#### STP: Self-Play Theorem Prover (2025)
- **Architecture**: Two components—**Conjecturer** (proposes challenging problems) and **Prover** (attempts proofs).
- **Training**: Jointly trained in adversarial feedback loop, creating implicit curriculum learning.
- **Performance**:
  - Lean: 28.5% on LeanWorkbook (vs. 13.2% for expert iteration)
  - miniF2F-test: 65.0% (pass@3200)
  - PutnamBench: 8/644 (pass@3200)
- **Key insight**: Circumvents sparse reward problem by generating provable conjectures instead of trying to prove all theorems.

#### Expert Iteration and Policy Distillation
- **Standard approach**: Alternate between (1) proof search generating successful proofs, (2) fine-tuning on successes.
- **Limitation**: Plateaus quickly due to sparse rewards—exponentially many samples needed for hard theorems.
- **Solution**: Combine with curriculum learning, synthetic data, and self-play.

---

## 10. Citation and Grounding: Linking Neural Reasoning to Formal Knowledge

### 10.1 The Retrieval Challenge

**Problem**: Mathlib4 contains 44K+ theorems. Contextualizing a proof requires selecting the right 10-50 lemmas from this vast space.

**Why it's hard:**
- **Semantic search**: Must understand what lemma is needed, not just keyword match.
- **Dynamic context**: As proof progresses, relevant lemmas change.
- **Formalization gap**: Human thinks "triangle inequality," Mathlib calls it `dist_triangle`.

### 10.2 Vector-Based Semantic Search

#### Moogle, LeanSearch, Search-Mathlib
- **Architecture**: Each theorem → semantic embedding (via sentence transformers like all-mpnet-base-v2).
- **Indexing**: Vector database (FAISS, Chroma DB) for fast cosine similarity search.
- **Query**: User types natural language, LaTeX, or Lean code → embedded → nearest neighbors retrieved.
- **Database entries**: Pair of (formal theorem statement, informal version).

#### A Semantic Search Engine for Mathlib4 (2024)
- **Benchmark**: First systematic evaluation of search engines for Mathlib4.
- **Available at**: https://leansearch.net/
- **Method**: Augment queries for improved context understanding before semantic search.

#### Formal vs. Neural Search
- **Formal (Loogle, exact?)**: Metaprogramming-based, returns all theorems matching query shape. Predictable but requires knowing statement structure.
- **Neural (Moogle, LeanSearch)**: LLM embeddings, approximate matches. Good for exploration and natural language queries.

### 10.3 Retrieval-Augmented Proving

#### LeanDojo (2023)
- **Innovation**: Extracts fine-grained premise annotations from Lean proofs.
- **ReProver**: LLM-based prover augmented with retrieval for premise selection.
- **Impact**: Enables training models to learn which lemmas are relevant in which contexts.

#### Rango (ICSE 2025)
- **Domain**: Coq software verification.
- **Innovation**: Retrieves both lemmas AND proofs adaptively at each proof step.
- **Performance**: 32.0% of theorems on CoqStoq benchmark (+29% vs. Tactician).
- **Key finding**: Adding relevant proofs increases success by 47%.

#### RagVerus (2025)
- **Domain**: Repository-level Verus verification.
- **Challenge**: Complex projects with vast semantic context and many dependent premises.
- **Performance**: 27% relative improvement on RepoVBench (383 tasks).
- **Architecture**: Modularizes task preparation, context retrieval, and proof generation.

#### LemmaHead (2025)
- **Focus**: RAG knowledge base from published textbooks for Lean proof generation.
- **Finding**: Formal mathematical language achieves 73% accuracy on Google's Mathematics Dataset vs. 54% for text-based RAG.
- **Implication**: Precision of formal language eliminates natural language ambiguity.

### 10.4 Premise Selection as Learned Skill

**Traditional approach**: Hand-crafted heuristics (name similarity, type matching).

**Modern approach**: BERT-based models embed proof states and premises into shared latent space, trained via contrastive learning.

**LeanSearch-PS**: Semantic premise selection engine identifying most relevant theorems for any proof state.

---

## 11. Validation and Verification: The Cascade Architecture

### 11.1 The Verification Cascade Concept

**Proposed cascade**: Computational → Symbolic → Semi-Formal → Formal (LEAN)

**Why this works**: Each level trades off cost vs. confidence:

| Level | Cost | Confidence | Examples |
|-------|------|-----------|----------|
| Computational | Low | Medium | Numerical evaluation, type checking, simple linting |
| Symbolic | Medium | High | SMT solving, symbolic algebra, constraint solving |
| Semi-Formal | High | Very High | Partial proof checking, tactic validation |
| Formal (Lean) | Very High | Perfect | Full kernel verification |

**Key insight from research**: Use cheap checks to filter out obvious failures before committing to expensive formal verification.

### 11.2 Extracting Information from Failures

#### How Testing Helps Diagnose Proof Failures (2018)
**Problem**: Proof failures have multiple causes:
- Non-compliance (error in code or spec)
- Missing/weak function specifications
- Missing loop invariants
- Prover incapacity/timeout

**Solution**: Test generation helps identify the root cause and produce counterexamples.

**Application**: When Lean proof fails, run concrete tests to distinguish between:
1. **Theorem is false** (test produces counterexample)
2. **Tactics are wrong** (theorem is true but proof strategy failed)
3. **Missing lemmas** (proof is possible but needs more premises)

#### Lean 4 Error Recovery in Tactics
- **Design**: `TacticM` has `MonadExcept` instance that backtracks state including error messages.
- **Mechanism**: `tryCatch` enables recovery from failures.
- **Use case**: `first | ... | ...` tries multiple tactics, recovering from failures.

**Implication for LLM systems**: Error messages from Lean are structured data that can guide the next attempt. Don't just retry—parse the error and adjust strategy.

#### Structured Feedback from Verification
**Research finding**: Multi-hop fact verification shows error propagation—errors in early steps distort conclusions.

**Solution**: Checkpoint verification at each step to isolate errors.

**For theorem proving**:
- After each tactic application, check proof state validity.
- If invalid, don't continue—fix current step first.
- Maintain "proof health metrics" (open subgoals, assumptions introduced, progress toward goal).

### 11.3 Confidence Estimation Before Formal Verification

**Challenge**: No direct research on confidence estimation for theorem proving found.

**Analogous approaches:**

1. **Ensemble-based uncertainty**: Train multiple models; disagreement indicates uncertainty.

2. **Proof state evaluation heuristics**:
   - Number of open subgoals
   - Syntactic distance to goal
   - Historical success rate of current tactic on similar goals

3. **Value function learning**: Train a model to estimate P(proof succeeds | current state).

**Proposed approach for your system**:

1. **Fast heuristics** (Computational level):
   - Type correctness
   - Tactic applicability (does tactic match goal structure?)
   - Syntactic progress (goal simplification)

2. **Symbolic checks** (Symbolic level):
   - SMT for decidable fragments
   - Numeric validation for inequalities
   - Constraint solving for existential goals

3. **Learned confidence** (Semi-Formal level):
   - Value function estimating P(provable)
   - Ensemble agreement across multiple proof attempts
   - Similarity to known successful proofs (retrieval-based confidence)

4. **Full verification** (Formal level):
   - Only commit to Lean kernel when confidence > threshold
   - Use Lean errors to update confidence model

### 11.4 Using Lean Errors as Structured Feedback

**Current state of the art:**

#### Lyra: Dual Correction Framework (2023)
- **Tool Correction**: Uses Sledgehammer to suggest alternative tactics when tactics fail.
- **Conjecture Correction**: Modifies the theorem statement when proof is impossible.
- **Performance**: 55.3% on miniF2F validation (+7.3% over previous best).

#### Adaptive Proof Refinement (Adapt, 2024)
- **Insight**: Different error types need different refinement strategies.
- **Method**: LLM-guided strategy selection based on error classification.
- **Performance**: 32.0% on CoqStoq benchmark.

#### ProofNet++: Neuro-Symbolic Self-Correction (2025)
- **Architecture**: Recursive verification-correction loop until all steps validated.
- **Pipeline**: LLM generates outline → verifier checks → correction module fixes errors → repeat.

**Key patterns:**

1. **Error classification**: Parse Lean error messages to determine error type (type mismatch, tactic failure, unknown identifier, etc.).

2. **Targeted repair**: Different repair strategies for different error types:
   - **Type errors** → adjust term construction
   - **Tactic failures** → try alternative tactics or decompose goal differently
   - **Unknown identifiers** → search for correct name or synthesize missing lemma

3. **Iterative refinement**: Don't start from scratch—locally repair the error and re-verify.

---

## 12. The Induction/Transduction Bridge for Transfer Learning

### 12.1 Analogical Reasoning in Mathematics

**Research finding**: Neural embeddings allow architectures to draw analogies between mathematical domains.

**Example**: Algebraic operators (associativity, commutativity, distributivity) appear across domains with different names. Graph neural networks can recognize these structural patterns as analogous.

### 12.2 Transfer Learning Across Mathematical Domains

#### Meta-Interpretive Learning with Reuse (2024)
- **Problem**: ILP program search space grows exponentially with clauses.
- **Solution**: Reuse auxiliary predicates learned from previous problems.
- **Analogy**: Transfer lemmas/tactics learned in one mathematical domain to another.

#### Thought Propagation (ICLR 2024)
- **Limitation of existing methods**: LLMs reason from scratch, cannot reuse insights from analogous problems.
- **Solution**: Propagate "thoughts" from solving similar problems to new problems.
- **Application**: When proving a theorem about group theory, retrieve similar proofs about rings and adapt the reasoning.

### 12.3 Cross-Domain Transfer in Theorem Proving

**What "induction/transduction bridge" might mean:**

1. **Inductive reasoning**: Generalize from specific examples to general principles.
   - Example: Prove 5 theorems about finite groups → induce general group theory tactics.

2. **Transductive reasoning**: Apply knowledge from one specific case to another.
   - Example: Proof about real numbers → adapt to complex numbers.

**Mechanisms:**

1. **Analogical mapping**:
   - Identify structural similarities between domains (categories, groups, topological spaces all have morphisms).
   - Transfer proof strategies that exploit these structures.

2. **Meta-tactics**:
   - Learn high-level proof strategies (proof by contradiction, induction, case analysis) that transcend specific domains.
   - Apply domain-agnostic strategies with domain-specific lemmas.

3. **Embedding-based transfer**:
   - Embed proof states and theorems into a latent space.
   - Theorems that are nearby in latent space likely admit similar proof strategies.

**Research gap**: Limited work on explicit transfer learning between mathematical domains in formal proving. Most work focuses on transfer within a single domain (e.g., Mathlib).

**Opportunity**: Build a **mathematical world model** that represents relationships between domains, enabling systematic transfer.

---

## 13. Emerging Approaches and Open Problems

### 13.1 What's Working in Practice (2024-2025)

**Proven architectures:**

1. **Test-time compute scaling** (AlphaProof, o3):
   - Spend more compute on hard problems via RL and tree search.
   - Generate problem variations to improve sample efficiency.
   - Scales multiple orders of magnitude with predictable performance gains.

2. **Hierarchical proof decomposition** (DeepSeek-Prover-V2, Aristotle):
   - High-level informal sketch → formal subgoals → solve recursively.
   - Bridges LLM strengths (informal reasoning) with symbolic strengths (verification).

3. **Retrieval-augmented proving** (Rango, RagVerus, LeanDojo):
   - Adaptive retrieval at each proof step dramatically improves success rate.
   - Retrieving proofs (not just lemmas) gives even larger gains.

4. **Synthetic data generation** (DeepSeek-Prover, STP, MUSTARD):
   - Auto-formalization of natural language problems.
   - Proof mining from existing libraries.
   - Self-play generation of provable conjectures.
   - Variation generation via tree search.

5. **Expert iteration + curriculum learning** (LeanAgent, GAR):
   - Interleave proof search with model training.
   - Optimize learning trajectory by difficulty.
   - Dramatically outperforms proof search alone at same compute.

### 13.2 What's Broken About Current Approaches

**Fundamental limitations:**

1. **Lack of lemma accumulation**:
   - Most models don't build reusable lemma libraries during proving.
   - Each proof starts from scratch, re-proving basic facts.
   - **Impact**: Hinders scalability to deep theories requiring many intermediate results.

2. **Geometry and visual reasoning**:
   - Most systems fail on geometry problems requiring diagram interpretation.
   - Hidden attributes (angles, lengths) must be inferred from graphics.
   - **Exception**: AlphaGeometry-2 specializes in geometry but doesn't generalize.

3. **Autoformalization brittleness**:
   - Minor paraphrasing of natural language dramatically changes formalization.
   - Semantic gaps (polysemy) and logic gaps (tacit knowledge) are unaddressed.
   - **Needed**: Interactive autoformalization with human-in-the-loop.

4. **Higher-order logic verification**:
   - Fully automated verification of higher-order provers remains unsolved.
   - Diversity of output encodings hinders proof checking and prover cooperation.

5. **Sparse rewards in RL**:
   - Expert iteration plateaus because hard theorems need exponentially many samples.
   - **Partial solution**: Self-play (STP) and synthetic data help but don't fully solve it.

6. **Context window limitations**:
   - Large proofs exceed LLM context windows.
   - Repository-level verification requires 100K+ token contexts.
   - **Workarounds**: Hierarchical summarization, retrieval (but lossy).

### 13.3 Promising Directions Not Yet Mainstream

**Research frontiers:**

1. **Neurosymbolic architectures** (Proof of Thought, 2024):
   - Integrate symbolic reasoning at the neuron level (symbolic activation functions).
   - Combine System 1 (neural pattern recognition) with System 2 (symbolic deliberation).
   - **Potential**: More interpretable and constrained than pure neural models.

2. **Proof repair and adaptive refinement** (Adapt, Baldur, Lyra):
   - When proof breaks (due to dependency changes), auto-repair instead of rewriting.
   - **LLM role**: Classify error type and suggest targeted fixes.
   - **Practical importance**: Essential for maintaining large formal codebases.

3. **Curriculum learning with adaptive difficulty**:
   - No curriculum dominates universally—depends on model capacity, task complexity, and metric.
   - **Insight**: Stronger models benefit from forward (easy→hard) on simple tasks; weaker models or hard tasks favor reverse.
   - **Opportunity**: Meta-learn curriculum strategy based on model performance.

4. **Goedel-Prover-V2 innovations**:
   - Scaffolded data synthesis with increasing difficulty.
   - Verifier-guided self-correction using Lean compiler feedback.
   - Model averaging to maintain output diversity.

5. **Library learning and tactic discovery**:
   - **TacMiner**: Uses Tactic Dependence Graphs to discover reusable tactics (3x more tactics than baselines, 26% proof size reduction).
   - **LEGO-Prover**: Growing skill library of verified lemmas.
   - **Data-driven lemma synthesis (lfind)**: 68% success rate synthesizing useful lemmas.

6. **Proof-state value functions**:
   - Learn to estimate remaining steps, proof complexity, and success probability.
   - **LeanProgress**: Lightweight progress prediction (+3.8% improvement).
   - **Potential**: Guide search more efficiently than handcrafted heuristics.

### 13.4 Open Research Questions

**Critical gaps:**

1. **How to build a mathematical world model?**
   - Represent relationships between domains, definitions, and theorems.
   - Enable reasoning about "what kind of math is this?" and "what techniques apply here?"
   - **Current work**: Ontological models (limited scope), knowledge graphs (static).

2. **Confidence calibration for theorem proving:**
   - When should the system attempt formal verification vs. give up?
   - How to estimate P(provable) before expensive proof search?
   - **Analogy**: Ensemble-based uncertainty, but for proofs.

3. **Interactive autoformalization:**
   - Autoformalization is not one-shot—needs human feedback loop.
   - How to present formalization candidates and collect feedback efficiently?
   - **Gap**: Most work assumes fully automated formalization.

4. **Proof compression and abstraction:**
   - Long proofs are hard to verify and maintain.
   - How to automatically identify reusable lemmas and factor them out?
   - **Related**: Proof refactoring, tactic synthesis.

5. **Cross-prover verification:**
   - Proofs generated in Lean should ideally be checkable in Coq, Isabelle, etc.
   - **Dedukti framework** attempts this via λΠ-calculus modulo.
   - **Challenge**: Prover-specific tactics and libraries make translation hard.

6. **Scaling to research-level mathematics:**
   - Current benchmarks (IMO, Putnam) are challenging but not research frontier.
   - Research math requires months of context, novel definitions, deep theory.
   - **Gap**: No clear path from IMO gold to automated research.

---

# Part III: The Real Problem — Tooling Gaps

## 14. The Tooling Gaps (Not Language Gaps)

A new language won't solve these. Better tools will.

### 14.1 Gap 1: Tactics Library

When filling a `sorry`:
```
Goal: ∀ n, Even (n^2) → Even n
```

Which tactic? `induction`? `cases`? `by_contra`? Wrong choice = Lean error = wasted iteration.

**What's needed:**

| Goal Pattern | Primary Tactics | Mathlib Lemmas |
|-------------|-----------------|----------------|
| `∀ n, P n` | intro, induction | Nat.recOn |
| `∃ x, P x` | use, constructor | exists_prop |
| `a = b` | rfl, calc, ring | eq_comm |
| `a ≤ b` | linarith, omega | le_trans |
| `P ∨ Q` | left, right | Or.intro_* |
| `¬ P` | by_contra, push_neg | not_* |

Plus: success rates by domain (number theory vs analysis vs algebra).

### 14.2 Gap 2: Error Feedback

Lean error:
```
tactic 'simp' failed, no progress was made
```

This tells you nothing. Need:
- Error category (tactic failed vs type mismatch vs unknown identifier)
- Alternative suggestions ("try `ring` or `norm_num` instead")
- Similar successful proofs in library

### 14.3 Gap 3: Proof Order Strategy

Which sorry to fill first? Options:
1. **Leaves first** — fill sorry's with no dependencies
2. **Core first** — fill the main insight, delegate helper lemmas
3. **Easiest first** — build momentum

No language change helps here. Need strategy heuristics based on proof structure.

---

## 15. What To Build Instead

**Priority order. All of these are tooling, not language design.**

### 15.1 Tactics Library (CRITICAL, WEEK 1)

Index Mathlib by goal pattern. Make it searchable:

```python
# Input: goal from Lean
goal = "∀ n : ℕ, Even (n^2) → Even n"

# Output: ranked tactics + lemmas
suggest(goal) → [
    ("by_contra", 0.65, "Common for Even proofs"),
    ("induction", 0.20, "But n^2 makes this hard"),
    ("Nat.even_pow", 0.80, "Direct Mathlib lemma"),
]
```

Source: Mine 44K Herald proofs for tactic → goal patterns.

### 15.2 Error Parser (WEEK 2)

Classify Lean errors into actionable categories:

```python
def classify_error(error: str) → ErrorType:
    if "failed, no progress" in error:
        return TacticNoProgress(suggest=["try different tactic"])
    if "type mismatch" in error:
        return TypeMismatch(expected=..., got=..., fix="cast or convert")
    if "unknown identifier" in error:
        return UnknownIdent(candidates=find_similar_names())
```

### 15.3 Sorry-Filling Loop (WEEK 3-4)

Automated iteration:

```python
def fill_sorries(skeleton, max_per_sorry=5):
    while has_sorries(skeleton):
        sorry = pick_easiest_sorry(skeleton)  # leaves first
        for _ in range(max_per_sorry):
            tactic = suggest_tactic(sorry.goal)
            result = try_tactic(skeleton, sorry, tactic)
            if result.success:
                skeleton = result.new_code
                break
            record_failure(sorry, tactic, result.error)
    return skeleton
```

### 15.4 LLM Integration (MONTH 1+)

Claude/Gemini as tactic advisor:
- Input: goal + context + failed attempts
- Output: next tactic to try + explanation

```python
def llm_suggest_tactic(goal, context, failures):
    prompt = f"""
    Goal: {goal}
    Available hypotheses: {context.hypotheses}
    Failed tactics: {failures}

    Suggest next tactic with reasoning.
    """
    return claude(prompt)
```

### 15.5 Blueprint Integration (MONTH 2+)

Connect our solver to Blueprint-style dependency tracking:
- Auto-generate dependency graph from proof
- Color nodes by status
- Enable distributed contribution

---

## 16. Existing System Analysis

### 16.1 What the Solver System Actually Has

**Re-examined `~/Documents/base/solver` (December 8, 2025):**

```
ACTUAL SYSTEM:
├── data/archive/problems/*.md       ← 84 years of Putnam (1938-2024)
├── data/archive/solutions/*.md      ← Reference solutions
├── data/memory/
│   ├── problems.json                ← Problem state, attempts list, techniques
│   ├── techniques.json              ← Success rates: {topic: {technique: {successes, attempts}}}
│   └── sessions/*.json              ← Session step logs with UUID
├── src/memory/
│   ├── world_model.py               ← WorldModel class with get_context(), record_attempt()
│   └── technique_tracker.py         ← TechniqueTracker with recommend(topics)
├── src/verify/
│   ├── cascade.py                   ← 3-stage: numerical(100) → symbolic → semiformal(1000)
│   └── lean/verifier.py             ← Subprocess to lake/lean, error classification
├── lean/Putnam/*.lean               ← ACTUAL Lean proofs (2025 problems formalized!)
└── src/orchestrator.py              ← Python pipeline (seems for programmatic use)
```

**What ACTUALLY works (verified by reading code):**

1. **Memory persists and updates:**
   - `record_attempt()` calls `technique_tracker.update()` for every attempt
   - Auto-saves to JSON after every record
   - `get_context()` returns similar_problems, recommended_techniques, known_pitfalls

2. **Technique tracking has real data:**
   ```json
   "number_theory": {
     "modular_arithmetic": {"successes": 14, "attempts": 20},
     "induction": {"successes": 12, "attempts": 18}
   }
   ```

3. **Lean verification works:**
   - `lean/Putnam/A6_2025_full_proven.lean` - fully formalized, no sorry
   - Verifier runs via subprocess, parses errors, classifies them

4. **Confidence is NOT always 0.0:**
   - `record_attempt()` takes confidence as parameter (default 0.5)
   - Cascade computes confidence from stages (0.7 for numerical, 0.85 for semiformal)

**What's ACTUALLY missing (honest assessment):**

1. **The /solve command is a manual workflow** - Claude follows it, but doesn't auto-call Python tools
2. **get_context() exists but isn't called in /solve** - the context data exists, /solve doesn't show it
3. **orchestrator.py seems designed for Python use**, not Claude Code invocation
4. **No automatic bridge** from "Claude solved something" → record_attempt()

**CORRECTION from previous analysis:**
- Flywheel partially works at Python level
- Gap is Claude↔Python integration, not missing Python functionality

### 16.2 Questions Answered (Clarified by User)

1. **Is orchestrator.py meant to be called by Claude?**
   - **NO.** Claude Code IS the orchestrator.
   - orchestrator.py is for Gemini integration (second opinion LLM)
   - There's a `/gemini` skill that Claude uses to call Gemini

2. **How were the Lean proofs created?**
   - **By Claude.** Claude generates the Lean proofs in `lean/Putnam/`

3. **What's the actual workflow?**
   - **REFERENCE.md defines it:** THINK → PROBE → OBSERVE → LEARN
   - Claude Code orchestrates this loop
   - Tools (Python) are called during PROBE phase
   - Memory is updated during LEARN phase

4. **Is the 2-system architecture (Python + Claude) intentional?**
   - **YES.** Design is:
     - Claude Code = orchestrator (brain)
     - Skills = reasoning strategies
     - Tools = Python executables Claude calls
     - Commands = workflow prompts
     - Memory = persistent state Claude reads/writes

**The architecture is correct. The question is: are the pieces connected properly?**

### 16.3 What About SFM (Semi-Formal Markdown)?

Proposed: A `.sfm` file format with Lean blocks, auto-verification, hooks.

**Honest assessment:**

| Component | Solves Real Gap? | Adds Complexity? |
|-----------|------------------|------------------|
| New `.sfm` format | No - current .md + .json works | Yes - new format to maintain |
| Lean block extractor | **Partial** - enables verification | Minimal |
| File watcher + hooks | **Yes** - could close loop | Moderate |
| Dashboard/renderer | No - observability != closure | Yes - web server overhead |

**The valuable part:** Not the file format. The hooks that auto-feed verification results back to technique_tracker and world_model.

### 16.4 What About a Dashboard?

Proposed: Live HTML showing proof status, where to focus, tactic suggestions.

**Current observability:**
- `problems.json` has status per problem
- `sessions/*.json` has step-by-step logs
- `technique_tracker` has success rates

**What dashboard would add:**
- Real-time view (vs reading JSON)
- Visual dependency graph
- "Focus here" pointer for humans

**Is it needed?**
- For solo Claude solving: **No** - Claude has context
- For human-in-the-loop: **Maybe** - helps see what's pending
- For team collaboration: **Yes** - shared visibility

**Verdict:** Dashboard is nice-to-have, not blocking. The core gap is feedback loop closure.

### 16.5 What's ACTUALLY Worth Building

**Priority 1: Close the Flywheel (code changes, not new formats)**

```python
# In /solve workflow, after verification:
def on_verification_complete(problem_id, result):
    # 1. Record with ACTUAL confidence (not hardcoded 0.0)
    world_model.record_attempt(
        problem_id=problem_id,
        confidence=result.cascade_confidence,  # Compute this
        stage_reached=result.stage,
    )

    # 2. Update technique effectiveness
    technique_tracker.update(
        topics=problem.topics,
        technique=used_technique,
        success=result.passed,
        stage_reached=result.stage,  # Partial credit
    )

    # 3. If failed, record WHY
    if not result.passed:
        world_model.record_failure_reason(
            problem_id,
            stage=result.failed_stage,
            error=result.error
        )
```

**Priority 2: Use History in /solve**

```python
# At start of /solve:
context = world_model.get_context(problem)

# ACTUALLY USE IT:
if context.similar_problems:
    print(f"Similar solved: {context.similar_problems}")
    print(f"Techniques that worked: {context.successful_techniques}")

if context.known_pitfalls:
    print(f"Watch out for: {context.known_pitfalls}")
```

**Priority 3: Lean Verification Hook (NOT new file format)**

```python
# Add to existing verification cascade:
def verify_solution(solution_md: str) -> CascadeResult:
    # Extract Lean blocks from existing markdown
    lean_blocks = extract_lean_blocks(solution_md)  # regex, not new parser

    # Verify each block
    for block in lean_blocks:
        result = lean_verify(block)
        if not result.success:
            return CascadeResult(stage='lean', passed=False, error=result.error)

    return CascadeResult(stage='formal', passed=True)
```

This works with current `.md` files. No new format needed.

**Priority 4: Dashboard (LATER, if needed)**

Build only if:
- Multiple humans collaborating on proofs
- Need real-time visibility into distributed work
- Current JSON inspection is too slow

Don't build if:
- Solo Claude solving (Claude has full context)
- Batch processing (just check final results)
- "Would be nice to have" (YAGNI)

### 16.6 Summary

| Build | Priority | Why |
|-------|----------|-----|
| Flywheel closure (code) | **P0** | Actual gap |
| History usage in /solve | **P1** | Data exists, not used |
| Lean extraction from .md | **P2** | Enables formal verification |
| New .sfm format | **P3/Skip** | Adds complexity, current .md works |
| Dashboard | **P4/Skip** | Nice-to-have, not blocking |

**The insight:** Don't build new formats/tools when the gap is using what already exists.

---

# Part IV: Architectural Recommendations

## 17. Core Architecture: Orchestration via Claude Code

**Design**: Claude Code orchestrates Skills, Tools, and Memory.

**Alignment with research**: This is exactly the right pattern. Here's how to make it concrete:

### 17.1 Skills (Reusable Strategies)

**What research shows:**
- LEGO-Prover: Growing library of verified lemmas as skills.
- TacMiner: Auto-discover tactics by analyzing Tactic Dependence Graphs.
- Meta-tactics in Lean 4: Code that generates tactics.

**Concrete implementation:**

```python
class Skill:
    name: str
    description: str  # Natural language explanation
    applicability_condition: Callable[[ProofState], bool]
    tactic_generator: Callable[[ProofState], Tactic]
    success_rate: float
    examples: List[ProofTrace]  # Successful applications

class SkillLibrary:
    def retrieve(self, proof_state: ProofState, k: int) -> List[Skill]:
        """Retrieve top-k relevant skills via semantic similarity"""

    def add(self, skill: Skill):
        """Add newly discovered skill"""

    def refine(self, skill: Skill, feedback: ProofTrace):
        """Update skill based on success/failure"""
```

**Skill discovery process:**

1. **Mining from successful proofs**: Extract common tactic patterns from proof traces (similar to TacMiner).
2. **LLM-generated meta-tactics**: Prompt Claude to generate Lean 4 tactics that implement high-level strategies.
3. **Expert iteration**: Learn skills that generalize across multiple problems.

### 17.2 Tools (Symbolic, Numerical, Code Exec, LEAN)

**Verification cascade mapping:**

| Tool | Cascade Level | Purpose |
|------|--------------|---------|
| Type checker | Computational | Fast syntax/type validation |
| Numerical evaluator | Computational | Concrete value checking |
| SMT solver (Z3) | Symbolic | Decidable fragment verification |
| Symbolic algebra (SymPy) | Symbolic | Equation manipulation |
| Partial proof checker | Semi-Formal | Tactic validation without kernel |
| Lean kernel | Formal | Ground truth verification |

**Tool selection strategy:**

```python
def select_verification_level(proof_state: ProofState, confidence: float) -> Tool:
    """Choose cheapest tool sufficient for current confidence level"""
    if confidence < 0.3:
        return ComputationalChecker()  # Fast rejection
    elif confidence < 0.7:
        return SymbolicVerifier()  # Medium confidence → SMT
    elif confidence < 0.9:
        return PartialProofChecker()  # High confidence → semi-formal
    else:
        return LeanKernel()  # Very high confidence → formal
```

**Key insight**: Don't always go to Lean first. Use cheap filters to reject bad candidates early.

### 17.3 Memory (Working, Library, World Model)

**Three-layer memory architecture:**

1. **Working Memory** (current proof context):
   - Current proof state (goals, assumptions)
   - Recently applied tactics and their results
   - Active retrieval results (relevant lemmas/proofs)
   - Error history and recovery attempts

2. **Library Memory** (repository knowledge):
   - Vector database of all theorems (semantic search)
   - Formal search index (structural queries)
   - Skill library (reusable tactics)
   - Proof trace database (successful proofs)

3. **World Model** (mathematical knowledge):
   - Domain ontology (what kind of math is this?)
   - Cross-domain analogies (group theory ↔ topology)
   - Difficulty estimator (how hard is this theorem?)
   - Strategy recommender (what approaches work for this domain?)

**World model construction:**

```python
class WorldModel:
    domain_embeddings: Dict[str, Vector]  # "group_theory" → embedding
    domain_relationships: Graph  # Morphisms, specializations, analogies

    def classify_theorem(self, theorem: Statement) -> List[Domain]:
        """What domains does this theorem belong to?"""

    def retrieve_analogies(self, theorem: Statement) -> List[Theorem]:
        """Find structurally similar theorems in other domains"""

    def suggest_strategy(self, theorem: Statement) -> List[Strategy]:
        """What proof strategies work for this type of problem?"""
```

**Training world model:**
- **Supervised**: Theorems labeled with domains/tags in Mathlib.
- **Contrastive**: Learn embeddings where similar theorems cluster.
- **Transfer learning**: Successful strategies in domain A → try in analogous domain B.

---

## 18. Reasoning Loop Implementation

**Design**: Hypothesis generation → Validation → Reflection → Meta-tactic building.

**Research alignment**: This is Draft, Sketch, and Prove (DSP) + reflection.

### 18.1 Phase 1: Hypothesis Generation (Informal)

```python
def generate_proof_hypothesis(theorem: Statement, memory: Memory) -> ProofSketch:
    """Generate high-level informal proof strategy"""

    # Retrieve analogous proofs
    similar_theorems = memory.library.retrieve_similar(theorem, k=5)

    # Retrieve relevant lemmas
    relevant_lemmas = memory.library.retrieve_lemmas(theorem, k=20)

    # World model suggests strategy
    strategies = memory.world_model.suggest_strategy(theorem)

    # Claude generates informal sketch
    prompt = f"""
    Prove: {theorem}

    Similar proved theorems:
    {similar_theorems}

    Potentially useful lemmas:
    {relevant_lemmas}

    Suggested strategies:
    {strategies}

    Generate a high-level proof sketch with key steps and subgoals.
    """

    sketch = claude.complete(prompt)
    return sketch
```

### 18.2 Phase 2: Validation (Cascade)

```python
def validate_hypothesis(sketch: ProofSketch) -> ValidationResult:
    """Validate sketch through cascade"""

    results = []

    # Computational validation
    comp_result = computational_check(sketch)
    results.append(comp_result)
    if not comp_result.passed:
        return ValidationResult(level="computational", passed=False, errors=comp_result.errors)

    # Symbolic validation
    sym_result = symbolic_check(sketch)
    results.append(sym_result)
    if not sym_result.passed:
        return ValidationResult(level="symbolic", passed=False, errors=sym_result.errors)

    # Semi-formal validation (tactic applicability)
    semi_result = semiformal_check(sketch)
    results.append(semi_result)
    if not semi_result.passed:
        return ValidationResult(level="semiformal", passed=False, errors=semi_result.errors)

    # If high confidence, attempt formal verification
    if semi_result.confidence > 0.8:
        formal_result = lean_verify(sketch)
        results.append(formal_result)

    return ValidationResult.aggregate(results)
```

### 18.3 Phase 3: Reflection (Error Analysis)

```python
def reflect_on_failure(sketch: ProofSketch, validation: ValidationResult) -> Reflection:
    """Analyze why proof attempt failed"""

    # Classify error type
    error_type = classify_error(validation.errors)

    # Retrieve similar failures and their fixes
    similar_failures = memory.retrieve_failures(error_type, sketch)

    # Generate reflection
    prompt = f"""
    Proof sketch: {sketch}

    Validation failed at {validation.level} level with errors:
    {validation.errors}

    Error type: {error_type}

    Similar past failures and their fixes:
    {similar_failures}

    Analyze:
    1. Why did this approach fail?
    2. What assumptions were wrong?
    3. What alternative strategies should we try?
    4. What lemmas are we missing?
    """

    reflection = claude.complete(prompt)
    return Reflection(
        failure_reason=reflection.reason,
        wrong_assumptions=reflection.assumptions,
        alternative_strategies=reflection.alternatives,
        missing_lemmas=reflection.lemmas
    )
```

### 18.4 Phase 4: Meta-Tactic Building

```python
def build_meta_tactic(successful_proofs: List[ProofTrace]) -> Skill:
    """Extract reusable tactic from successful proofs"""

    # Find common patterns using TDG analysis
    tactic_graph = TacticDependenceGraph(successful_proofs)
    patterns = tactic_graph.find_reusable_patterns()

    # Generate meta-tactic code
    for pattern in patterns:
        prompt = f"""
        These proofs share a common tactic pattern:
        {pattern}

        Generate a Lean 4 meta-tactic that implements this pattern.
        The tactic should:
        1. Check applicability conditions
        2. Decompose the goal appropriately
        3. Apply the core strategy
        4. Handle common edge cases
        """

        tactic_code = claude.complete(prompt)

        # Validate generated tactic
        if validate_lean_tactic(tactic_code):
            skill = Skill(
                name=pattern.name,
                description=pattern.description,
                tactic_code=tactic_code,
                success_rate=pattern.observed_success_rate,
                examples=successful_proofs
            )
            memory.skills.add(skill)

    return skill
```

---

## 19. Formalization Loop Implementation

**Design**: Natural language → Lean formalization → Compilation → Feedback.

**Research alignment**: Autoformalization + error-guided refinement.

### 19.1 Phase 1: Autoformalization

```python
def autoformalize(sketch: ProofSketch, context: ProofContext) -> LeanProof:
    """Convert informal sketch to formal Lean proof"""

    # Retrieve formalization examples
    examples = memory.retrieve_formalizations(sketch.domain, k=10)

    # Iterative formalization with increasing detail
    formalization = None
    for detail_level in ["statement", "subgoals", "tactics"]:
        prompt = f"""
        Formalize the following {detail_level}:

        Informal: {sketch}

        Context:
        - Imports: {context.imports}
        - Definitions: {context.definitions}
        - Available lemmas: {context.lemmas}

        Similar formalizations:
        {examples}

        Generate valid Lean 4 code.
        """

        formalization = claude.complete(prompt)

        # Validate at this level
        validation = validate_lean_syntax(formalization)
        if not validation.passed:
            # Repair and retry
            formalization = repair_lean_code(formalization, validation.errors)

    return formalization
```

### 19.2 Phase 2: Compilation and Error Handling

```python
def compile_and_refine(lean_code: str, max_iterations: int = 5) -> CompilationResult:
    """Compile Lean code with iterative error correction"""

    for iteration in range(max_iterations):
        result = lean_compile(lean_code)

        if result.success:
            return result

        # Parse structured errors
        errors = parse_lean_errors(result.error_output)

        # Classify error types
        error_classes = classify_errors(errors)

        # Select repair strategy based on error type
        repair_strategy = select_repair_strategy(error_classes)

        # Apply repair
        if repair_strategy == "tool_correction":
            # Use Sledgehammer or similar to suggest alternative tactics
            lean_code = tool_correction(lean_code, errors)
        elif repair_strategy == "conjecture_correction":
            # Maybe the statement is wrong
            lean_code = conjecture_correction(lean_code, errors)
        elif repair_strategy == "premise_retrieval":
            # Missing lemmas
            missing_lemmas = retrieve_missing_premises(errors)
            lean_code = add_premises(lean_code, missing_lemmas)
        elif repair_strategy == "type_correction":
            # Type errors
            lean_code = fix_types(lean_code, errors)
        else:
            # Unknown error type—ask Claude
            lean_code = llm_repair(lean_code, errors)

    return CompilationResult(success=False, error="Max iterations exceeded")
```

### 19.3 Phase 3: Structured Feedback Extraction

```python
def extract_feedback(compilation_result: CompilationResult) -> Feedback:
    """Extract structured learning from compilation"""

    if compilation_result.success:
        # Success feedback
        feedback = Feedback(
            type="success",
            proof_trace=compilation_result.proof_trace,
            tactics_used=compilation_result.tactics,
            lemmas_used=compilation_result.lemmas,
            proof_length=len(compilation_result.tactics),
            novel_patterns=detect_novel_patterns(compilation_result)
        )

        # Update skill library
        memory.skills.update_success_rates(compilation_result)

        # Extract reusable meta-tactics
        if feedback.novel_patterns:
            build_meta_tactic(feedback.novel_patterns)
    else:
        # Failure feedback
        errors = compilation_result.errors
        feedback = Feedback(
            type="failure",
            error_type=classify_errors(errors),
            failed_tactics=compilation_result.failed_tactics,
            failure_point=compilation_result.failure_point,
            context_at_failure=compilation_result.proof_state,
            suggested_fixes=generate_repair_suggestions(errors)
        )

        # Store failure for future learning
        memory.add_failure(feedback)

    return feedback
```

---

## 20. Scaling Strategy: Test-Time Compute

**Research finding**: AlphaProof and o3 demonstrate that test-time compute scaling is the key to superhuman performance.

**Implementation for your system:**

```python
def adaptive_compute_allocation(theorem: Statement, compute_budget: float) -> Proof:
    """Allocate compute based on estimated difficulty"""

    # Estimate difficulty using world model
    difficulty = memory.world_model.estimate_difficulty(theorem)

    # Allocate compute budget
    if difficulty < 0.3:
        # Easy theorem—fast path
        return quick_proof_attempt(theorem, budget=0.1 * compute_budget)
    elif difficulty < 0.7:
        # Medium theorem—standard search
        return standard_proof_search(theorem, budget=0.3 * compute_budget)
    else:
        # Hard theorem—intensive search with RL
        return intensive_proof_search_with_rl(theorem, budget=compute_budget)
```

**Intensive search strategy** (AlphaProof-style):

```python
def intensive_proof_search_with_rl(theorem: Statement, budget: float) -> Proof:
    """Test-time RL for hard theorems"""

    # Generate problem variations
    variations = generate_variations(theorem, n=1000)

    # Attempt to prove variations
    solved_variations = []
    for var in variations:
        proof = attempt_proof(var, budget=budget/1000)
        if proof:
            solved_variations.append((var, proof))

    # Train specialized model on solved variations
    specialized_model = fine_tune_on_variations(solved_variations)

    # Use specialized model to prove original theorem
    proof = specialized_model.prove(theorem, budget=budget/2)

    return proof
```

---

## 21. Key Architectural Principles

**Based on research synthesis:**

1. **Hybrid symbolic-neural by default**
   - Never use pure neural or pure symbolic.
   - Neural for semantic understanding and hypothesis generation.
   - Symbolic for verification and guarantees.

2. **Retrieval at every step**
   - Before every major decision (tactic choice, lemma application), retrieve relevant context.
   - Retrieve both lemmas (what to use) and proofs (how to use them).

3. **Cascade before commitment**
   - Always run cheap checks before expensive verification.
   - Confidence gates: only proceed to next level if confidence threshold met.

4. **Structured error feedback**
   - Parse Lean errors into structured data.
   - Classify error types and apply targeted repairs.
   - Store failures for meta-learning.

5. **Skills as first-class objects**
   - Reusable tactics are the unit of learning.
   - Continuously mine skills from successful proofs.
   - Compose skills hierarchically (meta-tactics).

6. **World model for transfer**
   - Represent mathematical knowledge at multiple levels (domains, techniques, analogies).
   - Use world model to guide exploration (what strategies work here?).
   - Enable transfer between domains via analogical reasoning.

7. **Adaptive compute allocation**
   - Easy theorems: fast heuristics.
   - Medium theorems: standard search with retrieval.
   - Hard theorems: test-time RL with variation generation.

8. **Continuous learning loop**
   - Every proof attempt (success or failure) updates the system.
   - Expert iteration: proof search → training data → improved model.
   - Curriculum: solve easier problems first to build skill library.

---

# Part V: Critical Open Questions and Next Steps

## 22. Critical Open Questions for Your System

**Research gaps you should investigate:**

1. **How to build an effective world model for mathematics?**
   - Current: Domain tags, embeddings.
   - Needed: Explicit representation of mathematical structures, relationships, and proof strategies.
   - **Approach**: Knowledge graph + embedding + LLM reasoning.

2. **Confidence calibration for the verification cascade:**
   - When is computational check sufficient? When must we go to Lean?
   - **Approach**: Train confidence estimator on historical proof attempts.

3. **Optimal retrieval strategy:**
   - What should we retrieve? Lemmas, proofs, tactics, or all three?
   - **Finding**: Research shows proofs are especially valuable (+47% success in Rango).
   - **Approach**: Adaptive retrieval—retrieve different things at different proof stages.

4. **Skill library organization:**
   - How to index/search a library of 10,000+ tactics?
   - How to handle skill composition (meta-meta-tactics)?
   - **Approach**: Hierarchical skill taxonomy + semantic search + applicability predicates.

5. **Transfer learning mechanism:**
   - How to systematically transfer proof strategies between domains?
   - **Approach**: Embed theorems/proofs, cluster by strategy, enable cross-domain retrieval.

6. **Synthetic data generation:**
   - What data should we generate? Variations, conjectures, auto-formalizations?
   - **Finding**: Self-play (STP) generates valuable training data by creating provable conjectures.
   - **Approach**: Multi-strategy synthesis (variations + self-play + auto-formalization).

7. **Error recovery vs. restart:**
   - When to repair a failed proof vs. start over with new strategy?
   - **Approach**: Estimate repair cost vs. new proof cost; choose cheaper option.

8. **Scaling to research mathematics:**
   - Current benchmarks are competition problems. How to tackle novel research?
   - **Gap**: Research requires months of context, novel definitions, no clear solution path.
   - **Approach**: Interactive system with human mathematician in the loop.

---

## 23. Concrete Next Steps

**To build your system, prioritize:**

### Phase 1: Foundation (Months 1-3)

1. **Implement verification cascade**
   - Computational: Type checker + numerical validator.
   - Symbolic: Z3 integration for decidable fragments.
   - Semi-formal: Tactic applicability checker.
   - Formal: Lean 4 kernel integration.

2. **Build retrieval infrastructure**
   - Vector database for semantic search (FAISS/Chroma).
   - Index all of Mathlib4 (theorems + proofs).
   - Implement adaptive retrieval (lemmas + proofs).

3. **Structured error handling**
   - Parser for Lean error messages.
   - Error classifier (type error, tactic failure, missing lemma, etc.).
   - Basic repair strategies for each error type.

### Phase 2: Skills and Memory (Months 4-6)

1. **Skill library**
   - Manual seed with common tactics.
   - Implement TDG-based skill mining.
   - Build skill retrieval (semantic + applicability).

2. **Memory architecture**
   - Working memory (proof state + context).
   - Library memory (theorems + proofs + skills).
   - World model v1 (domain classifier + strategy recommender).

3. **Basic reasoning loop**
   - Hypothesis generation (informal sketch).
   - Validation (through cascade).
   - Reflection (error analysis).

### Phase 3: Learning and Scaling (Months 7-12)

1. **Formalization loop**
   - Autoformalization (natural language → Lean).
   - Iterative compilation with error repair.
   - Feedback extraction and storage.

2. **Expert iteration**
   - Proof search generates training data.
   - Fine-tune model on successful proofs.
   - Curriculum learning (easy → hard).

3. **Test-time compute scaling**
   - Difficulty estimator.
   - Adaptive compute allocation.
   - Variation generation for hard problems.

4. **Meta-tactic synthesis**
   - Automatic Lean 4 tactic generation.
   - Validation and testing of generated tactics.
   - Hierarchical skill composition.

### Phase 4: Advanced Capabilities (Months 12+)

1. **World model v2**
   - Knowledge graph of mathematical concepts.
   - Cross-domain analogy finder.
   - Transfer learning between domains.

2. **Self-improvement**
   - Self-play conjecture generation (STP-style).
   - Bootstrapped skill discovery.
   - Meta-learning of proof strategies.

3. **Production deployment**
   - Interactive mode (assist human mathematician).
   - Proof repair for library maintenance.
   - Auto-formalization pipeline for new theorems.

---

## 24. Actual Gaps to Fix Now

**Architecture is now clear:**

```
CLAUDE CODE = THE ORCHESTRATOR
├── Commands (/solve, /verify, /approaches) → workflow prompts
├── Skills (14 reasoning strategies) → Claude invokes as needed
├── Tools (Python via bash) → Claude calls during PROBE
├── Memory (world_model, techniques) → Claude reads/writes
└── Gemini (second opinion) → via skill, for validation
```

**Do we need a new semi-formal language?**

No. The workflow in REFERENCE.md is correct:
- Semi-formal = md + latex + lean (with sorry) + citations + references
- Claude generates this during solving
- Lean verifies structure

**What ARE the actual gaps?**

| Gap | Description | Fix |
|-----|-------------|-----|
| /solve doesn't call get_context() | Context exists but not shown | Update /solve.md |
| Claude doesn't auto-record to memory | world_model not called post-solve | Add to /solve workflow |
| No sorry-filling loop guidance | Claude does it ad-hoc | Document in REFERENCE.md |
| Tactics library not indexed | Exists in description, not searchable | Build index tool |

**Priority:**

1. **Update /solve.md** to explicitly call:
   - `get_context()` at start (show similar problems, recommended techniques)
   - `record_attempt()` at end (update technique tracker)

2. **Document the THINK→PROBE→OBSERVE→LEARN loop** more explicitly in /solve

3. **Tactics library index** (if sorry-filling is slow)

4. **Dashboard/SFM** (if observability becomes a problem)

**The system is architecturally sound. The gap is making the pieces call each other.**

---

# Part VI: Summary

## 25. Final Summary: What to Build

**Your system should be:**

1. **Hybrid by design**: Neural reasoning + symbolic verification at every step.

2. **Cascade-based**: Cheap filters before expensive verification (Computational → Symbolic → Semi-Formal → Formal).

3. **Retrieval-augmented**: Always retrieve relevant lemmas, proofs, and skills before action.

4. **Self-improving**: Every proof attempt feeds back into skill library and world model.

5. **Hierarchical**: High-level strategy (informal) → mid-level tactics (semi-formal) → low-level proof steps (formal).

6. **Adaptive**: Allocate compute based on problem difficulty; use test-time scaling for hard problems.

7. **Error-aware**: Treat Lean errors as structured data guiding repair, not just binary failure.

8. **Transfer-capable**: World model enables cross-domain reasoning and analogical transfer.

**The critical insight**: Don't try to solve theorem proving end-to-end with one model. Instead, **orchestrate multiple specialized components** (Skills, Tools, Memory) through an intelligent cascade that balances cost and confidence.

**What makes this different from existing systems:**
- Most current work focuses on one piece (retrieval OR synthesis OR verification).
- Your architecture **integrates all pieces into a coherent system with feedback loops**.
- The verification cascade is novel—current systems typically jump straight to Lean without intermediate checks.
- Explicit world model for transfer learning is underexplored in formal proving.

**The research strongly supports your architecture. Build it.**

---

# Part VII: References

## Proof Assistants
- [Lean proof assistant](https://lean-lang.org/)
- [Lean community](https://leanprover-community.github.io/)
- [Theorem Proving in Lean 4](https://lean-lang.org/theorem_proving_in_lean4/propositions_and_proofs.html)
- [Proof assistant comparison](https://proofassistants.stackexchange.com/questions/43/proof-assistants-for-beginners-a-comparison)

## AI + Formal Methods
- [AlphaProof announcement](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [AlphaProof Nature paper](https://www.nature.com/articles/s41586-025-09833-y)

## Workflows
- [Terry Tao's Blueprint post](https://terrytao.wordpress.com/2023/11/18/formalizing-the-proof-of-pfr-in-lean4-using-blueprint-a-short-tour/)
- [Terry Tao's longer Lean tour](https://terrytao.wordpress.com/2023/12/05/a-slightly-longer-lean-4-proof-tour/)

## Literate Programming
- [Literate Programming (Wikipedia)](https://en.wikipedia.org/wiki/Literate_programming)
- [WEB programming system](https://en.wikipedia.org/wiki/Web_(programming_system))
- [Literate programming criticism](http://akkartik.name/post/literate-programming)
- [Jupyter pitfalls](https://scicomp.aalto.fi/scicomp/jupyter-pitfalls/)
- [Org-mode Babel](https://orgmode.org/worg/org-contrib/babel/intro.html)

## Formal Methods
- [Dijkstra: On the Cruelty of Really Teaching Computing Science](https://en.wikipedia.org/wiki/On_the_Cruelty_of_Really_Teaching_Computer_Science)
- [TLA+ Wikipedia](https://en.wikipedia.org/wiki/TLA+)
- [TLA+ at Amazon](https://lamport.azurewebsites.net/tla/formal-methods-amazon.pdf)

## Program Synthesis and Scaling
- [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)
- [AlphaProof Paper](https://www.nature.com/articles/s41586-025-09833-y)
- [DeepSeekMath-V2](https://arxiv.org/html/2511.22570v1)
- [Scaling Enumerative Synthesis](https://www.cis.upenn.edu/~alur/Tacas17.pdf)
- [CEGIS Overview](https://remy.wang/blog/cegis.html)
- [Llemma](https://arxiv.org/abs/2310.10631)

## Tactic Prediction and Proof Generation
- [LeanProgress](https://arxiv.org/html/2502.17925)
- [llmstep](https://arxiv.org/abs/2310.18457)
- [Lean-STaR](https://arxiv.org/html/2407.10040v1)
- [Generating Millions of Lean Theorems](https://arxiv.org/html/2503.04772v1)

## Retrieval-Augmented Proving
- [LeanDojo](https://arxiv.org/abs/2306.15626)
- [Rango](https://arxiv.org/html/2412.14063v2)
- [RagVerus](https://arxiv.org/html/2502.05344)
- [LemmaHead](https://arxiv.org/html/2501.15797v4)
- [A Semantic Search Engine for Mathlib4](https://arxiv.org/abs/2403.13310)

## Proof Decomposition and Strategy
- [Draft, Sketch, and Prove](https://arxiv.org/abs/2210.12283)
- [ProofSketch](https://arxiv.org/html/2510.24811)
- [Hilbert Framework](https://arxiv.org/html/2509.22819)

## Autoformalization
- [Lean Workbook](https://arxiv.org/html/2406.03847v3)
- [Process-Driven Autoformalization](https://arxiv.org/html/2406.01940v1)
- [Evaluating Autoformalization Robustness](https://arxiv.org/html/2511.12784)

## Error Correction and Repair
- [Lyra](https://arxiv.org/html/2309.15806)
- [Adaptive Proof Refinement](https://arxiv.org/html/2510.25103)
- [ProofNet++](https://arxiv.org/html/2505.24230)
- [Baldur](https://arxiv.org/pdf/2303.04910)
- [How Testing Helps Diagnose Proof Failures](https://link.springer.com/article/10.1007/s00165-018-0456-4)

## Learning and Self-Improvement
- [STP: Self-play Theorem Prover](https://arxiv.org/abs/2502.00212)
- [Meta-Rewarding Language Models](https://arxiv.org/abs/2407.19594)
- [Meta Prompting](https://arxiv.org/html/2311.11482v7)
- [LeanAgent](https://arxiv.org/html/2410.06209v6)
- [Expert Iteration Survey](https://github.com/zhaoyu-li/DL4TP)

## Library Learning and Skill Discovery
- [LEGO-Prover](https://openreview.net/forum?id=3f5PALef5B)
- [Learning-assisted Theorem Proving](https://arxiv.org/abs/1402.3578)
- [Data-driven Lemma Synthesis](https://dl.acm.org/doi/10.1145/3563306)
- [Proof Strategy Extraction](https://arxiv.org/html/2510.10131)

## Transfer Learning and Analogical Reasoning
- [Formal Mathematical Reasoning](https://arxiv.org/html/2412.16075v1)
- [Meta-Interpretive Learning with Reuse](https://www.mdpi.com/2227-7390/12/6/916)
- [Thought Propagation](https://github.com/Samyu0304/thought-propagation)
- [Transfer Learning vs Analogical Inference](https://www.mdpi.com/1999-4893/16/3/146)

## Neurosymbolic AI
- [Neuro-Symbolic AI 2024 Review](https://arxiv.org/html/2501.05435v1)
- [Proof of Thought](https://www.aimodels.fyi/papers/arxiv/proof-thought-neurosymbolic-program-synthesis-allows-robust)

## Test-Time Compute and Search
- [AlphaProof Analysis](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/)
- [ThetaEvolve](https://arxiv.org/html/2511.23473)
- [Scaling Test-Time Compute](https://arxiv.org/html/2512.02008)
- [Monte Carlo Tree Search for Theorem Proving](https://arxiv.org/pdf/1611.05990)
- [Proof Number Search](https://arxiv.org/html/2303.09449v4)

## Synthetic Data Generation
- [DeepSeek-Prover](https://arxiv.org/abs/2405.14333)
- [LLM-based ATP and Synthetic Data](https://arxiv.org/abs/2505.12031)
- [Spark-Prover-X1](https://arxiv.org/html/2511.13043)
- [Synthetic Theorem Generation in Lean](https://openreview.net/forum?id=EeDSMy5Ruj)
- [Theorem Prover as a Judge](https://arxiv.org/html/2502.13137)

## Lean 4 Metaprogramming
- [Metaprogramming in Lean 4](https://leanprover-community.github.io/lean4-metaprogramming-book/)
- [Mathlib4 Metaprogramming Guide](https://github.com/leanprover-community/mathlib4/wiki/Metaprogramming-for-dummies)
- [Lean 4 Tactics](https://leanprover-community.github.io/lean4-metaprogramming-book/main/09_tactics.html)

## Knowledge Representation
- [Mathematical Knowledge Representation](https://link.springer.com/article/10.1134/S1995080214040143)
- [Automating Mathematical Proof with Knowledge Graphs](https://arxiv.org/html/2503.11657v1)
- [LLMs for Mathematical Reasoning](https://arxiv.org/html/2402.00157v1)

## Formal Verification in Practice
- [Lessons from Verified Deployed Systems](https://arxiv.org/html/2301.02206v3)
- [Formal Verification of Cyber-Physical Systems](https://arxiv.org/abs/2003.03729)
- [Formal Methods Overview](https://www.galois.com/what-are-formal-methods)

## Recent Conferences and Workshops
- [AI for Math Workshop @ ICML 2024](https://sites.google.com/view/ai4mathworkshopicml2024)
- [AI for Math Workshop @ ICML 2025](https://sites.google.com/view/ai4mathworkshopicml2025/home)
- [Survey: Deep Learning for Theorem Proving](https://github.com/zhaoyu-li/DL4TP)
- [Aristotle: IMO-level ATP](https://arxiv.org/html/2510.01346v1)

## Architecture and Inductive Bias
- [Graph Inductive Biases in Transformers](https://arxiv.org/abs/2305.17589)
- [Graph Transformers Survey](https://arxiv.org/html/2407.09777v1)
- [Inductive Bias in Transformers](https://arxiv.org/abs/2402.05173)

## Curriculum Learning
- [What Makes a Good Curriculum](https://arxiv.org/abs/2510.19099)
- [GAR: Generative Adversarial RL](https://www.researchgate.net/publication/396499692_GAR_Generative_Adversarial_Reinforcement_Learning_for_Formal_Theorem_Proving)
- [Formal Mathematics Statement Curriculum Learning](https://paperswithcode.com/paper/formal-mathematics-statement-curriculum)

## Open Problems
- [Autoformalization Survey](https://arxiv.org/html/2505.23486v1)
- [Formal Mathematical Reasoning: A New Frontier](https://arxiv.org/html/2412.16075v1)
- [MATHESIS: Towards Formal Theorem Proving](https://openreview.net/pdf/600dc6f6d9d25d53a7fc6460423b585343099993.pdf)

## Meta
- [When to create your own language](https://softwareengineering.stackexchange.com/questions/62727/when-is-it-reasonable-to-create-my-own-programming-language)
