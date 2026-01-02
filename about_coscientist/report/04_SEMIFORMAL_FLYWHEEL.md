# The Semi-Formal Flywheel: Core of the System

## What is Semi-Formal?

**Semi-formal is a mixed representation combining:**

```
SEMI-FORMAL DOCUMENT
├── Markdown      (prose, structure, reasoning explanation)
├── LaTeX         (mathematical notation, formulas, equations)
├── Lean          (machine-checkable fragments, sorry's allowed)
└── Citations     (references to Mathlib, Herald, known theorems)
```

**Key distinction:**
- **Informal**: Natural language, intuition, exploration
- **Semi-formal**: md + latex + lean (with sorry's) + citations — type-checked, structure-verified
- **Formal**: Complete Lean proof, no sorry's — machine-verified

---

## Why Semi-Formal Matters

```
INFORMAL ─────────► SEMI-FORMAL ◄─────────► FORMAL
                         ▲
    exploration          │           certification
    intuition            │           ground truth
                         │
                    ITERATE HERE
                    md+latex+lean+sorry
                         │
                    fast enough to try many things
                    rigorous enough to prune bad ideas
```

**Key Insight**: Semi-formal is where experimentation happens:
- Lean checks types and structure even with sorry's
- Fast iteration: write skeleton → Lean verifies types → fill sorry's one by one
- Immediate feedback: type errors caught instantly
- Only fully complete proofs graduate to formal

---

## The Flywheel Concept

### Basic Loop
```
Problem → Attempt → Verify → Learn → Better Attempt → ...
```

### Full Loop with Semi-Formal
```
Problem
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ EXPLORATION (Informal)                                   │
│                                                          │
│  hypothesis skill → enumerate approaches                 │
│  explore skill → small cases, patterns                   │
│                                                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ EXPERIMENTATION (Semi-Formal)     ◄──── iterate here    │
│                                                          │
│  Numerical check (10 cases)                              │
│      ↓ Pass                                              │
│  Symbolic check (algebraic)                              │
│      ↓ Pass                                              │
│  Extended numerical (1000 cases)                         │
│      ↓ Pass                                              │
│  LLM critique (Claude + Gemini)                          │
│      ↓ High confidence                                   │
│                                                          │
│  On ANY failure: analyze, refine, retry                  │
│                                                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ CERTIFICATION (Formal)                                   │
│                                                          │
│  Autoformalize to Lean skeleton                          │
│  Fill tactics (LLM)                                      │
│  Lean verify                                             │
│      ↓ Pass → CERTIFIED SOLUTION                         │
│      ↓ Fail → error feedback → back to semi-formal       │
│                                                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ LEARNING                                                 │
│                                                          │
│  Record attempt in world model                           │
│  Update technique tracker                                │
│  If novel: add to proof library                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## The Semi-Formal Iteration Loop

**This is where most work happens:**

```
1. Write Lean skeleton with sorry
2. Lean checks types, structure
3. Error? Fix and retry
4. Fill one sorry with tactic
5. Lean compiles? Move to next sorry
6. Error? Try different tactic
7. All sorry's filled = FORMAL (certified)
```

### Example Semi-Formal Document

```markdown
## Theorem: √2 is irrational

**Approach:** Proof by contradiction (cite: classical technique)

Assume $\sqrt{2} = \frac{p}{q}$ where $\gcd(p,q) = 1$.

Then $p^2 = 2q^2$, so $p^2$ is even, hence $p$ is even.

Let $p = 2k$. Then $4k^2 = 2q^2$, so $q^2 = 2k^2$, hence $q$ is even.

Contradiction with $\gcd(p,q) = 1$. ∎

**Lean fragment:**
```lean
theorem sqrt2_irrational : Irrational (Real.sqrt 2) := by
  intro ⟨p, q, hq, h⟩
  have hp : Even (p^2) := by sorry  -- TODO: from p^2 = 2*q^2
  have hp' : Even p := by sorry     -- TODO: even square → even
  sorry  -- finish contradiction
```

**Citations:**
- Mathlib: `Nat.even_pow`, `Int.even_or_odd`
- Similar: Herald #4521 (irrationality proofs)
- Tactic: `omega` for arithmetic, `ring` for algebra
```

---

## Verification Cascade Stages

### Stage 1: Computational (src/verify/numerical.py)
```python
# Quick sanity check
quick_check("n*(n+1)/2", lambda n: sum(range(1,n+1)), max_n=10)
```
- Fast: milliseconds
- Catches: wrong formulas, off-by-one errors
- Trust: high (if fails, solution is wrong)

### Stage 2: Symbolic (src/verify/symbolic.py)
```python
# Algebraic verification
verify_equality("(a+b)**2", "a**2 + 2*a*b + b**2")
```
- Medium speed: seconds
- Catches: algebraic errors, simplification mistakes
- Trust: high (SymPy is reliable)

### Stage 3: Semi-Formal (md + latex + lean + citations)
```
Write markdown document with:
- Prose explanation of proof
- LaTeX for mathematical notation
- Lean code with sorry's (type-checked)
- Citations to Mathlib/Herald
```
- Iteration speed: seconds per change
- Catches: type errors, structural issues, logic gaps
- Trust: medium-high (Lean validates structure)

### Stage 4: Formal (complete Lean, no sorry's)
```python
# Full Lean 4 proof
lean_verify(lean_code)  # All sorry's filled
```
- Slow: seconds to minutes (compilation)
- Catches: all logical errors
- Trust: absolute (machine-verified)

---

## Iteration Patterns

### Pattern 1: Numerical Failure
```
Solution: f(n) = n^2
Numerical check: FAIL at n=3 (expected 6, got 9)

Analysis: Wrong formula
Action: Re-examine derivation
Refined: f(n) = n*(n+1)/2
Retry...
```

### Pattern 2: Symbolic Failure
```
Solution: (a+b)^2 = a^2 + b^2
Symbolic check: FAIL (not equal to a^2 + 2ab + b^2)

Analysis: Missing cross term
Action: Add 2ab
Refined: (a+b)^2 = a^2 + 2ab + b^2
Retry...
```

### Pattern 3: Semiformal Failure
```
Solution: Works for n = 1..100
Extended check: FAIL at n=0

Analysis: Edge case not handled
Action: Add n=0 case or exclude from domain
Refined: For n >= 1, f(n) = ...
Retry...
```

### Pattern 4: Lean Failure
```
Solution: Semiformal verified
Lean check: FAIL - "tactic failed" at line 15

Analysis: Proof step not automatable
Action: Break into smaller lemmas or add explicit steps
Refined: Add intermediate lemma
Retry...
```

---

## Connecting to Existing System

### What Exists
- `src/verify/cascade.py`: Orchestrates stages 1-3
- `src/verify/lean/verifier.py`: Stage 4
- `src/verify/lean/formalizer.py`: NL → Lean translation

### What Needs Connection
1. **Failure feedback loop**: When stage N fails, feedback to stage N-1
2. **Lean error classification**: Parse Lean errors into actionable categories
3. **Automatic retry**: Refine and retry instead of just failing

### Proposed Addition
```python
def semiformal_iterate(problem, solution, max_iterations=5):
    """Iterate in semiformal space until confident or exhausted."""

    current = solution

    for i in range(max_iterations):
        # Run cascade up to semiformal
        result = verify_cascade(current, stages=[1, 2, 3])

        if result.passed:
            return current, result.confidence

        # Analyze failure
        failure = analyze_failure(result)

        # Refine based on failure type
        if failure.type == "numerical":
            current = refine_for_numerical(current, failure)
        elif failure.type == "symbolic":
            current = refine_for_symbolic(current, failure)
        elif failure.type == "edge_case":
            current = refine_for_edge_case(current, failure)

    return None, 0.0  # Exhausted
```

---

## Learning from Attempts

### Every Attempt Should Record
```python
world.record_attempt(
    problem_id=problem.id,
    techniques_tried=["induction", "algebra"],
    stages_passed=["numerical", "symbolic"],
    failure_stage="semiformal",
    failure_reason="edge case n=0",
    time_spent=45.0,
    notes="Forgot to handle n=0 case"
)
```

### Every Failure Should Update
```python
technique_tracker.update(
    topics=["number_theory"],
    technique="induction",
    success=False,
    failure_reason="edge_case"
)
```

### Every Success Should Share
```python
if success:
    # Add to proof library if novel
    if is_novel(solution):
        library.add(problem, solution)

    # Update success rates
    technique_tracker.update(
        topics=problem.topics,
        technique=winning_technique,
        success=True
    )
```

---

## Tactics Library (CRITICAL NEED)

**The LLM needs to know WHICH tactic to try. This requires a massive indexed library:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TACTICS LIBRARY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CORE TACTICS (Lean 4 built-in)                                             │
│  ├── intro, apply, exact, rfl, rw, simp, ring, omega, linarith              │
│  ├── cases, induction, rcases, obtain                                       │
│  ├── constructor, use, exists, left, right                                  │
│  ├── have, let, suffices, calc                                              │
│  └── contradiction, exfalso, by_contra                                      │
│                                                                             │
│  MATHLIB TACTICS                                                            │
│  ├── norm_num, positivity, polyrith, nlinarith                              │
│  ├── field_simp, ring_nf, group                                             │
│  ├── gcongr, rel_congr                                                      │
│  ├── continuity, measurability                                              │
│  └── aesop, decide, native_decide                                           │
│                                                                             │
│  DOMAIN-SPECIFIC PATTERNS                                                   │
│  ├── Number Theory: mod_cast, push_cast, zify, nat_abs                      │
│  ├── Algebra: linear_combination, polyrith                                  │
│  ├── Analysis: filter_upwards, tendsto_nhds                                 │
│  ├── Combinatorics: Finset.sum_*, Finset.card_*                             │
│  └── Topology: continuity, isOpen_*, isClosed_*                             │
│                                                                             │
│  PROOF PATTERNS (indexed by problem type)                                   │
│  ├── "prove for all n" → induction, strong_induction, Nat.rec              │
│  ├── "find all x" → constructor + cases                                     │
│  ├── "show exists" → use, constructor                                       │
│  ├── "prove equality" → calc, rw, simp                                      │
│  ├── "prove inequality" → linarith, nlinarith, positivity                   │
│  └── "prove by contradiction" → by_contra, exfalso                          │
│                                                                             │
│  TACTIC SEQUENCES (common patterns)                                         │
│  ├── "intro h; cases h with..." (destruct hypothesis)                       │
│  ├── "induction n with n ih; simp; ..." (standard induction)                │
│  ├── "by_contra h; push_neg at h; ..." (contradiction setup)                │
│  ├── "obtain ⟨x, hx⟩ := h; use x; ..." (existential)                        │
│  └── "calc x = ... := by ring; _ = ... := by rw [h]" (chain)                │
│                                                                             │
│  LEMMA INDEX (searchable by pattern)                                        │
│  ├── Nat.add_comm, Nat.mul_comm, ...                                        │
│  ├── Int.even_add, Int.odd_mul, ...                                         │
│  ├── Real.sqrt_*, Real.exp_*, Real.log_*, ...                               │
│  └── [44K+ Herald proofs indexed by technique]                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- Wrong tactic = Lean error = wasted iteration
- Good tactic suggestions = faster convergence
- Pattern matching: "problem looks like X" → "try tactics Y, Z"

---

## Key Principles

### 1. Fail Fast, Fail Cheap
```
Numerical (fast) → Symbolic (medium) → Semiformal (medium) → Formal (slow)
```
Don't spend compute on formal verification until semiformal passes.

### 2. Learn from Every Failure
Every failed attempt is data. Record it. Use it.

### 3. Iterate in Semi-Formal
Most refinement happens here. It's the right balance of speed and rigor.

### 4. Formal is Ground Truth
When you need certainty, formalize. But only for high-confidence solutions.

### 5. The Flywheel Compounds
Each solved problem makes the next easier:
- Better technique recommendations
- Similar problem references
- Accumulated proof patterns
