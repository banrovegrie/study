System (with Claude Code as ORCHESTRATOR) = SKILLS + COMMANDS + TOOLS + RUNTIME + MEMORY, where:

- SKILLS = reusable callable strategies [and also decontextualized techniques—methods extracted from their original context, indexed by structural applicability (not topic), transferable across domains]
- COMMANDS = entry points, actions, slash commands
- TOOLS = executables, verification, web search, formalization (Lean), db access, retrieval
- RUNTIME = environment, sandbox, execution context, resource management
- MEMORY = world model, databases, graphs, KGs, graph of thought, persistence

Core Principle: **Retrieve by applicability, not aboutness.** Don't find papers *about* similar topics—find techniques that solved structurally similar goals, regardless of original domain.

# Architecture

```
                                   PROBLEM
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│                        CLAUDE CODE AS THE ORCHESTRATOR                         │
│                                                                                │
│         ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │
│         │    SKILLS    │      │    TOOLS     │      │    MEMORY    │           │
│         │              │      │              │      │              │           │
│         │  reusable    │      │  symbolic    │      │  working     │           │
│         │  strategies  │      │  numerical   │      │  library     │           │
│         │              │      │  code exec   │      │  world model │           │
│         │              │      │  LEAN        │      │  graph of    │           │
│         │              │      │              │      │  thought     │           │
│         └──────────────┘      └──────────────┘      └──────────────┘           │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                ┌────────────────────┐
                                │    EXPLORATION     │
                                │                    │
                                │  branch → explore  │
                                │  prune  → merge    │
                                └─────────┬──────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│                            VERIFICATION CASCADE                                │
│                                                                                │
│    COMPUTATIONAL ────▶ SYMBOLIC ────▶ SEMI-FORMAL ────▶ FORMAL (LEAN)          │
│                                                                                │
│ fast/cheap                     ◀── iterate here ──▶               slow/certain │
│ sanity checks                  md+latex+lean+sorry                no sorry's   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
                    ┌───────────┐                    ┌─────────────┐
                    │ BACKTRACK │                    │  CERTIFIED  │
                    │ + learn   │    <───────────>   │  SOLUTION   │
                    └─────┬─────┘                    └──────┬──────┘
                          │                                 │
                          └────────────────┬────────────────┘
                                           ▼
                                    ┌─────────────┐
                                    │  FLYWHEEL   │
                                    │             │
                                    │ solve →     │
                                    │ learn →     │
                                    │ solve better│
                                    └─────────────┘
```

## Semi-Formal

```
SEMI-FORMAL
├── Markdown      (prose, structure, reasoning explanation)
├── LaTeX         (mathematical notation, formulas, equations)
├── Lean          (machine-checkable fragments, sorry's allowed)
└── Citations     (references to Mathlib, Herald, known theorems)
└── References    (all non grounded references to ideas, other files, folders, systems, web etc.)
```

```
    INFORMAL ──────────▶ SEMI-FORMAL ──────────▶ FORMAL

    exploration          md + latex + lean       complete lean proof
    intuition            + citations             no sorry's
    natural language     + references            machine-verified
                         structure-verified
                              ▲
                              │
                         ITERATE HERE
```

```
THINK → EXPERIMENT → OBSERVE → LEARN → (back to THINK)
```

**THINK**: Generate hypotheses, reflect, notice patterns, recall similar situations, ground in domain knowledge, merge promising threads.

**EXPERIMENT**: Run Lean, execute code, compute examples, check edge cases, try a tactic, fill a sorry, test conjectures numerically, query Mathlib.

**OBSERVE**: What happened? What does the error say? Success → progress. Failure → information (not just "retry"). Partial → extract what worked, diagnose what didn't.

**LEARN**: Update world model, prune dead ends, notice new paths, refine hypotheses, build meta-tactics.

The point: **iteration is not "run tool and retry"**. It's a continuous loop of thinking and probing. The execution—whether Lean compilation, running Python, checking examples—is how we ask questions of reality. The answers (including errors, type mismatches, and partial successes) are data that reshape our understanding.

A Lean error isn't a failure to process—it's information:

- "type mismatch: expected Nat, got Int" → we misunderstood the domain
- "unknown identifier" → we need a different import or the theorem doesn't exist
- "tactic failed" → this approach doesn't work here, try another
- "sorry filled successfully" → this subgoal is tractable, the structure is sound

The thinking happens in natural language. The probing happens through tools. They weave together—you can't separate "the reasoning loop" from "the execution loop" because every probe / run / experiment is motivated by reasoning and every result feeds back into thought.

## Tactics Library

We need a massive indexed library of tactics.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TACTICS LIBRARY                                   │
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
│  ├── "prove for all n" → induction, strong_induction, Nat.rec               │
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

Why this matters:

- LLM needs to know WHICH tactic to try
- Wrong tactic = Lean error = wasted iteration
- Good tactic suggestions = faster convergence
- Pattern matching: "problem looks like X" → "try tactics Y, Z"

Every attempt feeds back:

- Which tactics worked for which goals?
- Which proof patterns applied?
- What errors did we see and how did we fix them?
- What similar problems have we solved?

## Exploration (Graph of Thought)

```
                         ┌─────────────────┐
                         │    PROBLEM      │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        ┌───────────┐       ┌───────────┐       ┌───────────┐
        │ Approach A│       │ Approach B│       │ Approach C│
        └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
              │                   │                   │
              ▼                   ▼                   ▼
           [stuck]            [partial]           [promising]
              │                   │                   │
           prune              continue             refine
                                  │                   │
                                  └─────────┬─────────┘
                                            ▼
                                        [merge]
                                            │
                                            ▼
                                       [solution]
```

## Memory Structure

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY                                          │
├───────────────────┬───────────────────┬──────────────────────────────────────┤
│   WORKING         │   LIBRARY         │   WORLD MODEL                        │
├───────────────────┼───────────────────┼──────────────────────────────────────┤
│ • current problem │ • solved proofs   │                                      │
│ • active branches │ • technique index │   LOCAL (problem-specific)           │
│ • failed attempts │ • LEAN tactics    │   ├── current problem state          │
│ • partial results │ • embeddings      │   ├── attempted approaches           │
│ • sorry's to fill │ • Mathlib index   │   ├── discovered constraints         │
│                   │                   │   ├── local concept relationships    │
│                   │                   │   ├── context engineering state      │
│                   │                   │   └── continuously updated           │
│                   │                   │                                      │
│                   │                   │   GLOBAL (domain knowledge)          │
│                   │                   │   ├── field syllabus & scope         │
│                   │                   │   ├── theorem graph                  │
│                   │                   │   ├── concept relations              │
│                   │                   │   ├── domain heuristics              │
│                   │                   │   ├── tactic success rates           │
│                   │                   │   └── subject matter coverage        │
├───────────────────┴───────────────────┴──────────────────────────────────────┤
│                         GRAPH OF THOUGHT                                     │
│              branch → explore → prune/merge → backtrack                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

Local vs Global World Model:

- Local World Model: Maintains context for the specific problem being solved. Tracks what we've tried, what constraints we've discovered, how sub-problems relate. Continuously updated as exploration proceeds. Essential for context engineering—knowing where we are, what's been ruled out, what's promising.

- Global World Model: Captures understanding of the entire field. For Putnam-style problems: the full syllabus, known techniques, which tactics apply to which problem types, conceptual relationships across mathematics. Relatively stable, grows as we solve more problems.

Knowledge Graph Implementation:

- **Concept Graph** (internal): `ConceptNode` + `MathDomain` structures in memory. We build and maintain this—domain hierarchies, concept relationships, tactic patterns.
- **Theorem Graph** (external): Lean/Mathlib's dependency structure. We don't rebuild this—we access it via tools. Lean already knows which theorems depend on which lemmas.

Principle: Build what Lean doesn't provide (domain heuristics, problem-specific context). Access what Lean already has (theorem dependencies, type information).

(Future) Neural Memory:

- **Associative Memory**: Modern Hopfield networks, Titans for test-time learning
- **Memory Diffusion**: Generate memories via denoising, not just retrieve
- **Neural KGs**: Graph neural networks over concept/theorem structures
- Current design is neural-ready: abstract interfaces allow swapping structured storage for neural backends
