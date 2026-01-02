# Kepler

SYSTEM (with Claude Code as ORCHESTRATOR) = META + SKILLS + TOOLS + RUNTIME + MEMORY, where:

META = Alignment layer (goals, values, philosophy, strategic planning)
SKILLS = Reusable callable strategies, decontextualized techniques
TOOLS = Lean verification, computation, knowledge base queries
RUNTIME = Execution environment, sandbox, resource management
MEMORY = WORKING (local world model) + LIBRARY (global world model)

## Architecture

```
                                    PROBLEM
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                         CLAUDE CODE AS THE ORCHESTRATOR                          │
│                                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │    META    │  │   SKILLS   │  │   TOOLS    │  │  RUNTIME   │  │   MEMORY   │  │
│  │            │  │            │  │            │  │            │  │            │  │
│  │ CLAUDE.md  │  │ reusable   │  │ lean_*     │  │ execution  │  │ WORKING    │  │
│  │ planning   │  │ strategies │  │ compute_*  │  │ sandbox    │  │ LIBRARY    │  │
│  │ goals      │  │ prompts    │  │ kb_*       │  │ resources  │  │            │  │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                             ┌──────────────────┐
                             │   EXPLORATION    │
                             │                  │
                             │ branch → explore │
                             │ prune  → merge   │
                             └────────┬─────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                           VERIFICATION CASCADE                                   │
│                                                                                  │
│   NUMERICAL ──▶ SYMBOLIC ──▶ TYPE-LEVEL ──▶ TACTIC ──▶ PROOF SEARCH ──▶ FULL     │
│                                                                                  │
│   fast/cheap              ◀── iterate here ──▶                    slow/certain   │
│   sanity check            md + latex + lean + sorry               no sorry's     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
                ┌───────────┐                    ┌─────────────┐
                │ BACKTRACK │                    │  CERTIFIED  │
                │ + learn   │◄──────────────────▶│  SOLUTION   │
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

## META: The Alignment and Planning Layer

META is the "constitution" that guides all decisions—the WHY behind what Kepler does.

```
META
├── SYSTEM CONSTITUTION (CLAUDE.md)
│   └── Persistent guidance for all runs
│       ├── GOAL: What Kepler is trying to achieve
│       ├── VALUES: What makes a good proof/discovery
│       ├── PHILOSOPHY: How to approach mathematics
│       ├── ACCESS: How to use MEMORY, TOOLS, SKILLS
│       └── PRIORITIES: How to make tradeoffs
│
└── STRATEGIC STATE (planning.json)
    └── Per-run planning for current problem
        ├── Current goal & strategy
        ├── Assumptions being made
        ├── Checkpoints for self-correction
        ├── Blockers & open questions
        └── Insights gained
```

## MEMORY: Working + Library

```
MEMORY
├── WORKING (Local World Model)
│   └── Files in /session/ — ephemeral, per-problem
│       ├── planning.json      Strategic state (high-level decisions)
│       ├── exploration.json   Tactical state (proof tree, all branches)
│       ├── proof.lean         Current Lean code
│       └── goals.json         Extracted goals/sorries from Lean
│
└── LIBRARY (Global World Model)
    └── PostgreSQL + Neo4j — persistent, grows forever
        ├── Theorems (50K+ from Mathlib, indexed by type + embedding)
        ├── Concepts (mathematical concepts, hierarchically organized)
        ├── Techniques (learned patterns, indexed by applicability)
        ├── Problem Types (classification with technique mappings)
        ├── Proofs (completed proofs with full traces)
        ├── Papers (indexed literature, decontextualized)
        └── Communities (auto-clustered themes)
```

### WORKING: Local World Model

The local world model tracks the **current problem**:

#### exploration.json (Tactical State)

The proof tree—every branch, every attempt:

```json
{
  "problem_id": "putnam_2023_a1",
  "root": {
    "id": "root",
    "type": "goal",
    "informal": "Prove the main theorem",
    "formal": "theorem main : ∀ n ≥ 1, ...",
    "status": "in_progress",
    "children": [
      {
        "id": "approach_1",
        "type": "approach",
        "informal": "Strong induction on n",
        "status": "in_progress",
        "confidence": 0.7,
        "children": [
          {
            "id": "step_1_1",
            "type": "step",
            "informal": "Base case n = 1",
            "formal": "theorem base : P 1 := by ...",
            "status": "done",
            "techniques_used": ["simp", "ring"]
          },
          {
            "id": "step_1_2",
            "type": "step",
            "informal": "Inductive step",
            "status": "in_progress",
            "attempts": [
              {"tactic": "induction", "result": "progress", "new_goals": [...]}
            ]
          }
        ]
      },
      {
        "id": "approach_2",
        "type": "approach",
        "informal": "Direct computation",
        "status": "not_started"
      }
    ]
  },
  "constraints": [
    {"discovered_at": "step_1_2", "content": "Need n > 0 for division"}
  ],
  "current_focus": "step_1_2"
}
```

#### goals.json (Extracted from Lean)

Auto-updated whenever proof.lean changes:

```json
{
  "goals": [
    {
      "target": "P n",
      "hypotheses": [
        {"name": "n", "type": "ℕ"},
        {"name": "ih", "type": "∀ k < n, P k"}
      ],
      "location": {"line": 23}
    }
  ],
  "sorries": [{"location": "line 25", "goal": "P n"}],
  "errors": []
}
```

### LIBRARY: Global World Model

The global world model is the **accumulated knowledge** of all runs.

#### Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LIBRARY STORAGE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                     POSTGRESQL (Documents + Vectors)                      │  │
│  │                                                                           │  │
│  │  theorems                         proofs                                  │  │
│  │  ├── id, name, module             ├── id, problem_id, problem_text        │  │
│  │  ├── statement, type_signature    ├── solution_lean                       │  │
│  │  ├── docstring, tags[]            ├── exploration_tree (jsonb)            │  │
│  │  ├── embedding (vector)           ├── technique_trace[]                   │  │
│  │  └── neo4j_node_id                ├── problem_embedding (vector)          │  │
│  │                                   └── lessons[]                           │  │
│  │  documents                                                                │  │
│  │  ├── id, title, content, source                                           │  │
│  │  ├── embedding (vector)                                                   │  │
│  │  └── extracted_entity_ids[]                                               │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│                                       │ linked by ID                            │
│                                       ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                     NEO4J (Relationships + Structure)                     │  │
│  │                                                                           │  │
│  │  NODES                                                                    │  │
│  │  ├── (:Theorem {id, name, pg_id})                                         │  │
│  │  ├── (:Concept {id, name, description, embedding})                        │  │
│  │  ├── (:Technique {id, name, goal_patterns, success_rate})                 │  │
│  │  ├── (:ProblemType {id, name, description})                               │  │
│  │  └── (:Community {id, level, summary, embedding})                         │  │
│  │                                                                           │  │
│  │  RELATIONSHIPS                                                            │  │
│  │  ├── (Theorem)-[:DEPENDS_ON]->(Theorem)                                   │  │
│  │  ├── (Theorem)-[:USES]->(Concept)                                         │  │
│  │  ├── (Concept)-[:RELATED_TO {weight}]->(Concept)                          │  │
│  │  ├── (Concept)-[:PARENT_OF]->(Concept)                                    │  │
│  │  ├── (Technique)-[:APPLIES_TO]->(ProblemType)                             │  │
│  │  ├── (Technique)-[:OFTEN_FOLLOWED_BY {count}]->(Technique)                │  │
│  │  ├── (Technique)-[:IF_STUCK_TRY {condition}]->(Technique)                 │  │
│  │  └── (Community)-[:CONTAINS]->(*)                                         │  │
│  │                                                                           │  │
│  │  COMMUNITIES (auto-generated via Leiden algorithm)                        │  │
│  │  ├── L0: Fine-grained clusters (~100s)                                    │  │
│  │  ├── L1: Medium clusters (~20s)                                           │  │
│  │  └── L2: Broad themes (~5)                                                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Technique Indexing (By Applicability, Not Aboutness)

```
TECHNIQUE: "Infinite descent"

WRONG (by aboutness):
  topics: ["number_theory", "fermat"]

RIGHT (by applicability):
  goal_patterns: ["∄ x : ℕ, P(x)", "¬∃ minimal element"]
  preconditions: ["well_ordering_available", "can_construct_smaller"]
  structural_signature: "prove_nonexistence_via_infinite_regress"
```

A technique from number theory becomes applicable to combinatorics if the **goal structure** matches.

#### Library Building Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LIBRARY BUILDING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. INGEST                                                                  │
│     ├── Papers (arXiv, journals)                                            │
│     ├── Mathlib updates (new theorems)                                      │
│     ├── Competition problems (Putnam, IMO)                                  │
│     └── Our completed proofs                                                │
│                                                                             │
│  2. EXTRACT                                                                 │
│     ├── Theorems: statement, type, dependencies                             │
│     ├── Concepts: mathematical ideas present                                │
│     ├── Techniques: proof methods used                                      │
│     └── Relationships: what connects to what                                │
│                                                                             │
│  3. DECONTEXTUALIZE                                                         │
│     ├── Extract technique from original context                             │
│     ├── Index by STRUCTURAL APPLICABILITY                                   │
│     │   • Goal pattern it solves                                            │
│     │   • Preconditions required                                            │
│     │   • What it enables next                                              │
│     └── NOT by topic                                                        │
│                                                                             │
│  4. EMBED                                                                   │
│     └── Generate vector embeddings for semantic search                      │
│                                                                             │
│  5. GRAPH                                                                   │
│     ├── Add nodes to Neo4j                                                  │
│     ├── Create relationships                                                │
│     └── Recompute communities (Leiden)                                      │
│                                                                             │
│  6. PROMOTE                                                                 │
│     ├── High success rate techniques → higher retrieval weight              │
│     ├── Frequently used theorems → prioritize                               │
│     └── Community summaries → regenerate periodically                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## TOOLS

### Lean Tools (Verification)

```python
lean_elaborate(statement: str) -> {expr, type, errors}
    # Translate informal to Lean type

lean_check(code: str) -> {success, goals, sorries, errors}
    # Check Lean code, extract goals

lean_tactic(goal: str, hypotheses: list, tactic: str) -> {result, new_goals, error}
    # Apply a tactic, return result

lean_search(query: str, by: "embedding" | "type") -> [theorems]
    # Search Mathlib
```

### Compute Tools (Verification Cascade)

```python
compute_numerical(claim: str, examples: list) -> {holds, counterexample}
    # Quick numerical check

compute_symbolic(expr: str, operation: str) -> {result}
    # SymPy: simplify, expand, solve, limit, series
```

### Knowledge Base Tools

```python
kb_search(query: str, entity_type: str) -> [results]
    # Semantic search over theorems, concepts, proofs

kb_traverse(node: str, relationship: str, depth: int) -> {graph}
    # Graph traversal from a node

kb_path(from: str, to: str) -> [paths]
    # Find connections between entities

kb_community(node: str) -> {community, summary, members}
    # Get community membership and summary

kb_update_technique(name: str, success: bool, followed_by: str) -> {}
    # Update technique statistics after use

kb_save_proof(problem_id, solution, trace, lessons) -> {}
    # Save completed proof
```

### Planning Tools

```python
plan_read() -> planning.json
    # Read current planning state

plan_update(updates: dict) -> {}
    # Update planning state (high-signal events only)

plan_review() -> {recommendations}
    # Periodic review: check assumptions, checkpoints
```

### Exploration Tools

```python
tree_read() -> exploration.json
    # Read exploration tree

tree_update(node_id: str, updates: dict) -> {}
    # Update a node

tree_add_child(parent_id: str, child: dict) -> node_id
    # Add new branch

tree_set_focus(node_id: str) -> {}
    # Set current focus
```

---

## SKILLS

Reusable prompt strategies:

| Skill | Purpose |
|-------|---------|
| `/skill/understand` | Classify problem, find similar solved, identify structure |
| `/skill/strategize` | Generate approaches, evaluate tradeoffs, choose strategy |
| `/skill/explore` | Generate 2-3 distinct approaches for a goal |
| `/skill/formalize` | Convert informal step to Lean (possibly with sorry) |
| `/skill/diagnose` | Interpret Lean error, extract constraint, suggest fix |
| `/skill/reflect` | Review strategy, assumptions, checkpoints; trigger self-correction |
| `/skill/extract` | From completed proof, extract techniques, lessons, reusable lemmas |

---

## Verification Cascade

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VERIFICATION CASCADE                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 0: NUMERICAL              "Does this even make sense?"                │
│  ───────────────────────────────────────────────────────────                 │
│  Tool: compute_numerical                                                     │
│  Check: Evaluate for n=1,2,3,10,100                                          │
│  Output: Counterexample OR confidence boost                                  │
│  Time: <1 second                                                             │
│                                                                              │
│  LEVEL 1: SYMBOLIC               "Can we simplify/verify algebraically?"     │
│  ───────────────────────────────────────────────────────────                 │
│  Tool: compute_symbolic                                                      │
│  Check: Symbolic manipulation, limits, series                                │
│  Output: Simplified form OR stuck point                                      │
│  Time: 1-30 seconds                                                          │
│                                                                              │
│  LEVEL 2: TYPE-LEVEL             "Does this typecheck in Lean?"              │
│  ───────────────────────────────────────────────────────────                 │
│  Tool: lean_elaborate                                                        │
│  Check: Can we STATE this formally?                                          │
│  Output: Lean type OR type errors (constraint discovery)                     │
│  Time: 1-10 seconds                                                          │
│                                                                              │
│  LEVEL 3: TACTIC PROBING         "Which tactics make progress?"              │
│  ───────────────────────────────────────────────────────────                 │
│  Tool: lean_tactic                                                           │
│  Check: Try simp, ring, omega, linarith, aesop on each sorry                 │
│  Output: Closed sorries OR remaining goals                                   │
│  Time: 1-60 seconds per sorry                                                │
│                                                                              │
│  LEVEL 4: PROOF SEARCH           "Can automation find the proof?"            │
│  ───────────────────────────────────────────────────────────                 │
│  Tool: lean_check with Duper/Hammer/Polyrith                                 │
│  Check: Let ATP/SMT solvers try                                              │
│  Output: Complete proof OR timeout                                           │
│  Time: 1-300 seconds                                                         │
│                                                                              │
│  LEVEL 5: HUMAN-GUIDED           "What insight closes remaining gaps?"       │
│  ───────────────────────────────────────────────────────────                 │
│  Tool: Claude reasoning + all above                                          │
│  Check: Iterate on remaining sorries with informal insight                   │
│  Output: Complete proof OR refined gaps                                      │
│  Time: Minutes to hours                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Each level produces **information** that feeds back:
- Numerical counterexample → abandon approach
- Type error → discovered constraint
- Tactic failure → prune technique, try alternative
- Partial success → update gap tracker

---

## Semi-Formal: The Iteration Sweet Spot

```
SEMI-FORMAL
├── Markdown      (prose, structure, reasoning explanation)
├── LaTeX         (mathematical notation, formulas)
├── Lean          (machine-checkable fragments, sorry's allowed)
├── Citations     (references to Mathlib, known theorems)
└── References    (pointers to other files, papers, ideas)
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

The semi-formal representation is where most productive work happens. Full formalization is the end goal; informal is the starting point; semi-formal is where you iterate.

---

## The Reasoning Loop

```
THINK → EXPERIMENT → REFLECT → LEARN → (back to THINK)
```

**THINK**: Generate hypotheses, reflect, notice patterns, recall similar problems, check LIBRARY for applicable techniques.

**EXPERIMENT**: Run Lean, execute code, compute examples, try a tactic, fill a sorry, probe numerically.

**REFLECT**: What happened? Success → progress. Failure → information (not just "retry"). Partial → extract what worked.

**LEARN**: Update exploration tree, record in planning if significant, update LIBRARY with new patterns.

**Key insight**: Iteration is not "run tool and retry." It's a continuous loop of thinking and probing. Every execution is a question to reality. Every result reshapes understanding.

### Lean Errors Are Information

- `type mismatch: expected Nat, got Int` → misunderstood the domain
- `unknown identifier` → need different import or theorem doesn't exist
- `tactic failed` → this approach doesn't work, try another
- `sorry filled successfully` → subgoal is tractable, structure is sound

---

## Exploration: Graph of Thought

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

The exploration tree (`exploration.json`) captures this structure:
- Multiple approaches can be explored
- Each approach decomposes into steps
- Steps can have sub-approaches
- Branches can be pruned (abandoned) or merged
- Status tracks: `not_started`, `in_progress`, `done`, `abandoned`

---

## Tactics Library

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
│  └── aesop, decide, native_decide                                           │
│                                                                             │
│  PROOF PATTERNS (indexed by goal structure, not topic)                      │
│  ├── "∀ n, P(n)" → induction, strong_induction, Nat.rec                     │
│  ├── "∃ x, P(x)" → use, constructor                                         │
│  ├── "a = b" → calc, rw, simp, ring                                         │
│  ├── "a < b" → linarith, nlinarith, positivity                              │
│  ├── "¬P" → by_contra, exfalso                                              │
│  └── "P ∨ Q" → left, right, cases                                           │
│                                                                             │
│  TACTIC SEQUENCES (learned from successful proofs)                          │
│  ├── "intro h; cases h with..." (destruct hypothesis)                       │
│  ├── "induction n with n ih; simp; ..." (standard induction)                │
│  ├── "by_contra h; push_neg at h; ..." (contradiction setup)                │
│  └── "obtain ⟨x, hx⟩ := h; use x; ..." (existential instantiation)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Stored in LIBRARY (Neo4j Technique nodes), indexed by goal pattern, learned from successful proofs.

---

## The Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: LOAD META                                                         │
│                                                                             │
│  Read CLAUDE.md → internalize goals, values, philosophy                     │
│  Check planning.json → any previous strategic state?                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: UNDERSTAND + STRATEGIZE                                           │
│                                                                             │
│  1. Read problem                                                            │
│  2. /skill/understand: classify, find similar solved                        │
│  3. compute_numerical: sanity check                                         │
│  4. kb_search: relevant theorems, techniques                                │
│  5. kb_community: get context for problem area                              │
│  6. /skill/strategize: choose approach                                      │
│  7. Initialize planning.json (goal, strategy, initial assumptions)          │
│  8. Initialize exploration.json (root + approaches)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: WORK (iterative loop)                                             │
│                                                                             │
│  While exploration tree has in_progress nodes:                              │
│                                                                             │
│    1. tree_read → understand current state                                  │
│    2. plan_read → check assumptions, checkpoints                            │
│    3. Find current_focus (deepest in_progress node)                         │
│                                                                             │
│    4. Based on node state:                                                  │
│       IF informal → /skill/formalize → lean_elaborate                       │
│       IF has Lean with sorries → lean_check → lean_tactic probe             │
│       IF stuck → /skill/diagnose → add constraint, try alternative          │
│       IF complete → mark done, move to sibling/parent                       │
│                                                                             │
│    5. Update exploration.json with results                                  │
│    6. IF significant event → plan_update                                    │
│    7. kb_update_technique for each technique used                           │
│                                                                             │
│    8. IF checkpoint triggered → /skill/reflect → reassess                   │
│    9. IF periodic review due → /skill/reflect                               │
│                                                                             │
│  Until: root complete OR all approaches exhausted                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: LEARN                                                             │
│                                                                             │
│  1. /skill/extract: technique trace, lessons, reusable lemmas               │
│  2. kb_update_technique: update success rates, sequences                    │
│  3. kb_save_proof: store for future retrieval                               │
│  4. Update Neo4j: new relationships discovered                              │
│  5. If new lemma proven: add to theorems                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Future: Neural Memory

Current design is **neural-ready**:

- **Associative Memory**: Modern Hopfield networks, Titans for test-time learning
- **Memory Diffusion**: Generate memories via denoising, not just retrieve
- **Neural KGs**: Graph neural networks over concept/theorem structures
- **Learned Embeddings**: Fine-tuned on mathematical text

The interfaces stay the same; the backend can evolve from structured storage to neural backends.

---

## Summary

```
KEPLER = Claude Code (ORCHESTRATOR)
       │
       ├── META
       │   ├── CLAUDE.md        Goals, values, philosophy (system constitution)
       │   └── planning.json    Strategic state per problem
       │
       ├── SKILLS
       │   └── /skill/*         Reusable prompt strategies
       │
       ├── TOOLS
       │   ├── lean_*           Verification (ground truth)
       │   ├── compute_*        Numerical + symbolic
       │   ├── kb_*             Knowledge base queries
       │   ├── plan_*           Planning state management
       │   └── tree_*           Exploration tree management
       │
       ├── RUNTIME
       │   └── Execution environment, sandbox, resources
       │
       └── MEMORY
           ├── WORKING (Local World Model)
           │   └── /session/ files
           │       ├── planning.json     (strategic)
           │       ├── exploration.json  (tactical)
           │       ├── proof.lean        (artifact)
           │       └── goals.json        (extracted state)
           │
           └── LIBRARY (Global World Model)
               └── PostgreSQL + Neo4j
                   ├── Theorems (50K+, embeddings, type signatures)
                   ├── Concepts (hierarchy, relationships)
                   ├── Techniques (indexed by applicability)
                   ├── Problem Types (classification)
                   ├── Proofs (with traces)
                   ├── Papers (decontextualized)
                   └── Communities (auto-clustered)
```

**Core Principle**: Retrieve by applicability, not aboutness.

**Key Innovation**: Bidirectional formal-informal reasoning loop where Lean verification informs exploration and exploration drives formalization.

**The Flywheel**: Solve → Learn → Solve Better. Every proof attempt teaches something. The LIBRARY grows. Future proofs benefit.
