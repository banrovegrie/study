# System Architecture: Claude Code as Orchestrator

## The Correct Mental Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        CLAUDE CODE (THE BRAIN)                              │
│                                                                             │
│   "I read commands and skills, call tools, use memory, and reason"          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐
    │COMMANDS │          │ SKILLS  │          │  TOOLS  │          │ MEMORY  │
    │         │          │         │          │         │          │         │
    │Workflows│          │Reasoning│          │Executables         │State    │
    │I follow │          │patterns │          │I call   │          │I use    │
    └─────────┘          └─────────┘          └─────────┘          └─────────┘
```

**Key Distinction:**
- Commands/Skills = **prompts that guide Claude's reasoning** (not code that runs)
- Tools = **Python executables Claude can call** (actual code)
- Memory = **persistent state Claude can read/write** (JSON files)

---

## Commands (.claude/commands/)

Commands are structured workflows. When Claude sees `/solve`, it follows the workflow.

### /solve - Main Solving Workflow
```
1. Parse & Classify problem
2. Enumerate ALL cases (CRITICAL!)
3. Generate hypotheses
4. Explore small cases
5. Attack best approach
6. Verify (MANDATORY)
7. Self-evaluate (0, 0.5, or 1)
8. Refinement loop if needed
9. Output structured answer
```

Key lessons embedded:
- "Lesson from 2021 A1: Missed (0,+/-5) hops -> wrong answer"
- "Referencing theorems without proof does NOT count"
- DeepSeekMath-style self-evaluation

### /verify - Verification Protocol
```
1. Numerical testing (MANDATORY)
2. Symbolic verification
3. Step-by-step audit
4. Check common errors
5. Meta-verification (optional)
```

### /approaches - Technique Suggestions
Returns ranked techniques from 30+ method taxonomy by topic.

---

## Skills (.claude/skills/)

Skills are reusable reasoning strategies. Claude invokes them as needed.

### Core Skills

| Skill | When to Use | Key Insight |
|-------|-------------|-------------|
| **hypothesis** | Starting problem, stuck | Generate ALL possible approaches first |
| **explore** | Sequences, parameters | "Less is more" - small n reveals structure |
| **critique** | After solution, low confidence | Find errors before claiming success |
| **backtrack** | Current approach failing | Document failures, learn, try different |
| **evolution** | Construction, optimization | Evolve SEARCH STRATEGIES, not solutions |
| **mcts** | 3+ approaches failed | Systematic fallback, not default |

### Key Principles in Skills

**From hypothesis skill:**
> "Evolve programs that SEARCH for constructions, not programs that directly generate them."

**From explore skill:**
> "The point of rigor is to guide or build intuition."

**From evolution skill:**
> "LLMs are better at improving Python code than mathematical objects."

**From backtrack skill:**
> "Every failed approach teaches something"

---

## Tools (src/tools.py)

Claude calls these via `python -m src.tools <tool> <args>`.

### Tool Catalog

| Tool | Usage | Returns |
|------|-------|---------|
| `verify` | `verify "n*(n+1)/2" --compute "sum(range(1,n+1))"` | passed, confidence, stage |
| `search` | `search "problem text" --iterations 100 --method mcts` | best_path, best_reward |
| `library` | `library "pigeonhole"` | matching proofs |
| `approaches` | `approaches "number_theory"` | ranked techniques |
| `gemini` | `gemini "problem" --question "What do you think?"` | Gemini's response |
| `parse` | `parse "Prove that..."` | type, topics, goal |
| `simplify` | `simplify "(a+b)^2" --operation expand` | result |
| `archive` | `archive 2023 A1` | problem statement |
| `evaluate` | `evaluate --proof "step1\nstep2"` | score, issues |
| `benchmark` | `benchmark --year 2023 --problem_id A1` | analysis, gaps |

### Verification Cascade

The `verify` tool runs a multi-stage cascade:
```
Stage 1: Numerical (quick sanity check)
         ↓ Pass
Stage 2: Symbolic (algebraic verification)
         ↓ Pass
Stage 3: Semiformal (1000+ cases)
         ↓ Pass
Stage 4: Formal (Lean proof)
         ↓ Verified
```

---

## Memory (src/memory/)

### World Model (world_model.py)
Tracks:
- Problems: id, status (unseen→attempted→solved→verified→formalized)
- Attempts: techniques tried, success, confidence, time spent
- Solutions: best solution, Lean proof
- Relationships: similar problems

### Technique Tracker (technique_tracker.py)
Tracks effectiveness by topic:
```python
# Example structure
stats["number_theory"]["induction"] = TechniqueStats(successes=15, attempts=25)
# success_rate = 60%, confidence = min(1, 25/10) = 1.0
```

Priors from 144 Putnam problems:
- Number theory: modular_arithmetic (70%), divisibility (65%), induction (60%)
- Algebra: telescoping (70%), am_gm (55%)
- Combinatorics: pigeonhole (65%), bijection (60%)

### Proof Library (library/)
- Herald library: 44K formal proofs
- Embeddings: semantic search capability
- Retrieval: `search_proofs(query)` returns relevant proofs

---

## What About orchestrator.py?

The `orchestrator.py` file is **NOT the primary orchestration mechanism**.

It appears to be:
1. An alternative implementation (possibly for Gemini)
2. A utility that COULD be called as a tool
3. Contains good logic but Claude Code itself should orchestrate

**Claude Code orchestrates by:**
- Following /solve command workflow
- Invoking skills as needed
- Calling tools via bash
- Using memory for context

---

## Data Flow Example

```
User: "Solve Putnam 2023 A1"

Claude Code:
1. Load problem: python -m src.tools archive 2023 A1
2. Parse: python -m src.tools parse "..."
3. Get approaches: python -m src.tools approaches "..."
4. [Invoke hypothesis skill] - Generate all approaches
5. [Invoke explore skill] - Test small cases
6. Develop solution using reasoning
7. Verify: python -m src.tools verify "formula" --compute "fn"
8. [Invoke critique skill] - Self-check
9. If errors: [invoke backtrack skill], try different approach
10. Output final solution with confidence
11. Update world model (record attempt)
```

---

## File Structure

```
solver/
├── .claude/
│   ├── commands/           # Workflow prompts
│   │   ├── solve.md        # Main solving workflow
│   │   ├── verify.md       # Verification protocol
│   │   └── approaches.md   # Technique lookup
│   └── skills/             # Reasoning strategies
│       ├── hypothesis/     # Generate approaches
│       ├── explore/        # Small case exploration
│       ├── critique/       # Self-verification
│       ├── evolution/      # Strategy mutation
│       ├── mcts/           # Systematic search
│       ├── backtrack/      # Failure recovery
│       └── ...
├── src/
│   ├── tools.py            # CLI tools for Claude
│   ├── core/               # Problem parsing
│   ├── search/             # MCTS, approaches
│   ├── verify/             # Verification cascade
│   │   ├── numerical.py
│   │   ├── symbolic.py
│   │   ├── semiformal.py
│   │   └── lean/           # Lean integration
│   ├── memory/             # World model, technique tracker
│   ├── library/            # Proof search
│   └── llm/                # Gemini integration
├── data/
│   ├── archive/            # 87 years Putnam
│   ├── memory/             # Persistent state
│   └── library/            # 44K proofs
└── lean/                   # Lean 4 proofs
```
