# 08. Orchestration: Claude Code as the Reasoning Core

## Prime Principle

**ORCHESTRATION IS DONE BY CLAUDE CODE ONLY.**

Claude Code is not just a tool - it IS the orchestrator. The meta-reasoner. The decision-maker. Everything else is what Claude Code uses.

---

## The System

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                        CLAUDE CODE = THE ORCHESTRATOR                            │
│                                                                                  │
│   Reasons about problems. Decides what to try. Explores. Backtracks. Learns.    │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           THE SYSTEM                                     │   │
│   │                                                                          │   │
│   │   SKILLS          Reusable strategies, callable units                    │   │
│   │                   "try induction", "use pigeonhole", "factor polynomial" │   │
│   │                                                                          │   │
│   │   COMMANDS        Slash commands, actions                                │   │
│   │                   /solve, /verify, /formalize, /search                   │   │
│   │                                                                          │   │
│   │   TOOLS           Code execution, Lean formalization, CAS, tests         │   │
│   │                   Execute Python, run SymPy, compile Lean, run tests     │   │
│   │                                                                          │   │
│   │   ENVIRONMENT     Setups for experimentation/validation/verification     │   │
│   │                   Lean project, test harness, computation sandbox        │   │
│   │                                                                          │   │
│   │   MEMORY          World model, database, graphs, context management      │   │
│   │                   Graph of thought, branching/merging, persistence       │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Skills

Reusable, callable strategies that encapsulate proven approaches.

```
SKILLS
├── Techniques        "induction", "contradiction", "pigeonhole"
├── Patterns          "reduce to known problem", "find invariant"
├── Domain-specific   "use Cauchy-Schwarz", "apply generating functions"
└── Meta-strategies   "try small cases first", "work backwards"
```

**How Claude Code uses skills:**
- Query relevant skills for a problem type
- Execute skill as a structured approach
- Learn which skills work for which problems
- Compose skills into multi-step strategies

See: `03_SKILLS_LIBRARY.md`

### 2. Commands

Entry points and actions Claude Code can invoke.

```
COMMANDS
├── /solve <problem>     Start solving attempt
├── /verify <proof>      Run verification cascade
├── /formalize <proof>   Convert to Lean
├── /search <query>      Find relevant proofs/lemmas
├── /branch              Create exploration branch
├── /merge               Merge successful branches
└── /reflect             Analyze attempt, update memory
```

### 3. Tools

Executable capabilities for computation, verification, formalization.

```
TOOLS
├── Code Execution
│   ├── Python/SymPy    Symbolic computation, CAS
│   ├── NumPy/SciPy     Numerical computation
│   └── Custom scripts  Problem-specific computations
│
├── Lean Formalization
│   ├── Compile         Check if Lean code compiles
│   ├── Type-check      Verify types and goals
│   ├── Tactic suggest  Get tactic suggestions for goal
│   └── Sorry-fill      Iteratively fill sorry placeholders
│
├── Verification
│   ├── Numerical       Test against examples
│   ├── Symbolic        CAS verification
│   ├── Semiformal      LLM-judged rigor check
│   └── Formal          Full Lean verification
│
└── Search/Retrieval
    ├── Mathlib search  Find relevant Mathlib lemmas
    ├── Proof search    Find similar proofs in library
    └── Web search      Find relevant papers/discussions
```

### 4. Environment

Setups for experimentation, validation, verification, grounding.

```
ENVIRONMENT
├── Lean Project
│   ├── Mathlib imports
│   ├── Custom lemmas
│   └── Problem-specific definitions
│
├── Computation Sandbox
│   ├── Python environment
│   ├── Test harness
│   └── Resource limits
│
├── Verification Harness
│   ├── Test case generators
│   ├── Counter-example search
│   └── Grounding checks
│
└── Experiment Tracking
    ├── Approach logging
    ├── Result recording
    └── Failure analysis
```

### 5. Memory

World model, persistence, graphs, context management.

```
MEMORY
├── World Model (Kosmos/FutureHouse-inspired)
│   ├── Local Context      Current problem state, constraints discovered
│   ├── Global Knowledge   Domain syllabus, theorem relationships
│   └── Grounding          Facts established, verified claims
│
├── Graph of Thought
│   ├── Nodes              Reasoning states, partial proofs
│   ├── Branches           Exploration paths
│   ├── Merges             Successful path combinations
│   └── Pruning            Dead-end detection
│
├── Technique Tracker
│   ├── Success rates      Per technique, per domain
│   ├── Failure patterns   Why techniques fail
│   └── Recommendations    What to try next
│
└── Persistence
    ├── Problem database   All attempted problems
    ├── Proof library      Successful proofs
    └── Session state      Current exploration state
```

**World Model Reference:**
- Paper: Kosmos (FutureHouse, 2024) - arxiv.org/abs/2511.02824
- Reference impl: https://github.com/jimmc414/Kosmos
- Kosmos uses structured world model (graph DB) shared across agents - NOT neural
- Our approach: Kosmos-style structured world model + mathematical domain knowledge
- Future: Neural memory backends (see 01_MEMORY_SYSTEM.md)

See: `01_MEMORY_SYSTEM.md`, `02_GRAPH_OF_THOUGHT.md`

---

## How Claude Code Orchestrates

Claude Code doesn't follow a fixed algorithm. It **reasons** about what to do next using:

### The THINK → ACT → OBSERVE → LEARN Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│   THINK     What do I know? What should I try? What's promising? │
│      │      Query memory, assess state, consider options          │
│      ▼                                                            │
│   ACT       Execute skill, run tool, make computation            │
│      │      Use the system components                             │
│      ▼                                                            │
│   OBSERVE   What happened? Did it work? What did I learn?        │
│      │      Check results, analyze output                         │
│      ▼                                                            │
│   LEARN     Update memory, record outcome, adjust strategy       │
│      │      Close the flywheel                                    │
│      └──────────────────────────────────────────────────────────▶│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key Orchestration Patterns

**1. Problem Analysis**
```
Claude Code receives problem
  → Parse and understand
  → Query memory for similar problems
  → Identify domain, difficulty, structure
  → Retrieve relevant skills
```

**2. Strategy Selection**
```
Consider available approaches
  → What worked on similar problems?
  → What skills are relevant?
  → What's the verification path?
  → Balance exploration vs exploitation
```

**3. Exploration with Branching**
```
Try approach A
  → Branch: save current state
  → Execute approach
  → If stuck: backtrack to branch point
  → If progress: continue or branch again
  → Merge successful paths
```

**4. Verification Escalation**
```
Have candidate solution
  → Level 1: Computational check (fast)
  → Level 2: Semiformal check (rigorous-NL, Lean skeleton)
  → Level 3: Formal verification (full Lean)
  → At each level: if fail, learn and retry
```

**5. Learning and Recording**
```
Attempt complete (success or failure)
  → Record what was tried
  → Record what worked/didn't
  → Update technique tracker
  → Update world model
  → Persist for future problems
```

---

## Tool Interfaces

How Claude Code accesses system components:

### MCP Servers

```yaml
# Example MCP server configuration
servers:
  lean:
    command: "lean-mcp-server"
    tools:
      - compile
      - typecheck
      - suggest_tactic
      - fill_sorry

  memory:
    command: "memory-mcp-server"
    tools:
      - get_context
      - record_attempt
      - query_similar
      - update_world_model

  skills:
    command: "skills-mcp-server"
    tools:
      - search_skills
      - execute_skill
      - record_outcome

  compute:
    command: "compute-mcp-server"
    tools:
      - run_python
      - run_sympy
      - run_tests
```

### Tool Definitions

```python
# Tools exposed to Claude Code

@tool
def verify(proof: str, level: int = 2) -> VerificationResult:
    """
    Verify proof at specified certification level.

    Levels:
      0 - Informal (exploration check)
      1 - Computational (tests + CAS)
      2 - Semiformal (rigorous-NL, Lean+sorry, programs)
      3 - Formal (complete Lean)
    """
    pass

@tool
def search_skills(problem: str, domain: str = None) -> List[Skill]:
    """Find relevant skills for a problem."""
    pass

@tool
def execute_skill(skill_id: str, context: dict) -> SkillResult:
    """Execute a skill with given context."""
    pass

@tool
def get_context(problem_id: str) -> Context:
    """Get all relevant context for a problem from memory."""
    pass

@tool
def record_attempt(problem_id: str, attempt: Attempt) -> None:
    """Record an attempt for learning."""
    pass

@tool
def branch(name: str = None) -> BranchId:
    """Create a new exploration branch."""
    pass

@tool
def merge(branch_ids: List[BranchId]) -> MergeResult:
    """Merge successful branches."""
    pass

@tool
def run_lean(code: str) -> LeanResult:
    """Compile and run Lean code."""
    pass

@tool
def run_computation(code: str, language: str = "python") -> ComputeResult:
    """Run computation in sandbox."""
    pass
```

---

## Orchestration Prompts

Claude Code's behavior is guided by system prompts and context:

### Problem-Solving System Prompt

```markdown
You are solving mathematical problems with access to:

**Skills**: Reusable strategies you can query and execute
**Tools**: Computation (Python, SymPy), Verification (Lean), Search
**Memory**: World model, technique tracker, proof library

**Process**:
1. ANALYZE the problem - what type is it? what domain?
2. QUERY memory - similar problems? relevant techniques?
3. SELECT approach - which skill/technique to try?
4. EXECUTE - use tools to implement approach
5. VERIFY - check work at appropriate level
6. LEARN - record outcome regardless of success

**Verification Levels**:
- Level 0: Informal exploration
- Level 1: Computational (tests + CAS)
- Level 2: Semiformal (rigorous-NL, Lean+sorry, programs)
- Level 3: Formal (complete Lean proof)

**Principles**:
- Verification is ground truth (Lean acceptance = success)
- Learn from every attempt
- Branch early, merge successful paths
- Know when to pivot vs persist
```

### Context Injection

```markdown
## Current Problem
{problem_statement}

## Memory Context
Similar problems: {similar_problems}
Relevant techniques: {relevant_techniques}
Domain knowledge: {domain_context}

## Available Skills
{skill_list}

## Current State
Approaches tried: {attempts}
Best partial result: {best_partial}
Verification level reached: {current_level}
```

---

## Integration Points

How the orchestrator (Claude Code) connects to other specs:

| Spec | Integration |
|------|-------------|
| `01_MEMORY_SYSTEM` | Tools: get_context, record_attempt, world_model |
| `02_GRAPH_OF_THOUGHT` | Tools: branch, merge, prune, get_frontier |
| `03_SKILLS_LIBRARY` | Tools: search_skills, execute_skill, record_outcome |
| `04_VERIFICATION` | Tools: verify (levels 0-3), lean_compile, lean_check |
| `05_AUTOFORMALIZATION` | Tools: formalize, fill_sorry, suggest_tactic |
| `06_SEARCH_AND_MCTS` | Strategy patterns Claude Code uses, not separate search |
| `07_RETRIEVAL` | Tools: search_proofs, search_lemmas, search_mathlib |

---

## What This Spec Does NOT Include

**No Python orchestrator class.** Claude Code IS the orchestrator.

**No separate search algorithm.** Claude Code does the search by reasoning.

**No fixed control flow.** Claude Code decides what to do based on context.

The other specs define the **components** Claude Code uses. This spec defines **how they're exposed** to Claude Code and **patterns** for using them effectively.

---

## Design Principles

### Principle 1: Claude Code Reasons, Tools Execute

Claude Code does the thinking. Tools do the computation/verification/storage.

Bad: Tool that "solves the problem"
Good: Tool that "verifies this proof" or "runs this computation"

### Principle 2: Rich Context, Simple Tools

Give Claude Code rich context (memory, similar problems, technique history).
Keep individual tools simple and focused.

### Principle 3: Everything Persists

Every attempt, every branch, every outcome goes to memory.
Claude Code learns from history.

### Principle 4: Verification as Ground Truth

Claude Code can explore freely at Levels 0-2.
Level 3 (Lean) is the only absolute truth.

---

## Summary

```
ORCHESTRATION = CLAUDE CODE

SYSTEM = Skills + Commands + Tools + Environment + Memory

Claude Code uses the system to:
  - Understand problems (memory, analysis)
  - Try approaches (skills, tools)
  - Verify solutions (verification cascade)
  - Learn from attempts (memory, flywheel)

All other specs define components.
This spec defines how Claude Code accesses them.
```
