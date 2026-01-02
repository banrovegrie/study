# Implementation Overview: Gaps, Priorities, and Architecture

## Executive Summary

This document maps the gap between the **vision** (ARCHITECTURE.md, RESEARCH.md) and **reality** (solver codebase). The goal is to build a system that scales test-time compute for mathematical problem solving with formal verification.

**Key Insight**: The solver codebase has solid foundations (memory persistence, verification cascade Levels 0-2, approach taxonomy) but is missing the **dynamic** components that make the architecture come alive:
- Graph of Thought (exploration, pruning, merging)
- Skills as callable units (not just descriptions)
- Full World Model (domain knowledge, not just technique tracking)
- Working MCTS with test-time compute scaling
- Autoformalization loop (sorry-filling)
- Semantic retrieval from proof library

---

## Gap Analysis

### What Exists vs What's Needed

| Component | Exists | Vision (ARCHITECTURE.md) | Gap |
|-----------|--------|-------------------------|-----|
| **Memory** | | | |
| Problem tracking | ✅ Full | Track problems, attempts | None |
| Technique tracker | ✅ Full | Learn technique effectiveness | None |
| Session manager | ✅ Full | Step-level tracking | None |
| World Model (local) | ⚠️ Partial | Current problem state, constraints | Missing constraint discovery |
| World Model (global) | ❌ Missing | Domain syllabus, theorem graph, heuristics | Full implementation needed |
| Graph of Thought | ❌ Missing | Branch → explore → prune → merge | Full implementation needed |
| **Skills** | | | |
| Approach taxonomy | ✅ Full | 30+ techniques documented | None |
| Skills as callables | ❌ Missing | Reusable, composable strategies | Full implementation needed |
| Meta-tactics | ❌ Missing | LLM-generated tactics | Full implementation needed |
| Skill discovery | ❌ Missing | Mine from successful proofs | Full implementation needed |
| **Verification** | | | |
| Level 0: Informal | ✅ Full | Exploration, hypothesis generation | None |
| Level 1: Computational+Symbolic | ✅ Full | Tests + CAS verification | None |
| Level 2: Semiformal | ⚠️ Partial | Rigorous-NL + Lean+sorry + Programs | Needs LLM logical flow check |
| Level 3: Formal | ⚠️ Scaffolded | Complete Lean verification | Needs error parsing improvement |
| Sorry-filling loop | ❌ Missing | Iterative refinement | Full implementation needed |
| **Search** | | | |
| Proof state | ✅ Full | Immutable state snapshots | None |
| MCTS nodes | ⚠️ Scaffolded | Node structure | Needs traversal algorithm |
| MCTS search | ❌ Missing | Selection, expansion, simulation, backprop | Full implementation needed |
| Test-time scaling | ❌ Missing | Adaptive compute allocation | Full implementation needed |
| PRM evaluation | ❌ Missing | Process reward model | Full implementation needed |
| **Retrieval** | | | |
| Proof library | ✅ 44K proofs | Herald proofs available | None |
| Keyword search | ⚠️ Partial | Basic structure | Needs implementation |
| Semantic search | ❌ Missing | Embedding-based retrieval | Full implementation needed |
| **Orchestration** | | | |
| Pipeline stages | ✅ Defined | PARSE → CONTEXT → ... → FORMALIZE | None |
| Main solve loop | ❌ Missing | Coordinate all components | Full implementation needed |

---

## Priority Ordering

Based on the architecture and research findings, here's the implementation order:

### Phase 1: Core Loop (P0 - Critical)
**Goal**: Make the system actually solve problems end-to-end.

1. **08_ORCHESTRATOR.md** - Claude Code as the reasoning core
   - Claude Code IS the orchestrator (not a separate implementation)
   - System = Skills + Commands + Tools + Environment + Memory
   - Tools exposed via MCP servers

2. **06_SEARCH_AND_MCTS.md** - Search strategies and tools
   - MCTS as a tool Claude Code can invoke
   - Search patterns for exploration
   - Test-time compute scaling strategies

3. **04_VERIFICATION_CASCADE.md** - Complete cascade
   - 4-level certification (Informal → Computational → Semiformal → Formal)
   - Lean error parsing → actionable feedback
   - Human review flagging for uncertain steps

### Phase 2: Intelligence (P1 - High)
**Goal**: Make the system learn and improve.

4. **01_MEMORY_SYSTEM.md** - Full memory
   - Local world model (per-problem context)
   - Global world model (domain knowledge)
   - Flywheel closure (record → learn → recommend)

5. **02_GRAPH_OF_THOUGHT.md** - Exploration
   - Branch creation
   - Pruning dead ends
   - Merging successful paths
   - Backtracking with learning

6. **03_SKILLS_LIBRARY.md** - Reusable strategies
   - Skills as callable objects
   - Skill retrieval by goal pattern
   - Success rate tracking per skill

### Phase 3: Formalization (P2 - Medium)
**Goal**: Produce verified proofs in Lean.

7. **05_AUTOFORMALIZATION.md** - Sorry-filling
   - LLM-driven tactic suggestion
   - Iterative refinement loop
   - Error-guided repair

8. **07_RETRIEVAL_SYSTEM.md** - Semantic search
   - Embedding generation for proofs
   - Vector database (FAISS/Chroma)
   - Retrieval-augmented proving

### Phase 4: Scaling (P3 - Future)
**Goal**: Handle harder problems with more compute.

9. **Advanced MCTS** - Test-time RL
   - Problem variation generation
   - Difficulty-adaptive compute allocation
   - Self-play conjecture generation

10. **Meta-tactic synthesis** - Automatic skill discovery
    - Mine tactics from successful proofs
    - Generate Lean 4 meta-tactics
    - Hierarchical skill composition

---

## Architecture Alignment

### From ARCHITECTURE.md

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
```

### Implementation Mapping

**SYSTEM = Skills + Commands + Tools + Environment + Memory**

| System Component | Implementation Module | Status |
|------------------|----------------------|--------|
| **Skills** (reusable strategies) | 03_SKILLS_LIBRARY.md | ❌ |
| **Commands** (slash commands) | 08_ORCHESTRATOR.md | ❌ |
| **Tools** (verification, Lean, compute) | 04_VERIFICATION_CASCADE.md | ⚠️ |
| **Tools** (search, retrieval) | 07_RETRIEVAL_SYSTEM.md | ❌ |
| **Environment** (Lean project, sandbox) | 04_VERIFICATION_CASCADE.md | ⚠️ |
| **Memory** (world model) | 01_MEMORY_SYSTEM.md | ❌ |
| **Memory** (graph of thought) | 02_GRAPH_OF_THOUGHT.md | ❌ |
| **Memory** (technique tracker) | 01_MEMORY_SYSTEM.md | ⚠️ |
| **Orchestrator** = Claude Code | 08_ORCHESTRATOR.md | ✅ Defined |

### From RESEARCH.md Key Findings

1. **Build on Lean, not beside it** → 04_VERIFICATION_CASCADE.md, 05_AUTOFORMALIZATION.md
2. **Verification cascade** → 04_VERIFICATION_CASCADE.md (expand existing)
3. **Test-time compute scaling** → 06_SEARCH_AND_MCTS.md
4. **Retrieval at every step** → 07_RETRIEVAL_SYSTEM.md
5. **Skills as first-class objects** → 03_SKILLS_LIBRARY.md
6. **World model for transfer** → 01_MEMORY_SYSTEM.md
7. **Structured error feedback** → 04_VERIFICATION_CASCADE.md, 05_AUTOFORMALIZATION.md
8. **Self-improving flywheel** → 01_MEMORY_SYSTEM.md, 03_SKILLS_LIBRARY.md

---

## Implementation Dependencies

```
                    08_ORCHESTRATOR (P0)
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    06_SEARCH         04_VERIFY         01_MEMORY
         │                 │                 │
         │                 ▼                 ▼
         │         05_AUTOFORMALIZE    02_GRAPH_OF_THOUGHT
         │                 │                 │
         └────────────────►│◄────────────────┘
                           ▼
                   03_SKILLS_LIBRARY
                           │
                           ▼
                   07_RETRIEVAL_SYSTEM
```

**Dependencies:**
- Orchestrator needs: Search, Verify, Memory
- Autoformalize needs: Verify (error feedback)
- Graph of Thought needs: Memory (persistence)
- Skills Library needs: Graph of Thought (discovery), Memory (tracking)
- Retrieval needs: Skills (for matching)

---

## Success Criteria

### Phase 1 Complete (Core Loop)
- [ ] Can solve a Putnam problem end-to-end via orchestrator
- [ ] MCTS explores at least 10 approaches per problem
- [ ] Verification cascade rejects incorrect solutions
- [ ] Technique tracker updates after each attempt

### Phase 2 Complete (Intelligence)
- [ ] World model provides relevant context for new problems
- [ ] Graph of thought prunes >30% of explored branches
- [ ] Skills library has 50+ callable skills
- [ ] Flywheel demonstrably improves success rate over time

### Phase 3 Complete (Formalization)
- [ ] Can fill 50% of sorry's automatically
- [ ] Lean errors map to specific repair strategies
- [ ] Retrieval finds relevant lemmas for 80% of goals

### Phase 4 Complete (Scaling)
- [ ] Harder problems get more compute automatically
- [ ] Self-play generates useful training problems
- [ ] Meta-tactics generalize across domains

---

## File Index

| File | Purpose | Priority |
|------|---------|----------|
| 00_OVERVIEW.md | This document - gaps and priorities | - |
| 01_MEMORY_SYSTEM.md | Full memory architecture | P1 |
| 02_GRAPH_OF_THOUGHT.md | Exploration, pruning, merging | P1 |
| 03_SKILLS_LIBRARY.md | Skills as first-class objects | P1 |
| 04_VERIFICATION_CASCADE.md | Complete verification with Lean | P0 |
| 05_AUTOFORMALIZATION.md | LLM-driven sorry-filling | P2 |
| 06_SEARCH_AND_MCTS.md | Test-time compute scaling | P0 |
| 07_RETRIEVAL_SYSTEM.md | Semantic proof retrieval | P2 |
| 08_ORCHESTRATOR.md | Main solving loop | P0 |

---

## Research Foundation

Each module is grounded in state-of-the-art research. Key references:

| Research | Insight | Used In |
|----------|---------|---------|
| **AlphaProof** (DeepMind, 2024) | MCTS + formal verification | 06_SEARCH, 04_VERIFY |
| **AI Co-scientist** (Google, 2024) | Tournament of ideas, self-critique | 08_ORCHESTRATOR |
| **Sakana AI-Scientist** (2024) | Full automation loop | 08_ORCHESTRATOR |
| **DeepSeek-Prover-V2** (2024) | Cold-start problem, progressive training | 05_AUTOFORMALIZE |
| **LeanDojo** (Yang et al., 2023) | Premise selection for proving | 07_RETRIEVAL |
| **Rango** (Sprint et al., 2024) | Hybrid BM25 + dense retrieval | 07_RETRIEVAL |
| **Harmonic** | Lean verification as ground truth | 04_VERIFY, 05_AUTOFORMALIZE |
| **LEGO-Prover** | Skill libraries, composable proofs | 03_SKILLS |
| **Draft-Sketch-Prove** | Informal-to-formal pipeline | 05_AUTOFORMALIZE |

---

## Questioning Assumptions

Every module includes a "Questioning Assumptions" section. Key open questions:

1. **Is embedding the right retrieval approach?** Mathematical similarity ≠ text similarity
2. **Should orchestration be learned?** Current design is hand-crafted
3. **Is more context always better?** Too much failure history might confuse LLMs
4. **Are sequential strategies optimal?** Parallel hypothesis generation might be faster
5. **Is Lean feedback sufficient?** Maybe we need richer error explanations
6. **Can skills generalize across domains?** Number theory tactics may not help in geometry

These are not blockers but areas requiring validation through experimentation.

---

## Design Principles

All modules follow these principles:

1. **Verification as Ground Truth**: Lean acceptance is the only truth
2. **Build on existing foundations**: Extend solver codebase, don't rewrite
3. **Learn from every attempt**: Success and failure both teach
4. **Resource awareness**: Compute is finite, allocate wisely
5. **First principles over cargo-culting**: Question every assumption

---

## Next Steps

1. **Read each module spec** in priority order (P0 → P1 → P2)
2. **Implement incrementally** - each module should be testable alone
3. **Close the flywheel** - ensure every attempt updates memory
4. **Measure and iterate** - track success rates, learn from failures
5. **Validate assumptions** - test whether design choices actually work

The system is architecturally sound. The task is implementation with continuous validation.
