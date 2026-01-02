# Putnam Solver System Review: Executive Summary

**Review Date**: December 8, 2025
**Reviewer**: Claude Opus 4.5
**System Location**: ~/Documents/base/solver

---

## Core Architecture Understanding

**CRITICAL INSIGHT**: Claude Code IS the orchestrator. The Python code provides tools and memory - Claude Code does the reasoning.

```
                           PROBLEM
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                 CLAUDE CODE = THE ORCHESTRATOR                      │
│                                                                     │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │   COMMANDS   │   │    SKILLS    │   │    TOOLS     │           │
│   │              │   │              │   │              │           │
│   │ /solve       │   │ hypothesis   │   │ verify       │           │
│   │ /verify      │   │ explore      │   │ search       │           │
│   │ /approaches  │   │ critique     │   │ library      │           │
│   │              │   │ evolution    │   │ simplify     │           │
│   │              │   │ mcts         │   │ gemini       │           │
│   │              │   │ backtrack    │   │              │           │
│   └──────────────┘   └──────────────┘   └──────────────┘           │
│                              │                                      │
│                     ┌────────┴────────┐                             │
│                     │     MEMORY      │                             │
│                     │                 │                             │
│                     │ world_model     │                             │
│                     │ technique_tracker│                            │
│                     │ proof_library   │                             │
│                     └─────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │      VERIFICATION CASCADE         │
              │                                   │
              │  Computational → Symbolic →       │
              │  Semi-formal → Formal (Lean)      │
              └───────────────────────────────────┘
                              │
                              ▼
                      ┌─────────────┐
                      │  FLYWHEEL   │
                      │             │
                      │ solve →     │
                      │ learn →     │
                      │ solve better│
                      └─────────────┘
```

---

## System Components

### Commands (Entry Points)
3 structured workflows that guide Claude's reasoning:
- `/solve` - Full 9-step problem-solving workflow
- `/verify` - Solution verification protocol
- `/approaches` - Get technique suggestions

### Skills (Reasoning Strategies)
14 reusable reasoning patterns:
| Skill | Purpose |
|-------|---------|
| `hypothesis` | Generate all possible approaches |
| `critique` | Self-verification, error detection |
| `explore` | Small case exploration, pattern finding |
| `evolution` | LLM-guided strategy mutation (AlphaEvolve-style) |
| `mcts` | Fallback systematic search |
| `backtrack` | Explicit failure documentation and recovery |
| `cases` | Case analysis |
| `bound` | Establish bounds |
| `experiment` | Numerical experimentation |
| `library` | Search proof library |
| `gemini` | Get second opinion |
| `parallel` | Multiple approach exploration |
| `pattern` | Pattern recognition |
| `search` | General search |

### Tools (Python Executables)
11 callable tools via `python -m src.tools`:
1. `verify` - Verification cascade
2. `search` - MCTS/beam search
3. `library` - 44K proof search
4. `approaches` - Technique recommendations
5. `gemini` - Second opinion
6. `parse` - Problem classification
7. `simplify` - Symbolic manipulation
8. `archive` - Load Putnam problems
9. `evaluate` - Score proof rigor
10. `benchmark` - Flywheel testing
11. `flywheel_status` - Check learning status

### Memory (Persistent State)
- **World Model**: Problems, attempts, solutions, Lean proofs
- **Technique Tracker**: Success rates by topic (priors from 144 problems)
- **Proof Library**: 44K Herald proofs with embeddings

---

## Current State Assessment

### What Works
- Commands: Well-designed 9-step solve workflow
- Skills: Comprehensive reasoning strategies
- Tools: 10 functional utilities
- Memory: World model and technique tracker implemented
- Verification: 4-stage cascade (numerical → symbolic → semiformal → formal)
- Data: 87 years Putnam archive, 30+ technique taxonomy

### What Needs Work
1. **Flywheel not active**: Solve → Learn → Solve Better loop not automated
2. **Skills not auto-invoked**: Claude must manually choose skills
3. **Semi-formal iteration**: The key experimentation loop needs strengthening
4. **Lean integration**: Formalizer generates skeletons, needs LLM tactics
5. **Memory utilization**: World model exists but not actively used in solving

---

## The Semi-Formal Sweet Spot

**Semi-formal = md + latex + lean (with sorry's) + citations**

```
INFORMAL ──────► SEMI-FORMAL ◄───────► FORMAL
                      ▲
exploration     │     │          certification
intuition       │  iterate       ground truth
                │   here
                │ (md+latex+lean+sorry)
            MOST WORK HAPPENS HERE
```

**Key Insight**: Semi-formal is where experimentation happens:
- Write Lean skeleton with sorry's
- Lean checks types and structure immediately
- Fill sorry's one at a time with tactics
- Error? Try different tactic. Success? Move to next.
- All sorry's filled = formal proof

**This is NOT just numerical/symbolic testing. It includes running Lean with sorry's.**

---

## Primary Recommendations

### Immediate (Day 1)
1. Test current system: Does `/solve` + skills actually work end-to-end?
2. Build semantic index for library (~30 min)
3. Verify world model persists across sessions

### Week 1
1. Auto-invoke skills based on problem type
2. Connect Gemini for multi-model verification
3. Strengthen semi-formal iteration loop

### Month 1
1. Add LLM tactic generation for Lean formalizer
2. Implement active flywheel: solve → learn → solve better
3. Add skill routing based on technique tracker

---

## Bottom Line

The system has **excellent architecture** designed around the correct insight: **Claude Code IS the orchestrator**. The components (commands, skills, tools, memory) are well-designed.

The primary work is:
1. **Activation**: Ensure components actually work together
2. **Iteration**: Strengthen the semi-formal experimentation loop
3. **Learning**: Close the flywheel loop

**Note**: The `orchestrator.py` file appears to be for Gemini or alternative use - Claude Code itself does orchestration through commands and skills.
