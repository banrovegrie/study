# Putnam Solver System Review

**Date:** December 8, 2025
**Reviewer:** Claude Opus 4.5
**System:** ~/Documents/base/solver

---

## Critical Architecture Insight

**Claude Code IS the orchestrator.**

The Python code provides tools and memory. Claude Code does the reasoning using commands, skills, and tools.

```
SYSTEM = Claude Code (brain)
       + Commands (workflows)
       + Skills (reasoning strategies)
       + Tools (Python executables)
       + Memory (persistent state)
```

---

## Report Structure

| File | Contents |
|------|----------|
| `00_EXECUTIVE_SUMMARY.md` | High-level architecture, current state, recommendations |
| `01_ARCHITECTURE.md` | Detailed component breakdown: commands, skills, tools, memory |
| `02_GAP_ANALYSIS.md` | What works vs what needs connection |
| `03_RECOMMENDATIONS.md` | Prioritized actions from Day 1 to Month 2+ |
| `04_SEMIFORMAL_FLYWHEEL.md` | The core iteration loop concept |
| `05_EXTERNAL_CONTEXT.md` | SOTA systems and research alignment |

---

## Key Findings

### The Good
- **Correct architecture**: Claude Code as orchestrator
- **Well-designed commands**: 9-step /solve workflow
- **Comprehensive skills**: 14 reasoning strategies
- **Functional tools**: 10 Python utilities
- **Rich data**: 87 years Putnam, 44K proofs, 30+ techniques

### The Critical Gap
**Components exist but may not work together.**

The flywheel (solve → learn → solve better) is not active. Skills are not auto-invoked. Memory exists but may not be used.

### Primary Recommendation
**Test first, then connect.**
1. Verify current system works end-to-end
2. Connect memory to solving workflow
3. Add skill routing
4. Close the flywheel loop

---

## How to Use This Report

1. **Start with** `00_EXECUTIVE_SUMMARY.md` for the big picture
2. **Understand architecture** via `01_ARCHITECTURE.md`
3. **See gaps** in `02_GAP_ANALYSIS.md`
4. **Prioritize work** via `03_RECOMMENDATIONS.md`
5. **Understand the core loop** in `04_SEMIFORMAL_FLYWHEEL.md`

---

## The Semi-Formal Sweet Spot

**Semi-formal = md + latex + lean (with sorry's) + citations**

```
Informal ──► SEMI-FORMAL ◄──► Formal
                  ▲
              iterate here
          (md+latex+lean+sorry)

Write Lean skeleton → Lean checks types → Fill sorry's → Formal
```

**This includes running Lean with sorry's, not just numerical testing.**

---

## Immediate Actions

### Day 1
1. Test /solve workflow end-to-end
2. Build semantic index (~30 min)
3. Verify world model persists

### Week 1
1. Add memory queries to /solve
2. Add skill routing logic
3. Test flywheel: solve → record → learn

---

## Caveats

- This review is based on code reading, not execution testing
- Some claims should be verified by actually running the system
- Time estimates are rough approximations
- The orchestrator.py file appears to be for Gemini or alternative use
