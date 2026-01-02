# Gap Analysis: What's Working vs What's Needed

## Current State Summary

The system has the right architecture (Claude Code as orchestrator) with well-designed components. The question is: **are they actually working together?**

---

## Component Status

### Commands (Workflows)
| Component | Status | Assessment |
|-----------|--------|------------|
| `/solve` workflow | Well-designed | 9-step structure is solid |
| `/verify` protocol | Well-designed | Numerical + symbolic + audit |
| `/approaches` lookup | Implemented | Returns ranked techniques |

**Gap**: Need to verify these are actually followed end-to-end.

### Skills (Reasoning Strategies)
| Skill | Status | Gap |
|-------|--------|-----|
| hypothesis | Complete | Key insights embedded |
| explore | Complete | Generalizer mode documented |
| critique | Complete | Checklist comprehensive |
| evolution | Complete | AlphaEvolve principles |
| backtrack | Complete | Failure documentation |
| mcts | Complete | But marked as fallback |

**Gap**: Skills exist but need testing. Are they invoked at right times?

### Tools (Executables)
| Tool | Status | Gap |
|------|--------|-----|
| verify | Implemented | Cascade works |
| search | Implemented | MCTS available |
| library | Implemented | May need index |
| approaches | Implemented | 30+ techniques |
| gemini | Implemented | API key needed |
| parse | Implemented | Topic detection |
| simplify | Implemented | SymPy backend |
| archive | Implemented | 87 years data |
| evaluate | Implemented | Rigor scoring |
| benchmark | Implemented | Flywheel tool |

**Gap**: Tools exist. Need to test integration.

### Memory (Persistent State)
| Component | Status | Gap |
|-----------|--------|-----|
| World Model | Implemented | Need to verify persistence |
| Technique Tracker | Implemented | Priors seeded |
| Proof Library | Exists | Semantic index may need building |

**Gap**: Memory infrastructure exists but is it being used during solving?

---

## Critical Gaps

### Gap 1: Flywheel Not Active

**The Problem:**
The solve → learn → solve better loop is not automated.

```
Current:
  Solve problem → Done

Should be:
  Solve problem → Record attempt → Update technique tracker →
  Similar problem uses learned data → Solve better
```

**What Exists:**
- World model has `record_attempt()`
- Technique tracker has `update()` and `recommend()`
- But nothing connects them automatically

**Fix:**
After each solve attempt, explicitly:
1. Record to world model
2. Update technique tracker
3. On next problem, fetch context from world model

---

### Gap 2: Skills Not Auto-Invoked

**The Problem:**
Skills must be manually invoked. No routing logic.

**Example:**
```
Problem type = "prove for all n"
→ Should auto-invoke: hypothesis, explore, critique
→ Currently: Claude must remember to invoke them
```

**What Exists:**
- Skills are well-documented
- Problem parser detects type and topics
- Technique tracker knows what works

**Fix:**
Add skill routing in /solve command:
```
if problem.type == "prove":
    invoke hypothesis skill
    invoke critique skill
if problem.topics contains "number_theory":
    invoke explore skill for small cases
```

---

### Gap 3: Semi-Formal Iteration Weak

**The Problem:**
The semi-formal space (where most experimentation should happen) isn't well-connected.

**Semi-formal = md + latex + lean (with sorry's) + citations**

```
Informal ──► Semi-formal ◄──► Formal
                ▲
                │ ITERATE HERE
                │ (md+latex+lean+sorry)
            Lean runs with sorry's here!
```

**The Iteration Loop:**
```
1. Write Lean skeleton with sorry
2. Lean checks types, structure
3. Error? Fix and retry
4. Fill one sorry with tactic
5. Lean compiles? Move to next sorry
6. Error? Try different tactic
7. All sorry's filled = FORMAL
```

**What Exists:**
- Numerical verification (stage 1)
- Symbolic verification (stage 2)
- Lean verifier (stage 4)
- Formalizer generates skeletons

**Gap:**
The iteration loop where we fill sorry's with tactics needs to be explicit and guided. When Lean fails on a tactic, need structured approach to try alternatives.

**Fix:**
1. Parse Lean errors into actionable categories
2. Suggest alternative tactics based on error type
3. Iterate filling sorry's one at a time
4. Use tactics library for guidance

---

### Gap 4: Lean Formalizer Incomplete

**The Problem:**
Formalizer generates skeletons with `sorry`, needs LLM for tactics.

**What Exists:**
- `formalizer.py`: 490 lines
- Pattern detection (induction, contradiction, etc.)
- Type/tactic mappings
- Iterative repair loop

**Gap:**
`sorry` placeholders need LLM-generated tactics.

**Fix:**
Integrate Claude/Gemini for tactic generation:
1. Formalizer generates skeleton
2. LLM fills tactics
3. Lean verifies
4. On error, LLM repairs
5. Iterate

---

### Gap 5: Memory Not Used in Solving

**The Problem:**
World model and technique tracker exist but may not be queried during solving.

**What Exists:**
- `world_model.get_context(problem)` returns similar problems, recommended techniques
- `technique_tracker.recommend(topics)` returns ranked techniques

**Gap:**
Does /solve actually call these before solving?

**Fix:**
At start of /solve:
```python
context = world_model.get_context(problem)
print(f"Similar problems: {context.similar_problems}")
print(f"Recommended techniques: {context.recommended_techniques}")
```

---

### Gap 6: Tactics Library Missing

**The Problem:**
LLM needs to know WHICH tactic to try when filling sorry's. Without indexed tactics, it's trial and error.

**What's Needed:**
```
TACTICS LIBRARY
├── Core Tactics (intro, apply, simp, ring, omega, linarith, etc.)
├── Mathlib Tactics (norm_num, positivity, polyrith, etc.)
├── Domain-Specific Patterns (number theory, algebra, analysis)
├── Proof Patterns ("prove for all n" → induction)
├── Tactic Sequences (common multi-step patterns)
└── Lemma Index (searchable by pattern)
```

**What Exists:**
- Formalizer has some tactic mappings
- 44K Herald proofs (could be indexed by technique)

**Gap:**
No structured, searchable tactics library. LLM guesses tactics instead of pattern-matching.

**Fix:**
1. Build indexed tactics library
2. Map problem patterns → suggested tactics
3. Track tactic success rates by context
4. Integrate with sorry-filling iteration

---

## Gap Summary Matrix

| Gap | Severity | Effort | Impact |
|-----|----------|--------|--------|
| Flywheel not active | High | Medium | Learning from attempts |
| Skills not auto-invoked | High | Low | Better reasoning flow |
| Semi-formal iteration weak | High | Medium | Core experimentation loop |
| Lean formalizer incomplete | Medium | High | Formal verification |
| Memory not used | Medium | Low | Context utilization |
| **Tactics library missing** | **High** | **High** | **Tactic selection for Lean** |

---

## Priority Order

### Day 1: Verify and Test
1. Test /solve command end-to-end
2. Verify world model persistence
3. Build semantic index if needed

### Week 1: Connect Components
1. Add memory queries to /solve start
2. Add skill routing logic
3. Test flywheel: solve → record → learn

### Week 2: Semi-Formal Iteration
1. Implement sorry-filling loop with Lean feedback
2. Parse Lean errors into actionable categories
3. Test on known problems with sorry→tactic→verify cycle

### Month 1: Tactics and Pipeline
1. Build indexed tactics library (core, mathlib, patterns)
2. Map problem patterns to suggested tactics
3. LLM tactic generation with library guidance
4. Full flywheel automation

### Month 2+: Scale
1. Index 44K Herald proofs by tactic patterns
2. Track tactic success rates by context
3. Multi-model verification (Claude + Gemini)
