# Recommendations: Activating the System

## Guiding Principles

### Principle 1: Claude Code IS the Orchestrator
Don't build another orchestration layer. Enhance the skills, commands, and tools that Claude uses.

### Principle 2: Semi-Formal is the Sweet Spot

**Semi-formal = md + latex + lean (with sorry's) + citations**

```
Informal ──► SEMI-FORMAL ◄──► Formal
                  ▲
              iterate here
          (md+latex+lean+sorry)
```
Most work happens in semi-formal: write Lean skeleton with sorry's → Lean checks types → fill sorry's with tactics → iterate until formal.

### Principle 3: Flywheel = Learning
```
Solve → Learn → Solve Better
```
Every attempt should feed back into the system.

---

## Immediate Actions (Day 1)

### Action 1: Test End-to-End
Before changing anything, test what exists:

```bash
# Test /solve workflow manually
# Load a simple problem
python -m src.tools archive 2023 A1

# Get approaches
python -m src.tools approaches "number_theory"

# Verify a known formula
python -m src.tools verify "n*(n+1)/2" --compute "sum(range(1,n+1))"
```

**Question to answer:** Does the current system actually work?

### Action 2: Build Semantic Index
```python
from src.library.semantic import SemanticRetriever
retriever = SemanticRetriever()
retriever.build_index()  # ~30 min one-time
```

### Action 3: Verify Memory Persistence
```python
from src.memory import get_world_model
world = get_world_model()
# Check: data/memory/*.json exists and updates
```

---

## Week 1 Actions

### Action 4: Add Memory to /solve

Update `/solve` command to query memory at start:

```markdown
## 0. Context (Before solving)

```python
from src.memory import get_world_model

world = get_world_model()
context = world.get_context(problem)

if context.similar_problems:
    print("Similar problems solved before:")
    for pid, summary in context.similar_problems:
        print(f"  {pid}: {summary}")

if context.recommended_techniques:
    print("Recommended techniques:")
    for technique, rate in context.recommended_techniques[:5]:
        print(f"  {technique}: {rate:.0%} success rate")
```
```

### Action 5: Add Skill Routing

Add to `/solve` command:

```markdown
## 1b. Skill Selection (Auto-invoke based on problem)

Based on problem type and topics:
- If type = "prove": Use hypothesis skill first, critique skill after
- If type = "find" or "construct": Use explore skill, consider evolution skill
- If topics include "number_theory": Always explore small cases
- If 3+ approaches tried: Consider mcts skill as fallback
```

### Action 6: Close Flywheel Loop

Add to end of `/solve`:

```markdown
## 10. Learn (Record for flywheel)

```python
from src.memory import get_world_model

world = get_world_model()
world.record_attempt(
    problem_id=problem.id,
    techniques_tried=["..."],
    success=True/False,
    confidence=0.85,
    notes="Key insight: ..."
)
world.save()  # Persist for next session
```
```

---

## Week 2-4 Actions

### Action 7: Strengthen Semi-Formal Iteration (Lean with Sorry's)

The key iteration is filling sorry's one at a time:

```python
def semiformal_iterate(problem, solution, max_attempts=5):
    """Iterate in semi-formal space (Lean with sorry's)."""

    # 1. Generate Lean skeleton with sorry's
    skeleton = formalizer.generate_skeleton(problem, solution)

    # 2. Verify skeleton compiles (with sorry's)
    result = lean_verify(skeleton)
    if not result.success:
        # Fix structural/type issues first
        skeleton = repair_structure(skeleton, result.error)

    # 3. Fill sorry's one at a time
    for sorry_loc in find_sorries(skeleton):
        for attempt in range(max_attempts):
            # Get tactic suggestion from library
            tactic = suggest_tactic(sorry_loc.goal, sorry_loc.context)

            # Try the tactic
            new_code = replace_sorry(skeleton, sorry_loc, tactic)
            result = lean_verify(new_code)

            if result.success:
                skeleton = new_code
                break  # Move to next sorry
            else:
                # Try different tactic
                continue

    # 4. Check if all sorry's filled
    if count_sorries(skeleton) == 0:
        return skeleton, "formal_verified"
    else:
        return skeleton, "partial_with_sorries"
```

### Action 8: Add Lean Error Feedback

When Lean verification fails:

```python
def lean_with_feedback(code, max_repairs=3):
    for repair in range(max_repairs):
        result = lean_verify(code)
        if result.success:
            return code, "verified"

        # Parse error
        error_type = classify_lean_error(result.error)

        # Feed back for repair
        if error_type == "tactic_failed":
            code = generate_alternative_tactic(code, result.error)
        elif error_type == "type_mismatch":
            code = fix_type_issue(code, result.error)
        # ... etc

    return None, "formalization_failed"
```

### Action 9: Multi-Model Verification

Add Gemini verification step:

```python
def multi_model_verify(solution, problem):
    # Claude's assessment (implicit - we're in Claude Code)
    claude_conf = self_evaluate(solution)

    # Gemini's assessment
    gemini_resp = tool_gemini(
        problem.statement,
        question=f"Is this solution correct?\n{solution}",
    )
    gemini_conf = parse_confidence(gemini_resp)

    # Require agreement
    if claude_conf > 0.8 and gemini_conf > 0.8:
        return "high_confidence"
    elif claude_conf > 0.5 or gemini_conf > 0.5:
        return "medium_confidence"
    else:
        return "low_confidence"
```

---

## Month 2+ Actions

### Action 10: Build Tactics Library (CRITICAL)

Before LLM tactic generation works well, need indexed tactics:

```python
# Tactics Library Structure
tactics_library = {
    "core": ["intro", "apply", "exact", "rfl", "rw", "simp", "ring", "omega"],
    "mathlib": ["norm_num", "positivity", "polyrith", "field_simp"],
    "domain": {
        "number_theory": ["mod_cast", "push_cast", "zify"],
        "algebra": ["linear_combination", "polyrith"],
        "analysis": ["continuity", "filter_upwards"],
    },
    "patterns": {
        "prove_forall": ["induction", "strong_induction"],
        "prove_exists": ["use", "constructor"],
        "prove_equality": ["calc", "rw", "simp"],
        "prove_inequality": ["linarith", "nlinarith", "positivity"],
        "by_contradiction": ["by_contra", "exfalso"],
    }
}

def suggest_tactic(goal, context):
    """Suggest tactics based on goal pattern."""
    if "∀" in goal:
        return tactics_library["patterns"]["prove_forall"]
    if "∃" in goal:
        return tactics_library["patterns"]["prove_exists"]
    # ... pattern matching
```

### Action 11: LLM Tactic Generation

Integrate LLM into formalizer with tactics library:

```python
def formalize_with_llm(problem, solution):
    # Generate skeleton
    skeleton = formalizer.generate_skeleton(problem, solution)

    # Fill tactics with LLM
    for sorry_loc in find_sorries(skeleton):
        tactic = llm_generate_tactic(
            goal=sorry_loc.goal,
            context=sorry_loc.context,
            available_lemmas=sorry_loc.available
        )
        skeleton = replace_sorry(skeleton, sorry_loc, tactic)

    # Verify and repair
    return lean_with_feedback(skeleton)
```

### Action 11: Active Learning

Prioritize which solutions to formalize:

```python
def should_formalize(solution, confidence):
    # High value: novel proof patterns
    if solution.uses_novel_technique:
        return True

    # High value: high confidence solutions
    if confidence > 0.9:
        return True

    # Low value: already have similar formal proof
    if library_has_similar(solution):
        return False

    return False
```

---

## Success Metrics

| Metric | Current | Week 4 | Month 3 |
|--------|---------|--------|---------|
| E2E workflow tested | Unknown | Yes | Yes |
| Memory used in solving | No | Yes | Yes |
| Skills auto-invoked | No | Partial | Yes |
| Flywheel active | No | Basic | Full |
| Multi-model verification | No | Yes | Yes |
| LLM tactic generation | No | No | Partial |

---

## What NOT to Do

1. **Don't build another orchestrator** - Claude Code IS the orchestrator
2. **Don't over-formalize early** - semi-formal iteration first
3. **Don't ignore failures** - every failure should update technique tracker
4. **Don't skip numerical verification** - it's the cheapest sanity check
5. **Don't use MCTS as default** - it's a fallback for when exploration fails
