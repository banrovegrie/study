# Autoformalization Implementation

## Overview

From ARCHITECTURE.md:

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

Autoformalization is the process of converting informal mathematical reasoning into formal Lean proofs. The key insight from RESEARCH.md:

> **The language already exists. The workflow needs tooling.**
> Lean's `sorry` IS the semi-formal language. It lets you write skeleton proofs that Lean type-checks.

---

## Current State

### What Exists
- `PutnamFormalizer` in `lean/formalizer.py`:
  - Pattern detection (induction, contradiction, cases)
  - Type mapping (natural language → Lean types)
  - Proof templates for common patterns
  - Basic repair loop (adds imports, replaces failed tactics with sorry)

### What's Missing
- **LLM-driven tactic suggestion** - Currently marked as TODO
- **Iterative sorry-filling loop** - No systematic approach
- **Error-guided repair** - Basic, needs Lean error → repair strategy mapping
- **Retrieval integration** - No lookup of similar proofs/lemmas

---

## Design: The Sorry-Filling Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SORRY-FILLING LOOP                               │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │   INFORMAL   │  "Prove by induction. Base case: trivial.         │
│  │   PROOF      │   Inductive step: use hypothesis twice."          │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  SKELETON    │  theorem foo : ∀ n, P n := by                     │
│  │  (sorry's)   │    induction n with n ih                          │
│  │              │    · sorry  -- base case                          │
│  │              │    · sorry  -- inductive step                     │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     ITERATION LOOP                            │   │
│  │                                                               │   │
│  │   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌────────┐ │   │
│  │   │ SELECT  │────▶│ SUGGEST │────▶│  TRY    │────▶│OBSERVE │ │   │
│  │   │ SORRY   │     │ TACTIC  │     │ TACTIC  │     │ RESULT │ │   │
│  │   └─────────┘     └─────────┘     └─────────┘     └───┬────┘ │   │
│  │        ▲                                              │      │   │
│  │        └──────────────────────────────────────────────┘      │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │   FORMAL     │  theorem foo : ∀ n, P n := by                     │
│  │   PROOF      │    induction n with n ih                          │
│  │  (no sorry)  │    · simp                                         │
│  │              │    · simp [ih]                                    │
│  └──────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Core Autoformalization Pipeline

**File**: `src/formalize/pipeline.py`

```python
"""
Autoformalization Pipeline: Informal → Semi-formal → Formal

The pipeline:
1. Parse informal proof into structure
2. Generate Lean skeleton with sorry's
3. Iteratively fill sorry's with LLM-suggested tactics
4. Verify and repair until complete
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import re
from datetime import datetime

class SorryStatus(Enum):
    UNFILLED = "unfilled"
    IN_PROGRESS = "in_progress"
    FILLED = "filled"
    STUCK = "stuck"

@dataclass
class Sorry:
    """A sorry placeholder in a proof."""
    id: str
    line_number: int
    goal_context: str  # The Lean goal at this point
    informal_hint: str  # Informal description of what's needed
    status: SorryStatus = SorryStatus.UNFILLED
    attempts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    filled_tactic: Optional[str] = None

@dataclass
class ProofSkeleton:
    """A Lean proof with sorry placeholders."""
    statement: str
    code: str
    sorries: List[Sorry]
    imports: List[str]

    @property
    def sorry_count(self) -> int:
        return len([s for s in self.sorries if s.status != SorryStatus.FILLED])

    @property
    def is_complete(self) -> bool:
        return self.sorry_count == 0

    def to_lean(self) -> str:
        """Generate complete Lean code."""
        imports_str = "\n".join(f"import {imp}" for imp in self.imports)
        return f"{imports_str}\n\n{self.code}"

@dataclass
class FormalizationResult:
    """Result of formalization attempt."""
    success: bool
    skeleton: ProofSkeleton
    iterations: int
    time_spent: float
    final_code: str
    errors: List[str]


class AutoFormalizationPipeline:
    """
    Converts informal proofs to formal Lean proofs.

    Uses LLM for:
    1. Skeleton generation
    2. Tactic suggestion
    3. Error repair
    """

    def __init__(self, lean_verifier, skills_library, retrieval_system=None):
        self.verifier = lean_verifier
        self.skills = skills_library
        self.retrieval = retrieval_system

        # Configuration
        self.max_iterations = 20
        self.max_attempts_per_sorry = 5

    def formalize(self, informal_proof: str, statement: str,
                  domain: str = None, tactic_suggester: Callable = None) -> FormalizationResult:
        """
        Main entry point: convert informal proof to formal Lean.

        Args:
            informal_proof: Natural language proof
            statement: The theorem statement
            domain: Mathematical domain (for skill lookup)
            tactic_suggester: Function(goal, context, hints) -> List[str]
                              If None, uses default skill-based suggestions

        Returns:
            FormalizationResult with final code and status
        """
        import time
        start_time = time.time()

        # Step 1: Generate skeleton
        skeleton = self._generate_skeleton(informal_proof, statement, domain)

        # Step 2: Initial verification (identify sorry positions)
        skeleton = self._identify_sorries(skeleton)

        # Step 3: Iterative sorry-filling
        iterations = 0
        while not skeleton.is_complete and iterations < self.max_iterations:
            iterations += 1

            # Select a sorry to fill
            sorry = self._select_sorry(skeleton)
            if sorry is None:
                break

            # Get tactic suggestions
            if tactic_suggester:
                suggestions = tactic_suggester(
                    sorry.goal_context,
                    self._get_context(skeleton, sorry),
                    sorry.informal_hint
                )
            else:
                suggestions = self._suggest_tactics(sorry, skeleton, domain)

            # Try suggestions
            filled = False
            for tactic in suggestions[:self.max_attempts_per_sorry]:
                sorry.attempts.append(tactic)
                result = self._try_tactic(skeleton, sorry, tactic)

                if result["success"]:
                    sorry.status = SorryStatus.FILLED
                    sorry.filled_tactic = tactic
                    skeleton.code = result["new_code"]
                    filled = True
                    break
                else:
                    sorry.errors.append(result.get("error", "Unknown error"))

            if not filled:
                sorry.status = SorryStatus.STUCK

            # Re-identify sorries (code may have changed)
            skeleton = self._identify_sorries(skeleton)

        # Final verification
        final_result = self.verifier.verify(skeleton.to_lean())

        return FormalizationResult(
            success=skeleton.is_complete and final_result["success"],
            skeleton=skeleton,
            iterations=iterations,
            time_spent=time.time() - start_time,
            final_code=skeleton.to_lean(),
            errors=[s.errors[-1] for s in skeleton.sorries if s.errors]
        )

    def _generate_skeleton(self, informal: str, statement: str, domain: str) -> ProofSkeleton:
        """
        Generate initial Lean skeleton from informal proof.

        This is where we translate structure:
        - "by induction" → induction tactic
        - "base case" → first goal
        - "WLOG" → wlog tactic
        - etc.
        """
        # Detect proof pattern
        pattern = self._detect_pattern(informal)

        # Generate skeleton based on pattern
        if pattern == "induction":
            code = self._skeleton_induction(statement, informal)
        elif pattern == "contradiction":
            code = self._skeleton_contradiction(statement, informal)
        elif pattern == "cases":
            code = self._skeleton_cases(statement, informal)
        elif pattern == "construction":
            code = self._skeleton_construction(statement, informal)
        else:
            code = self._skeleton_default(statement, informal)

        # Determine imports
        imports = self._suggest_imports(statement, informal, domain)

        return ProofSkeleton(
            statement=statement,
            code=code,
            sorries=[],  # Will be filled by _identify_sorries
            imports=imports
        )

    def _detect_pattern(self, informal: str) -> str:
        """Detect the main proof pattern."""
        informal_lower = informal.lower()

        patterns = [
            ("induction", ["by induction", "induct on", "base case", "inductive step"]),
            ("contradiction", ["contradiction", "assume.*false", "suppose.*not", "by_contra"]),
            ("cases", ["case 1", "case 2", "either.*or", "split into cases"]),
            ("construction", ["construct", "define", "let .* be", "consider the"]),
        ]

        for pattern_name, keywords in patterns:
            if any(kw in informal_lower for kw in keywords):
                return pattern_name

        return "direct"

    def _skeleton_induction(self, statement: str, informal: str) -> str:
        """Generate induction skeleton."""
        # Extract variable to induct on
        var_match = re.search(r'induct(?:ion)?\s+on\s+(\w+)', informal, re.IGNORECASE)
        var = var_match.group(1) if var_match else "n"

        return f"""theorem statement : {statement} := by
  induction {var} with {var} ih
  · -- Base case
    sorry
  · -- Inductive step
    sorry"""

    def _skeleton_contradiction(self, statement: str, informal: str) -> str:
        """Generate contradiction skeleton."""
        return f"""theorem statement : {statement} := by
  by_contra h
  push_neg at h
  -- Derive contradiction
  sorry"""

    def _skeleton_cases(self, statement: str, informal: str) -> str:
        """Generate case split skeleton."""
        return f"""theorem statement : {statement} := by
  rcases _ with _ | _
  · -- Case 1
    sorry
  · -- Case 2
    sorry"""

    def _skeleton_construction(self, statement: str, informal: str) -> str:
        """Generate construction skeleton."""
        return f"""theorem statement : {statement} := by
  use _  -- Construct witness
  constructor
  · -- Property 1
    sorry
  · -- Property 2
    sorry"""

    def _skeleton_default(self, statement: str, informal: str) -> str:
        """Generate default (direct proof) skeleton."""
        return f"""theorem statement : {statement} := by
  sorry"""

    def _suggest_imports(self, statement: str, informal: str, domain: str) -> List[str]:
        """Suggest necessary imports."""
        imports = ["Mathlib.Tactic"]  # Always need basic tactics

        # Domain-specific imports
        domain_imports = {
            "number_theory": ["Mathlib.NumberTheory.Basic", "Mathlib.Data.Nat.Prime"],
            "algebra": ["Mathlib.Algebra.Ring.Basic", "Mathlib.Algebra.Field.Basic"],
            "analysis": ["Mathlib.Analysis.SpecialFunctions.Basic", "Mathlib.Topology.Basic"],
            "combinatorics": ["Mathlib.Combinatorics.Basic", "Mathlib.Data.Finset.Basic"],
        }

        if domain and domain in domain_imports:
            imports.extend(domain_imports[domain])

        # Keyword-based imports
        keywords_imports = {
            "sum": "Mathlib.Algebra.BigOperators.Basic",
            "product": "Mathlib.Algebra.BigOperators.Basic",
            "prime": "Mathlib.Data.Nat.Prime",
            "sqrt": "Mathlib.Analysis.SpecialFunctions.Sqrt",
            "continuous": "Mathlib.Topology.Basic",
        }

        combined = (statement + " " + informal).lower()
        for keyword, imp in keywords_imports.items():
            if keyword in combined and imp not in imports:
                imports.append(imp)

        return imports

    def _identify_sorries(self, skeleton: ProofSkeleton) -> ProofSkeleton:
        """
        Identify sorry positions and their goal contexts.

        This requires running Lean to get goal states.
        """
        # Find sorry positions in code
        sorry_pattern = re.compile(r'sorry(\s*--.*)?')
        sorries = []

        for i, match in enumerate(sorry_pattern.finditer(skeleton.code)):
            line_number = skeleton.code[:match.start()].count('\n') + 1
            comment = match.group(1).strip() if match.group(1) else ""

            sorries.append(Sorry(
                id=f"sorry_{i}",
                line_number=line_number,
                goal_context="",  # Would be filled by Lean interaction
                informal_hint=comment.replace("--", "").strip(),
                status=SorryStatus.UNFILLED
            ))

        skeleton.sorries = sorries
        return skeleton

    def _select_sorry(self, skeleton: ProofSkeleton) -> Optional[Sorry]:
        """
        Select which sorry to fill next.

        Strategy: Fill in order (typically top-down, leaves first)
        """
        for sorry in skeleton.sorries:
            if sorry.status == SorryStatus.UNFILLED:
                return sorry
        return None

    def _suggest_tactics(self, sorry: Sorry, skeleton: ProofSkeleton,
                         domain: str) -> List[str]:
        """
        Suggest tactics for a sorry.

        Uses:
        1. Skills library
        2. Goal-based heuristics
        3. Domain-specific patterns
        """
        suggestions = []

        # Get suggestions from skills library
        if self.skills:
            skill_results = self.skills.search(
                goal=sorry.goal_context or sorry.informal_hint,
                domain=domain,
                k=5
            )
            for skill, score in skill_results:
                suggestions.extend(skill.tactics_sequence)

        # Goal-based heuristics
        goal = sorry.goal_context or ""
        hint = sorry.informal_hint.lower()

        if "base case" in hint or "trivial" in hint:
            suggestions.extend(["simp", "rfl", "norm_num", "decide"])

        if "inductive" in hint:
            suggestions.extend(["simp [ih]", "exact ih _", "apply ih"])

        if "contradiction" in hint:
            suggestions.extend(["contradiction", "exact absurd _ _", "linarith"])

        if "=" in goal:
            suggestions.extend(["ring", "field_simp", "rfl"])

        if "<" in goal or ">" in goal or "≤" in goal or "≥" in goal:
            suggestions.extend(["linarith", "nlinarith", "omega", "positivity"])

        # Default fallbacks
        suggestions.extend(["simp", "ring", "linarith", "omega", "exact?", "apply?"])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique

    def _try_tactic(self, skeleton: ProofSkeleton, sorry: Sorry,
                    tactic: str) -> dict:
        """
        Try replacing a sorry with a tactic.

        Returns:
        - success: bool
        - new_code: str (if success)
        - error: str (if failure)
        """
        # Replace the sorry with the tactic
        lines = skeleton.code.split('\n')
        old_line = lines[sorry.line_number - 1]
        new_line = re.sub(r'sorry(\s*--.*)?', tactic, old_line)
        lines[sorry.line_number - 1] = new_line
        new_code = '\n'.join(lines)

        # Verify
        result = self.verifier.verify(
            ProofSkeleton(
                statement=skeleton.statement,
                code=new_code,
                sorries=[],
                imports=skeleton.imports
            ).to_lean()
        )

        if result["success"]:
            return {"success": True, "new_code": new_code}
        else:
            # Check if error is at this line or elsewhere
            errors = result.get("errors", [])
            error_at_line = any(
                e.get("line") == sorry.line_number
                for e in errors if isinstance(e, dict)
            )

            if error_at_line:
                return {
                    "success": False,
                    "error": errors[0]["message"] if errors else "Tactic failed"
                }
            else:
                # Error elsewhere - tactic might have worked but revealed other issues
                return {"success": True, "new_code": new_code}

    def _get_context(self, skeleton: ProofSkeleton, sorry: Sorry) -> dict:
        """Get context for tactic suggestion."""
        return {
            "statement": skeleton.statement,
            "code_before": "\n".join(skeleton.code.split('\n')[:sorry.line_number]),
            "informal_hint": sorry.informal_hint,
            "previous_attempts": sorry.attempts,
            "previous_errors": sorry.errors
        }

    # === Repair Strategies ===

    def repair_from_error(self, skeleton: ProofSkeleton, error: dict) -> ProofSkeleton:
        """
        Apply repair strategy based on error type.
        """
        error_type = error.get("type", "unknown")

        if error_type == "unknown_identifier":
            return self._repair_unknown_identifier(skeleton, error)
        elif error_type == "type_mismatch":
            return self._repair_type_mismatch(skeleton, error)
        elif error_type == "tactic_failed":
            # Already handled in sorry-filling
            pass

        return skeleton

    def _repair_unknown_identifier(self, skeleton: ProofSkeleton, error: dict) -> ProofSkeleton:
        """Try to fix unknown identifier errors."""
        message = error.get("message", "")

        # Extract the unknown name
        name_match = re.search(r"'([^']+)'", message)
        if not name_match:
            return skeleton

        unknown_name = name_match.group(1)

        # Try to find correct import
        # This would ideally use Mathlib search
        possible_imports = {
            "Nat": "Mathlib.Data.Nat.Basic",
            "Int": "Mathlib.Data.Int.Basic",
            "Real": "Mathlib.Data.Real.Basic",
            "Finset": "Mathlib.Data.Finset.Basic",
        }

        for prefix, imp in possible_imports.items():
            if unknown_name.startswith(prefix) and imp not in skeleton.imports:
                skeleton.imports.append(imp)
                return skeleton

        return skeleton

    def _repair_type_mismatch(self, skeleton: ProofSkeleton, error: dict) -> ProofSkeleton:
        """Try to fix type mismatch errors."""
        # This would need to insert type coercions
        # For now, just return unchanged
        return skeleton
```

---

### LLM Tactic Suggester

**File**: `src/formalize/tactic_suggester.py`

```python
"""
LLM-based tactic suggestion for sorry-filling.

This is where Claude's reasoning power is leveraged.
"""

from typing import List, Callable

def create_tactic_suggester(llm_call: Callable[[str], str]) -> Callable:
    """
    Create a tactic suggester that uses an LLM.

    Args:
        llm_call: Function that takes a prompt and returns LLM response

    Returns:
        A function(goal, context, hint) -> List[str] of tactic suggestions
    """

    def suggest_tactics(goal: str, context: dict, hint: str) -> List[str]:
        prompt = f"""You are helping fill a `sorry` in a Lean 4 proof.

## Current Goal
{goal if goal else "Unknown (check context)"}

## Context
Statement: {context.get('statement', 'Unknown')}

Code so far:
```lean
{context.get('code_before', '')}
```

## Hint from informal proof
{hint}

## Previous attempts that failed
{context.get('previous_attempts', [])}

## Previous errors
{context.get('previous_errors', [])}

## Your task
Suggest 5 Lean 4 tactics to try for this goal, in order of likelihood to succeed.

Format: Return ONLY the tactics, one per line, no explanation.
Example:
simp
ring
linarith
exact ih n
apply Nat.add_comm
"""

        response = llm_call(prompt)

        # Parse response into list of tactics
        tactics = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Remove markdown code formatting if present
                line = line.strip('`')
                if line:
                    tactics.append(line)

        return tactics[:10]  # Limit to 10

    return suggest_tactics


def create_skeleton_generator(llm_call: Callable[[str], str]) -> Callable:
    """
    Create a skeleton generator that uses an LLM.
    """

    def generate_skeleton(statement: str, informal: str, domain: str) -> str:
        prompt = f"""Convert this informal proof to a Lean 4 proof skeleton with `sorry` placeholders.

## Theorem Statement
{statement}

## Informal Proof
{informal}

## Domain
{domain or "general mathematics"}

## Instructions
1. Write valid Lean 4 syntax
2. Use `sorry` for parts that need filling
3. Include comments describing what each sorry should prove
4. Use appropriate tactics for the proof structure
5. Include necessary imports at the top

## Output
Return ONLY the Lean code, nothing else.
"""

        return llm_call(prompt)

    return generate_skeleton
```

---

### Integration with Claude Code

**File**: `src/formalize/__init__.py`

```python
from .pipeline import AutoFormalizationPipeline, FormalizationResult, ProofSkeleton
from .tactic_suggester import create_tactic_suggester, create_skeleton_generator

__all__ = [
    "AutoFormalizationPipeline",
    "FormalizationResult",
    "ProofSkeleton",
    "create_tactic_suggester",
    "create_skeleton_generator"
]
```

---

### Tool Interface

**File**: `src/tools.py` (add)

```python
def tool_formalize(statement: str, informal_proof: str,
                   domain: str = None, max_iterations: int = 20) -> dict:
    """
    Formalize an informal proof to Lean.

    Args:
        statement: The theorem statement (in mathematical notation)
        informal_proof: Natural language proof
        domain: Mathematical domain (number_theory, algebra, etc.)
        max_iterations: Max sorry-filling iterations

    Returns:
        - success: bool
        - code: str (final Lean code)
        - sorry_count: int (remaining sorries)
        - iterations: int
        - time_spent: float
        - errors: list
    """
    from src.formalize import AutoFormalizationPipeline, create_tactic_suggester
    from src.verify.lean.verifier import LeanVerifier
    from src.skills import SkillsLibrary

    verifier = LeanVerifier(LEAN_PROJECT_PATH)
    skills = SkillsLibrary(Path(DATA_DIR) / "skills")

    # Create tactic suggester (uses Claude via the current conversation)
    # In practice, this would call back to Claude
    def claude_call(prompt: str) -> str:
        # This is a placeholder - in reality, Claude would generate this
        # The actual implementation depends on how tools callback
        return ""

    tactic_suggester = create_tactic_suggester(claude_call)

    pipeline = AutoFormalizationPipeline(
        lean_verifier=verifier,
        skills_library=skills
    )
    pipeline.max_iterations = max_iterations

    result = pipeline.formalize(
        informal_proof=informal_proof,
        statement=statement,
        domain=domain,
        tactic_suggester=tactic_suggester
    )

    return {
        "success": result.success,
        "code": result.final_code,
        "sorry_count": result.skeleton.sorry_count,
        "iterations": result.iterations,
        "time_spent": result.time_spent,
        "errors": result.errors
    }
```

---

## Workflow in /solve

### Phase 1: Generate Skeleton

```markdown
After solving informally, generate Lean skeleton:

1. State the theorem in Lean syntax
2. Identify proof structure (induction, contradiction, etc.)
3. Generate skeleton with sorry's for each step

Example:
```lean
theorem sum_first_n (n : ℕ) : 2 * (∑ i in Finset.range (n + 1), i) = n * (n + 1) := by
  induction n with n ih
  · -- Base case: n = 0
    sorry
  · -- Inductive step
    sorry
```
```

### Phase 2: Fill Sorries

```markdown
For each sorry, in order:

1. **Understand the goal**: What does this sorry need to prove?
2. **Retrieve relevant skills**: Search skills library for matching tactics
3. **Try tactics**: Apply suggestions, observe errors
4. **Iterate**: If stuck, try alternative approaches or ask for help

Tips:
- `simp` and `ring` solve many goals
- Type errors usually mean wrong coercion
- `exact?` and `apply?` help find the right lemma
```

### Phase 3: Verify and Repair

```markdown
After filling all sorries:

1. Run full verification
2. If errors remain, apply repair strategies:
   - Missing import → add import
   - Type mismatch → add coercion
   - Unknown identifier → search Mathlib

3. Iterate until clean compile or stuck
```

---

## Example Session

```
User: Formalize this proof that sum of first n naturals is n(n+1)/2

Informal: By induction on n. Base case n=0: sum is 0 and 0*1/2=0.
Inductive step: Assume true for n. For n+1, sum is (sum for n) + (n+1)
= n(n+1)/2 + (n+1) = (n+1)(n+2)/2 by algebra.

Claude: I'll formalize this step by step.

**Step 1: Skeleton**
```lean
import Mathlib.Tactic
import Mathlib.Algebra.BigOperators.Basic

theorem sum_first_n (n : ℕ) : 2 * (∑ i in Finset.range (n + 1), i) = n * (n + 1) := by
  induction n with n ih
  · -- Base case: n = 0
    sorry
  · -- Inductive step: assume for n, prove for n + 1
    sorry
```

**Step 2: Fill base case**
Goal: `2 * (∑ i in Finset.range 1, i) = 0 * 1`
Trying: `simp` → Success!

**Step 3: Fill inductive step**
Goal: `2 * (∑ i in Finset.range (n + 2), i) = (n + 1) * (n + 2)`
Trying: `simp [Finset.sum_range_succ, ih]` → Partial...
Trying: `simp [Finset.sum_range_succ, ih]; ring` → Success!

**Final Code:**
```lean
import Mathlib.Tactic
import Mathlib.Algebra.BigOperators.Basic

theorem sum_first_n (n : ℕ) : 2 * (∑ i in Finset.range (n + 1), i) = n * (n + 1) := by
  induction n with n ih
  · simp
  · simp [Finset.sum_range_succ, ih]; ring
```

Verification: ✓ Compiles successfully with no sorry's!
```

---

## Summary

| Component | File | Purpose | Priority |
|-----------|------|---------|----------|
| ProofSkeleton | `pipeline.py` | Represent sorry-filled proofs | P2 |
| AutoFormalizationPipeline | `pipeline.py` | Main formalization loop | P2 |
| TacticSuggester | `tactic_suggester.py` | LLM-based suggestions | P2 |
| SkeletonGenerator | `tactic_suggester.py` | LLM-based skeleton creation | P2 |
| tool_formalize | `tools.py` | Claude Code interface | P2 |

Autoformalization is the bridge from informal reasoning to machine-verified proofs. The sorry-filling loop makes it iterative and debuggable.
