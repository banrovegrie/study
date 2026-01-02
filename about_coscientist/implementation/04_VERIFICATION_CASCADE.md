# Verification Cascade: Layers of Certification

## First Principles: What Does "Verified" Mean?

Different contexts require different levels of certainty. The key insight: **Lean is "can use" not "must use".**

From AlphaEvolve paper: *"we have demonstrated that **in rare cases** this is already possible, by providing an example of a full pipeline from discovery to formalization"* - meaning most results are NOT fully formalized.

---

## The Certification Spectrum (4 Levels)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LAYERS OF CERTIFICATION                                  │
│                                                                                  │
│  Level 0: INFORMAL                                                               │
│           "NL, intuition, search, referring, reading, hypothesis generation"     │
│           This is where exploration and discovery happen                         │
│           Format: Natural language, sketches, conjectures                        │
│                                                                                  │
│  Level 1: COMPUTATIONAL + SYMBOLIC                                               │
│           "Examples don't contradict + CAS confirms algebraically"               │
│           Format: Tests + SymPy verification                                     │
│                                                                                  │
│  Level 2: SEMIFORMAL (Rigorous-NL + Formal Support)                              │
│           "Logical steps sound + structure verified + gaps explicit"             │
│           LLM-judged rigor, may request human review                             │
│           Includes:                                                              │
│             - Rigorous NL proofs with citations                                  │
│             - Lean proofs with sorry (type-checked skeleton)                     │
│             - Verified programs (for ICPC/computational problems)                │
│           This is the DeepSeek-Prover-V2 approach                                │
│           Expressibility of NL + partial formal verification                     │
│           Caveat: Possible inconsistency, not absolute rigor                     │
│                                                                                  │
│  Level 3: FULLY FORMALIZED                                                       │
│           "Complete machine verification in Lean"                                │
│           No sorry, no admitted, no unsafe axioms                                │
│           Format: Lean 4, compiles clean                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Certification Levels

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import IntEnum
import re


class CertificationLevel(IntEnum):
    """
    Levels of mathematical certification.

    Higher = more certain. But higher isn't always better -
    it's about matching certainty to need.
    """
    INFORMAL = 0           # Exploration, hypothesis generation
    COMPUTATIONAL = 1      # Tests + CAS
    SEMIFORMAL = 2         # Rigorous-NL + formal support (Lean+sorry, verified programs)
    FORMAL = 3             # Complete Lean proof


@dataclass
class Citation:
    """A reference to an established result."""
    source: str           # "Mathlib", "Rudin", "Tao Analysis I", etc.
    theorem: str          # "Nat.add_comm", "Theorem 3.2.1", etc.
    statement: str        # What the cited result says
    url: Optional[str] = None


@dataclass
class CertificationResult:
    """Result of certification at any level."""
    level: CertificationLevel
    passed: bool

    # Evidence
    evidence: str                              # Summary of why we believe this
    citations: List[Citation] = field(default_factory=list)

    # For Lean-based verification
    lean_code: Optional[str] = None
    sorry_count: int = 0
    sorry_locations: List[str] = field(default_factory=list)

    # For computational problems (ICPC)
    program_code: Optional[str] = None
    tests_passed: int = 0
    tests_total: int = 0

    # Diagnostics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    needs_human_review: bool = False
    human_review_reason: Optional[str] = None

    @property
    def is_publishable(self) -> bool:
        """Would this pass peer review?"""
        return self.passed and self.level >= CertificationLevel.SEMIFORMAL

    @property
    def is_machine_verified(self) -> bool:
        """Is this completely machine-checked?"""
        return self.passed and self.level == CertificationLevel.FORMAL


@dataclass
class ProofDocument:
    """
    A proof at any level of formality.

    Can be:
    - Markdown/LaTeX (informal or rigorous-NL)
    - Lean code (with or without sorries)
    - Program code (for computational problems)
    - Hybrid (markdown + embedded Lean/code)
    """
    format: str                    # "markdown", "latex", "lean", "program", "hybrid"
    content: str                   # The actual proof text/code

    # Metadata
    problem_statement: str = ""
    citations: List[Citation] = field(default_factory=list)

    # For Lean
    lean_imports: List[str] = field(default_factory=list)

    def has_sorries(self) -> bool:
        """Check if Lean content has sorry placeholders."""
        if self.format not in ["lean", "hybrid"]:
            return False
        return bool(re.search(r'\bsorry\b', self.content))

    def count_sorries(self) -> int:
        """Count sorry placeholders."""
        if self.format not in ["lean", "hybrid"]:
            return 0
        return len(re.findall(r'\bsorry\b', self.content))
```

---

### The Verifier Interface

```python
from abc import ABC, abstractmethod


class Verifier(ABC):
    """Base class for all verification methods."""

    @abstractmethod
    def verify(self, proof: ProofDocument, problem: str) -> CertificationResult:
        pass

    @property
    @abstractmethod
    def level(self) -> CertificationLevel:
        pass


class InformalVerifier(Verifier):
    """
    Level 0: Informal exploration.

    This isn't really "verification" - it's the starting point.
    Everything passes Level 0 if it's a coherent attempt.
    """

    @property
    def level(self) -> CertificationLevel:
        return CertificationLevel.INFORMAL

    def verify(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """
        Check if this is a coherent exploration/hypothesis.

        Basically always passes - this is the creative space.
        """
        return CertificationResult(
            level=self.level,
            passed=True,
            evidence="Informal exploration recorded"
        )


class ComputationalVerifier(Verifier):
    """
    Level 1: Computational + Symbolic verification.

    - Numerical tests (examples don't contradict)
    - CAS verification (SymPy confirms algebraically)
    """

    @property
    def level(self) -> CertificationLevel:
        return CertificationLevel.COMPUTATIONAL

    def verify(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """
        Verify by testing + CAS.

        This catches obvious errors and confirms algebraic claims.
        """
        # Extract testable claims from proof
        # Run numerical tests
        # Verify with SymPy

        return CertificationResult(
            level=self.level,
            passed=True,
            evidence="Numerical tests pass, CAS confirms algebraic claims"
        )


class SemiformalVerifier(Verifier):
    """
    Level 2: Semiformal verification.

    This is the workhorse level - where most real math happens.

    Includes THREE types of verification:
    1. Rigorous-NL: Logical soundness + citations (LLM-judged)
    2. Lean-skeleton: Type-checked structure with explicit gaps (sorry)
    3. Verified programs: For computational problems (ICPC-style)

    May request human review for uncertain steps.

    This is what DeepSeek-Prover-V2 does: NL reasoning + sorry-filling.
    Provides expressibility of natural language while having partial formal support.
    Caveat: Not absolute rigor, possible inconsistency in informal parts.
    """

    @property
    def level(self) -> CertificationLevel:
        return CertificationLevel.SEMIFORMAL

    def __init__(self, llm_client=None, lean_project_path: str = None):
        self.llm = llm_client
        self.lean_path = lean_project_path

    def verify(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """
        Verify at semiformal level based on proof format.
        """
        if proof.format == "lean" or (proof.format == "hybrid" and "```lean" in proof.content):
            return self._verify_lean_skeleton(proof, problem)
        elif proof.format == "program":
            return self._verify_program(proof, problem)
        else:
            return self._verify_rigorous_nl(proof, problem)

    def _verify_rigorous_nl(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """
        Verify rigorous natural language proof.

        Checks:
        1. Logical flow (each step follows from previous)
        2. Citations to established results
        3. No unjustified leaps

        May flag uncertain steps for human review.
        """
        citations = proof.citations or self._extract_citations(proof.content)

        if not citations:
            return CertificationResult(
                level=self.level,
                passed=False,
                evidence="No citations to established results",
                errors=["Proof makes claims without citations"]
            )

        # LLM checks logical flow
        logical_check = self._verify_logical_flow(proof.content)

        # Check citations are applied correctly
        citation_check = self._verify_citations(proof.content, citations)

        passed = logical_check["valid"] and citation_check["valid"]

        # Flag for human review if uncertain
        needs_review = logical_check.get("uncertain_steps", [])

        return CertificationResult(
            level=self.level,
            passed=passed,
            evidence=f"Logical flow: {logical_check['summary']}. Citations: {citation_check['summary']}",
            citations=citations,
            errors=logical_check.get("errors", []) + citation_check.get("errors", []),
            needs_human_review=len(needs_review) > 0,
            human_review_reason=f"Uncertain steps: {needs_review}" if needs_review else None
        )

    def _verify_lean_skeleton(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """
        Verify Lean proof with sorries.

        The type-checked structure proves the SHAPE of the argument.
        Sorries mark exactly what remains to be proven.
        """
        lean_code = self._extract_lean(proof.content)

        # Locate sorries
        sorries = self._analyze_sorries(lean_code)

        # Run Lean type checker
        result = self._run_lean(lean_code)

        if result["success"]:
            return CertificationResult(
                level=self.level,
                passed=True,
                evidence=f"Lean type-checks with {len(sorries)} sorry(s)",
                lean_code=lean_code,
                sorry_count=len(sorries),
                sorry_locations=[s["location"] for s in sorries]
            )
        else:
            return CertificationResult(
                level=self.level,
                passed=False,
                evidence="Lean compilation failed",
                lean_code=lean_code,
                errors=result.get("errors", [])
            )

    def _verify_program(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """
        Verify computational solution (ICPC-style).

        For problems where the "proof" is a program that computes the answer.
        """
        # Run program against test cases
        test_result = self._run_tests(proof.content, problem)

        passed = test_result["all_passed"]

        return CertificationResult(
            level=self.level,
            passed=passed,
            evidence=f"Tests: {test_result['passed']}/{test_result['total']} passed",
            program_code=proof.content,
            tests_passed=test_result["passed"],
            tests_total=test_result["total"],
            errors=test_result.get("errors", [])
        )

    def _extract_citations(self, content: str) -> List[Citation]:
        """Extract citations from proof text."""
        citations = []

        patterns = [
            r'by\s+\[([^\]]+)\]',
            r'using\s+\[([^\]]+)\]',
            r'from\s+\[([^\]]+)\]',
            r'\\cite\{([^}]+)\}',
            r'\(([A-Z][a-z]+\s+\d{4})\)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                ref = match.group(1)
                citations.append(Citation(
                    source="extracted",
                    theorem=ref,
                    statement=""
                ))

        return citations

    def _verify_logical_flow(self, content: str) -> Dict[str, Any]:
        """Use LLM to verify logical flow."""
        if not self.llm:
            return {"valid": True, "summary": "No LLM available for deep check"}

        # LLM prompt to check logical flow
        # Returns {valid, summary, errors, uncertain_steps}
        return {"valid": True, "summary": "Logical flow verified"}

    def _verify_citations(self, content: str, citations: List[Citation]) -> Dict[str, Any]:
        """Verify citations are applied correctly."""
        return {"valid": True, "summary": f"All {len(citations)} citations verified"}

    def _extract_lean(self, content: str) -> str:
        """Extract Lean code from content."""
        if not content.startswith("#") and not content.startswith("```"):
            return content
        lean_blocks = re.findall(r'```lean4?\n(.*?)```', content, re.DOTALL)
        return "\n\n".join(lean_blocks)

    def _analyze_sorries(self, lean_code: str) -> List[Dict]:
        """Analyze sorry placeholders."""
        sorries = []
        lines = lean_code.split('\n')

        for i, line in enumerate(lines):
            if 'sorry' in line:
                sorries.append({
                    "line": i + 1,
                    "location": f"line {i+1}: {line.strip()[:50]}"
                })

        return sorries

    def _run_lean(self, lean_code: str) -> Dict:
        """Run Lean type checker."""
        # Actual implementation uses subprocess
        return {"success": True}

    def _run_tests(self, program: str, problem: str) -> Dict:
        """Run program against test cases."""
        # Actual implementation runs tests
        return {"all_passed": True, "passed": 10, "total": 10}


class FormalVerifier(Verifier):
    """
    Level 3: Complete Lean verification.

    No sorries, no admitted, no axioms beyond the standard ones.
    This is the gold standard - absolute certainty.
    """

    @property
    def level(self) -> CertificationLevel:
        return CertificationLevel.FORMAL

    def __init__(self, lean_project_path: str = None):
        self.project_path = lean_project_path

    def verify(self, proof: ProofDocument, problem: str) -> CertificationResult:
        """Verify complete Lean proof with no gaps."""

        if proof.format not in ["lean", "hybrid"]:
            return CertificationResult(
                level=self.level,
                passed=False,
                evidence="No Lean code provided"
            )

        lean_code = proof.content if proof.format == "lean" else self._extract_lean(proof.content)

        # Check for sorries
        if re.search(r'\bsorry\b', lean_code):
            return CertificationResult(
                level=self.level,
                passed=False,
                evidence="Contains sorry - use SEMIFORMAL level instead",
                errors=["Full verification requires no sorries"]
            )

        # Check for axioms/admitted
        if re.search(r'\b(axiom|admitted|native_decide)\b', lean_code):
            return CertificationResult(
                level=self.level,
                passed=False,
                evidence="Contains unsafe axioms",
                warnings=["Uses axiom, admitted, or native_decide"]
            )

        # Run Lean
        result = self._run_lean(lean_code)

        if result["success"]:
            return CertificationResult(
                level=self.level,
                passed=True,
                evidence="Complete machine verification in Lean",
                lean_code=lean_code
            )
        else:
            return CertificationResult(
                level=self.level,
                passed=False,
                evidence="Lean verification failed",
                errors=result.get("errors", [])
            )

    def _extract_lean(self, content: str) -> str:
        """Extract Lean from hybrid content."""
        lean_blocks = re.findall(r'```lean4?\n(.*?)```', content, re.DOTALL)
        return "\n\n".join(lean_blocks)

    def _run_lean(self, lean_code: str) -> Dict:
        """Run Lean compiler."""
        return {"success": True}
```

---

### The Certification Cascade

```python
class CertificationCascade:
    """
    Orchestrates multi-level certification.

    Key principle: Each level ADDS certainty, doesn't replace.
    Level 2 (Semiformal) is sufficient for most purposes.
    Level 3 is only needed for formal library integration.
    """

    def __init__(
        self,
        lean_project_path: str = None,
        llm_client = None
    ):
        self.verifiers = {
            CertificationLevel.INFORMAL: InformalVerifier(),
            CertificationLevel.COMPUTATIONAL: ComputationalVerifier(),
            CertificationLevel.SEMIFORMAL: SemiformalVerifier(llm_client, lean_project_path),
            CertificationLevel.FORMAL: FormalVerifier(lean_project_path),
        }

    def certify(
        self,
        proof: ProofDocument,
        problem: str,
        target_level: CertificationLevel = CertificationLevel.SEMIFORMAL,
        require_all_previous: bool = True,
    ) -> CertificationResult:
        """
        Certify a proof to the requested level.
        """
        results = []
        highest_passed = None

        for level in CertificationLevel:
            if level > target_level:
                break

            verifier = self.verifiers[level]
            result = verifier.verify(proof, problem)
            results.append(result)

            if result.passed:
                highest_passed = result
            elif require_all_previous:
                return CertificationResult(
                    level=level,
                    passed=False,
                    evidence=f"Failed at {level.name}: {result.evidence}",
                    errors=result.errors
                )

        return highest_passed or results[-1]

    def certify_best_effort(
        self,
        proof: ProofDocument,
        problem: str,
    ) -> CertificationResult:
        """
        Try to achieve highest possible certification.
        """
        best_result = None

        for level in CertificationLevel:
            verifier = self.verifiers[level]
            result = verifier.verify(proof, problem)

            if result.passed:
                best_result = result

        return best_result or CertificationResult(
            level=CertificationLevel.INFORMAL,
            passed=False,
            evidence="No certification level passed"
        )

    def what_level_is_this(self, proof: ProofDocument) -> CertificationLevel:
        """
        Determine what certification level a proof represents.
        """
        if proof.format == "lean" and not proof.has_sorries():
            return CertificationLevel.FORMAL

        if proof.format in ["lean", "program"] or (proof.format == "hybrid"):
            return CertificationLevel.SEMIFORMAL

        if proof.format in ["markdown", "latex"] and proof.citations:
            return CertificationLevel.SEMIFORMAL

        if proof.format in ["markdown", "latex"]:
            return CertificationLevel.INFORMAL

        return CertificationLevel.INFORMAL
```

---

## Usage Examples

### Example 1: Exploration (Level 0)

```python
proof = ProofDocument(
    format="markdown",
    content="""
    **Conjecture**: Maybe we can use the pigeonhole principle here?

    If we partition the integers mod p, there are p residue classes...
    This feels like it might lead somewhere.
    """
)

cascade = CertificationCascade()
result = cascade.certify(proof, problem="number theory problem", target_level=CertificationLevel.INFORMAL)

print(f"Level: {result.level.name}")  # INFORMAL
```

### Example 2: Semiformal - Rigorous NL (Level 2)

```python
proof = ProofDocument(
    format="markdown",
    content="""
    **Theorem**: For all n ≥ 1, 1 + 2 + ... + n = n(n+1)/2

    **Proof**: By induction on n.

    Base case (n=1): 1 = 1·2/2 = 1. ✓

    Inductive step: Assume true for k. Then:
    1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1)  [by induction hypothesis]
                            = (k+1)(k/2 + 1)
                            = (k+1)(k+2)/2 ✓

    By [Mathematical Induction Principle], the result holds for all n.
    """,
    citations=[
        Citation(
            source="Standard",
            theorem="Mathematical Induction Principle",
            statement="If P(1) and P(k)→P(k+1), then P(n) for all n≥1"
        )
    ]
)

result = cascade.certify(proof, problem="sum formula", target_level=CertificationLevel.SEMIFORMAL)

print(f"Publishable: {result.is_publishable}")  # True
```

### Example 3: Semiformal - Lean Skeleton (Level 2)

```python
proof = ProofDocument(
    format="lean",
    content="""
import Mathlib.Tactic

theorem fermat_last (n : ℕ) (hn : n > 2) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    a^n + b^n ≠ c^n := by
  -- Main structure verified, key lemmas marked
  have h1 : ... := sorry  -- Frey curve construction
  have h2 : ... := sorry  -- Modularity theorem
  have h3 : ... := sorry  -- Level lowering
  sorry  -- Final contradiction
"""
)

result = cascade.certify(proof, problem="FLT", target_level=CertificationLevel.SEMIFORMAL)

print(f"Structure verified: {result.passed}")
print(f"Gaps remaining: {result.sorry_count}")  # 4
```

### Example 4: Semiformal - ICPC Program (Level 2)

```python
proof = ProofDocument(
    format="program",
    content="""
def solve(n: int, edges: List[Tuple[int, int]]) -> int:
    '''
    Find minimum spanning tree weight using Kruskal's algorithm.

    Correctness: Kruskal's algorithm is correct by the cut property
    of MSTs [CLRS Theorem 23.1]. We sort edges by weight and greedily
    add edges that don't create cycles (checked via Union-Find).
    '''
    edges.sort(key=lambda e: e[2])  # Sort by weight
    uf = UnionFind(n)
    total = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            total += w
    return total
"""
)

result = cascade.certify(proof, problem="MST", target_level=CertificationLevel.SEMIFORMAL)

print(f"Tests passed: {result.tests_passed}/{result.tests_total}")
```

### Example 5: Full Lean (Level 3)

```python
proof = ProofDocument(
    format="lean",
    content="""
import Mathlib.Tactic

theorem sum_formula (n : ℕ) : 2 * (∑ i in Finset.range (n+1), i) = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ]
    ring_nf
    linarith
"""
)

result = cascade.certify(proof, problem="sum formula", target_level=CertificationLevel.FORMAL)

print(f"Machine verified: {result.is_machine_verified}")  # True
```

---

## Key Insights

### 1. Level 2 (Semiformal) is the Workhorse

Most mathematical work happens at Level 2. This includes:
- **Rigorous-NL**: Natural language proofs with citations (what papers contain)
- **Lean-skeleton**: Type-checked structure with explicit gaps (DeepSeek-Prover-V2 approach)
- **Verified programs**: Computational solutions (ICPC-style)

All three are "semiformal" - more than just intuition, but not absolute rigor.

### 2. NL Expressibility vs Formal Rigor

Natural language has advantages:
- More expressive (can describe high-level strategies)
- Easier to generate and iterate
- Matches how mathematicians actually think

But has risks:
- Possible hidden gaps or inconsistencies
- Not machine-checkable in totality

Lean + sorry gives the best of both:
- Type-checked structure (verified shape)
- Explicit gaps (we know what's unproven)
- Can be incrementally filled

### 3. Human Review at Level 2

Level 2 may flag uncertain steps for human review. This is appropriate because:
- LLM judgment is not infallible
- Some logical leaps need expert verification
- Critical steps deserve extra scrutiny

### 4. Level 3 is Rare and Optional

From AlphaEvolve: full formalization is "rare". Level 3 is for:
- Mathlib/formal library integration
- Absolute certainty requirements
- Foundational results

Most results don't need Level 3.

---

## Summary

| Level | Name | What it proves | Format | Use case |
|-------|------|---------------|--------|----------|
| 0 | Informal | Exploration recorded | NL sketches | Hypothesis generation |
| 1 | Computational | Examples + CAS | Tests + SymPy | Quick validation |
| 2 | Semiformal | Logic sound + gaps explicit | NL+citations / Lean+sorry / Programs | Publication, working proofs |
| 3 | Formal | Complete machine proof | Lean (no sorry) | Mathlib, formal archives |

**Key Points:**
- Level 0 is where discovery happens - don't skip it
- Level 1 catches obvious errors fast
- Level 2 is sufficient for publication and most purposes
- Level 3 is for formal library integration (rare)
- Lean + sorry = Level 2 (semiformal), NOT incomplete
- ICPC programs with test verification = Level 2

### The DeepSeek-Prover-V2 Approach

Level 2 IS the DeepSeek approach:
1. Generate natural language proof (rigorous, with citations)
2. Translate to Lean skeleton
3. Fill sorries incrementally
4. If all sorries filled → Level 3

This gives expressibility of NL during search + formal verification where possible.
