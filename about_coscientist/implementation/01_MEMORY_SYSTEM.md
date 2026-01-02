# Memory System Implementation

## Overview

The memory system is the persistent "brain" that enables learning across problems and sessions. ARCHITECTURE.md describes three memory types:

1. **Working Memory** - Current problem context
2. **Library Memory** - Solved proofs, technique index, embeddings
3. **World Model** - Domain knowledge, concept relationships

The existing solver has **partial** implementation. This spec completes it.

---

## Current State (What Exists)

### world_model.py - Problem Tracking ✅
```python
class ProblemState:
    problem_id: str
    statement_hash: str  # Deduplication
    topics: List[str]
    status: ProblemStatus  # UNSEEN → ATTEMPTED → SOLVED → VERIFIED → FORMALIZED
    attempts: List[SolvingAttempt]
    best_solution: Optional[str]
    lean_proof: Optional[str]
```

### technique_tracker.py - Technique Learning ✅
```python
class TechniqueTracker:
    # Per-topic technique effectiveness
    topic_techniques: Dict[str, Dict[str, TechniqueStats]]
    # Prior knowledge from "Putnam and Beyond"
    _initialize_priors()
    # Bayesian updating
    update(topics, technique, success, stage_reached)
    recommend(topics) -> List[Tuple[technique, score]]
```

### session.py - Session Management ✅
```python
class Session:
    session_id: str
    problem_id: str
    status: SessionStatus  # ACTIVE, PAUSED, COMPLETED, ABANDONED
    steps: List[SessionStep]
    # Step-level tracking with timing
```

---

## What's Missing

### 1. Local World Model (Per-Problem Context)

The current `ProblemState` tracks attempts but doesn't maintain:
- Discovered constraints (e.g., "n must be > 0")
- Failed approaches with WHY they failed
- Partial progress (what's been established)
- Active hypotheses being explored

### 2. Global World Model (Domain Knowledge)

Completely missing:
- Mathematical domain syllabus (what topics exist, their relationships)
- Theorem graph (dependencies between theorems)
- Concept embeddings (for similarity search)
- Domain-specific heuristics (what tactics work where)

### 3. Flywheel Closure

The pieces exist but aren't connected:
- `world_model.record_attempt()` is called... when?
- `technique_tracker.recommend()` is available... but not used in /solve
- Learning happens... but doesn't feed back to solving

---

## Implementation Plan

### Phase 1: Local World Model

**File**: `src/memory/local_context.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime

class ConstraintType(Enum):
    DOMAIN = "domain"           # x > 0, n ∈ ℕ
    STRUCTURE = "structure"     # sequence, function, etc.
    RELATIONSHIP = "relationship"  # a < b, f(x) = g(x)
    IMPOSSIBILITY = "impossibility"  # proved something can't happen

@dataclass
class Constraint:
    """A discovered constraint on the problem."""
    type: ConstraintType
    description: str
    formal: Optional[str]  # Lean expression if formalized
    discovered_at: datetime = field(default_factory=datetime.now)
    source: str = ""  # Which approach discovered this

@dataclass
class FailedApproach:
    """Record of why an approach didn't work."""
    approach_name: str
    reason: str
    error_type: Optional[str]  # Type mismatch, tactic failed, etc.
    partial_progress: Optional[str]  # What was established before failure
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EstablishedFact:
    """Something proven during exploration."""
    statement: str
    formal: Optional[str]  # Lean version
    proof_sketch: str
    confidence: float
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ActiveHypothesis:
    """A hypothesis being explored."""
    statement: str
    plausibility: float  # 0-1
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    child_hypotheses: List[str] = field(default_factory=list)

@dataclass
class LocalContext:
    """Per-problem working memory."""
    problem_id: str

    # What we've discovered
    constraints: List[Constraint] = field(default_factory=list)
    established_facts: List[EstablishedFact] = field(default_factory=list)

    # What we've tried
    failed_approaches: List[FailedApproach] = field(default_factory=list)
    approaches_in_progress: Set[str] = field(default_factory=set)

    # What we're exploring
    hypotheses: List[ActiveHypothesis] = field(default_factory=list)

    # Current focus
    current_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)

    def add_constraint(self, constraint: Constraint):
        """Add a discovered constraint."""
        # Avoid duplicates
        for existing in self.constraints:
            if existing.description == constraint.description:
                return
        self.constraints.append(constraint)

    def record_failure(self, approach: str, reason: str, error_type: str = None):
        """Record why an approach failed."""
        self.failed_approaches.append(FailedApproach(
            approach_name=approach,
            reason=reason,
            error_type=error_type
        ))
        self.approaches_in_progress.discard(approach)

    def establish_fact(self, statement: str, proof_sketch: str, confidence: float):
        """Record a proven fact."""
        self.established_facts.append(EstablishedFact(
            statement=statement,
            proof_sketch=proof_sketch,
            confidence=confidence
        ))

    def get_untried_approaches(self, all_approaches: List[str]) -> List[str]:
        """Return approaches not yet tried or in progress."""
        tried = {f.approach_name for f in self.failed_approaches}
        in_progress = self.approaches_in_progress
        return [a for a in all_approaches if a not in tried and a not in in_progress]

    def summarize(self) -> str:
        """Create a summary for LLM context."""
        lines = []

        if self.constraints:
            lines.append("CONSTRAINTS:")
            for c in self.constraints:
                lines.append(f"  - {c.description}")

        if self.established_facts:
            lines.append("\nESTABLISHED:")
            for f in self.established_facts:
                lines.append(f"  - {f.statement} (confidence: {f.confidence:.1%})")

        if self.failed_approaches:
            lines.append("\nFAILED APPROACHES:")
            for f in self.failed_approaches[-5:]:  # Last 5
                lines.append(f"  - {f.approach_name}: {f.reason}")

        if self.hypotheses:
            lines.append("\nACTIVE HYPOTHESES:")
            for h in self.hypotheses:
                lines.append(f"  - {h.statement} (plausibility: {h.plausibility:.1%})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "problem_id": self.problem_id,
            "constraints": [asdict(c) for c in self.constraints],
            "established_facts": [asdict(f) for f in self.established_facts],
            "failed_approaches": [asdict(f) for f in self.failed_approaches],
            "approaches_in_progress": list(self.approaches_in_progress),
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "current_goal": self.current_goal,
            "subgoals": self.subgoals
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalContext":
        """Deserialize."""
        ctx = cls(problem_id=data["problem_id"])
        ctx.constraints = [Constraint(**c) for c in data.get("constraints", [])]
        ctx.established_facts = [EstablishedFact(**f) for f in data.get("established_facts", [])]
        ctx.failed_approaches = [FailedApproach(**f) for f in data.get("failed_approaches", [])]
        ctx.approaches_in_progress = set(data.get("approaches_in_progress", []))
        ctx.hypotheses = [ActiveHypothesis(**h) for h in data.get("hypotheses", [])]
        ctx.current_goal = data.get("current_goal")
        ctx.subgoals = data.get("subgoals", [])
        return ctx
```

---

### Phase 2: Global World Model (Knowledge Graph)

**File**: `src/memory/global_knowledge.py`

This implements the **Concept Graph** portion of the system's knowledge graph. The structures below form our internal KG for domain knowledge.

**Core Principle: Techniques Over Topics**

The KG indexes **decontextualized techniques**—methods stripped from their original context, indexed by structural applicability. This enables cross-domain transfer:

```
Traditional KG:                    Our KG:
───────────────                    ───────
Topics → Papers                    Techniques → Goal Patterns → Domains Used
"number theory" → [papers]         "infinite descent" → "∀n, prove smaller exists" → [NT, combinatorics, analysis]
```

**Note on Theorem Graph**: Lean/Mathlib already maintains theorem dependencies. We access this via tools rather than rebuilding it. See `04_VERIFICATION_CASCADE.md` and `07_RETRIEVAL_SYSTEM.md` for how we query Lean's structure.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

@dataclass
class MathDomain:
    """A node in the domain hierarchy graph."""
    name: str
    description: str
    parent: Optional[str] = None  # Edge to parent domain
    children: List[str] = field(default_factory=list)  # Edges to child domains
    key_concepts: List[str] = field(default_factory=list)  # Links to ConceptNodes
    typical_tactics: List[str] = field(default_factory=list)

@dataclass
class ConceptNode:
    """A node in the mathematical concept graph (KG)."""
    name: str
    domain: str  # Edge to MathDomain
    definition: str
    related_concepts: List[str] = field(default_factory=list)  # Edges to other concepts
    mathlib_names: List[str] = field(default_factory=list)  # Links to Lean's theorem graph
    embedding: Optional[List[float]] = None  # For semantic search

@dataclass
class TacticPattern:
    """A pattern for when to use a tactic."""
    tactic_name: str
    goal_pattern: str  # Regex or structural pattern
    domain_affinity: Dict[str, float]  # domain -> affinity score
    success_rate: float
    example_uses: List[str] = field(default_factory=list)

@dataclass
class TransferableTechnique:
    """
    A decontextualized technique—the core of cross-domain transfer.

    Indexed by WHAT IT DOES (goal pattern), not WHERE IT CAME FROM (domain).
    """
    id: str
    name: str  # e.g., "infinite descent", "probabilistic method"
    description: str  # Context-free description of the technique

    # Structural indexing (NOT topic-based)
    goal_patterns: List[str]  # What goals this technique helps prove
    structure_patterns: List[str]  # What proof structures it produces

    # Transfer history
    origin_domain: str  # Where it was first used
    domains_transferred_to: List[str]  # Where it's been successfully applied
    transfer_examples: List[Dict[str, str]]  # {domain, problem, how_it_applied}

    # Applicability
    preconditions: List[str]  # When can this technique apply?
    key_insight: str  # The core idea, domain-independent

    # Links
    related_techniques: List[str]  # Other techniques often used together
    mathlib_tactics: List[str]  # Lean tactics that implement this
    source_papers: List[str]  # Original papers (for provenance, not retrieval)

class GlobalKnowledge:
    """Domain knowledge for mathematical problem solving."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.domains: Dict[str, MathDomain] = {}
        self.concepts: Dict[str, ConceptNode] = {}
        self.tactic_patterns: List[TacticPattern] = []
        self._load()

    def _load(self):
        """Load from persistent storage."""
        domains_file = self.data_dir / "domains.json"
        if domains_file.exists():
            with open(domains_file) as f:
                data = json.load(f)
                self.domains = {d["name"]: MathDomain(**d) for d in data}
        else:
            self._initialize_domains()

        concepts_file = self.data_dir / "concepts.json"
        if concepts_file.exists():
            with open(concepts_file) as f:
                data = json.load(f)
                self.concepts = {c["name"]: ConceptNode(**c) for c in data}

        patterns_file = self.data_dir / "tactic_patterns.json"
        if patterns_file.exists():
            with open(patterns_file) as f:
                data = json.load(f)
                self.tactic_patterns = [TacticPattern(**p) for p in data]
        else:
            self._initialize_tactic_patterns()

    def _initialize_domains(self):
        """Initialize domain hierarchy from Putnam syllabus."""
        # Based on RESEARCH.md and Putnam and Beyond structure
        self.domains = {
            "number_theory": MathDomain(
                name="number_theory",
                description="Properties of integers, divisibility, primes, modular arithmetic",
                children=["divisibility", "primes", "modular_arithmetic", "diophantine"],
                key_concepts=["gcd", "lcm", "prime", "congruence", "quadratic_residue"],
                typical_tactics=["induction", "modular_arithmetic", "divisibility", "p_adic"]
            ),
            "algebra": MathDomain(
                name="algebra",
                description="Equations, inequalities, polynomials, sequences",
                children=["polynomials", "inequalities", "sequences", "functional_equations"],
                key_concepts=["root", "coefficient", "inequality", "recurrence"],
                typical_tactics=["telescoping", "am_gm", "cauchy_schwarz", "substitution"]
            ),
            "combinatorics": MathDomain(
                name="combinatorics",
                description="Counting, arrangements, discrete structures",
                children=["counting", "graph_theory", "extremal"],
                key_concepts=["permutation", "combination", "bijection", "pigeonhole"],
                typical_tactics=["pigeonhole", "double_counting", "bijection", "recursion"]
            ),
            "analysis": MathDomain(
                name="analysis",
                description="Limits, continuity, derivatives, integrals",
                children=["real_analysis", "sequences_series", "calculus"],
                key_concepts=["limit", "continuity", "derivative", "integral", "convergence"],
                typical_tactics=["ivt", "mvt", "squeeze", "epsilon_delta"]
            ),
            "geometry": MathDomain(
                name="geometry",
                description="Shapes, transformations, coordinates",
                children=["euclidean", "coordinate", "transformations"],
                key_concepts=["point", "line", "circle", "angle", "area"],
                typical_tactics=["coordinates", "vectors", "complex_numbers", "trigonometry"]
            ),
            "linear_algebra": MathDomain(
                name="linear_algebra",
                description="Matrices, vectors, linear transformations",
                children=["matrices", "eigenvalues", "vector_spaces"],
                key_concepts=["matrix", "determinant", "eigenvalue", "rank", "kernel"],
                typical_tactics=["eigenvalues", "determinants", "cayley_hamilton", "rank_nullity"]
            )
        }
        self._save_domains()

    def _initialize_tactic_patterns(self):
        """Initialize tactic patterns from RESEARCH.md and solver/approaches.py."""
        # From RESEARCH.md Tactics Library section
        self.tactic_patterns = [
            # Goal patterns
            TacticPattern(
                tactic_name="intro",
                goal_pattern=r"∀|→|->",
                domain_affinity={"*": 1.0},
                success_rate=0.95
            ),
            TacticPattern(
                tactic_name="induction",
                goal_pattern=r"∀\s*n\s*[∈:]\s*ℕ|for all.*natural|prove for all n",
                domain_affinity={"number_theory": 0.9, "combinatorics": 0.7},
                success_rate=0.65
            ),
            TacticPattern(
                tactic_name="by_contra",
                goal_pattern=r"¬|not|no.*exists|impossible",
                domain_affinity={"*": 0.8},
                success_rate=0.60
            ),
            TacticPattern(
                tactic_name="use",
                goal_pattern=r"∃|exists|find",
                domain_affinity={"*": 0.9},
                success_rate=0.70
            ),
            TacticPattern(
                tactic_name="ring",
                goal_pattern=r"=.*\+|\*|polynomial|algebraic",
                domain_affinity={"algebra": 0.95, "number_theory": 0.6},
                success_rate=0.85
            ),
            TacticPattern(
                tactic_name="linarith",
                goal_pattern=r"[<>≤≥]|inequality|less|greater",
                domain_affinity={"analysis": 0.9, "algebra": 0.8},
                success_rate=0.75
            ),
            TacticPattern(
                tactic_name="omega",
                goal_pattern=r"ℕ|ℤ|Int|Nat.*[<>≤≥=]",
                domain_affinity={"number_theory": 0.95},
                success_rate=0.80
            ),
            TacticPattern(
                tactic_name="simp",
                goal_pattern=r".*",  # Always worth trying
                domain_affinity={"*": 0.5},
                success_rate=0.40
            ),
            TacticPattern(
                tactic_name="norm_num",
                goal_pattern=r"\d+|concrete.*value|numerical",
                domain_affinity={"*": 0.9},
                success_rate=0.90
            ),
            TacticPattern(
                tactic_name="cases",
                goal_pattern=r"∨|or|either|if.*then.*else",
                domain_affinity={"*": 0.85},
                success_rate=0.70
            ),
        ]
        self._save_tactic_patterns()

    def _save_domains(self):
        """Persist domains."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / "domains.json", "w") as f:
            json.dump([asdict(d) for d in self.domains.values()], f, indent=2)

    def _save_tactic_patterns(self):
        """Persist tactic patterns."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / "tactic_patterns.json", "w") as f:
            json.dump([asdict(p) for p in self.tactic_patterns], f, indent=2)

    def classify_problem(self, statement: str, topics: List[str]) -> List[MathDomain]:
        """Classify a problem into domains."""
        matching = []
        for topic in topics:
            if topic in self.domains:
                matching.append(self.domains[topic])

        # Fallback: keyword matching
        if not matching:
            statement_lower = statement.lower()
            for domain in self.domains.values():
                for concept in domain.key_concepts:
                    if concept in statement_lower:
                        matching.append(domain)
                        break

        return matching

    def suggest_tactics(self, goal: str, domain: str = None) -> List[Tuple[str, float]]:
        """Suggest tactics for a goal."""
        import re
        suggestions = []

        for pattern in self.tactic_patterns:
            # Check goal pattern match
            if re.search(pattern.goal_pattern, goal, re.IGNORECASE):
                score = pattern.success_rate

                # Adjust by domain affinity
                if domain and domain in pattern.domain_affinity:
                    score *= pattern.domain_affinity[domain]
                elif "*" in pattern.domain_affinity:
                    score *= pattern.domain_affinity["*"]

                suggestions.append((pattern.tactic_name, score))

        # Sort by score
        suggestions.sort(key=lambda x: -x[1])
        return suggestions

    def get_related_concepts(self, concept: str) -> List[ConceptNode]:
        """Get concepts related to a given concept."""
        if concept in self.concepts:
            node = self.concepts[concept]
            return [self.concepts[r] for r in node.related_concepts if r in self.concepts]
        return []

    def update_tactic_success(self, tactic: str, domain: str, success: bool):
        """Update tactic success rate based on experience."""
        for pattern in self.tactic_patterns:
            if pattern.tactic_name == tactic:
                # Bayesian update (simplified)
                n = 10  # Prior weight
                current_rate = pattern.success_rate
                new_rate = (current_rate * n + (1 if success else 0)) / (n + 1)
                pattern.success_rate = new_rate

                # Update domain affinity
                if domain not in pattern.domain_affinity:
                    pattern.domain_affinity[domain] = 0.5
                affinity = pattern.domain_affinity[domain]
                pattern.domain_affinity[domain] = (affinity * n + (1 if success else 0)) / (n + 1)

        self._save_tactic_patterns()
```

---

### Phase 3: Flywheel Integration

**File**: `src/memory/flywheel.py`

```python
"""
Flywheel: Solve → Learn → Solve Better

This module ensures every solving attempt feeds back into the system.
"""

from typing import Optional
from dataclasses import dataclass
from .world_model import WorldModel, SolvingAttempt
from .technique_tracker import TechniqueTracker
from .local_context import LocalContext
from .global_knowledge import GlobalKnowledge

@dataclass
class SolveResult:
    """Result of a solving attempt."""
    success: bool
    solution: Optional[str]
    techniques_used: list
    approaches_tried: list
    confidence: float
    stage_reached: str  # informal, computational, semiformal, formal (Level 0-3)
    errors: list
    time_spent: float

class Flywheel:
    """
    Coordinates learning across memory components.

    CRITICAL: Every attempt must call record() to close the loop.
    """

    def __init__(self, data_dir: str):
        self.world_model = WorldModel(data_dir)
        self.technique_tracker = TechniqueTracker(data_dir)
        self.global_knowledge = GlobalKnowledge(Path(data_dir) / "knowledge")
        self.local_contexts: dict[str, LocalContext] = {}

    def get_context(self, problem_id: str, statement: str, topics: list) -> dict:
        """
        Get all relevant context for solving a problem.

        THIS SHOULD BE CALLED AT START OF /solve.
        """
        # Get or create local context
        if problem_id not in self.local_contexts:
            self.local_contexts[problem_id] = LocalContext(problem_id=problem_id)
        local = self.local_contexts[problem_id]

        # Get similar problems from world model
        similar = self.world_model.get_similar_problems(statement, k=5)

        # Get technique recommendations
        recommended = self.technique_tracker.recommend(topics)

        # Get domain knowledge
        domains = self.global_knowledge.classify_problem(statement, topics)

        # Get tactic suggestions (if we have a goal)
        tactic_suggestions = []
        if local.current_goal:
            domain = domains[0].name if domains else None
            tactic_suggestions = self.global_knowledge.suggest_tactics(
                local.current_goal, domain
            )

        return {
            "local_context": local.summarize(),
            "similar_problems": similar,
            "recommended_techniques": recommended,
            "domains": [d.name for d in domains],
            "domain_tactics": [d.typical_tactics for d in domains],
            "tactic_suggestions": tactic_suggestions,
            "failed_approaches": [f.approach_name for f in local.failed_approaches],
            "established_facts": [f.statement for f in local.established_facts],
        }

    def record(self, problem_id: str, topics: list, result: SolveResult):
        """
        Record a solving attempt and update all learning components.

        THIS MUST BE CALLED AFTER EVERY ATTEMPT.
        """
        # 1. Update world model
        attempt = SolvingAttempt(
            techniques_tried=result.techniques_used,
            success=result.success,
            confidence=result.confidence,
            time_spent=result.time_spent,
            errors=result.errors
        )
        self.world_model.record_attempt(problem_id, attempt)

        # 2. Update technique tracker
        for technique in result.techniques_used:
            self.technique_tracker.update(
                topics=topics,
                technique=technique,
                success=result.success,
                stage_reached=result.stage_reached
            )

        # 3. Update global knowledge (tactic success rates)
        domains = self.global_knowledge.classify_problem("", topics)
        domain = domains[0].name if domains else "general"

        for technique in result.techniques_used:
            self.global_knowledge.update_tactic_success(
                tactic=technique,
                domain=domain,
                success=result.success
            )

        # 4. Update local context
        if problem_id in self.local_contexts:
            local = self.local_contexts[problem_id]
            for approach in result.approaches_tried:
                if approach not in result.techniques_used:
                    # This approach was tried but not in final solution
                    local.record_failure(approach, "Did not lead to solution")

        # 5. Save state
        self.world_model.save()
        self.technique_tracker.save()

    def record_failure(self, problem_id: str, approach: str, reason: str, error_type: str = None):
        """Record a failed approach for local context."""
        if problem_id not in self.local_contexts:
            self.local_contexts[problem_id] = LocalContext(problem_id=problem_id)
        self.local_contexts[problem_id].record_failure(approach, reason, error_type)

    def record_constraint(self, problem_id: str, constraint: str, source: str):
        """Record a discovered constraint."""
        if problem_id not in self.local_contexts:
            self.local_contexts[problem_id] = LocalContext(problem_id=problem_id)
        from .local_context import Constraint, ConstraintType
        self.local_contexts[problem_id].add_constraint(Constraint(
            type=ConstraintType.DOMAIN,
            description=constraint,
            source=source
        ))

    def record_fact(self, problem_id: str, statement: str, proof_sketch: str, confidence: float):
        """Record an established fact."""
        if problem_id not in self.local_contexts:
            self.local_contexts[problem_id] = LocalContext(problem_id=problem_id)
        self.local_contexts[problem_id].establish_fact(statement, proof_sketch, confidence)

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "problems_attempted": len(self.world_model.problems),
            "techniques_tracked": self.technique_tracker.get_stats(),
            "domains_known": list(self.global_knowledge.domains.keys()),
            "tactic_patterns": len(self.global_knowledge.tactic_patterns),
        }
```

---

## Integration Points

### 1. Update /solve Command

The `/solve` command should:

```markdown
## At Start
1. Call `flywheel.get_context(problem_id, statement, topics)`
2. Display similar problems and recommended techniques
3. Show any established facts or failed approaches

## During Solving
1. Call `flywheel.record_failure()` when an approach fails
2. Call `flywheel.record_constraint()` when discovering constraints
3. Call `flywheel.record_fact()` when establishing intermediate results

## At End
1. ALWAYS call `flywheel.record()` with the result
2. This updates world model, technique tracker, and global knowledge
```

### 2. Tool Interface

Add to `tools.py`:

```python
def tool_get_context(problem_id: str, statement: str, topics: List[str]) -> dict:
    """Get solving context from memory."""
    flywheel = Flywheel(DATA_DIR)
    return flywheel.get_context(problem_id, statement, topics)

def tool_record_attempt(problem_id: str, topics: List[str], result: dict) -> bool:
    """Record a solving attempt."""
    flywheel = Flywheel(DATA_DIR)
    flywheel.record(problem_id, topics, SolveResult(**result))
    return True

def tool_record_failure(problem_id: str, approach: str, reason: str) -> bool:
    """Record a failed approach."""
    flywheel = Flywheel(DATA_DIR)
    flywheel.record_failure(problem_id, approach, reason)
    return True
```

---

## Data Schema

### File: `data/memory/problems.json` (existing, enhanced)
```json
{
  "problem_id": {
    "statement_hash": "abc123",
    "statement": "...",
    "topics": ["number_theory"],
    "status": "SOLVED",
    "attempts": [...],
    "best_solution": "...",
    "lean_proof": "..."
  }
}
```

### File: `data/memory/techniques.json` (existing)
```json
{
  "number_theory": {
    "modular_arithmetic": {"successes": 14, "attempts": 20},
    "induction": {"successes": 12, "attempts": 18}
  }
}
```

### File: `data/memory/knowledge/domains.json` (new)
```json
[
  {
    "name": "number_theory",
    "description": "...",
    "children": ["divisibility", "primes"],
    "key_concepts": ["gcd", "prime"],
    "typical_tactics": ["induction", "modular_arithmetic"]
  }
]
```

### File: `data/memory/knowledge/tactic_patterns.json` (new)
```json
[
  {
    "tactic_name": "induction",
    "goal_pattern": "∀\\s*n\\s*[∈:]\\s*ℕ",
    "domain_affinity": {"number_theory": 0.9},
    "success_rate": 0.65
  }
]
```

### File: `data/memory/local/{problem_id}.json` (new)
```json
{
  "problem_id": "putnam_2024_a1",
  "constraints": [...],
  "established_facts": [...],
  "failed_approaches": [...],
  "hypotheses": [...],
  "current_goal": "..."
}
```

---

## Testing

### Unit Tests

```python
def test_local_context():
    ctx = LocalContext(problem_id="test")
    ctx.add_constraint(Constraint(
        type=ConstraintType.DOMAIN,
        description="n > 0"
    ))
    ctx.record_failure("induction", "Base case failed")
    ctx.establish_fact("n is even", "Direct computation", 0.9)

    summary = ctx.summarize()
    assert "n > 0" in summary
    assert "induction" in summary
    assert "n is even" in summary

def test_global_knowledge():
    gk = GlobalKnowledge(Path("/tmp/test_knowledge"))
    suggestions = gk.suggest_tactics("∀ n : ℕ, P n", "number_theory")
    assert ("induction", _) in suggestions

def test_flywheel():
    fw = Flywheel("/tmp/test_flywheel")
    ctx = fw.get_context("test", "For all n...", ["number_theory"])
    assert "recommended_techniques" in ctx

    fw.record("test", ["number_theory"], SolveResult(
        success=True,
        solution="...",
        techniques_used=["induction"],
        approaches_tried=["induction", "cases"],
        confidence=0.9,
        stage_reached="semiformal",
        errors=[],
        time_spent=120.0
    ))

    # Verify learning
    ctx2 = fw.get_context("test2", "For all m...", ["number_theory"])
    # induction should be recommended higher
```

---

## Future: Neural Memory Systems

The current implementation uses structured storage (JSON, graphs). Future versions should explore neural approaches for richer, more flexible memory.

### Research Directions

**1. Associative Memory**
- **Modern Hopfield Networks** (Ramsauer et al., 2020): Continuous associative memory with exponential storage capacity. Could replace exact-match retrieval with pattern-completion.
- **Titans** (Google, 2024): Neural long-term memory architecture that learns to memorize at test time. Directly relevant to learning from solving attempts.
- **Memory Transformers**: Attention as soft associative lookup over stored representations.

**2. Memory Diffusion**
- **Diffusion Models for Retrieval**: Generate relevant memories via iterative denoising rather than nearest-neighbor search.
- **Memory-Conditioned Generation**: Use stored proof patterns as conditioning signals for generating new proofs.
- **Retrieval-Augmented Diffusion**: Combine retrieval with diffusion for structured generation.

**3. World Models → Neural World Models**
- **Kosmos** (FutureHouse, 2024): AI Scientist with structured world model shared across agents. Uses structured storage (graph DB) - NOT neural. Our design references this. Paper: arxiv.org/abs/2511.02824
- **Future neural direction**: Replace structured graph with learned representations
- **Genie 2** (DeepMind, 2024): Learned world models for simulation. Relevant for "what-if" reasoning.
- **World Models in Transformers**: Implicitly learned dynamics in large-scale pretraining.

**4. Neural Knowledge Graphs**
- **Knowledge Graph Embeddings** (TransE, RotatE, etc.): Learn vector representations of entities/relations for fuzzy matching.
- **Neural-Symbolic Integration**: Combine symbolic KG structure with neural retrieval/reasoning.
- **Graph Neural Networks for KGs**: Message passing over concept/theorem graphs for richer representations.

**5. Memory-Augmented Architectures**
- **Differentiable Neural Computer** (DeepMind): External memory with learned read/write operations.
- **Memory Networks** (Facebook): End-to-end memory access for QA.
- **Memorizing Transformers** (Google, 2022): kNN-augmented attention over large external memory.

### Integration Path

```
Phase 1 (Current): Structured storage + embedding retrieval
Phase 2: Neural retrieval (learned similarity, not just cosine)
Phase 3: Associative memory (pattern completion, not just lookup)
Phase 4: Full neural world model (learned dynamics, counterfactual reasoning)
```

### Key Papers

| Paper | Relevance |
|-------|-----------|
| Ramsauer et al., "Hopfield Networks is All You Need" (2020) | Modern associative memory theory |
| Google, "Titans: Learning to Memorize at Test Time" (2024) | Test-time memory learning |
| Borgeaud et al., "RETRO" (2022) | Retrieval-enhanced transformers |
| Graves et al., "Neural Turing Machines" (2014) | Differentiable external memory |
| Wu et al., "Memorizing Transformers" (2022) | kNN-augmented attention |
| FutureHouse, "Kosmos" (arxiv 2511.02824) | Structured world model for AI scientist |

### Design Principle

Current implementation should be **neural-ready**: clean interfaces that can swap structured storage for neural alternatives without changing the rest of the system.

```python
# Abstract interface - implementation can be symbolic or neural
class MemoryBackend(ABC):
    @abstractmethod
    def store(self, key: str, value: Any, context: Dict) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Tuple[Any, float]]: ...

    @abstractmethod
    def associate(self, partial_pattern: Dict) -> List[Dict]: ...  # Pattern completion
```

---

## Summary

| Component | File | Status | Priority |
|-----------|------|--------|----------|
| LocalContext | `local_context.py` | TO IMPLEMENT | P1 |
| GlobalKnowledge | `global_knowledge.py` | TO IMPLEMENT | P1 |
| Flywheel | `flywheel.py` | TO IMPLEMENT | P0 |
| Tool Integration | `tools.py` | TO UPDATE | P0 |
| /solve Integration | `.claude/commands/solve.md` | TO UPDATE | P0 |
| Neural Memory Backend | `neural_memory.py` | FUTURE | P3 |

The memory system is foundational. Without it, the system doesn't learn. Implementing Flywheel first ensures the loop is closed.
