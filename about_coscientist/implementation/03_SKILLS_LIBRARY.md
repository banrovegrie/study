# Skills Library Implementation

## Overview

From ARCHITECTURE.md:

```
┌──────────────┐
│    SKILLS    │
│              │
│  reusable    │
│  strategies  │
└──────────────┘
```

Skills are **reusable reasoning strategies** that can be:
1. **Invoked** - Applied to a specific problem/goal
2. **Composed** - Combined to form meta-strategies
3. **Learned** - Discovered from successful proofs
4. **Tracked** - Success rates monitored per domain

From RESEARCH.md:
- **LEGO-Prover**: Growing library of verified lemmas as skills
- **TacMiner**: Auto-discover tactics from Tactic Dependence Graphs
- **Meta-tactics in Lean 4**: Code that generates tactics

---

## Current State

### What Exists
- `approaches.py` - 30+ techniques with descriptions, when-to-use, steps
- These are **static descriptions**, not callable objects
- No composition, no discovery, no per-skill tracking

### What's Missing
- Skills as callable units with execution logic
- Skill applicability predicates (can this skill apply to this goal?)
- Skill composition (combine skills into meta-skills)
- Skill discovery (extract skills from successful proofs)
- Per-skill success tracking (distinct from technique tracker)

---

## Design Principles

### Core Philosophy: Decontextualized Techniques

**The Problem with Current Literature Search:**
Current SOTA retrieval finds papers/proofs *about* similar topics—surface-level keyword/semantic matching. This misses the deeper structure: techniques that transfer across domains.

**Our Approach: Extract the Transferable Atoms**

A "skill" in our system is a **decontextualized technique**—the method/trick/idea stripped from its original problem context, indexed by its *structural applicability*, not its topic.

```
Traditional Search:              Our Approach:
─────────────────────           ─────────────────────
"Find papers about primes"       "Find techniques applicable to
                                  'prove X has property P for all n'"

Returns: Papers mentioning        Returns: Induction, descent,
         primes                            probabilistic method, etc.
                                          (from ANY domain)
```

**Why This Matters:**

- Probabilistic method (Erdős) → born in combinatorics, now used in number theory, CS, analysis
- Fourier techniques → number theory (circle method), signal processing, ML
- Fixed-point theorems → analysis, economics, game theory, CS, topology
- Compression arguments → information theory, learning theory, complexity

The skill library doesn't organize by topic ("number theory skills"). It organizes by **structural pattern** ("techniques for proving ∀n statements", "techniques for showing existence").

**Extraction Process:**

```
Paper/Proof → Identify core technique → Strip context → Index by:
  1. Goal pattern (what it proves)
  2. Structure pattern (how it proves)
  3. Transfer history (where else it worked)
```

**Retrieval by Applicability:**

When solving a new problem, we don't ask "what papers are about similar topics?"
We ask "what techniques have solved structurally similar goals?"

This enables discoveries like: "This optimization problem has the same structure as that number theory proof—try descent."

### Skills vs Techniques vs Tactics

| Concept | Level | Example | Callable? |
|---------|-------|---------|-----------|
| **Tactic** | Low | `ring`, `simp`, `induction` | Yes (Lean) |
| **Technique** | Medium | "Use modular arithmetic" | Description |
| **Skill** | High | "Prove by contradiction via parity" | Yes (our system) |

Skills are **orchestrations of techniques and tactics** with:
- Preconditions (when to apply)
- Steps (how to apply)
- Postconditions (what's achieved)

### Skill Composition

```
Meta-Skill: "Solve Diophantine by descent"
├── Skill: "Assume minimal counterexample"
│   ├── Technique: "Proof by contradiction"
│   └── Tactic: "by_contra"
├── Skill: "Find smaller counterexample"
│   ├── Technique: "Algebraic manipulation"
│   └── Tactics: "ring", "omega"
└── Skill: "Derive contradiction"
    └── Tactic: "linarith"
```

---

## Implementation Plan

### Core Skill Class

**File**: `src/skills/skill.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import uuid
import json
from datetime import datetime

class SkillType(Enum):
    TACTIC = "tactic"           # Single Lean tactic
    TECHNIQUE = "technique"     # Multi-step technique
    META_SKILL = "meta_skill"   # Composition of skills
    LEMMA = "lemma"             # Proven lemma to apply

@dataclass
class SkillStats:
    """Tracking statistics for a skill."""
    total_applications: int = 0
    successful_applications: int = 0
    partial_success: int = 0  # Made progress but didn't fully solve
    failures: int = 0

    # Domain-specific stats
    by_domain: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Time tracking
    total_time_spent: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_applications == 0:
            return 0.5  # Prior
        return (self.successful_applications + 0.5 * self.partial_success) / self.total_applications

    def record(self, success: bool, partial: bool, domain: str, time_spent: float):
        self.total_applications += 1
        self.total_time_spent += time_spent

        if success:
            self.successful_applications += 1
        elif partial:
            self.partial_success += 1
        else:
            self.failures += 1

        if domain not in self.by_domain:
            self.by_domain[domain] = {"success": 0, "partial": 0, "fail": 0}

        if success:
            self.by_domain[domain]["success"] += 1
        elif partial:
            self.by_domain[domain]["partial"] += 1
        else:
            self.by_domain[domain]["fail"] += 1

    def to_dict(self) -> dict:
        return {
            "total_applications": self.total_applications,
            "successful_applications": self.successful_applications,
            "partial_success": self.partial_success,
            "failures": self.failures,
            "by_domain": self.by_domain,
            "total_time_spent": self.total_time_spent
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillStats":
        stats = cls()
        stats.total_applications = data.get("total_applications", 0)
        stats.successful_applications = data.get("successful_applications", 0)
        stats.partial_success = data.get("partial_success", 0)
        stats.failures = data.get("failures", 0)
        stats.by_domain = data.get("by_domain", {})
        stats.total_time_spent = data.get("total_time_spent", 0.0)
        return stats


@dataclass
class Skill:
    """
    A reusable reasoning strategy.

    Skills can be:
    - Applied to goals/problems
    - Composed into meta-skills
    - Tracked for effectiveness
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    skill_type: SkillType = SkillType.TECHNIQUE

    # Applicability
    applicable_domains: List[str] = field(default_factory=list)
    goal_patterns: List[str] = field(default_factory=list)  # Regex patterns
    preconditions: List[str] = field(default_factory=list)   # Natural language

    # Execution
    steps: List[str] = field(default_factory=list)
    tactics_sequence: List[str] = field(default_factory=list)  # Lean tactics
    key_insights: List[str] = field(default_factory=list)

    # Composition
    component_skills: List[str] = field(default_factory=list)  # Skill IDs
    parent_skills: List[str] = field(default_factory=list)     # Skills that use this

    # Tracking
    stats: SkillStats = field(default_factory=SkillStats)

    # Metadata
    source: str = ""  # Where this skill came from
    created_at: datetime = field(default_factory=datetime.now)
    examples: List[str] = field(default_factory=list)  # Problem IDs where used

    def matches_goal(self, goal: str) -> bool:
        """Check if this skill is applicable to a goal."""
        import re
        for pattern in self.goal_patterns:
            if re.search(pattern, goal, re.IGNORECASE):
                return True
        return False

    def matches_domain(self, domain: str) -> bool:
        """Check if this skill applies to a domain."""
        if not self.applicable_domains:
            return True  # Applies to all
        return domain in self.applicable_domains

    def applicability_score(self, goal: str, domain: str) -> float:
        """
        Score how applicable this skill is.

        Considers:
        - Goal pattern match
        - Domain match
        - Historical success in domain
        """
        score = 0.0

        # Goal match
        if self.matches_goal(goal):
            score += 0.4

        # Domain match
        if self.matches_domain(domain):
            score += 0.2

            # Domain-specific success rate
            if domain in self.stats.by_domain:
                domain_stats = self.stats.by_domain[domain]
                domain_total = sum(domain_stats.values())
                if domain_total > 0:
                    domain_success = (domain_stats["success"] + 0.5 * domain_stats["partial"]) / domain_total
                    score += 0.3 * domain_success

        # Overall success rate
        score += 0.1 * self.stats.success_rate

        return score

    def to_prompt(self) -> str:
        """Generate a prompt describing this skill."""
        lines = [
            f"SKILL: {self.name}",
            f"Type: {self.skill_type.value}",
            "",
            f"Description: {self.description}",
            "",
            "Steps:",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")

        if self.tactics_sequence:
            lines.append("")
            lines.append(f"Lean tactics: {' >> '.join(self.tactics_sequence)}")

        if self.key_insights:
            lines.append("")
            lines.append("Key insights:")
            for insight in self.key_insights:
                lines.append(f"  - {insight}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type.value,
            "applicable_domains": self.applicable_domains,
            "goal_patterns": self.goal_patterns,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "tactics_sequence": self.tactics_sequence,
            "key_insights": self.key_insights,
            "component_skills": self.component_skills,
            "parent_skills": self.parent_skills,
            "stats": self.stats.to_dict(),
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "examples": self.examples
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        skill = cls()
        skill.id = data.get("id", skill.id)
        skill.name = data.get("name", "")
        skill.description = data.get("description", "")
        skill.skill_type = SkillType(data.get("skill_type", "technique"))
        skill.applicable_domains = data.get("applicable_domains", [])
        skill.goal_patterns = data.get("goal_patterns", [])
        skill.preconditions = data.get("preconditions", [])
        skill.steps = data.get("steps", [])
        skill.tactics_sequence = data.get("tactics_sequence", [])
        skill.key_insights = data.get("key_insights", [])
        skill.component_skills = data.get("component_skills", [])
        skill.parent_skills = data.get("parent_skills", [])
        skill.stats = SkillStats.from_dict(data.get("stats", {}))
        skill.source = data.get("source", "")
        skill.examples = data.get("examples", [])
        return skill
```

---

### Skills Library

**File**: `src/skills/library.py`

```python
"""
The Skills Library: A searchable, growing collection of reusable strategies.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import re
from .skill import Skill, SkillType, SkillStats

class SkillsLibrary:
    """
    Manages the collection of skills.

    Features:
    - Search by goal/domain
    - Composition of meta-skills
    - Discovery from proofs
    - Persistence
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.skills: Dict[str, Skill] = {}
        self._load()

    def _load(self):
        """Load skills from persistent storage."""
        skills_file = self.data_dir / "skills.json"
        if skills_file.exists():
            with open(skills_file) as f:
                data = json.load(f)
                self.skills = {
                    s["id"]: Skill.from_dict(s)
                    for s in data.get("skills", [])
                }
        else:
            self._initialize_core_skills()

    def save(self):
        """Save skills to persistent storage."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / "skills.json", "w") as f:
            json.dump({
                "skills": [s.to_dict() for s in self.skills.values()]
            }, f, indent=2, default=str)

    def _initialize_core_skills(self):
        """Initialize with core skills from approaches.py patterns."""
        core_skills = [
            # Proof Methods
            Skill(
                name="proof_by_contradiction",
                description="Assume the negation, derive a contradiction",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["*"],
                goal_patterns=[r"¬|not|no.*exists|impossible|prove.*false"],
                preconditions=["Goal involves negation or non-existence"],
                steps=[
                    "Assume the negation of what we want to prove",
                    "Derive consequences from this assumption",
                    "Show these consequences lead to a contradiction",
                    "Conclude the original statement must be true"
                ],
                tactics_sequence=["by_contra", "push_neg", "...", "exact ⟨_, _⟩ <;> linarith"],
                key_insights=["Look for parity, divisibility, or size contradictions"],
                source="core"
            ),
            Skill(
                name="strong_induction",
                description="Prove P(n) assuming P(k) for all k < n",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["number_theory", "combinatorics"],
                goal_patterns=[r"∀.*n.*ℕ|for all.*natural|for all.*positive"],
                preconditions=["Goal is universally quantified over naturals", "Need multiple previous cases"],
                steps=[
                    "Establish base case(s)",
                    "For inductive step, assume P(k) for all k < n",
                    "Prove P(n) using these assumptions",
                    "Apply well-founded recursion or Nat.strong_induction_on"
                ],
                tactics_sequence=["induction n using Nat.strong_induction_on with n ih", "...", "exact ih _ _"],
                key_insights=["Use when simple induction isn't strong enough"],
                source="core"
            ),
            Skill(
                name="induction",
                description="Prove base case, then P(n) → P(n+1)",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["number_theory", "combinatorics", "algebra"],
                goal_patterns=[r"∀.*n.*ℕ|for all.*n|sum.*i=|product.*i="],
                preconditions=["Goal is universally quantified over naturals"],
                steps=[
                    "Prove base case P(0) or P(1)",
                    "Assume P(n) (inductive hypothesis)",
                    "Prove P(n+1) using the hypothesis",
                    "Conclude by induction"
                ],
                tactics_sequence=["induction n with n ih", "· simp", "· simp [ih]"],
                key_insights=["Check if base case is n=0 or n=1", "IH appears as 'ih' in Lean"],
                source="core"
            ),
            Skill(
                name="cases_on_parity",
                description="Split into even and odd cases",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["number_theory"],
                goal_patterns=[r"even|odd|2.*\||divisib|parity"],
                preconditions=["Problem involves integers or naturals", "Parity seems relevant"],
                steps=[
                    "Consider two cases: n is even or n is odd",
                    "For even: write n = 2k for some integer k",
                    "For odd: write n = 2k + 1 for some integer k",
                    "Prove the goal in each case"
                ],
                tactics_sequence=["rcases Nat.even_or_odd n with ⟨k, rfl⟩ | ⟨k, rfl⟩"],
                key_insights=["Use Int.even_or_odd for integers", "Often simplifies mod 2 arguments"],
                source="core"
            ),
            Skill(
                name="modular_arithmetic",
                description="Work modulo some number to find constraints",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["number_theory"],
                goal_patterns=[r"mod|≡|divis|remainder|congruent"],
                preconditions=["Divisibility is relevant", "Looking for periodicity"],
                steps=[
                    "Identify a useful modulus (often from the problem)",
                    "Reduce all quantities modulo this number",
                    "Use properties: a ≡ b → a + c ≡ b + c, etc.",
                    "Derive the desired conclusion"
                ],
                tactics_sequence=["omega", "mod_cast", "push_cast", "ring_nf", "norm_num"],
                key_insights=["Common moduli: 2, 3, 4, p (primes)", "mod_cast converts between ℕ and ℤ"],
                source="core"
            ),
            Skill(
                name="am_gm",
                description="Apply AM-GM inequality",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["algebra", "analysis"],
                goal_patterns=[r"≥|≤|sum.*product|product.*sum|minimum|maximum|inequality"],
                preconditions=["All quantities are non-negative", "Inequality involves sums and products"],
                steps=[
                    "Identify the quantities to apply AM-GM to",
                    "Verify all quantities are non-negative",
                    "Apply: (a₁ + ... + aₙ)/n ≥ (a₁·...·aₙ)^(1/n)",
                    "Equality holds when a₁ = ... = aₙ"
                ],
                tactics_sequence=["nlinarith [mul_self_nonneg _, sq_nonneg _]"],
                key_insights=["Choose terms so equality gives desired bound", "May need weighted AM-GM"],
                source="core"
            ),
            Skill(
                name="cauchy_schwarz",
                description="Apply Cauchy-Schwarz inequality",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["algebra", "analysis", "linear_algebra"],
                goal_patterns=[r"sum.*square|inner.*product|dot.*product|≤.*sum"],
                preconditions=["Have sums of products", "Looking for upper/lower bound"],
                steps=[
                    "Identify sequences (aᵢ) and (bᵢ)",
                    "Apply: (Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²)",
                    "Choose aᵢ, bᵢ strategically",
                    "Equality when aᵢ/bᵢ constant"
                ],
                tactics_sequence=["have h := inner_mul_le_norm_mul_norm _ _", "nlinarith [sq_nonneg _]"],
                key_insights=["Engel form: Σ(aᵢ²/bᵢ) ≥ (Σaᵢ)²/Σbᵢ", "Often set one sequence to constants"],
                source="core"
            ),
            Skill(
                name="telescoping",
                description="Simplify sum/product by cancellation",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["algebra", "combinatorics"],
                goal_patterns=[r"sum|Σ|∑|product|Π|∏|closed.*form"],
                preconditions=["Sum or product of similar terms", "Terms might cancel"],
                steps=[
                    "Try to write aₙ = f(n) - f(n-1) or aₙ = f(n)/f(n-1)",
                    "Sum/product collapses to boundary terms",
                    "For sums: Σ(f(n) - f(n-1)) = f(N) - f(0)",
                    "For products: Π(f(n)/f(n-1)) = f(N)/f(0)"
                ],
                tactics_sequence=["simp only [Finset.sum_range_sub']", "ring"],
                key_insights=["Partial fractions often reveal telescoping structure", "1/(n(n+1)) = 1/n - 1/(n+1)"],
                source="core"
            ),
            Skill(
                name="pigeonhole",
                description="If n+1 objects in n boxes, some box has 2+",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["combinatorics"],
                goal_patterns=[r"exist.*two|at least|must.*same|some.*equal|coincide"],
                preconditions=["More objects than categories", "Looking for collision/repeat"],
                steps=[
                    "Identify the 'pigeons' (objects)",
                    "Identify the 'holes' (categories)",
                    "Verify pigeons > holes",
                    "Conclude some hole has multiple pigeons"
                ],
                tactics_sequence=["apply Finset.exists_lt_card_fiber_of_mul_lt_card", "..."],
                key_insights=["Generalized: n objects in k holes → some hole has ⌈n/k⌉", "Think creatively about what are pigeons/holes"],
                source="core"
            ),
            Skill(
                name="construction",
                description="Construct an explicit example",
                skill_type=SkillType.TECHNIQUE,
                applicable_domains=["*"],
                goal_patterns=[r"∃|exists|find|construct|give.*example"],
                preconditions=["Goal requires exhibiting an object"],
                steps=[
                    "Understand what properties the object needs",
                    "Try simple cases: small numbers, known sequences",
                    "Build up from components if complex",
                    "Verify all required properties"
                ],
                tactics_sequence=["use _", "constructor", "· exact _", "· exact _"],
                key_insights=["Start with simplest possible construction", "Parametric families often work"],
                source="core"
            ),
            # Meta-skills (compositions)
            Skill(
                name="descent",
                description="Infinite descent / minimal counterexample",
                skill_type=SkillType.META_SKILL,
                applicable_domains=["number_theory"],
                goal_patterns=[r"no.*solution|impossible|diophantine"],
                preconditions=["Dealing with integer equations", "Contradiction approach seems promising"],
                steps=[
                    "Assume a solution exists",
                    "Among all solutions, consider one that minimizes some measure",
                    "Show this 'minimal' solution leads to a smaller one",
                    "Contradiction: no truly minimal solution exists"
                ],
                component_skills=["proof_by_contradiction", "modular_arithmetic"],
                key_insights=["Choose measure carefully (often sum of squares)", "Vieta jumping is a variant"],
                source="core"
            ),
        ]

        for skill in core_skills:
            self.skills[skill.id] = skill

        self.save()

    # === Search ===

    def search(self, goal: str, domain: str = None, k: int = 5) -> List[Tuple[Skill, float]]:
        """
        Search for applicable skills.

        Returns list of (skill, score) tuples, sorted by score descending.
        """
        scored = []
        for skill in self.skills.values():
            score = skill.applicability_score(goal, domain or "*")
            if score > 0:
                scored.append((skill, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def get_by_name(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        for skill in self.skills.values():
            if skill.name == name:
                return skill
        return None

    def get_by_id(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.skills.get(skill_id)

    def get_by_type(self, skill_type: SkillType) -> List[Skill]:
        """Get all skills of a given type."""
        return [s for s in self.skills.values() if s.skill_type == skill_type]

    def get_by_domain(self, domain: str) -> List[Skill]:
        """Get all skills applicable to a domain."""
        return [s for s in self.skills.values() if s.matches_domain(domain)]

    # === Skill Management ===

    def add(self, skill: Skill) -> str:
        """Add a new skill to the library."""
        self.skills[skill.id] = skill
        self.save()
        return skill.id

    def update_stats(self, skill_id: str, success: bool, partial: bool,
                     domain: str, time_spent: float):
        """Update skill statistics after use."""
        if skill_id in self.skills:
            self.skills[skill_id].stats.record(success, partial, domain, time_spent)
            self.save()

    def add_example(self, skill_id: str, problem_id: str):
        """Add a problem as an example of skill use."""
        if skill_id in self.skills:
            if problem_id not in self.skills[skill_id].examples:
                self.skills[skill_id].examples.append(problem_id)
                self.save()

    # === Composition ===

    def compose(self, name: str, description: str, component_ids: List[str],
                goal_patterns: List[str] = None, domains: List[str] = None) -> Skill:
        """
        Create a meta-skill from component skills.
        """
        components = [self.skills[cid] for cid in component_ids if cid in self.skills]

        # Aggregate steps
        all_steps = []
        all_tactics = []
        all_insights = []

        for comp in components:
            all_steps.extend(comp.steps)
            all_tactics.extend(comp.tactics_sequence)
            all_insights.extend(comp.key_insights)

        meta_skill = Skill(
            name=name,
            description=description,
            skill_type=SkillType.META_SKILL,
            applicable_domains=domains or ["*"],
            goal_patterns=goal_patterns or [],
            steps=all_steps,
            tactics_sequence=all_tactics,
            key_insights=list(set(all_insights)),  # Dedupe
            component_skills=component_ids,
            source="composed"
        )

        # Update parent references
        for comp in components:
            if meta_skill.id not in comp.parent_skills:
                comp.parent_skills.append(meta_skill.id)

        self.skills[meta_skill.id] = meta_skill
        self.save()
        return meta_skill

    # === Discovery ===

    def discover_from_proof(self, proof_trace: dict) -> Optional[Skill]:
        """
        Attempt to extract a skill from a successful proof.

        proof_trace should contain:
        - problem_id: str
        - goal: str
        - domain: str
        - steps: List[str]  # Reasoning steps
        - tactics: List[str]  # Lean tactics used
        - key_insight: str
        """
        # Check if this pattern is already captured
        for skill in self.skills.values():
            # Similar tactics sequence?
            if skill.tactics_sequence == proof_trace.get("tactics", []):
                # Just add as example
                self.add_example(skill.id, proof_trace["problem_id"])
                return None

        # Create new skill
        new_skill = Skill(
            name=f"discovered_{len(self.skills)}",
            description=f"Discovered from {proof_trace['problem_id']}",
            skill_type=SkillType.TECHNIQUE,
            applicable_domains=[proof_trace.get("domain", "*")],
            goal_patterns=[],  # Would need pattern extraction
            steps=proof_trace.get("steps", []),
            tactics_sequence=proof_trace.get("tactics", []),
            key_insights=[proof_trace.get("key_insight", "")],
            source=f"discovered:{proof_trace['problem_id']}",
            examples=[proof_trace["problem_id"]]
        )

        self.skills[new_skill.id] = new_skill
        self.save()
        return new_skill

    # === Retrieval for LLM ===

    def get_prompts_for_goal(self, goal: str, domain: str, k: int = 3) -> str:
        """
        Generate skill prompts for LLM context.
        """
        skills = self.search(goal, domain, k)
        if not skills:
            return "No specific skills found for this goal."

        prompts = []
        for skill, score in skills:
            prompts.append(f"[Score: {score:.2f}]")
            prompts.append(skill.to_prompt())
            prompts.append("")

        return "\n".join(prompts)

    # === Statistics ===

    def get_stats(self) -> dict:
        """Get library statistics."""
        by_type = {}
        for skill in self.skills.values():
            t = skill.skill_type.value
            by_type[t] = by_type.get(t, 0) + 1

        top_skills = sorted(
            self.skills.values(),
            key=lambda s: s.stats.success_rate * (s.stats.total_applications + 1),
            reverse=True
        )[:5]

        return {
            "total_skills": len(self.skills),
            "by_type": by_type,
            "total_applications": sum(s.stats.total_applications for s in self.skills.values()),
            "top_skills": [(s.name, s.stats.success_rate) for s in top_skills]
        }
```

---

### Skill Executor

**File**: `src/skills/executor.py`

```python
"""
Skill execution: turning skills into actions.
"""

from typing import Optional, Tuple
from .skill import Skill, SkillType
from .library import SkillsLibrary

class SkillExecutor:
    """
    Executes skills in the context of problem solving.

    This bridges the gap between skill descriptions and actual actions.
    """

    def __init__(self, library: SkillsLibrary):
        self.library = library

    def execute(self, skill: Skill, context: dict) -> Tuple[bool, dict]:
        """
        Execute a skill given context.

        context should contain:
        - goal: str (current goal)
        - hypotheses: List[str] (available assumptions)
        - domain: str

        Returns (success, result_dict)
        """
        # For tactics, we generate Lean code
        if skill.skill_type == SkillType.TACTIC:
            return self._execute_tactic(skill, context)

        # For techniques, we generate a reasoning prompt
        elif skill.skill_type == SkillType.TECHNIQUE:
            return self._execute_technique(skill, context)

        # For meta-skills, we execute components in sequence
        elif skill.skill_type == SkillType.META_SKILL:
            return self._execute_meta_skill(skill, context)

        # For lemmas, we generate an application
        elif skill.skill_type == SkillType.LEMMA:
            return self._execute_lemma(skill, context)

        return False, {"error": "Unknown skill type"}

    def _execute_tactic(self, skill: Skill, context: dict) -> Tuple[bool, dict]:
        """Execute a single tactic."""
        tactic_code = " <;> ".join(skill.tactics_sequence) if skill.tactics_sequence else skill.name
        return True, {
            "type": "tactic",
            "code": tactic_code,
            "skill": skill.name
        }

    def _execute_technique(self, skill: Skill, context: dict) -> Tuple[bool, dict]:
        """Execute a multi-step technique."""
        prompt = self._generate_technique_prompt(skill, context)
        return True, {
            "type": "technique",
            "prompt": prompt,
            "steps": skill.steps,
            "tactics": skill.tactics_sequence,
            "skill": skill.name
        }

    def _execute_meta_skill(self, skill: Skill, context: dict) -> Tuple[bool, dict]:
        """Execute a composed meta-skill."""
        results = []
        current_context = context.copy()

        for comp_id in skill.component_skills:
            component = self.library.get_by_id(comp_id)
            if component:
                success, result = self.execute(component, current_context)
                results.append((component.name, result))
                if not success:
                    break
                # Update context with result
                # (In practice, this would integrate with the proof state)

        return True, {
            "type": "meta_skill",
            "components": results,
            "skill": skill.name
        }

    def _execute_lemma(self, skill: Skill, context: dict) -> Tuple[bool, dict]:
        """Apply a lemma."""
        return True, {
            "type": "lemma",
            "application": f"apply {skill.name}",
            "skill": skill.name
        }

    def _generate_technique_prompt(self, skill: Skill, context: dict) -> str:
        """Generate a reasoning prompt for a technique."""
        goal = context.get("goal", "")
        hypotheses = context.get("hypotheses", [])

        lines = [
            f"Apply the technique: {skill.name}",
            "",
            f"Goal: {goal}",
            "",
            "Available hypotheses:",
        ]
        for h in hypotheses:
            lines.append(f"  - {h}")

        lines.extend([
            "",
            "Steps to follow:",
        ])
        for i, step in enumerate(skill.steps, 1):
            lines.append(f"  {i}. {step}")

        if skill.key_insights:
            lines.extend([
                "",
                "Key insights:",
            ])
            for insight in skill.key_insights:
                lines.append(f"  - {insight}")

        return "\n".join(lines)
```

---

### Integration

**File**: `src/skills/__init__.py`

```python
from .skill import Skill, SkillType, SkillStats
from .library import SkillsLibrary
from .executor import SkillExecutor

__all__ = ["Skill", "SkillType", "SkillStats", "SkillsLibrary", "SkillExecutor"]
```

---

## Usage in /solve

```python
from src.skills import SkillsLibrary, SkillExecutor

# Initialize
library = SkillsLibrary(Path("data/memory/skills"))
executor = SkillExecutor(library)

# Search for applicable skills
skills = library.search(
    goal="∀ n : ℕ, n^2 + 1 > n",
    domain="algebra",
    k=3
)

# Get prompts for LLM
skill_context = library.get_prompts_for_goal(goal, domain)

# Execute a skill
skill = skills[0][0]
success, result = executor.execute(skill, {
    "goal": goal,
    "hypotheses": ["n : ℕ"],
    "domain": "algebra"
})

# Update stats after use
library.update_stats(
    skill.id,
    success=True,
    partial=False,
    domain="algebra",
    time_spent=5.0
)
```

---

## Persistence

### File: `data/memory/skills/skills.json`

```json
{
  "skills": [
    {
      "id": "abc123",
      "name": "proof_by_contradiction",
      "description": "Assume the negation, derive a contradiction",
      "skill_type": "technique",
      "applicable_domains": ["*"],
      "goal_patterns": ["¬|not|no.*exists"],
      "steps": ["Assume negation", "Derive consequences", "Find contradiction"],
      "tactics_sequence": ["by_contra", "push_neg"],
      "stats": {
        "total_applications": 15,
        "successful_applications": 10,
        "by_domain": {"number_theory": {"success": 5, "fail": 2}}
      },
      "examples": ["putnam_2024_a1", "putnam_2023_b2"]
    }
  ]
}
```

---

## Summary

| Component | File | Purpose | Priority |
|-----------|------|---------|----------|
| Skill | `skill.py` | Skill data structure | P1 |
| SkillStats | `skill.py` | Per-skill tracking | P1 |
| SkillsLibrary | `library.py` | Search and management | P1 |
| SkillExecutor | `executor.py` | Execute skills | P1 |
| Core Skills | `library.py` | Initialize from approaches | P1 |
| Skill Discovery | `library.py` | Extract from proofs | P2 |

Skills transform static technique descriptions into dynamic, trackable, composable strategies.
