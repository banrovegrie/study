# Search and MCTS Implementation

## Relationship to Orchestration

**Claude Code is the orchestrator.** This spec defines:
1. **Search strategies** Claude Code can employ
2. **MCTS as a tool** Claude Code can invoke for algorithmic search
3. **Patterns** for exploration, not a separate control flow

Claude Code decides WHEN and HOW to search. This spec provides the tools and strategies.

---

## Overview

From RESEARCH.md:

> **Test-time compute scaling** (AlphaProof, o3): Spend more compute on hard problems via RL and tree search. Generate problem variations to improve sample efficiency. Scales multiple orders of magnitude with predictable performance gains.

The key insight: **harder problems should get more compute**. This is adaptive, not fixed.

---

## Current State

### What Exists
- `MCTSNode` in `mcts.py` - Node structure with visit counts, rewards
- `ProofState` in `state.py` - Immutable state snapshots
- `approaches.py` - 30+ techniques as possible actions

### What's Missing
- **MCTS traversal algorithm** - Selection, expansion, simulation, backpropagation
- **Value function** - Estimate proof completion probability
- **Difficulty estimation** - Adaptive compute allocation
- **Problem variation generation** - Test-time RL training data

---

## Design: Adaptive Proof Search

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ADAPTIVE PROOF SEARCH                               │
│                                                                         │
│   ┌───────────────┐                                                     │
│   │   ESTIMATE    │  How hard is this problem?                          │
│   │   DIFFICULTY  │  → Easy: Quick search                               │
│   └───────┬───────┘  → Hard: Full MCTS + variations                     │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │                                                               │     │
│   │                        MCTS SEARCH                            │     │
│   │                                                               │     │
│   │    SELECT ──▶ EXPAND ──▶ SIMULATE ──▶ BACKPROP               │     │
│   │       │                                     │                 │     │
│   │       └──────────────◀──────────────────────┘                 │     │
│   │                                                               │     │
│   │    UCB selection:  exploit promising + explore uncertain      │     │
│   │    Expansion:      try untried approaches                     │     │
│   │    Simulation:     quick rollout / value function             │     │
│   │    Backprop:       update node statistics                     │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐                                                     │
│   │   SOLUTION    │  Return best path found                             │
│   │   or TIMEOUT  │  or best partial result                             │
│   └───────────────┘                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### MCTS Core

**File**: `src/search/mcts.py` (complete rewrite)

```python
"""
Monte Carlo Tree Search for theorem proving.

Based on:
- AlphaProof's test-time RL approach
- CMCTS (constrained MCTS) from rStar-Math
- OmegaPRM for process reward modeling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import math
import random
import time
from datetime import datetime

from .state import ProofState
from ..skills import SkillsLibrary

@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    # UCB exploration weight
    exploration_weight: float = 1.414  # sqrt(2) is theoretical optimum

    # Search limits
    max_iterations: int = 1000
    max_depth: int = 20
    time_limit_seconds: float = 300.0

    # Simulation
    simulation_depth: int = 5
    use_value_function: bool = True

    # Progressive widening
    progressive_widening: bool = True
    widening_alpha: float = 0.5  # n^alpha determines max children

    # Pruning
    prune_threshold: float = 0.01  # Prune nodes with very low value
    min_visits_to_prune: int = 10

@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    id: str
    state: ProofState
    parent: Optional["MCTSNode"] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)

    # Statistics
    visits: int = 0
    total_reward: float = 0.0
    squared_reward: float = 0.0  # For variance calculation

    # Action that led here
    action: Optional[str] = None  # The approach/tactic used

    # Untried actions
    untried_actions: List[str] = field(default_factory=list)

    # Process Reward Model score (if available)
    prm_score: Optional[float] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_solved or self.state.is_stuck

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def avg_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    @property
    def reward_variance(self) -> float:
        if self.visits < 2:
            return float('inf')
        mean = self.avg_reward
        return (self.squared_reward / self.visits) - (mean * mean)

    def ucb_score(self, parent_visits: int, exploration_weight: float) -> float:
        """Calculate UCB1 score for this node."""
        if self.visits == 0:
            return float('inf')  # Always try unvisited

        exploitation = self.avg_reward
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )

        return exploitation + exploration

    def ucb_tuned_score(self, parent_visits: int, exploration_weight: float) -> float:
        """UCB1-Tuned: accounts for variance."""
        if self.visits == 0:
            return float('inf')

        exploitation = self.avg_reward
        variance_term = min(0.25, self.reward_variance + math.sqrt(
            2 * math.log(parent_visits) / self.visits
        ))
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits * variance_term
        )

        return exploitation + exploration


class MCTS:
    """
    Monte Carlo Tree Search for theorem proving.

    The search proceeds in four phases:
    1. SELECT: Traverse tree using UCB to find promising leaf
    2. EXPAND: Add new child node for untried action
    3. SIMULATE: Estimate value via rollout or value function
    4. BACKPROPAGATE: Update statistics along path to root
    """

    def __init__(self, config: MCTSConfig = None,
                 skills_library: SkillsLibrary = None,
                 value_function: Callable[[ProofState], float] = None,
                 action_generator: Callable[[ProofState], List[str]] = None):
        self.config = config or MCTSConfig()
        self.skills = skills_library
        self.value_fn = value_function or self._default_value_function
        self.action_gen = action_generator or self._default_action_generator

        self.root: Optional[MCTSNode] = None
        self.best_solution: Optional[MCTSNode] = None

        # Statistics
        self.total_iterations = 0
        self.total_time = 0.0

    def search(self, initial_state: ProofState) -> Tuple[Optional[ProofState], Dict]:
        """
        Run MCTS search from initial state.

        Returns:
            (best_state, statistics)
        """
        start_time = time.time()

        # Initialize root
        self.root = MCTSNode(
            id="root",
            state=initial_state,
            untried_actions=self.action_gen(initial_state)
        )

        # Main MCTS loop
        iteration = 0
        while iteration < self.config.max_iterations:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > self.config.time_limit_seconds:
                break

            # One iteration of MCTS
            self._iterate()
            iteration += 1

            # Check for solution
            if self.best_solution and self.best_solution.state.is_solved:
                break

        self.total_iterations = iteration
        self.total_time = time.time() - start_time

        # Return best result
        best_node = self._get_best_node()
        stats = self._get_statistics()

        return (best_node.state if best_node else None, stats)

    def _iterate(self):
        """One iteration of MCTS: select, expand, simulate, backprop."""
        # SELECT: Find leaf node
        node = self._select(self.root)

        # EXPAND: Add new child if not terminal
        if not node.is_terminal and not node.is_fully_expanded:
            node = self._expand(node)

        # SIMULATE: Estimate value
        reward = self._simulate(node)

        # BACKPROPAGATE: Update statistics
        self._backpropagate(node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using UCB.

        Traverses down the tree, always picking child with highest UCB score.
        """
        while not node.is_terminal and node.is_fully_expanded:
            if not node.children:
                break

            # Progressive widening: limit children based on visits
            if self.config.progressive_widening:
                max_children = int(math.ceil(
                    node.visits ** self.config.widening_alpha
                ))
                if len(node.children) < max_children and node.untried_actions:
                    break  # Should expand instead

            # Select best child by UCB
            node = self._best_child(node)

        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB-Tuned."""
        best_score = float('-inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_tuned_score(
                node.visits,
                self.config.exploration_weight
            )
            if score > best_score:
                best_score = score
                best_child = child

        return best_child or node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand node by adding a new child.

        Picks an untried action and creates child node.
        """
        if not node.untried_actions:
            return node

        # Pick action (could be random or heuristic-guided)
        action = self._pick_action(node)
        node.untried_actions.remove(action)

        # Apply action to get new state
        new_state = self._apply_action(node.state, action)

        # Create child node
        child = MCTSNode(
            id=f"{node.id}_{action}",
            state=new_state,
            parent=node,
            action=action,
            untried_actions=self.action_gen(new_state) if not new_state.is_solved else []
        )

        node.children[action] = child

        # Track best solution
        if new_state.is_solved:
            if (self.best_solution is None or
                new_state.confidence > self.best_solution.state.confidence):
                self.best_solution = child

        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate/evaluate the value of a node.

        Can use:
        1. Value function (learned or heuristic)
        2. Random rollout
        3. Process Reward Model
        """
        if node.state.is_solved:
            return 1.0

        if node.state.is_stuck:
            return 0.0

        if self.config.use_value_function:
            return self.value_fn(node.state)

        # Random rollout
        return self._random_rollout(node.state)

    def _random_rollout(self, state: ProofState, depth: int = 0) -> float:
        """Random rollout simulation."""
        if state.is_solved:
            return 1.0
        if state.is_stuck or depth >= self.config.simulation_depth:
            return state.confidence

        actions = self.action_gen(state)
        if not actions:
            return state.confidence

        action = random.choice(actions)
        new_state = self._apply_action(state, action)

        return self._random_rollout(new_state, depth + 1)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagate reward up the tree.

        Updates visit counts and reward statistics for all ancestors.
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node.squared_reward += reward * reward
            node = node.parent

    def _pick_action(self, node: MCTSNode) -> str:
        """
        Pick which untried action to try next.

        Can use:
        1. Random selection
        2. Prior from skills library
        3. PRM-guided selection
        """
        if not node.untried_actions:
            return None

        # Use skills library to rank actions
        if self.skills:
            scored = []
            for action in node.untried_actions:
                skill = self.skills.get_by_name(action)
                if skill:
                    score = skill.stats.success_rate
                else:
                    score = 0.5  # Default prior
                scored.append((action, score))

            # Weighted random selection
            total = sum(s for _, s in scored)
            if total > 0:
                r = random.random() * total
                cumsum = 0
                for action, score in scored:
                    cumsum += score
                    if cumsum >= r:
                        return action

        # Fallback: random
        return random.choice(node.untried_actions)

    def _apply_action(self, state: ProofState, action: str) -> ProofState:
        """Apply an action to get a new state."""
        # This would integrate with the actual proof system
        # For now, create a modified state
        new_state = state.branch(approach=action)

        # Simulate some progress/failure
        # In reality, this would run verification
        if random.random() < 0.3:  # 30% chance of progress
            new_state = new_state.mark_progress(confidence=state.confidence + 0.1)
        elif random.random() < 0.1:  # 10% chance of solving
            new_state = new_state.mark_complete()

        return new_state

    def _default_value_function(self, state: ProofState) -> float:
        """Default heuristic value function."""
        # Base value from confidence
        value = state.confidence

        # Bonus for progress
        if state.subgoals_solved:
            value += 0.1 * len(state.subgoals_solved)

        # Penalty for being stuck
        if state.is_stuck:
            value *= 0.5

        # Penalty for depth (prefer shorter proofs)
        value *= 0.95 ** state.depth

        return min(1.0, max(0.0, value))

    def _default_action_generator(self, state: ProofState) -> List[str]:
        """Generate possible actions for a state."""
        # Get recommended approaches from skills
        if self.skills:
            goal = state.current_goal or state.problem_statement
            domain = state.topics[0] if state.topics else None
            skills = self.skills.search(goal, domain, k=10)
            return [s.name for s, _ in skills]

        # Fallback: generic approaches
        return [
            "direct_proof", "induction", "contradiction",
            "cases", "construction", "algebraic_manipulation"
        ]

    def _get_best_node(self) -> Optional[MCTSNode]:
        """Get the best node found (solved or highest value)."""
        if self.best_solution:
            return self.best_solution

        # Find node with highest average reward
        best = None
        best_value = float('-inf')

        def search_best(node):
            nonlocal best, best_value
            if node.avg_reward > best_value and node.visits > 0:
                best_value = node.avg_reward
                best = node
            for child in node.children.values():
                search_best(child)

        if self.root:
            search_best(self.root)

        return best

    def _get_statistics(self) -> Dict:
        """Get search statistics."""
        total_nodes = 0
        max_depth = 0
        solutions_found = 0

        def count_nodes(node, depth=0):
            nonlocal total_nodes, max_depth, solutions_found
            total_nodes += 1
            max_depth = max(max_depth, depth)
            if node.state.is_solved:
                solutions_found += 1
            for child in node.children.values():
                count_nodes(child, depth + 1)

        if self.root:
            count_nodes(self.root)

        return {
            "iterations": self.total_iterations,
            "time_seconds": self.total_time,
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "solutions_found": solutions_found,
            "best_value": self.best_solution.state.confidence if self.best_solution else 0.0
        }

    # === Pruning ===

    def prune_low_value_nodes(self):
        """Prune nodes with consistently low value."""
        def should_prune(node):
            return (node.visits >= self.config.min_visits_to_prune and
                    node.avg_reward < self.config.prune_threshold)

        def prune_subtree(node):
            children_to_remove = []
            for action, child in node.children.items():
                if should_prune(child):
                    children_to_remove.append(action)
                else:
                    prune_subtree(child)

            for action in children_to_remove:
                del node.children[action]

        if self.root:
            prune_subtree(self.root)
```

---

### Difficulty Estimation

**File**: `src/search/difficulty.py`

```python
"""
Problem difficulty estimation for adaptive compute allocation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import re

class Difficulty(Enum):
    TRIVIAL = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4
    BRUTAL = 5

@dataclass
class DifficultyEstimate:
    """Estimated difficulty with reasoning."""
    level: Difficulty
    confidence: float
    reasons: List[str]
    suggested_iterations: int
    suggested_time_seconds: float

class DifficultyEstimator:
    """
    Estimates problem difficulty for compute allocation.

    Factors considered:
    1. Problem source (Putnam A1 vs A6)
    2. Statement complexity
    3. Required techniques
    4. Historical success rates
    """

    def __init__(self, technique_tracker=None, world_model=None):
        self.technique_tracker = technique_tracker
        self.world_model = world_model

        # Compute allocation by difficulty
        self.compute_budget = {
            Difficulty.TRIVIAL: {"iterations": 10, "time": 30},
            Difficulty.EASY: {"iterations": 50, "time": 60},
            Difficulty.MEDIUM: {"iterations": 200, "time": 300},
            Difficulty.HARD: {"iterations": 500, "time": 600},
            Difficulty.VERY_HARD: {"iterations": 1000, "time": 1200},
            Difficulty.BRUTAL: {"iterations": 2000, "time": 3600},
        }

    def estimate(self, problem: dict) -> DifficultyEstimate:
        """
        Estimate difficulty of a problem.

        problem should contain:
        - statement: str
        - source: str (optional, e.g., "putnam_2024_a6")
        - topics: List[str]
        """
        reasons = []
        score = 0.0  # 0 = trivial, 1 = brutal

        # Factor 1: Source-based difficulty
        source = problem.get("source", "")
        source_score, source_reason = self._score_by_source(source)
        score += source_score * 0.4
        if source_reason:
            reasons.append(source_reason)

        # Factor 2: Statement complexity
        statement = problem.get("statement", "")
        complexity_score, complexity_reason = self._score_by_complexity(statement)
        score += complexity_score * 0.3
        if complexity_reason:
            reasons.append(complexity_reason)

        # Factor 3: Required techniques
        topics = problem.get("topics", [])
        technique_score, technique_reason = self._score_by_techniques(topics)
        score += technique_score * 0.2
        if technique_reason:
            reasons.append(technique_reason)

        # Factor 4: Historical success (if available)
        if self.world_model:
            historical_score, historical_reason = self._score_by_history(problem)
            score += historical_score * 0.1
            if historical_reason:
                reasons.append(historical_reason)

        # Convert score to difficulty level
        if score < 0.1:
            level = Difficulty.TRIVIAL
        elif score < 0.25:
            level = Difficulty.EASY
        elif score < 0.5:
            level = Difficulty.MEDIUM
        elif score < 0.7:
            level = Difficulty.HARD
        elif score < 0.9:
            level = Difficulty.VERY_HARD
        else:
            level = Difficulty.BRUTAL

        budget = self.compute_budget[level]

        return DifficultyEstimate(
            level=level,
            confidence=0.7,  # Estimate confidence
            reasons=reasons,
            suggested_iterations=budget["iterations"],
            suggested_time_seconds=budget["time"]
        )

    def _score_by_source(self, source: str) -> tuple:
        """Score based on problem source."""
        source_lower = source.lower()

        # Putnam scoring (A1=easy, A6=hard)
        putnam_difficulty = {
            "a1": 0.1, "b1": 0.1,
            "a2": 0.3, "b2": 0.3,
            "a3": 0.4, "b3": 0.4,
            "a4": 0.6, "b4": 0.6,
            "a5": 0.8, "b5": 0.8,
            "a6": 0.95, "b6": 0.95,
        }

        for position, score in putnam_difficulty.items():
            if position in source_lower:
                return score, f"Putnam {position.upper()} position"

        # IMO problems
        if "imo" in source_lower:
            match = re.search(r'p(\d)', source_lower)
            if match:
                problem_num = int(match.group(1))
                score = min(1.0, problem_num / 6)
                return score, f"IMO Problem {problem_num}"

        return 0.5, None  # Unknown source

    def _score_by_complexity(self, statement: str) -> tuple:
        """Score based on statement complexity."""
        # Length heuristic
        length = len(statement)
        length_score = min(1.0, length / 500)

        # Quantifier depth
        quantifiers = len(re.findall(r'∀|∃|for all|exists|for every', statement, re.I))
        quantifier_score = min(1.0, quantifiers / 4)

        # Multiple parts
        parts = len(re.findall(r'\([a-z]\)|part [a-z]|\d\)', statement, re.I))
        parts_score = min(1.0, parts / 3)

        score = (length_score + quantifier_score + parts_score) / 3

        reasons = []
        if length > 300:
            reasons.append("Long statement")
        if quantifiers > 2:
            reasons.append(f"{quantifiers} quantifiers")
        if parts > 1:
            reasons.append(f"{parts} parts")

        return score, "; ".join(reasons) if reasons else None

    def _score_by_techniques(self, topics: list) -> tuple:
        """Score based on required techniques."""
        hard_topics = {
            "p-adic": 0.8,
            "vieta_jumping": 0.9,
            "lifting_the_exponent": 0.7,
            "generating_functions": 0.6,
            "probabilistic_method": 0.8,
            "algebraic_geometry": 0.9,
        }

        max_score = 0.0
        hard_topic = None

        for topic in topics:
            topic_lower = topic.lower().replace(" ", "_")
            if topic_lower in hard_topics:
                if hard_topics[topic_lower] > max_score:
                    max_score = hard_topics[topic_lower]
                    hard_topic = topic

        if hard_topic:
            return max_score, f"Requires {hard_topic}"

        # Default based on topic count
        return min(1.0, len(topics) * 0.2), None

    def _score_by_history(self, problem: dict) -> tuple:
        """Score based on historical success rates."""
        if not self.world_model:
            return 0.5, None

        # Look for similar problems
        similar = self.world_model.get_similar_problems(
            problem.get("statement", ""), k=5
        )

        if not similar:
            return 0.5, None

        # Calculate average difficulty from similar problems
        solved = sum(1 for p in similar if p.get("status") == "SOLVED")
        solve_rate = solved / len(similar) if similar else 0.5

        score = 1.0 - solve_rate  # Higher solve rate = lower difficulty

        return score, f"Similar problems: {solve_rate:.0%} solved"
```

---

### Beam Search Alternative

**File**: `src/search/beam.py`

```python
"""
Beam search: simpler alternative to MCTS for easier problems.
"""

from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from .state import ProofState

@dataclass
class BeamConfig:
    beam_width: int = 5
    max_depth: int = 15
    early_stop_on_solution: bool = True

class BeamSearch:
    """
    Beam search for proof exploration.

    Simpler than MCTS, good for:
    - Easier problems
    - When we have good heuristics
    - Limited compute budget
    """

    def __init__(self, config: BeamConfig = None,
                 value_function: Callable[[ProofState], float] = None,
                 action_generator: Callable[[ProofState], List[str]] = None):
        self.config = config or BeamConfig()
        self.value_fn = value_function or (lambda s: s.confidence)
        self.action_gen = action_generator

    def search(self, initial_state: ProofState) -> Tuple[Optional[ProofState], dict]:
        """
        Run beam search from initial state.

        Returns:
            (best_state, statistics)
        """
        beam: List[Tuple[float, ProofState]] = [(self.value_fn(initial_state), initial_state)]
        best_solution = None
        iterations = 0

        for depth in range(self.config.max_depth):
            iterations += 1
            candidates = []

            for _, state in beam:
                if state.is_solved:
                    if best_solution is None or state.confidence > best_solution.confidence:
                        best_solution = state
                    if self.config.early_stop_on_solution:
                        return best_solution, {"iterations": iterations, "depth": depth}
                    continue

                if state.is_stuck:
                    continue

                # Generate successors
                actions = self.action_gen(state) if self.action_gen else []
                for action in actions:
                    new_state = state.branch(approach=action)
                    value = self.value_fn(new_state)
                    candidates.append((value, new_state))

            if not candidates:
                break

            # Keep top-k by value
            candidates.sort(key=lambda x: -x[0])
            beam = candidates[:self.config.beam_width]

        # Return best found
        if best_solution:
            return best_solution, {"iterations": iterations, "found_solution": True}

        # Return highest-value non-solution
        if beam:
            return beam[0][1], {"iterations": iterations, "found_solution": False}

        return None, {"iterations": iterations, "found_solution": False}
```

---

### Unified Search Interface

**File**: `src/search/search.py`

```python
"""
Unified search interface with adaptive algorithm selection.
"""

from typing import Optional, Tuple, Dict
from .mcts import MCTS, MCTSConfig
from .beam import BeamSearch, BeamConfig
from .difficulty import DifficultyEstimator, Difficulty
from .state import ProofState

class AdaptiveSearch:
    """
    Adaptive proof search that selects algorithm and compute based on difficulty.
    """

    def __init__(self, skills_library=None, technique_tracker=None, world_model=None):
        self.skills = skills_library
        self.difficulty_estimator = DifficultyEstimator(
            technique_tracker=technique_tracker,
            world_model=world_model
        )

    def search(self, problem: dict, initial_state: ProofState = None) -> Tuple[Optional[ProofState], Dict]:
        """
        Search for a proof using adaptive algorithm selection.

        Args:
            problem: Problem dict with statement, source, topics
            initial_state: Optional starting state

        Returns:
            (solution_state, statistics)
        """
        # Estimate difficulty
        difficulty = self.difficulty_estimator.estimate(problem)

        # Create initial state if not provided
        if initial_state is None:
            initial_state = ProofState(
                problem_statement=problem.get("statement", ""),
                topics=problem.get("topics", [])
            )

        # Select algorithm based on difficulty
        if difficulty.level in [Difficulty.TRIVIAL, Difficulty.EASY]:
            return self._beam_search(initial_state, difficulty)
        else:
            return self._mcts_search(initial_state, difficulty)

    def _beam_search(self, state: ProofState, difficulty) -> Tuple[Optional[ProofState], Dict]:
        """Run beam search for easy problems."""
        config = BeamConfig(
            beam_width=5,
            max_depth=10
        )

        beam = BeamSearch(
            config=config,
            value_function=self._value_function,
            action_generator=self._action_generator
        )

        result, stats = beam.search(state)
        stats["algorithm"] = "beam_search"
        stats["difficulty"] = difficulty.level.name
        return result, stats

    def _mcts_search(self, state: ProofState, difficulty) -> Tuple[Optional[ProofState], Dict]:
        """Run MCTS for harder problems."""
        config = MCTSConfig(
            max_iterations=difficulty.suggested_iterations,
            time_limit_seconds=difficulty.suggested_time_seconds,
            exploration_weight=1.414
        )

        mcts = MCTS(
            config=config,
            skills_library=self.skills,
            value_function=self._value_function,
            action_generator=self._action_generator
        )

        result, stats = mcts.search(state)
        stats["algorithm"] = "mcts"
        stats["difficulty"] = difficulty.level.name
        return result, stats

    def _value_function(self, state: ProofState) -> float:
        """Heuristic value function."""
        if state.is_solved:
            return 1.0
        if state.is_stuck:
            return 0.0
        return state.confidence * (0.95 ** state.depth)

    def _action_generator(self, state: ProofState) -> List[str]:
        """Generate possible actions."""
        if self.skills:
            goal = state.current_goal or state.problem_statement
            domain = state.topics[0] if state.topics else None
            skills = self.skills.search(goal, domain, k=10)
            return [s.name for s, _ in skills]
        return ["direct", "induction", "contradiction", "cases"]
```

---

## Tool Interface

**File**: `src/tools.py` (add)

```python
def tool_search(problem: dict, max_iterations: int = None,
                algorithm: str = "auto") -> dict:
    """
    Search for a proof using adaptive MCTS/beam search.

    Args:
        problem: dict with statement, source, topics
        max_iterations: Override automatic iteration limit
        algorithm: "auto", "mcts", or "beam"

    Returns:
        - success: bool
        - solution: str (if found)
        - confidence: float
        - iterations: int
        - algorithm: str
        - difficulty: str
        - stats: dict
    """
    from src.search.search import AdaptiveSearch
    from src.search.state import ProofState
    from src.skills import SkillsLibrary

    skills = SkillsLibrary(Path(DATA_DIR) / "skills")

    search = AdaptiveSearch(skills_library=skills)

    result_state, stats = search.search(problem)

    if result_state and result_state.is_solved:
        return {
            "success": True,
            "solution": result_state.solution,
            "confidence": result_state.confidence,
            "iterations": stats.get("iterations", 0),
            "algorithm": stats.get("algorithm", "unknown"),
            "difficulty": stats.get("difficulty", "unknown"),
            "stats": stats
        }
    else:
        return {
            "success": False,
            "solution": None,
            "confidence": result_state.confidence if result_state else 0.0,
            "iterations": stats.get("iterations", 0),
            "algorithm": stats.get("algorithm", "unknown"),
            "difficulty": stats.get("difficulty", "unknown"),
            "stats": stats
        }
```

---

## Summary

| Component | File | Purpose | Priority |
|-----------|------|---------|----------|
| MCTSNode | `mcts.py` | Tree node structure | P0 |
| MCTS | `mcts.py` | Full MCTS algorithm | P0 |
| DifficultyEstimator | `difficulty.py` | Adaptive compute allocation | P1 |
| BeamSearch | `beam.py` | Simpler search for easy problems | P1 |
| AdaptiveSearch | `search.py` | Unified interface | P0 |
| tool_search | `tools.py` | Claude Code interface | P0 |

Test-time compute scaling is what separates toy systems from IMO-level performance.
