# Graph of Thought Implementation

## Overview

From ARCHITECTURE.md:

```
                         ┌─────────────────┐
                         │    PROBLEM      │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        ┌───────────┐       ┌───────────┐       ┌───────────┐
        │ Approach A│       │ Approach B│       │ Approach C│
        └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
              │                   │                   │
              ▼                   ▼                   ▼
           [stuck]            [partial]           [promising]
              │                   │                   │
           prune              continue             refine
                                  │                   │
                                  └─────────┬─────────┘
                                            ▼
                                        [merge]
                                            │
                                            ▼
                                       [solution]
```

The Graph of Thought (GoT) enables:
1. **Branching** - Explore multiple approaches in parallel
2. **Pruning** - Abandon dead ends early
3. **Merging** - Combine successful partial results
4. **Backtracking** - Return to promising states when stuck

---

## Current State

### What Exists
- `ProofState` in `state.py` - Immutable state snapshots with parent_id, depth
- `MCTSNode` in `mcts.py` - Basic node structure with visit counts
- No graph structure, no pruning, no merging

### What's Missing
- Graph data structure connecting states
- Branching logic (when to create new branches)
- Pruning criteria (when to abandon a branch)
- Merging logic (how to combine partial results)
- Backtracking mechanism (return to previous good states)

---

## Implementation Plan

### Core Data Structures

**File**: `src/search/graph.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import uuid
import json

class NodeStatus(Enum):
    ACTIVE = "active"           # Currently being explored
    PROMISING = "promising"     # Good progress, continue
    PARTIAL = "partial"         # Some progress, might merge
    STUCK = "stuck"             # No progress possible
    PRUNED = "pruned"           # Abandoned
    SOLVED = "solved"           # Reached solution
    MERGED = "merged"           # Combined into another node

@dataclass
class ThoughtNode:
    """A node in the Graph of Thought."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

    # Content
    approach: str = ""
    hypothesis: str = ""
    reasoning: str = ""
    conclusion: str = ""

    # State
    status: NodeStatus = NodeStatus.ACTIVE
    depth: int = 0

    # Results
    established_facts: List[str] = field(default_factory=list)
    subgoals_solved: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)

    # Metrics
    confidence: float = 0.5
    progress_score: float = 0.0  # 0 = no progress, 1 = solved
    compute_spent: float = 0.0   # Time or tokens spent
    visits: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_terminal(self) -> bool:
        return self.status in [NodeStatus.SOLVED, NodeStatus.PRUNED, NodeStatus.MERGED]

    def is_expandable(self) -> bool:
        return self.status in [NodeStatus.ACTIVE, NodeStatus.PROMISING, NodeStatus.PARTIAL]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "children": self.children,
            "approach": self.approach,
            "hypothesis": self.hypothesis,
            "reasoning": self.reasoning,
            "conclusion": self.conclusion,
            "status": self.status.value,
            "depth": self.depth,
            "established_facts": self.established_facts,
            "subgoals_solved": self.subgoals_solved,
            "errors_encountered": self.errors_encountered,
            "confidence": self.confidence,
            "progress_score": self.progress_score,
            "compute_spent": self.compute_spent,
            "visits": self.visits,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThoughtNode":
        node = cls()
        node.id = data["id"]
        node.parent_id = data.get("parent_id")
        node.children = data.get("children", [])
        node.approach = data.get("approach", "")
        node.hypothesis = data.get("hypothesis", "")
        node.reasoning = data.get("reasoning", "")
        node.conclusion = data.get("conclusion", "")
        node.status = NodeStatus(data.get("status", "active"))
        node.depth = data.get("depth", 0)
        node.established_facts = data.get("established_facts", [])
        node.subgoals_solved = data.get("subgoals_solved", [])
        node.errors_encountered = data.get("errors_encountered", [])
        node.confidence = data.get("confidence", 0.5)
        node.progress_score = data.get("progress_score", 0.0)
        node.compute_spent = data.get("compute_spent", 0.0)
        node.visits = data.get("visits", 0)
        return node


@dataclass
class MergeResult:
    """Result of merging multiple nodes."""
    merged_node_id: str
    source_node_ids: List[str]
    combined_facts: List[str]
    combined_subgoals: List[str]
    merge_reasoning: str


class GraphOfThought:
    """
    The exploration graph for problem solving.

    Implements branching, pruning, and merging of thought chains.
    """

    def __init__(self, problem_id: str):
        self.problem_id = problem_id
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self.solution_ids: List[str] = []
        self.merge_history: List[MergeResult] = []

        # Pruning thresholds (configurable)
        self.max_depth = 20
        self.min_progress_to_continue = 0.1
        self.max_errors_before_prune = 3
        self.confidence_threshold = 0.2

    # === Node Operations ===

    def create_root(self, problem_statement: str) -> ThoughtNode:
        """Create the root node for a problem."""
        root = ThoughtNode(
            approach="problem_analysis",
            hypothesis=problem_statement,
            status=NodeStatus.ACTIVE
        )
        self.nodes[root.id] = root
        self.root_id = root.id
        return root

    def branch(self, parent_id: str, approach: str, hypothesis: str) -> ThoughtNode:
        """Create a new branch from a parent node."""
        parent = self.nodes[parent_id]

        child = ThoughtNode(
            parent_id=parent_id,
            approach=approach,
            hypothesis=hypothesis,
            depth=parent.depth + 1,
            status=NodeStatus.ACTIVE,
            # Inherit established facts
            established_facts=list(parent.established_facts)
        )

        self.nodes[child.id] = child
        parent.children.append(child.id)
        parent.updated_at = datetime.now()

        return child

    def update_node(self, node_id: str, **kwargs):
        """Update a node's state."""
        node = self.nodes[node_id]
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        node.updated_at = datetime.now()

    # === Status Transitions ===

    def mark_progress(self, node_id: str, facts: List[str], subgoals: List[str],
                      confidence: float, reasoning: str):
        """Record progress on a node."""
        node = self.nodes[node_id]
        node.established_facts.extend(facts)
        node.subgoals_solved.extend(subgoals)
        node.confidence = confidence
        node.reasoning = reasoning
        node.visits += 1
        node.updated_at = datetime.now()

        # Calculate progress score
        total_subgoals = len(node.subgoals_solved) + 1  # +1 for main goal
        node.progress_score = len(subgoals) / total_subgoals

        # Update status based on progress
        if node.progress_score > 0.7:
            node.status = NodeStatus.PROMISING
        elif node.progress_score > 0.3:
            node.status = NodeStatus.PARTIAL
        elif confidence < self.confidence_threshold:
            node.status = NodeStatus.STUCK

    def mark_stuck(self, node_id: str, reason: str, error: str = None):
        """Mark a node as stuck."""
        node = self.nodes[node_id]
        node.status = NodeStatus.STUCK
        node.conclusion = reason
        if error:
            node.errors_encountered.append(error)
        node.updated_at = datetime.now()

    def mark_solved(self, node_id: str, solution: str):
        """Mark a node as solved."""
        node = self.nodes[node_id]
        node.status = NodeStatus.SOLVED
        node.conclusion = solution
        node.progress_score = 1.0
        node.confidence = 1.0
        node.updated_at = datetime.now()
        self.solution_ids.append(node_id)

    # === Pruning ===

    def should_prune(self, node_id: str) -> Tuple[bool, str]:
        """
        Determine if a node should be pruned.

        Returns (should_prune, reason)
        """
        node = self.nodes[node_id]

        # Already terminal
        if node.is_terminal():
            return False, "already terminal"

        # Depth limit
        if node.depth > self.max_depth:
            return True, f"exceeded max depth ({self.max_depth})"

        # Too many errors
        if len(node.errors_encountered) >= self.max_errors_before_prune:
            return True, f"too many errors ({len(node.errors_encountered)})"

        # Low confidence after significant exploration
        if node.visits > 3 and node.confidence < self.confidence_threshold:
            return True, f"low confidence ({node.confidence:.2f}) after {node.visits} visits"

        # No progress after multiple visits
        if node.visits > 5 and node.progress_score < self.min_progress_to_continue:
            return True, f"insufficient progress ({node.progress_score:.2f})"

        return False, ""

    def prune(self, node_id: str, reason: str):
        """Prune a node and all its descendants."""
        node = self.nodes[node_id]
        node.status = NodeStatus.PRUNED
        node.conclusion = f"Pruned: {reason}"
        node.updated_at = datetime.now()

        # Recursively prune children
        for child_id in node.children:
            if self.nodes[child_id].status not in [NodeStatus.SOLVED, NodeStatus.MERGED]:
                self.prune(child_id, "parent pruned")

    def auto_prune(self) -> List[str]:
        """Automatically prune nodes that meet pruning criteria."""
        pruned = []
        for node_id, node in self.nodes.items():
            if node.is_expandable():
                should, reason = self.should_prune(node_id)
                if should:
                    self.prune(node_id, reason)
                    pruned.append(node_id)
        return pruned

    # === Merging ===

    def find_mergeable_nodes(self) -> List[List[str]]:
        """
        Find groups of nodes that can be merged.

        Nodes are mergeable if:
        1. They have complementary established facts
        2. They're working on the same or related subgoals
        3. Neither is pruned or solved
        """
        mergeable_groups = []

        # Get all partial nodes
        partial_nodes = [
            n for n in self.nodes.values()
            if n.status in [NodeStatus.PARTIAL, NodeStatus.PROMISING]
        ]

        # Find nodes with overlapping subgoals but different facts
        for i, node1 in enumerate(partial_nodes):
            for node2 in partial_nodes[i+1:]:
                # Check for complementary facts
                facts1 = set(node1.established_facts)
                facts2 = set(node2.established_facts)

                # They have different facts
                if facts1 != facts2 and (facts1 - facts2) and (facts2 - facts1):
                    # Combined facts are more than either alone
                    combined = facts1 | facts2
                    if len(combined) > max(len(facts1), len(facts2)):
                        mergeable_groups.append([node1.id, node2.id])

        return mergeable_groups

    def merge(self, node_ids: List[str], merge_reasoning: str) -> ThoughtNode:
        """
        Merge multiple nodes into a new node.

        The merged node:
        - Combines established facts from all sources
        - Combines solved subgoals
        - Has the highest parent (closest to root)
        """
        nodes = [self.nodes[nid] for nid in node_ids]

        # Find the parent (node closest to root)
        min_depth = min(n.depth for n in nodes)
        best_parent = next(n for n in nodes if n.depth == min_depth)

        # Combine facts and subgoals
        all_facts = set()
        all_subgoals = set()
        for node in nodes:
            all_facts.update(node.established_facts)
            all_subgoals.update(node.subgoals_solved)

        # Create merged node
        merged = ThoughtNode(
            parent_id=best_parent.parent_id,
            approach="merged",
            hypothesis=f"Merged from: {[n.approach for n in nodes]}",
            reasoning=merge_reasoning,
            depth=best_parent.depth,
            established_facts=list(all_facts),
            subgoals_solved=list(all_subgoals),
            status=NodeStatus.PROMISING,
            confidence=max(n.confidence for n in nodes),
            progress_score=max(n.progress_score for n in nodes)
        )

        # Add to graph
        self.nodes[merged.id] = merged

        # Mark source nodes as merged
        for node in nodes:
            node.status = NodeStatus.MERGED
            node.conclusion = f"Merged into {merged.id}"

        # Record merge
        self.merge_history.append(MergeResult(
            merged_node_id=merged.id,
            source_node_ids=node_ids,
            combined_facts=list(all_facts),
            combined_subgoals=list(all_subgoals),
            merge_reasoning=merge_reasoning
        ))

        return merged

    # === Navigation ===

    def get_frontier(self) -> List[ThoughtNode]:
        """Get all expandable nodes (the current frontier)."""
        return [n for n in self.nodes.values() if n.is_expandable()]

    def get_best_node(self) -> Optional[ThoughtNode]:
        """Get the most promising node to expand next."""
        frontier = self.get_frontier()
        if not frontier:
            return None

        # Score by: confidence * progress + exploration bonus
        def score(node):
            exploration_bonus = 1.0 / (node.visits + 1)
            return node.confidence * node.progress_score + 0.3 * exploration_bonus

        return max(frontier, key=score)

    def backtrack_to(self, node_id: str) -> ThoughtNode:
        """
        Backtrack to a previous node.

        Marks all nodes on the path from current frontier to target as candidates
        for re-exploration with different approaches.
        """
        target = self.nodes[node_id]

        # Find all nodes that are descendants of target and mark them for reconsideration
        # (but don't prune - they might be useful for merging)

        return target

    def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        """Get the path from a node back to root."""
        path = []
        current = self.nodes[node_id]
        while current:
            path.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                break
        return list(reversed(path))

    # === Serialization ===

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "root_id": self.root_id,
            "solution_ids": self.solution_ids,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "merge_history": [
                {
                    "merged_node_id": m.merged_node_id,
                    "source_node_ids": m.source_node_ids,
                    "merge_reasoning": m.merge_reasoning
                }
                for m in self.merge_history
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphOfThought":
        graph = cls(data["problem_id"])
        graph.root_id = data.get("root_id")
        graph.solution_ids = data.get("solution_ids", [])
        graph.nodes = {
            nid: ThoughtNode.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }
        return graph

    def save(self, path: str):
        """Save graph to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "GraphOfThought":
        """Load graph from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # === Visualization ===

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of the graph."""
        lines = ["graph TD"]

        status_styles = {
            NodeStatus.ACTIVE: "fill:#fff",
            NodeStatus.PROMISING: "fill:#90EE90",
            NodeStatus.PARTIAL: "fill:#FFD700",
            NodeStatus.STUCK: "fill:#FFA500",
            NodeStatus.PRUNED: "fill:#FF6B6B",
            NodeStatus.SOLVED: "fill:#32CD32",
            NodeStatus.MERGED: "fill:#87CEEB",
        }

        for node_id, node in self.nodes.items():
            label = f"{node.approach}\\n{node.status.value}"
            lines.append(f'    {node_id}["{label}"]')

            if node.parent_id:
                lines.append(f"    {node.parent_id} --> {node_id}")

            style = status_styles.get(node.status, "")
            if style:
                lines.append(f"    style {node_id} {style}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Get a text summary of the graph."""
        total = len(self.nodes)
        by_status = {}
        for node in self.nodes.values():
            by_status[node.status] = by_status.get(node.status, 0) + 1

        lines = [
            f"Graph of Thought for {self.problem_id}",
            f"Total nodes: {total}",
            f"Solutions found: {len(self.solution_ids)}",
            f"Merges performed: {len(self.merge_history)}",
            "",
            "By status:"
        ]
        for status, count in by_status.items():
            lines.append(f"  {status.value}: {count}")

        return "\n".join(lines)
```

---

### Exploration Strategy

**File**: `src/search/exploration.py`

```python
"""
Exploration strategies for Graph of Thought.

Implements the THINK → PROBE → OBSERVE → LEARN loop.
"""

from typing import List, Optional, Tuple
from .graph import GraphOfThought, ThoughtNode, NodeStatus
from ..memory.flywheel import Flywheel

class ExplorationStrategy:
    """
    Base class for exploration strategies.

    Strategies decide:
    1. Which node to explore next
    2. What approach to try
    3. When to branch/prune/merge
    """

    def __init__(self, graph: GraphOfThought, flywheel: Flywheel):
        self.graph = graph
        self.flywheel = flywheel

    def select_node(self) -> Optional[ThoughtNode]:
        """Select the next node to explore."""
        raise NotImplementedError

    def select_approach(self, node: ThoughtNode) -> str:
        """Select an approach to try for a node."""
        raise NotImplementedError

    def should_branch(self, node: ThoughtNode, approaches: List[str]) -> bool:
        """Decide whether to create multiple branches."""
        raise NotImplementedError

    def should_merge(self) -> Optional[List[str]]:
        """Check if any nodes should be merged."""
        raise NotImplementedError


class BestFirstExploration(ExplorationStrategy):
    """
    Best-first exploration with confidence-based selection.

    Always expands the most promising node.
    """

    def select_node(self) -> Optional[ThoughtNode]:
        return self.graph.get_best_node()

    def select_approach(self, node: ThoughtNode) -> str:
        # Get context from flywheel
        context = self.flywheel.get_context(
            self.graph.problem_id,
            node.hypothesis,
            []  # Topics would come from problem
        )

        # Get recommended approaches
        recommended = context.get("recommended_techniques", [])

        # Filter out already tried
        tried = {c.approach for c in self.graph.nodes.values() if c.parent_id == node.id}

        for technique, score in recommended:
            if technique not in tried:
                return technique

        # Fallback to first untried
        return recommended[0][0] if recommended else "direct_proof"

    def should_branch(self, node: ThoughtNode, approaches: List[str]) -> bool:
        # Branch if multiple high-scoring approaches
        high_scoring = [a for a, s in approaches if s > 0.5]
        return len(high_scoring) > 1 and node.depth < 5

    def should_merge(self) -> Optional[List[str]]:
        mergeable = self.graph.find_mergeable_nodes()
        return mergeable[0] if mergeable else None


class DepthFirstExploration(ExplorationStrategy):
    """
    Depth-first exploration with backtracking.

    Goes deep on one approach before trying others.
    """

    def __init__(self, graph: GraphOfThought, flywheel: Flywheel):
        super().__init__(graph, flywheel)
        self.current_path: List[str] = []

    def select_node(self) -> Optional[ThoughtNode]:
        # Continue on current path if possible
        if self.current_path:
            current_id = self.current_path[-1]
            current = self.graph.nodes.get(current_id)
            if current and current.is_expandable():
                return current

        # Backtrack to find expandable node
        while self.current_path:
            self.current_path.pop()
            if self.current_path:
                node = self.graph.nodes.get(self.current_path[-1])
                if node and node.is_expandable():
                    return node

        # Start fresh from best frontier node
        best = self.graph.get_best_node()
        if best:
            self.current_path = [n.id for n in self.graph.get_path_to_root(best.id)]
        return best

    def select_approach(self, node: ThoughtNode) -> str:
        # Same as best-first
        context = self.flywheel.get_context(
            self.graph.problem_id, node.hypothesis, []
        )
        recommended = context.get("recommended_techniques", [])
        tried = {c.approach for c in self.graph.nodes.values() if c.parent_id == node.id}

        for technique, score in recommended:
            if technique not in tried:
                return technique
        return "direct_proof"

    def should_branch(self, node: ThoughtNode, approaches: List[str]) -> bool:
        # Depth-first doesn't branch often
        return False

    def should_merge(self) -> Optional[List[str]]:
        # Check periodically
        if len(self.graph.nodes) % 10 == 0:
            mergeable = self.graph.find_mergeable_nodes()
            return mergeable[0] if mergeable else None
        return None


class BalancedExploration(ExplorationStrategy):
    """
    Balanced exploration using UCB-like selection.

    Balances exploitation (promising nodes) with exploration (less visited).
    """

    def __init__(self, graph: GraphOfThought, flywheel: Flywheel,
                 exploration_weight: float = 1.0):
        super().__init__(graph, flywheel)
        self.exploration_weight = exploration_weight

    def _ucb_score(self, node: ThoughtNode, total_visits: int) -> float:
        """Calculate UCB score for a node."""
        import math

        if node.visits == 0:
            return float('inf')  # Always explore unvisited

        exploitation = node.confidence * (1 + node.progress_score)
        exploration = self.exploration_weight * math.sqrt(
            math.log(total_visits + 1) / (node.visits + 1)
        )

        return exploitation + exploration

    def select_node(self) -> Optional[ThoughtNode]:
        frontier = self.graph.get_frontier()
        if not frontier:
            return None

        total_visits = sum(n.visits for n in self.graph.nodes.values())

        return max(frontier, key=lambda n: self._ucb_score(n, total_visits))

    def select_approach(self, node: ThoughtNode) -> str:
        context = self.flywheel.get_context(
            self.graph.problem_id, node.hypothesis, []
        )
        recommended = context.get("recommended_techniques", [])
        tried = {c.approach for c in self.graph.nodes.values() if c.parent_id == node.id}

        for technique, score in recommended:
            if technique not in tried:
                return technique
        return "direct_proof"

    def should_branch(self, node: ThoughtNode, approaches: List[str]) -> bool:
        # Branch at low depths with multiple good options
        high_scoring = [(a, s) for a, s in approaches if s > 0.4]
        return len(high_scoring) >= 2 and node.depth <= 3

    def should_merge(self) -> Optional[List[str]]:
        # More aggressive merging
        mergeable = self.graph.find_mergeable_nodes()
        if mergeable:
            # Merge the pair with most complementary facts
            return max(mergeable, key=lambda pair: len(
                set(self.graph.nodes[pair[0]].established_facts) ^
                set(self.graph.nodes[pair[1]].established_facts)
            ))
        return None
```

---

### Integration with Orchestrator

**File**: `src/search/exploration_loop.py`

```python
"""
The main exploration loop implementing THINK → PROBE → OBSERVE → LEARN.
"""

from typing import Optional
from .graph import GraphOfThought, NodeStatus
from .exploration import ExplorationStrategy, BalancedExploration
from ..memory.flywheel import Flywheel, SolveResult
from ..verify.cascade import VerificationCascade

class ExplorationLoop:
    """
    Orchestrates the exploration process.

    THINK:   Select node and approach based on current state
    PROBE:   Apply approach, run verification
    OBSERVE: Analyze results, extract facts
    LEARN:   Update graph, prune, merge, record to flywheel
    """

    def __init__(self, problem_id: str, statement: str, topics: list,
                 flywheel: Flywheel, verifier: VerificationCascade,
                 max_iterations: int = 100):
        self.problem_id = problem_id
        self.statement = statement
        self.topics = topics
        self.flywheel = flywheel
        self.verifier = verifier
        self.max_iterations = max_iterations

        # Initialize graph
        self.graph = GraphOfThought(problem_id)
        self.graph.create_root(statement)

        # Default strategy
        self.strategy = BalancedExploration(self.graph, flywheel)

    def run(self) -> Optional[str]:
        """
        Run the exploration loop until solution or max iterations.

        Returns the solution if found, None otherwise.
        """
        for iteration in range(self.max_iterations):
            # THINK: Select what to explore
            node = self.strategy.select_node()
            if node is None:
                break  # No expandable nodes

            approach = self.strategy.select_approach(node)

            # PROBE: Apply approach
            result = self._probe(node, approach)

            # OBSERVE: Analyze results
            self._observe(node, approach, result)

            # LEARN: Update graph and flywheel
            self._learn(node, result)

            # Check for solution
            if self.graph.solution_ids:
                solution_node = self.graph.nodes[self.graph.solution_ids[0]]
                return solution_node.conclusion

            # Check for merge opportunities
            merge_candidates = self.strategy.should_merge()
            if merge_candidates:
                self._perform_merge(merge_candidates)

            # Auto-prune
            self.graph.auto_prune()

        return None

    def _probe(self, node: ThoughtNode, approach: str) -> dict:
        """
        Apply an approach and get results.

        This is where LLM reasoning happens.
        """
        # Create child node for this attempt
        child = self.graph.branch(node.id, approach, f"Trying {approach}")

        # TODO: This is where Claude generates the reasoning
        # For now, return a placeholder
        result = {
            "success": False,
            "reasoning": f"Applied {approach}",
            "facts_established": [],
            "subgoals_solved": [],
            "errors": [],
            "confidence": 0.5,
            "child_id": child.id
        }

        return result

    def _observe(self, node: ThoughtNode, approach: str, result: dict):
        """
        Analyze the results of a probe.
        """
        child = self.graph.nodes[result["child_id"]]

        if result["success"]:
            self.graph.mark_solved(child.id, result["reasoning"])
        elif result["errors"]:
            self.graph.mark_stuck(
                child.id,
                f"Failed: {result['errors'][0]}",
                result["errors"][0]
            )
        elif result["facts_established"]:
            self.graph.mark_progress(
                child.id,
                facts=result["facts_established"],
                subgoals=result["subgoals_solved"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
        else:
            # No progress
            self.graph.mark_stuck(child.id, "No progress made")

    def _learn(self, node: ThoughtNode, result: dict):
        """
        Update flywheel with learnings.
        """
        child = self.graph.nodes[result["child_id"]]

        # Record failure if stuck
        if child.status == NodeStatus.STUCK:
            self.flywheel.record_failure(
                self.problem_id,
                child.approach,
                child.conclusion
            )

        # Record facts
        for fact in result["facts_established"]:
            self.flywheel.record_fact(
                self.problem_id,
                fact,
                result["reasoning"],
                result["confidence"]
            )

    def _perform_merge(self, node_ids: list):
        """Perform a merge of nodes."""
        nodes = [self.graph.nodes[nid] for nid in node_ids]

        # Generate merge reasoning
        reasoning = f"Merging {len(nodes)} partial solutions: "
        reasoning += ", ".join(n.approach for n in nodes)

        self.graph.merge(node_ids, reasoning)

    def get_state(self) -> dict:
        """Get current exploration state."""
        return {
            "problem_id": self.problem_id,
            "iterations": sum(n.visits for n in self.graph.nodes.values()),
            "nodes": len(self.graph.nodes),
            "solutions": len(self.graph.solution_ids),
            "frontier_size": len(self.graph.get_frontier()),
            "graph": self.graph.to_dict()
        }

    def save(self, path: str):
        """Save exploration state."""
        self.graph.save(path)

    @classmethod
    def load(cls, path: str, flywheel: Flywheel, verifier: VerificationCascade) -> "ExplorationLoop":
        """Load exploration state."""
        graph = GraphOfThought.load(path)
        loop = cls(graph.problem_id, "", [], flywheel, verifier)
        loop.graph = graph
        return loop
```

---

## Persistence

### File: `data/memory/graphs/{problem_id}.json`

```json
{
  "problem_id": "putnam_2024_a1",
  "root_id": "abc123",
  "solution_ids": [],
  "nodes": {
    "abc123": {
      "id": "abc123",
      "parent_id": null,
      "children": ["def456", "ghi789"],
      "approach": "problem_analysis",
      "hypothesis": "Find all positive integers...",
      "status": "partial",
      "depth": 0,
      "established_facts": ["n > 0"],
      "confidence": 0.6,
      "progress_score": 0.3
    }
  },
  "merge_history": []
}
```

---

## Integration with /solve

The `/solve` command should:

1. **Initialize**: Create or load GraphOfThought for problem
2. **Explore**: Run ExplorationLoop with configured strategy
3. **Display**: Show graph summary and current frontier
4. **Persist**: Save graph state after each iteration

```python
# In /solve workflow
from src.search.exploration_loop import ExplorationLoop

# Initialize
loop = ExplorationLoop(
    problem_id=problem.id,
    statement=problem.statement,
    topics=problem.topics,
    flywheel=flywheel,
    verifier=verifier,
    max_iterations=50
)

# Run with checkpoints
while not loop.graph.solution_ids:
    iteration_result = loop.run_one_iteration()

    # Display progress
    print(loop.graph.summary())

    # Save state
    loop.save(f"data/memory/graphs/{problem.id}.json")

    # Check for user interrupt
    if should_stop():
        break
```

---

## Summary

| Component | File | Purpose | Priority |
|-----------|------|---------|----------|
| ThoughtNode | `graph.py` | Node representation | P1 |
| GraphOfThought | `graph.py` | Graph structure | P1 |
| ExplorationStrategy | `exploration.py` | Selection logic | P1 |
| ExplorationLoop | `exploration_loop.py` | Main loop | P0 |
| Persistence | - | JSON serialization | P1 |

The Graph of Thought is the backbone of intelligent exploration. Without it, the system is just random search.
