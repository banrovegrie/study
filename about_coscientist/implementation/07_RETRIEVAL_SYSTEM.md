# 07. Retrieval System for Proof Assistance

## First Principles: What Are We Actually Retrieving?

Before diving into embeddings and vector databases, we need to ask fundamental questions:

### The Core Problem

When solving a mathematical problem, a human mathematician recalls:
1. **Similar problems** they've solved before
2. **Relevant lemmas** that might apply
3. **Proof patterns/techniques** that worked in similar contexts
4. **Failed approaches** to avoid

The question is: **what makes something "similar" in mathematics?**

### Why Naive Semantic Similarity Fails

Standard text embeddings capture lexical and semantic similarity, but mathematical similarity is structural:

```
Problem A: "Prove that √2 is irrational"
Problem B: "Prove that √3 is irrational"
Problem C: "The square root function is continuous"
```

Text embeddings might rank C closer to A/B (all mention "square root"), but A/B share the same **proof structure** (contradiction via divisibility). This is what matters.

### Core Principle: Retrieve by Applicability, Not Aboutness

**The goal**: Find techniques that can solve structurally similar problems, regardless of domain.

```
Traditional retrieval:          Our retrieval:
──────────────────────         ──────────────────────
Query: "prime number proof"     Query: "prove ∀n, property P(n)"

Returns: Papers about primes    Returns: Techniques for universal proofs
                                 - Induction (from NT, combinatorics)
                                 - Descent (from NT, analysis)
                                 - Probabilistic method (from combinatorics)
```

We retrieve **decontextualized techniques**—methods stripped from their original context—indexed by what goal patterns they solve, not what topics they're "about".

### Research Grounding

**LeanDojo (Yang et al., 2023)**: Retrieves premises (lemmas, theorems) for ReProver. Key insight: retrieves from the **accessible** premise set given current proof state, not just semantically similar text.

**Rango (Sprint et al., 2024)**: Retrieval-augmented generation for Isabelle. Uses BM25 + learned reranking. Found that combining sparse (BM25) and dense retrieval outperforms either alone.

**Mathematical Structure Awareness**: Recent work shows that encoding mathematical structure (AST, type information) significantly improves retrieval for theorem proving.

---

## Design Decision: Multi-Modal Retrieval

We don't just embed text. We build a **multi-modal retrieval system** that considers:

1. **Structural similarity** - AST/proof tree patterns
2. **Type-based matching** - problems involving similar Lean types
3. **Technique similarity** - what proof methods were used
4. **Semantic similarity** - as a fallback/tiebreaker

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum, auto
import numpy as np
from abc import ABC, abstractmethod


class RetrievalMode(Enum):
    """What aspect of mathematical similarity to prioritize."""
    STRUCTURAL = auto()   # Same proof structure/pattern
    TYPE_BASED = auto()   # Same mathematical types involved
    TECHNIQUE = auto()    # Same proof techniques used
    SEMANTIC = auto()     # Text/concept similarity
    HYBRID = auto()       # Weighted combination


@dataclass
class RetrievalQuery:
    """A query for retrieving relevant mathematical knowledge."""

    # The problem or goal we're trying to solve
    natural_language: str
    lean_goal: Optional[str] = None  # Current tactic state if available

    # Structural information
    goal_types: List[str] = field(default_factory=list)  # Types mentioned in goal
    hypotheses: List[str] = field(default_factory=list)  # Available hypotheses

    # Context about what we've tried
    attempted_techniques: List[str] = field(default_factory=list)
    failed_approaches: List[str] = field(default_factory=list)

    # What we're looking for
    retrieval_intent: str = "similar_proofs"  # or "relevant_lemmas", "applicable_tactics"


@dataclass
class RetrievedItem:
    """A single retrieved item with provenance and relevance scores."""

    id: str
    content: str
    source: str  # "herald", "mathlib", "session_cache", "skills_library"

    # Multi-dimensional relevance scores
    structural_score: float = 0.0
    type_score: float = 0.0
    technique_score: float = 0.0
    semantic_score: float = 0.0

    # Combined score (computed based on mode)
    final_score: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def explain_relevance(self) -> str:
        """Human-readable explanation of why this was retrieved."""
        reasons = []
        if self.structural_score > 0.7:
            reasons.append(f"similar proof structure ({self.structural_score:.2f})")
        if self.type_score > 0.7:
            reasons.append(f"matching types ({self.type_score:.2f})")
        if self.technique_score > 0.7:
            reasons.append(f"same techniques ({self.technique_score:.2f})")
        if self.semantic_score > 0.7:
            reasons.append(f"semantic similarity ({self.semantic_score:.2f})")
        return "; ".join(reasons) if reasons else "general relevance"
```

---

## The Index: What We Store

### Granularity Matters

**Question**: Should we index full proofs, individual proof steps, or lemma statements?

**Answer**: All three, with different use cases:

```python
class IndexGranularity(Enum):
    """Different levels of indexing granularity."""
    PROBLEM = auto()      # Full problem + solution
    PROOF = auto()        # Complete proof structure
    PROOF_STEP = auto()   # Individual tactic applications
    LEMMA = auto()        # Lemma/theorem statements
    TECHNIQUE = auto()    # Abstract proof techniques


@dataclass
class IndexedProof:
    """A proof indexed for retrieval."""

    id: str
    problem_statement: str
    lean_formalization: Optional[str]
    proof_text: Optional[str]  # Natural language proof
    lean_proof: Optional[str]   # Formal proof

    # Structural features (computed at index time)
    proof_structure: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"pattern": "contradiction", "steps": 5, "key_lemmas": [...]}

    # Type signature
    types_involved: List[str] = field(default_factory=list)

    # Techniques used
    techniques: List[str] = field(default_factory=list)

    # Embeddings (computed at index time)
    text_embedding: Optional[np.ndarray] = None
    structure_embedding: Optional[np.ndarray] = None

    # Metadata
    source: str = ""  # "herald", "mathlib", etc.
    difficulty: Optional[float] = None
    domain: Optional[str] = None


@dataclass
class IndexedLemma:
    """A lemma/theorem indexed for retrieval."""

    id: str
    name: str
    statement: str  # Natural language
    lean_statement: str  # Formal

    # Type information
    input_types: List[str] = field(default_factory=list)
    output_type: str = ""

    # When is this lemma useful?
    applicability_conditions: List[str] = field(default_factory=list)

    # Embeddings
    statement_embedding: Optional[np.ndarray] = None
    type_embedding: Optional[np.ndarray] = None

    # Metadata
    source: str = "mathlib"
    namespace: str = ""
```

---

## Embedding Strategy

### Critical Decision: What Gets Its Own Embedding Space?

Standard approach: one embedding model for everything.
Our approach: **separate embedding spaces for separate concerns**.

```python
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Base class for embedding models."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        pass


class TextEmbedder(Embedder):
    """Standard text embedding for semantic similarity."""

    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self.dimension = 3072  # For text-embedding-3-large

    def embed(self, text: str) -> np.ndarray:
        # Use OpenAI/local embedding model
        # In production: actual API call
        return np.zeros(self.dimension)  # Placeholder

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dimension))


class StructureEmbedder(Embedder):
    """
    Embeds proof structure, not content.

    Key insight: Two proofs can use completely different lemmas
    but have identical structure (both use contradiction, both
    proceed by case analysis, etc.)
    """

    def __init__(self):
        self.dimension = 256
        self.structure_vocab = self._build_structure_vocab()

    def _build_structure_vocab(self) -> Dict[str, int]:
        """Vocabulary of structural elements."""
        structures = [
            # Proof patterns
            "direct", "contradiction", "contrapositive", "induction",
            "strong_induction", "case_analysis", "construction",
            "pigeonhole", "extremal", "probabilistic",
            # Structural markers
            "base_case", "inductive_step", "assume_negation",
            "derive_contradiction", "cases_exhaustive",
            # Complexity markers
            "nested_induction", "mutual_recursion", "well_founded",
        ]
        return {s: i for i, s in enumerate(structures)}

    def _extract_structure(self, proof_text: str) -> List[str]:
        """
        Extract structural elements from proof.

        This is where LLM-based analysis helps - not for embedding,
        but for identifying structural patterns.
        """
        # In practice: use LLM to identify structure, or parse Lean proof
        structures = []

        # Simple heuristics as fallback
        text_lower = proof_text.lower()
        if "contradiction" in text_lower or "assume not" in text_lower:
            structures.append("contradiction")
        if "induction" in text_lower:
            structures.append("induction")
        if "cases" in text_lower:
            structures.append("case_analysis")
        # ... more pattern matching

        return structures

    def embed(self, text: str) -> np.ndarray:
        structures = self._extract_structure(text)
        # Create sparse embedding based on detected structures
        embedding = np.zeros(self.dimension)
        for struct in structures:
            if struct in self.structure_vocab:
                idx = self.structure_vocab[struct]
                embedding[idx] = 1.0
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class TypeEmbedder(Embedder):
    """
    Embeds based on mathematical types involved.

    Proofs involving ℕ, divisibility, and primes are related
    regardless of the specific statements.
    """

    def __init__(self):
        self.dimension = 512
        self.type_hierarchy = self._build_type_hierarchy()

    def _build_type_hierarchy(self) -> Dict[str, Set[str]]:
        """Mathematical type relationships."""
        return {
            "ℕ": {"number", "nat", "natural"},
            "ℤ": {"number", "int", "integer"},
            "ℚ": {"number", "rat", "rational"},
            "ℝ": {"number", "real"},
            "Set": {"collection", "set"},
            "List": {"collection", "sequence", "list"},
            "Group": {"algebraic", "group"},
            "Ring": {"algebraic", "ring"},
            "Field": {"algebraic", "field"},
            # ... extensive type vocabulary
        }

    def _extract_types(self, lean_code: str) -> List[str]:
        """Extract types from Lean code."""
        types = []
        # Parse type annotations and infer types
        # In practice: use Lean's type checker via LSP
        return types

    def embed(self, text: str) -> np.ndarray:
        types = self._extract_types(text)
        embedding = np.zeros(self.dimension)
        # Encode types with hierarchy-aware embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])
```

---

## The Retrieval Pipeline

### Multi-Stage Retrieval

Inspired by Rango's success with sparse + dense retrieval:

```python
class RetrievalPipeline:
    """
    Multi-stage retrieval combining multiple similarity measures.

    Stage 1: Broad candidate retrieval (fast, high recall)
    Stage 2: Multi-modal scoring (slower, high precision)
    Stage 3: Reranking with context (LLM-based)
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        structure_embedder: StructureEmbedder,
        type_embedder: TypeEmbedder,
    ):
        self.text_embedder = text_embedder
        self.structure_embedder = structure_embedder
        self.type_embedder = type_embedder

        # Indices (in practice: vector databases)
        self.proof_index: List[IndexedProof] = []
        self.lemma_index: List[IndexedLemma] = []

        # BM25 for sparse retrieval
        self.bm25_proofs = None  # BM25 index
        self.bm25_lemmas = None

    def retrieve(
        self,
        query: RetrievalQuery,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10,
        include_explanation: bool = True,
    ) -> List[RetrievedItem]:
        """
        Main retrieval entry point.
        """
        # Stage 1: Broad candidate retrieval
        candidates = self._stage1_candidate_retrieval(query, top_k=top_k * 5)

        # Stage 2: Multi-modal scoring
        scored = self._stage2_multimodal_scoring(query, candidates, mode)

        # Stage 3: Rerank top candidates with LLM
        if len(scored) > top_k:
            reranked = self._stage3_llm_rerank(query, scored[:top_k * 2])
        else:
            reranked = scored

        # Return top-k with explanations
        results = reranked[:top_k]

        if include_explanation:
            for item in results:
                item.metadata["explanation"] = item.explain_relevance()

        return results

    def _stage1_candidate_retrieval(
        self,
        query: RetrievalQuery,
        top_k: int,
    ) -> List[RetrievedItem]:
        """
        Fast, broad retrieval to get candidates.

        Uses combination of:
        - BM25 for keyword matching
        - Dense retrieval for semantic similarity
        - Type filtering for structural matching
        """
        candidates = []

        # BM25 retrieval
        bm25_results = self._bm25_search(query.natural_language, top_k // 2)
        candidates.extend(bm25_results)

        # Dense retrieval
        query_embedding = self.text_embedder.embed(query.natural_language)
        dense_results = self._dense_search(query_embedding, top_k // 2)
        candidates.extend(dense_results)

        # Type-based filtering if we have type information
        if query.goal_types:
            type_results = self._type_filtered_search(query.goal_types, top_k // 2)
            candidates.extend(type_results)

        # Deduplicate by ID
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.id not in seen:
                seen.add(c.id)
                unique_candidates.append(c)

        return unique_candidates

    def _stage2_multimodal_scoring(
        self,
        query: RetrievalQuery,
        candidates: List[RetrievedItem],
        mode: RetrievalMode,
    ) -> List[RetrievedItem]:
        """
        Score candidates on multiple dimensions.
        """
        query_text_emb = self.text_embedder.embed(query.natural_language)
        query_struct_emb = self.structure_embedder.embed(query.natural_language)
        query_type_emb = self.type_embedder.embed(query.lean_goal or "")

        for candidate in candidates:
            # Get cached embeddings from index
            proof = self._get_proof_by_id(candidate.id)
            if proof is None:
                continue

            # Compute similarity scores
            if proof.text_embedding is not None:
                candidate.semantic_score = self._cosine_similarity(
                    query_text_emb, proof.text_embedding
                )

            if proof.structure_embedding is not None:
                candidate.structural_score = self._cosine_similarity(
                    query_struct_emb, proof.structure_embedding
                )

            # Type matching - exact match gets bonus
            type_overlap = len(set(query.goal_types) & set(proof.types_involved))
            candidate.type_score = type_overlap / max(len(query.goal_types), 1)

            # Technique matching if we know what we're looking for
            if query.attempted_techniques:
                tech_overlap = len(
                    set(query.attempted_techniques) & set(proof.techniques)
                )
                candidate.technique_score = tech_overlap / len(query.attempted_techniques)

        # Compute final score based on mode
        for candidate in candidates:
            candidate.final_score = self._compute_final_score(candidate, mode)

        # Sort by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)

        return candidates

    def _compute_final_score(
        self,
        item: RetrievedItem,
        mode: RetrievalMode,
    ) -> float:
        """Compute final score based on retrieval mode."""

        if mode == RetrievalMode.STRUCTURAL:
            return item.structural_score * 0.7 + item.semantic_score * 0.3

        elif mode == RetrievalMode.TYPE_BASED:
            return item.type_score * 0.6 + item.structural_score * 0.3 + item.semantic_score * 0.1

        elif mode == RetrievalMode.TECHNIQUE:
            return item.technique_score * 0.5 + item.structural_score * 0.3 + item.semantic_score * 0.2

        elif mode == RetrievalMode.SEMANTIC:
            return item.semantic_score

        else:  # HYBRID - learned weights would be ideal
            return (
                item.structural_score * 0.35 +
                item.type_score * 0.25 +
                item.technique_score * 0.2 +
                item.semantic_score * 0.2
            )

    def _stage3_llm_rerank(
        self,
        query: RetrievalQuery,
        candidates: List[RetrievedItem],
    ) -> List[RetrievedItem]:
        """
        Use LLM to rerank based on deeper understanding.

        This is expensive, so only used on top candidates.
        """
        # Build prompt for reranking
        prompt = self._build_rerank_prompt(query, candidates)

        # In practice: call LLM and parse rankings
        # For now, return as-is
        return candidates

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedItem]:
        """BM25 sparse retrieval."""
        # In practice: use rank_bm25 or similar
        return []

    def _dense_search(self, embedding: np.ndarray, top_k: int) -> List[RetrievedItem]:
        """Dense vector retrieval."""
        # In practice: use FAISS, Pinecone, etc.
        return []

    def _type_filtered_search(self, types: List[str], top_k: int) -> List[RetrievedItem]:
        """Search filtered by type."""
        return []

    def _get_proof_by_id(self, proof_id: str) -> Optional[IndexedProof]:
        """Retrieve proof from index."""
        for proof in self.proof_index:
            if proof.id == proof_id:
                return proof
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between vectors."""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _build_rerank_prompt(
        self,
        query: RetrievalQuery,
        candidates: List[RetrievedItem],
    ) -> str:
        """Build prompt for LLM reranking."""
        return f"""Given the mathematical problem:
{query.natural_language}

Rank these candidate proofs by relevance (most relevant first):
{[c.content[:200] for c in candidates]}

Return a comma-separated list of indices in order of relevance."""
```

---

## Specialized Retrievers

### Lemma Retrieval for Tactic State

When we have a concrete Lean goal, we need lemmas that **apply**:

```python
class LemmaRetriever:
    """
    Retrieves applicable lemmas given current proof state.

    Inspired by LeanDojo's premise selection, but with
    additional filtering for actual applicability.
    """

    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    def retrieve_applicable_lemmas(
        self,
        goal_state: str,
        hypotheses: List[str],
        namespace: str,
        top_k: int = 20,
    ) -> List[IndexedLemma]:
        """
        Retrieve lemmas that could help prove the current goal.

        Key insight: A lemma is relevant if:
        1. Its conclusion unifies with our goal (or subgoal)
        2. Its hypotheses can be satisfied by our context
        3. It's accessible in the current namespace
        """
        # Parse goal to extract type information
        goal_types = self._extract_types_from_goal(goal_state)

        # Build query
        query = RetrievalQuery(
            natural_language=goal_state,
            lean_goal=goal_state,
            goal_types=goal_types,
            hypotheses=hypotheses,
            retrieval_intent="relevant_lemmas",
        )

        # Retrieve candidates
        candidates = self._retrieve_lemma_candidates(query, top_k * 3)

        # Filter for accessibility
        accessible = [
            l for l in candidates
            if self._is_accessible(l, namespace)
        ]

        # Filter for potential applicability
        applicable = [
            l for l in accessible
            if self._could_apply(l, goal_state, hypotheses)
        ]

        return applicable[:top_k]

    def _extract_types_from_goal(self, goal: str) -> List[str]:
        """Extract Lean types from goal string."""
        # In practice: parse with Lean or use regex patterns
        types = []
        # Simple patterns
        if "Nat" in goal or "ℕ" in goal:
            types.append("Nat")
        if "Int" in goal or "ℤ" in goal:
            types.append("Int")
        if "Real" in goal or "ℝ" in goal:
            types.append("Real")
        if "→" in goal:
            types.append("Function")
        if "∀" in goal:
            types.append("Universal")
        if "∃" in goal:
            types.append("Existential")
        return types

    def _retrieve_lemma_candidates(
        self,
        query: RetrievalQuery,
        top_k: int,
    ) -> List[IndexedLemma]:
        """Retrieve lemma candidates."""
        # Use type-based retrieval mode for lemmas
        items = self.pipeline.retrieve(
            query,
            mode=RetrievalMode.TYPE_BASED,
            top_k=top_k,
        )

        # Convert to IndexedLemma
        lemmas = []
        for item in items:
            if item.source == "mathlib":
                lemma = self._get_lemma_by_id(item.id)
                if lemma:
                    lemmas.append(lemma)
        return lemmas

    def _is_accessible(self, lemma: IndexedLemma, namespace: str) -> bool:
        """Check if lemma is accessible from namespace."""
        # In practice: check import hierarchy
        return True  # Simplified

    def _could_apply(
        self,
        lemma: IndexedLemma,
        goal: str,
        hypotheses: List[str],
    ) -> bool:
        """
        Heuristic check for whether lemma could apply.

        Full applicability requires actual unification,
        but we can filter obvious non-matches.
        """
        # Check if output type is compatible with goal
        # This is a heuristic - real check needs Lean
        return True  # Simplified

    def _get_lemma_by_id(self, lemma_id: str) -> Optional[IndexedLemma]:
        """Get lemma from index."""
        for lemma in self.pipeline.lemma_index:
            if lemma.id == lemma_id:
                return lemma
        return None
```

### Similar Problem Retrieval

For finding problems with similar structure:

```python
class SimilarProblemRetriever:
    """
    Retrieves problems that are structurally similar.

    Key insight: "Similar" means "solved with similar technique",
    not "mentions similar words".
    """

    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    def find_similar_problems(
        self,
        problem: str,
        known_techniques: Optional[List[str]] = None,
        exclude_ids: Optional[Set[str]] = None,
        top_k: int = 5,
    ) -> List[IndexedProof]:
        """
        Find problems solved with similar techniques.
        """
        exclude_ids = exclude_ids or set()

        # First, try to identify the problem's likely technique
        if known_techniques is None:
            known_techniques = self._identify_likely_techniques(problem)

        query = RetrievalQuery(
            natural_language=problem,
            attempted_techniques=known_techniques,
            retrieval_intent="similar_proofs",
        )

        # Use structural mode to find similar proof patterns
        items = self.pipeline.retrieve(
            query,
            mode=RetrievalMode.STRUCTURAL,
            top_k=top_k * 2,
        )

        # Filter excluded and convert
        proofs = []
        for item in items:
            if item.id not in exclude_ids:
                proof = self.pipeline._get_proof_by_id(item.id)
                if proof:
                    proofs.append(proof)

        return proofs[:top_k]

    def _identify_likely_techniques(self, problem: str) -> List[str]:
        """
        Use heuristics/LLM to identify likely proof techniques.
        """
        techniques = []
        problem_lower = problem.lower()

        # Pattern-based heuristics
        if any(w in problem_lower for w in ["irrational", "not rational"]):
            techniques.append("contradiction")
        if any(w in problem_lower for w in ["for all n", "every natural", "all integers"]):
            techniques.append("induction")
        if any(w in problem_lower for w in ["either", "or", "cases"]):
            techniques.append("case_analysis")
        if any(w in problem_lower for w in ["construct", "exists", "find"]):
            techniques.append("construction")
        if any(w in problem_lower for w in ["infinitely many", "unbounded"]):
            techniques.append("infinite_descent")

        return techniques if techniques else ["direct"]
```

---

## Index Building

### Building from Herald and Mathlib

```python
class IndexBuilder:
    """
    Builds the retrieval index from data sources.
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        structure_embedder: StructureEmbedder,
        type_embedder: TypeEmbedder,
    ):
        self.text_embedder = text_embedder
        self.structure_embedder = structure_embedder
        self.type_embedder = type_embedder

    def build_from_herald(
        self,
        herald_path: str,
        batch_size: int = 100,
    ) -> List[IndexedProof]:
        """
        Build proof index from Herald dataset.

        Herald contains ~44K proofs from Mathlib.
        """
        proofs = []

        # Load Herald data
        # In practice: iterate through the dataset

        # Process in batches for efficiency
        # For each proof:
        # 1. Extract problem statement
        # 2. Parse proof structure
        # 3. Identify types involved
        # 4. Identify techniques used
        # 5. Compute embeddings

        return proofs

    def build_from_mathlib(
        self,
        mathlib_path: str,
    ) -> List[IndexedLemma]:
        """
        Build lemma index from Mathlib.

        ~210K definitions/theorems to index.
        """
        lemmas = []

        # This is substantial - need incremental building
        # and caching of embeddings

        return lemmas

    def index_session_proof(
        self,
        problem: str,
        proof: str,
        lean_proof: Optional[str] = None,
    ) -> IndexedProof:
        """
        Index a proof from the current session.

        This supports the "learning from experience" loop.
        """
        # Analyze the proof
        structure = self._analyze_proof_structure(proof)
        types = self._extract_types(lean_proof or proof)
        techniques = self._identify_techniques(proof)

        # Compute embeddings
        text_emb = self.text_embedder.embed(f"{problem}\n{proof}")
        struct_emb = self.structure_embedder.embed(proof)

        indexed = IndexedProof(
            id=f"session_{hash(problem)}",
            problem_statement=problem,
            lean_formalization=lean_proof,
            proof_text=proof,
            lean_proof=lean_proof,
            proof_structure=structure,
            types_involved=types,
            techniques=techniques,
            text_embedding=text_emb,
            structure_embedding=struct_emb,
            source="session_cache",
        )

        return indexed

    def _analyze_proof_structure(self, proof: str) -> Dict[str, Any]:
        """Analyze proof to extract structural features."""
        return {
            "length": len(proof.split()),
            "has_cases": "case" in proof.lower(),
            "has_induction": "induction" in proof.lower(),
            "has_contradiction": "contradiction" in proof.lower(),
        }

    def _extract_types(self, code: str) -> List[str]:
        """Extract types from code."""
        return []  # Simplified

    def _identify_techniques(self, proof: str) -> List[str]:
        """Identify techniques in proof."""
        return []  # Simplified
```

---

## Integration with Memory System

The retrieval system connects directly to the memory architecture:

```python
class RetrievalIntegratedMemory:
    """
    Memory system with integrated retrieval capabilities.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        lemma_retriever: LemmaRetriever,
        similar_problem_retriever: SimilarProblemRetriever,
    ):
        self.retrieval = retrieval_pipeline
        self.lemma_retriever = lemma_retriever
        self.similar_retriever = similar_problem_retriever

        # Session cache for fast access to recent results
        self.session_cache: Dict[str, List[RetrievedItem]] = {}

    def get_relevant_knowledge(
        self,
        problem: str,
        goal_state: Optional[str] = None,
        hypotheses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve all relevant knowledge for a problem.

        Returns structured knowledge package for the solver.
        """
        result = {
            "similar_problems": [],
            "applicable_lemmas": [],
            "relevant_techniques": [],
            "cached_results": None,
        }

        # Check session cache first
        cache_key = f"{problem}:{goal_state}"
        if cache_key in self.session_cache:
            result["cached_results"] = self.session_cache[cache_key]
            return result

        # Retrieve similar problems (for proof strategies)
        similar = self.similar_retriever.find_similar_problems(
            problem, top_k=3
        )
        result["similar_problems"] = [
            {
                "statement": p.problem_statement,
                "technique": p.techniques[0] if p.techniques else "direct",
                "proof_sketch": p.proof_text[:500] if p.proof_text else None,
            }
            for p in similar
        ]

        # If we have a goal state, retrieve applicable lemmas
        if goal_state:
            lemmas = self.lemma_retriever.retrieve_applicable_lemmas(
                goal_state,
                hypotheses or [],
                namespace="Mathlib",
                top_k=10,
            )
            result["applicable_lemmas"] = [
                {
                    "name": l.name,
                    "statement": l.statement,
                    "lean": l.lean_statement,
                }
                for l in lemmas
            ]

        # Extract techniques from similar problems
        all_techniques = []
        for p in similar:
            all_techniques.extend(p.techniques)
        result["relevant_techniques"] = list(set(all_techniques))

        # Cache for this session
        self.session_cache[cache_key] = result

        return result

    def record_successful_retrieval(
        self,
        query: str,
        retrieved_item: RetrievedItem,
        was_helpful: bool,
    ):
        """
        Record whether a retrieved item was actually helpful.

        This feedback improves future retrieval.
        """
        # In practice: update retrieval model/weights
        pass
```

---

## Questioning Assumptions

### Assumption 1: Embeddings Capture Mathematical Similarity

**Challenge**: Standard embeddings are trained on general text, not mathematical structure.

**Mitigation**: Multi-modal approach with structure-aware embeddings. Consider fine-tuning embeddings on mathematical proof pairs.

**Open question**: Should we train a custom embedding model on (problem, useful_lemma) pairs from Mathlib?

### Assumption 2: More Retrieved Context = Better

**Challenge**: Too much context can confuse LLMs and exceed context windows.

**Mitigation**: Aggressive filtering and reranking. Quality over quantity.

**Open question**: What's the optimal number of retrieved items? Likely problem-dependent.

### Assumption 3: Retrieval from Text is Sufficient

**Challenge**: Some lemmas are useful not because of textual similarity but because of algebraic relationships.

**Mitigation**: Type-based retrieval as a separate dimension.

**Open question**: Should we incorporate Lean's actual type unification into retrieval?

### Assumption 4: Historical Proofs Are Good Examples

**Challenge**: Mathlib proofs are written for humans, optimized for readability, not for LLM consumption.

**Mitigation**: Post-process retrieved proofs to extract the essential technique.

**Open question**: Should we build a dataset of "LLM-optimized" proof examples?

---

## Research References

1. **LeanDojo** (Yang et al., 2023): Premise selection for ReProver
2. **Rango** (Sprint et al., 2024): BM25 + dense retrieval for Isabelle
3. **RagVerus**: Retrieval for verification
4. **Herald dataset**: 44K Mathlib proofs
5. **Retrieval-augmented generation** (general RAG literature)
6. **MathBERT/MathDistilBERT**: Mathematical text embeddings

---

## Implementation Priority

1. **Phase 1**: Basic text embedding retrieval from Herald
2. **Phase 2**: Add BM25 for hybrid sparse+dense
3. **Phase 3**: Add structure-aware embeddings
4. **Phase 4**: LLM-based reranking
5. **Phase 5**: Integration with Lean type system for true applicability filtering
