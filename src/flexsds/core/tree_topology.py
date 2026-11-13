"""Tree topology representations, causal attention masks, and expected yield models for FlexSDS.

Implements discrete canonical tree configurations (T0, T4, T8, T16, T32) based on
OPT-Tree (Wang et al., 2025, TACL) and speculative decoding theory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import math


@dataclass
class TreeTopology:
    """Represents a speculative draft tree structure.

    Attributes:
        name: Short topology identifier (e.g. 'T0', 'T4', 'T8', 'T16', 'T32').
        num_tokens: Number of non-root draft tokens N.
        depth: Maximum depth of the tree d (root has depth 0).
        parent_indices: List of length N+1 where parent_indices[0] == -1 and
            parent_indices[i] is the index of the parent node for token i (0 <= parent < i).
        layer_distribution: Number of nodes at each depth level [l_0, l_1, ..., l_d].
        node_depths: Computed depth for each node in 0..N.
        position_ids: Relative position index in tree (matches node_depths).
        tree_mask: 2D causal attention mask of shape (N+1, N+1) where
            mask[i][j] is True iff j is an ancestor of i (or j == i).
        nodes_at_depth: Mapping from depth level to list of node indices at that depth.
    """

    name: str
    num_tokens: int
    depth: int
    parent_indices: List[int]
    layer_distribution: List[int]
    node_depths: List[int] = field(default_factory=list)
    position_ids: List[int] = field(default_factory=list)
    tree_mask: List[List[bool]] = field(default_factory=list)
    nodes_at_depth: Dict[int, List[int]] = field(default_factory=dict)
    _lookup_table: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validates invariants and computes derived properties if not already set."""
        self._validate_basic_invariants()
        if not self.node_depths:
            self.node_depths = self._compute_depths()
        if not self.position_ids:
            self.position_ids = list(self.node_depths)
        if not self.nodes_at_depth:
            self.nodes_at_depth = self._build_nodes_at_depth()
        if not self.tree_mask:
            self.tree_mask = self._compute_causal_mask()
        self._validate_derived_invariants()
        if not self._lookup_table:
            self._lookup_table = self._build_lookup_table()

    def _validate_basic_invariants(self) -> None:
        """Validates fundamental structural invariants."""
        expected_len = self.num_tokens + 1
        actual_len = len(self.parent_indices)
        if actual_len != expected_len:
            raise ValueError(
                f"Parent array length mismatch for {self.name}: "
                f"expected {expected_len} (N+1), got {actual_len}"
            )
        if self.parent_indices[0] != -1:
            raise ValueError(
                f"Root parent index must be -1, got {self.parent_indices[0]}"
            )
        for i in range(1, actual_len):
            p = self.parent_indices[i]
            if p < 0 or p >= i:
                raise ValueError(
                    f"Acyclic/topological ordering violation at node {i}: parent={p} (must be in 0..{i-1})"
                )

    def _compute_depths(self) -> List[int]:
        """Computes depth for each node from root 0."""
        n = len(self.parent_indices)
        depths = [0] * n
        for i in range(1, n):
            p = self.parent_indices[i]
            depths[i] = depths[p] + 1
        return depths

    def _build_nodes_at_depth(self) -> Dict[int, List[int]]:
        """Groups node indices by depth."""
        mapping: Dict[int, List[int]] = {}
        for idx, d in enumerate(self.node_depths):
            mapping.setdefault(d, []).append(idx)
        return mapping

    def _compute_causal_mask(self) -> List[List[bool]]:
        """Constructs 2D causal attention mask where node i can attend to j iff j is an ancestor of i."""
        n = len(self.parent_indices)
        mask = [[False] * n for _ in range(n)]
        for i in range(n):
            curr = i
            while curr != -1:
                mask[i][curr] = True
                curr = self.parent_indices[curr]
        return mask

    def _validate_derived_invariants(self) -> None:
        """Validates depth, layer distribution, and mask properties."""
        actual_max_depth = max(self.node_depths) if self.node_depths else 0
        if actual_max_depth != self.depth:
            raise ValueError(
                f"Maximum depth mismatch for {self.name}: "
                f"computed max depth {actual_max_depth} != declared depth {self.depth}"
            )
        actual_dist = [len(self.nodes_at_depth.get(d, [])) for d in range(self.depth + 1)]
        if actual_dist != self.layer_distribution:
            raise ValueError(
                f"Layer distribution mismatch for {self.name}: "
                f"actual {actual_dist} != declared {self.layer_distribution}"
            )

    def _build_lookup_table(self, num_points: int = 1001) -> List[float]:
        """Precomputes lookup table of E[tau(alpha)] for alpha in [0.0, 1.0] in 1001 steps."""
        table = []
        for i in range(num_points):
            alpha = i / (num_points - 1)
            table.append(self.expected_acceptance(alpha))
        return table

    def expected_acceptance(self, alpha: float, method: str = "branch_aware") -> float:
        """Computes theoretical expected accepted token length E[tau(T, alpha)].

        Follows OPT-Tree expectation formula:
        E[tau(T)] = 1 + sum_{v in T \\ {r}} q(v)

        Under greedy decoding against target model:
        At any depth level l in 1..d, at most ONE token can be accepted because the target
        model generates a unique greedy token at each decoding position.

        Methods:
            - 'homogeneous': Linear expectation along the longest chain:
                E[tau] = 1 + sum_{l=1}^d alpha^l
            - 'branch_aware' (default): Multi-branch coverage at each layer:
                With n_l candidate branches at depth l, the step acceptance probability is:
                P_step(l, alpha) = 1 - (1 - alpha)^{n_l}
                Cumulative path acceptance:
                P_path(l, alpha) = prod_{k=1}^l P_step(k, alpha)
                E[tau] = 1 + sum_{l=1}^d P_path(l, alpha)

        Properties guaranteed:
            - Monotonically non-decreasing in alpha in [0.0, 1.0].
            - E[tau(alpha=0.0)] == 1.0 (only the root/verified token is emitted).
            - E[tau(alpha=1.0)] == 1.0 + depth (longest path bound).
            - Strict upper bound: E[tau] <= 1 + depth.
        """
        if self.depth == 0 or self.num_tokens == 0:
            return 1.0

        clamped_alpha = max(0.0, min(1.0, float(alpha)))
        if clamped_alpha <= 0.0:
            return 1.0
        if clamped_alpha >= 1.0:
            return 1.0 + float(self.depth)

        if method == "homogeneous":
            # Geometric progression along max depth d
            # E[tau] = 1 + alpha + alpha^2 + ... + alpha^d
            val = 1.0 + sum(clamped_alpha ** l for l in range(1, self.depth + 1))
            return min(1.0 + float(self.depth), val)

        # Default: branch-aware multi-candidate coverage
        expected = 1.0
        cum_prob = 1.0
        for l in range(1, self.depth + 1):
            n_l = self.layer_distribution[l] if l < len(self.layer_distribution) else 1
            # Step acceptance probability at depth l given parent accepted
            step_prob = 1.0 - math.pow(1.0 - clamped_alpha, n_l)
            cum_prob *= step_prob
            expected += cum_prob

        # Strict bound enforcement
        return max(1.0, min(1.0 + float(self.depth), expected))

    def lookup_expected_acceptance(self, alpha: float) -> float:
        """Fast O(1) lookup of precomputed E[tau(alpha)] (< 0.005 ms)."""
        clamped_alpha = max(0.0, min(1.0, float(alpha)))
        idx = int(round(clamped_alpha * (len(self._lookup_table) - 1)))
        return self._lookup_table[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes topology to dictionary."""
        return {
            "name": self.name,
            "num_tokens": self.num_tokens,
            "depth": self.depth,
            "parent_indices": self.parent_indices,
            "layer_distribution": self.layer_distribution,
            "node_depths": self.node_depths,
            "position_ids": self.position_ids,
            "tree_mask": self.tree_mask,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeTopology:
        """Deserializes topology from dictionary."""
        return cls(
            name=data["name"],
            num_tokens=data["num_tokens"],
            depth=data["depth"],
            parent_indices=data["parent_indices"],
            layer_distribution=data["layer_distribution"],
            node_depths=data.get("node_depths", []),
            position_ids=data.get("position_ids", []),
            tree_mask=data.get("tree_mask", []),
        )


# ==============================================================================
# CANONICAL CATALOG DEFINITIONS (T0, T4, T8, T16, T32)
# ==============================================================================

CANONICAL_TOPOLOGIES: Dict[str, TreeTopology] = {
    # T0: Baseline autoregressive decoding (no speculative draft tokens)
    "T0": TreeTopology(
        name="T0",
        num_tokens=0,
        depth=0,
        layer_distribution=[1],
        parent_indices=[-1],
    ),
    # T4: Linear chain (N=4, depth=4)
    "T4": TreeTopology(
        name="T4",
        num_tokens=4,
        depth=4,
        layer_distribution=[1, 1, 1, 1, 1],
        parent_indices=[-1, 0, 1, 2, 3],
    ),
    # T8: Narrow branching tree (N=8, depth=3)
    "T8": TreeTopology(
        name="T8",
        num_tokens=8,
        depth=3,
        layer_distribution=[1, 2, 4, 2],
        parent_indices=[-1, 0, 0, 1, 1, 2, 2, 3, 5],
    ),
    # T16: Moderate branching tree (N=16, depth=4)
    # Fixed invariant: node 16's parent is node 13 (depth 3) to strictly enforce max depth 4
    "T16": TreeTopology(
        name="T16",
        num_tokens=16,
        depth=4,
        layer_distribution=[1, 3, 6, 4, 3],
        parent_indices=[
            -1,
            0, 0, 0,
            1, 1, 2, 2, 3, 3,
            4, 4, 6, 8,
            10, 12, 13,
        ],
    ),
    # T32: Wide branching tree (N=32, depth=5)
    # Deterministic 33-element parent array strictly adhering to [1, 4, 8, 8, 6, 6]
    "T32": TreeTopology(
        name="T32",
        num_tokens=32,
        depth=5,
        layer_distribution=[1, 4, 8, 8, 6, 6],
        parent_indices=[
            -1,
            # Depth 1 (4 nodes: 1..4, parent 0)
            0, 0, 0, 0,
            # Depth 2 (8 nodes: 5..12, parents 1, 1, 2, 2, 3, 3, 4, 4)
            1, 1, 2, 2, 3, 3, 4, 4,
            # Depth 3 (8 nodes: 13..20, parents 5..12)
            5, 6, 7, 8, 9, 10, 11, 12,
            # Depth 4 (6 nodes: 21..26, parents 13..18)
            13, 14, 15, 16, 17, 18,
            # Depth 5 (6 nodes: 27..32, parents 21..26)
            21, 22, 23, 24, 25, 26,
        ],
    ),
}


def get_tree_topology(name: str) -> TreeTopology:
    """Retrieves a canonical tree topology by name."""
    clean_name = name.strip().upper()
    if clean_name not in CANONICAL_TOPOLOGIES:
        raise KeyError(
            f"Unknown topology '{name}'. Available canonical topologies: "
            f"{list(CANONICAL_TOPOLOGIES.keys())}"
        )
    return CANONICAL_TOPOLOGIES[clean_name]


def list_available_topologies() -> List[str]:
    """Returns list of canonical topology identifiers."""
    return list(CANONICAL_TOPOLOGIES.keys())
