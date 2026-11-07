"""Unit test suite for TreeTopology module (Task M4-02-01).

Validates:
1. Canonical catalog definitions (T0, T4, T8, T16, T32).
2. Topological invariants (acyclicity, exact depth, layer distribution).
3. 2D Causal Attention Mask (reflexivity, causality, branch exclusivity, no sibling leakage).
4. Expected acceptance E[tau(alpha)] monotonicity, bounds [1.0, 1+d], and lookup speed (<0.005ms).
5. Error handling and serialization round-trip.
"""
import time
import pytest

try:
    from flexsds.core.tree_topology import (
        TreeTopology,
        CANONICAL_TOPOLOGIES,
        get_tree_topology,
        list_available_topologies,
    )
except ModuleNotFoundError:
    from src.flexsds.core.tree_topology import (
        TreeTopology,
        CANONICAL_TOPOLOGIES,
        get_tree_topology,
        list_available_topologies,
    )


def test_list_and_get_topologies():
    """Verifies that all 5 canonical topologies are accessible."""
    available = list_available_topologies()
    expected = ["T0", "T4", "T8", "T16", "T32"]
    assert set(expected).issubset(set(available))

    for name in expected:
        topo = get_tree_topology(name)
        assert isinstance(topo, TreeTopology)
        assert topo.name == name

    with pytest.raises(KeyError):
        get_tree_topology("UNKNOWN_T64")


def test_canonical_catalog_invariants():
    """Verifies node counts, max depths, and layer distributions for all 5 canonical trees."""
    catalog_specs = {
        "T0": {"n": 0, "d": 0, "dist": [1]},
        "T4": {"n": 4, "d": 4, "dist": [1, 1, 1, 1, 1]},
        "T8": {"n": 8, "d": 3, "dist": [1, 2, 4, 2]},
        "T16": {"n": 16, "d": 4, "dist": [1, 3, 6, 4, 3]},
        "T32": {"n": 32, "d": 5, "dist": [1, 4, 8, 8, 6, 6]},
    }

    for name, spec in catalog_specs.items():
        topo = get_tree_topology(name)
        assert topo.num_tokens == spec["n"], f"Node count mismatch for {name}"
        assert topo.depth == spec["d"], f"Depth mismatch for {name}"
        assert topo.layer_distribution == spec["dist"], f"Layer distribution mismatch for {name}"
        assert len(topo.parent_indices) == spec["n"] + 1, f"Parent array length mismatch for {name}"
        assert topo.parent_indices[0] == -1, f"Root parent must be -1 for {name}"

        # Position IDs must match node depths
        assert topo.position_ids == topo.node_depths
        assert max(topo.node_depths) == spec["d"]


def test_acyclic_and_topological_ordering():
    """Verifies every non-root node has parent index strictly less than its own index."""
    for name in ["T0", "T4", "T8", "T16", "T32"]:
        topo = get_tree_topology(name)
        for i in range(1, len(topo.parent_indices)):
            p = topo.parent_indices[i]
            assert 0 <= p < i, f"Node {i} in {name} has invalid parent index {p}"


def test_tree_16_depth_invariant_fixed():
    """Verifies T16 maximum depth is strictly 4 and node 16 depth is exactly 4 (Review R2 S04 fix)."""
    t16 = get_tree_topology("T16")
    assert t16.depth == 4
    assert max(t16.node_depths) == 4
    # Node 16 parent is node 13
    assert t16.parent_indices[16] == 13
    # Trace path for node 16: 0 -> 3 -> 8 -> 13 -> 16 (length 4)
    assert t16.node_depths[16] == 4
    assert t16.node_depths[13] == 3
    assert t16.node_depths[8] == 2
    assert t16.node_depths[3] == 1
    assert t16.node_depths[0] == 0


def test_tree_32_unique_topology_and_depth():
    """Verifies T32 has 33 nodes, max depth 5, and exactly matches [1, 4, 8, 8, 6, 6] (Review R2 S05 fix)."""
    t32 = get_tree_topology("T32")
    assert len(t32.parent_indices) == 33
    assert t32.depth == 5
    assert max(t32.node_depths) == 5
    assert t32.layer_distribution == [1, 4, 8, 8, 6, 6]

    # Verify counts per depth
    for d, expected_count in enumerate([1, 4, 8, 8, 6, 6]):
        nodes = t32.nodes_at_depth[d]
        assert len(nodes) == expected_count, f"T32 depth {d} count {len(nodes)} != {expected_count}"


def test_causal_mask_invariants():
    """Verifies reflexive, lower-triangular, and ancestor-visibility properties of tree attention masks."""
    for name in ["T0", "T4", "T8", "T16", "T32"]:
        topo = get_tree_topology(name)
        mask = topo.tree_mask
        n = topo.num_tokens + 1

        assert len(mask) == n
        for row in mask:
            assert len(row) == n

        for i in range(n):
            # 1. Reflexive
            assert mask[i][i] is True, f"{name}: node {i} must attend to itself"

            # 2. Causality (no future attention)
            for j in range(i + 1, n):
                assert mask[i][j] is False, f"{name}: node {i} cannot attend to future node {j}"

            # 3. Ancestor trace
            curr = i
            ancestors = set()
            while curr != -1:
                ancestors.add(curr)
                curr = topo.parent_indices[curr]

            for j in range(n):
                expected_visible = (j in ancestors)
                assert mask[i][j] == expected_visible, (
                    f"{name}: mask[{i}][{j}]={mask[i][j]} != expected {expected_visible}"
                )


def test_branch_exclusivity_no_sibling_leakage():
    """Verifies that distinct sibling branches cannot attend to each other."""
    for name in ["T8", "T16", "T32"]:
        topo = get_tree_topology(name)
        mask = topo.tree_mask
        n = topo.num_tokens + 1

        for i in range(1, n):
            for j in range(1, n):
                if i != j and topo.parent_indices[i] == topo.parent_indices[j]:
                    # Siblings cannot attend to each other
                    assert mask[i][j] is False, f"{name}: sibling leakage from {j} to {i}"
                    assert mask[j][i] is False, f"{name}: sibling leakage from {i} to {j}"


def test_expected_acceptance_bounds_and_monotonicity():
    """Verifies that E[tau(alpha)] strictly stays within [1.0, 1 + d] and is monotonic."""
    for name in ["T0", "T4", "T8", "T16", "T32"]:
        topo = get_tree_topology(name)
        d = topo.depth

        # Boundary checks
        assert topo.expected_acceptance(0.0) == 1.0
        assert topo.expected_acceptance(1.0) == 1.0 + d

        prev_val = 1.0
        # Sweep alpha from 0.0 to 1.0 in steps of 0.05
        for step in range(21):
            alpha = step / 20.0
            val = topo.expected_acceptance(alpha)

            assert 1.0 <= val <= 1.0 + d, f"{name}: val {val} out of bounds [1.0, {1+d}] at alpha={alpha}"
            assert val >= prev_val - 1e-9, f"{name}: non-monotonic drop at alpha={alpha}: {val} < {prev_val}"
            prev_val = val


def test_fast_lookup_table_speed_and_accuracy():
    """Verifies lookup_expected_acceptance is accurate and runs in < 0.005 ms (< 5 microseconds)."""
    t16 = get_tree_topology("T16")

    # Accuracy against analytical calculation
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        analytical = t16.expected_acceptance(alpha)
        looked_up = t16.lookup_expected_acceptance(alpha)
        assert abs(analytical - looked_up) < 1e-3, f"Mismatch at alpha={alpha}: {analytical} vs {looked_up}"

    # Latency benchmark
    iterations = 10000
    start = time.perf_counter()
    for i in range(iterations):
        alpha = (i % 100) / 100.0
        _ = t16.lookup_expected_acceptance(alpha)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000.0

    print(f"\nAverage lookup latency: {avg_ms * 1000:.3f} microseconds ({avg_ms:.6f} ms)")
    # Must be well below 0.005 ms (5 microseconds)
    assert avg_ms < 0.005, f"Lookup too slow: {avg_ms:.6f} ms >= 0.005 ms"


def test_serialization_roundtrip():
    """Verifies to_dict and from_dict roundtrip preserve exact topology structure."""
    for name in ["T0", "T4", "T8", "T16", "T32"]:
        original = get_tree_topology(name)
        data = original.to_dict()
        restored = TreeTopology.from_dict(data)

        assert restored.name == original.name
        assert restored.num_tokens == original.num_tokens
        assert restored.depth == original.depth
        assert restored.parent_indices == original.parent_indices
        assert restored.layer_distribution == original.layer_distribution
        assert restored.node_depths == original.node_depths
        assert restored.position_ids == original.position_ids
        assert restored.tree_mask == original.tree_mask


def test_invalid_topologies_raise_error():
    """Verifies validation errors on corrupted parent indices or depth."""
    # 1. Invalid root parent != -1
    with pytest.raises(ValueError, match="Root parent index must be -1"):
        TreeTopology(
            name="T_BAD_ROOT",
            num_tokens=2,
            depth=2,
            layer_distribution=[1, 1, 1],
            parent_indices=[0, 0, 1],
        )

    # 2. Cycle / Topological order violation
    with pytest.raises(ValueError, match="Acyclic/topological ordering violation"):
        TreeTopology(
            name="T_CYCLE",
            num_tokens=2,
            depth=2,
            layer_distribution=[1, 1, 1],
            parent_indices=[-1, 2, 1],
        )

    # 3. Declared depth mismatch
    with pytest.raises(ValueError, match="Maximum depth mismatch"):
        TreeTopology(
            name="T_WRONG_DEPTH",
            num_tokens=2,
            depth=5,  # Actual is 2
            layer_distribution=[1, 1, 1],
            parent_indices=[-1, 0, 1],
        )
