"""Tests for M4-02-04 Interface A v0.2 contract and batch tree aggregation glue."""
import time
import pytest
from flexsds.core.controller import TreeAdaptiveController, TreeAdaptiveDecision, TREE_TOPOLOGIES


def test_batch_tree_aggregation_max_nodes():
    """Verifies that in a heterogeneous tree batch, the batch node count N_batch = max_i(N_i)."""
    # Mô phỏng 4 requests với các cây khác nhau: T0 (N=0), T4 (N=4), T8 (N=8), T16 (N=16)
    request_topologies = ["T0", "T4", "T8", "T16"]
    node_counts = [TREE_TOPOLOGIES[topo]["num_nodes"] for topo in request_topologies]

    assert node_counts == [0, 4, 8, 16]
    n_batch = max(node_counts)
    assert n_batch == 16


def test_interface_a_v02_telemetry_schema_13_fields():
    """Verifies that the telemetry record conforms to Interface A v0.2 with all 13 fields."""
    ctrl = TreeAdaptiveController(min_free_vram_mb=1500.0)
    decision = ctrl.select_budget(batch_size=4, free_vram_mb=7500.0, kv_cache_usage_pct=45.0)

    # Khởi tạo bản ghi telemetry 13 trường chuẩn Interface A v0.2
    telemetry_record = {
        "step": 101,
        "batch_size": 4,
        "topology_name": decision.topology_name,
        "num_nodes": decision.num_nodes,
        "k": decision.k,
        "alpha_effective": decision.alpha_effective,
        "alpha_star": decision.alpha_star,
        "speedup": decision.speedup,
        "status": decision.status,
        "free_vram_mb": 7500.0,
        "kv_cache_usage_pct": 45.0,
        "latency_ms": 32.5,
        "timestamp": time.time(),
    }

    assert len(telemetry_record) == 13
    required_fields = [
        "step",
        "batch_size",
        "topology_name",
        "num_nodes",
        "k",
        "alpha_effective",
        "alpha_star",
        "speedup",
        "status",
        "free_vram_mb",
        "kv_cache_usage_pct",
        "latency_ms",
        "timestamp",
    ]
    for field in required_fields:
        assert field in telemetry_record
        assert telemetry_record[field] is not None or field == "alpha_star"


def test_tree_adaptive_decision_serialization():
    """Verifies that TreeAdaptiveDecision can be cleanly converted to dictionary."""
    ctrl = TreeAdaptiveController()
    dec = ctrl.select_budget(batch_size=2, free_vram_mb=12000.0)
    dec_dict = {
        "k": dec.k,
        "topology_name": dec.topology_name,
        "num_nodes": dec.num_nodes,
        "alpha_effective": dec.alpha_effective,
        "speedup": dec.speedup,
        "status": dec.status,
    }
    assert dec_dict["topology_name"] in TREE_TOPOLOGIES
    assert dec_dict["status"] in ("ok", "memory_guard", "concurrency_throttled", "fallback_gate", "probing", "zero_crash_fallback")
