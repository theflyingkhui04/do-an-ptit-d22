"""FlexSDS Core module."""
try:
    from flexsds.core.tree_topology import (
        TreeTopology,
        CANONICAL_TOPOLOGIES,
        get_tree_topology,
        list_available_topologies,
    )
    from flexsds.core.cost_model import (
        FlexCostModel,
        SolverResult,
        load_profile_for_hardware,
    )
except ModuleNotFoundError:
    from src.flexsds.core.tree_topology import (
        TreeTopology,
        CANONICAL_TOPOLOGIES,
        get_tree_topology,
        list_available_topologies,
    )
    from src.flexsds.core.cost_model import (
        FlexCostModel,
        SolverResult,
        load_profile_for_hardware,
    )

__all__ = [
    "TreeTopology",
    "CANONICAL_TOPOLOGIES",
    "get_tree_topology",
    "list_available_topologies",
    "FlexCostModel",
    "SolverResult",
    "load_profile_for_hardware",
]
