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
        save_hardware_profile,
        run_hardware_micro_benchmark,
        USER_PROFILES_DIR,
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
        save_hardware_profile,
        run_hardware_micro_benchmark,
        USER_PROFILES_DIR,
    )

__all__ = [
    "TreeTopology",
    "CANONICAL_TOPOLOGIES",
    "get_tree_topology",
    "list_available_topologies",
    "FlexCostModel",
    "SolverResult",
    "load_profile_for_hardware",
    "save_hardware_profile",
    "run_hardware_micro_benchmark",
    "USER_PROFILES_DIR",
]
