"""FlexSDS Core module."""
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

__all__ = [
    "TreeTopology",
    "CANONICAL_TOPOLOGIES",
    "get_tree_topology",
    "list_available_topologies",
]
