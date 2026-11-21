"""Core exports, loaded on demand so scalar serving does not import tree code."""
from importlib import import_module

_EXPORTS = {
    "tree_topology": ("TreeTopology", "CANONICAL_TOPOLOGIES", "get_tree_topology", "list_available_topologies"),
    "cost_model": ("FlexCostModel", "SolverResult", "load_profile_for_hardware", "save_hardware_profile", "run_hardware_micro_benchmark", "USER_PROFILES_DIR"),
}
__all__ = [name for names in _EXPORTS.values() for name in names]

def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
