"""Server API, lazily imported to keep host metadata independent of policy."""

def __getattr__(name):
    if name in ("AdaptiveBudgetPlanner", "StepPlan"):
        from src.server import budget_planner
        value = getattr(budget_planner, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = ["AdaptiveBudgetPlanner", "StepPlan"]
