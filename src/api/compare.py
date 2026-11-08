"""Field-level comparison of a lane's Node 1 JSON with the vanilla lane's JSON."""

from __future__ import annotations

from typing import Any

SECTIONS = ("invoice", "seller", "buyer", "totals", "items", "taxes")


def leaf_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Leaf values of the compared sections, keyed by path (``totals.total_amount.value``, ``items[0].sku``)."""
    leaves: dict[str, Any] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{path}.{key}")
        elif isinstance(value, list):
            if not value:
                leaves[path] = []
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        else:
            leaves[path] = value

    for section in SECTIONS:
        walk(payload.get(section), section)
    return leaves


def compare_fields(payload: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    ours, theirs = leaf_fields(payload), leaf_fields(reference)
    paths = sorted(set(ours) | set(theirs))
    differing = [p for p in paths if p not in ours or p not in theirs or ours[p] != theirs[p]]
    return {"equal_fields": len(paths) - len(differing), "total_fields": len(paths),
            "differing_fields": differing}


def speedup(reference_ms: float | None, lane_ms: float | None) -> float | None:
    if not reference_ms or not lane_ms:
        return None
    return reference_ms / lane_ms
