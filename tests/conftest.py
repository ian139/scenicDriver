from __future__ import annotations

import pytest

# Marker taxonomy (registered in pyproject.toml). CI runs every marker; these
# exist for local selection, e.g. `pytest -m "not slow"` or `pytest -m native`.
_MODULE_MARKERS: dict[str, tuple[str, ...]] = {
    "test_compact_runtime.py": ("native",),
    "test_route_supervisor.py": ("integration",),
    "test_routing_benchmark.py": ("integration",),
    "test_smoke.py": ("integration",),
    "test_vast_train_runtime.py": ("integration",),
}

_SLOW_TESTS = frozenset(
    {
        "test_random_control_uses_full_pool_before_stratum_score_shortlist",
        "test_compact_comparator_alloc_failure_preserves_minus_one_and_frees_scratch",
        "test_compact_deep_equal_cost_shared_prefix_no_stack_overflow",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for marker in _MODULE_MARKERS.get(item.path.name, ()):
            item.add_marker(marker)
        if getattr(item, "originalname", item.name) in _SLOW_TESTS:
            item.add_marker("slow")
