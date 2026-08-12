from __future__ import annotations

import os
import sys
import ctypes
import math
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .cancellation import RoutingTimeout


_C_SOURCE = Path(__file__).with_name("_compact_search_c.c")
_C_LIB_PATH = Path(__file__).with_name("_compact_search_c.so")

_LIB: Optional[ctypes.CDLL] = None


class _CompactGraphSpec(ctypes.Structure):
    _fields_ = [
        ("node_count", ctypes.c_int64),
        ("edge_count", ctypes.c_int64),
        ("traversal_count", ctypes.c_int64),
        ("forward_indptr", ctypes.POINTER(ctypes.c_int64)),
        ("forward_indices", ctypes.POINTER(ctypes.c_int32)),
        ("reverse_indptr", ctypes.POINTER(ctypes.c_int64)),
        ("reverse_indices", ctypes.POINTER(ctypes.c_int32)),
        ("reverse_positions", ctypes.POINTER(ctypes.c_int64)),
        ("trav_travel_time", ctypes.POINTER(ctypes.c_double)),
        ("trav_highway_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("trav_scenic_score", ctypes.POINTER(ctypes.c_double)),
        ("trav_scenic_byway_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("trav_edge_rank", ctypes.POINTER(ctypes.c_int32)),
        ("trav_reverse", ctypes.POINTER(ctypes.c_uint8)),
        ("edge_id_strings", ctypes.POINTER(ctypes.c_uint8)),
        ("edge_id_offsets", ctypes.POINTER(ctypes.c_int64)),
    ]


class _CostSpec(ctypes.Structure):
    _fields_ = [
        ("scenic_weight", ctypes.c_double),
        ("strict_highways", ctypes.c_int),
        ("highway_preference", ctypes.c_double),
        ("travel_weight", ctypes.c_double),
        ("scenic_reward", ctypes.c_double),
        ("highway_penalty", ctypes.c_double),
        ("scenic_byway_bonus", ctypes.c_double),
        ("lagrangian_multiplier", ctypes.c_double),
        ("cost_limit", ctypes.c_double),
    ]


def _ensure_compiled_library() -> Optional[ctypes.CDLL]:
    global _LIB
    if _LIB is not None:
        return _LIB

    if not _C_SOURCE.exists():
        return None

    try:
        need_compile = not _C_LIB_PATH.exists()
        if not need_compile:
            need_compile = _C_SOURCE.stat().st_mtime > _C_LIB_PATH.stat().st_mtime

        if need_compile:
            compiler = "clang" if sys.platform == "darwin" else "gcc"
            cmd = [
                compiler,
                "-O3",
                "-shared",
                "-fPIC",
                str(_C_SOURCE),
                "-o",
                str(_C_LIB_PATH),
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=10)
            if res.returncode != 0 or not _C_LIB_PATH.exists():
                return None

        lib = ctypes.CDLL(str(_C_LIB_PATH))
        ranked_argtypes = [
            ctypes.POINTER(_CompactGraphSpec),
            ctypes.POINTER(_CostSpec),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_double,
        ]
        # The edge-score sidecar entry adds one argument after the cost spec:
        # the canonical-edge-rank-indexed scenic score array.  It is selected
        # only for active CompactRoadGraph score sidecars; the ranked and
        # legacy entries below keep their traversal-indexed contract.
        lib.compact_bidirectional_search_alloc_ranked_edge_scores.argtypes = (
            [ctypes.POINTER(_CompactGraphSpec), ctypes.POINTER(_CostSpec)]
            + [ctypes.POINTER(ctypes.c_double)]
            + ranked_argtypes[2:]
        )
        lib.compact_bidirectional_search_alloc_ranked_edge_scores.restype = ctypes.c_int32
        lib.compact_bidirectional_search_alloc.argtypes = [
            ctypes.POINTER(_CompactGraphSpec),
            ctypes.POINTER(_CostSpec),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_double,
        ]
        lib.compact_bidirectional_search_alloc.restype = ctypes.c_int32
        lib.compact_bidirectional_search_alloc_ranked.argtypes = ranked_argtypes
        lib.compact_bidirectional_search_alloc_ranked.restype = ctypes.c_int32
        lib.compact_free_positions.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        lib.compact_free_positions.restype = None
        _LIB = lib
        return _LIB
    except Exception:
        return None


def compact_search_available() -> bool:
    return _ensure_compiled_library() is not None


def run_compact_bidirectional_search(
    topology: Any,
    cost_function: Any,
    forward_seeds: List[Tuple[str, float, Tuple[object, ...], Any]],
    reverse_seeds: List[Tuple[str, float, Tuple[object, ...], Any]],
    *,
    lagrangian_multiplier: float = 0.0,
    cost_limit: Optional[float] = None,
    deadline_seconds: Optional[float] = None,
) -> Optional[Tuple[List[int], float, int, int]]:
    lib = _ensure_compiled_library()
    if lib is None:
        return None

    graph = topology.graph
    if hasattr(graph, "base_graph"):
        graph = graph.base_graph

    if not hasattr(graph, "_sections"):
        return None

    sections = graph._sections

    def get_ptr(name: str, ctype: Any) -> Any:
        arr = sections.get(name)
        if arr is None or not hasattr(arr, "ctypes"):
            return None
        return arr.ctypes.data_as(ctypes.POINTER(ctype))

    fwd_indptr = get_ptr("forward_indptr", ctypes.c_int64)
    fwd_indices = get_ptr("forward_indices", ctypes.c_int32)
    rev_indptr = get_ptr("reverse_indptr", ctypes.c_int64)
    rev_indices = get_ptr("reverse_indices", ctypes.c_int32)
    rev_positions = get_ptr("reverse_positions", ctypes.c_int64)
    trav_travel_time = get_ptr("trav_travel_time_minutes", ctypes.c_double)
    trav_highway_mask = get_ptr("trav_highway_mask", ctypes.c_uint8)
    trav_scenic_score = get_ptr("trav_scenic_score", ctypes.c_double)
    trav_byway_mask = get_ptr("trav_scenic_byway_mask", ctypes.c_uint8)
    trav_edge_rank = get_ptr("trav_edge_rank", ctypes.c_int32)
    trav_reverse = get_ptr("trav_reverse", ctypes.c_uint8)
    # The edge id string table is a raw (non-numeric) section; wrap the
    # memoryview in a numpy uint8 view so it can supply a ctypes pointer.
    # Both stay NULL when the section is missing and the native search then
    # falls back to numeric edge-rank ordering for tie-breaks.
    edge_id_strings = None
    edge_id_offsets = get_ptr("edge_id_offsets", ctypes.c_int64)
    raw_edge_ids = sections.get("edge_id_strings")
    if raw_edge_ids is not None:
        try:
            import numpy as _np

            edge_id_strings = _np.frombuffer(
                raw_edge_ids, dtype=_np.uint8
            ).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        except Exception:
            edge_id_strings = None

    if (
        fwd_indptr is None
        or fwd_indices is None
        or rev_indptr is None
        or rev_indices is None
        or rev_positions is None
        or trav_travel_time is None
        or trav_highway_mask is None
    ):
        return None

    # The compact score sidecar is indexed by canonical edge rank, not by
    # traversal position, so the sidecar-aware native entry receives the raw
    # sidecar array and indexes it via trav_edge_rank[position].  The legacy
    # trav_scenic_score section (traversal-indexed) stays untouched and keeps
    # serving the non-sidecar path below.
    sidecar = getattr(graph, "_active_score_sidecar", None)
    edge_scenic_score_by_rank = None
    if (
        sidecar is not None
        and hasattr(sidecar, "values")
        and hasattr(sidecar.values, "ctypes")
    ):
        if trav_edge_rank is None:
            # Edge-rank indexing is impossible without the rank table; fall
            # back to the Python CSR search rather than mis-index by position.
            return None
        if (
            getattr(
                lib, "compact_bidirectional_search_alloc_ranked_edge_scores", None
            )
            is None
        ):
            # Stale library without the sidecar-aware entry: never substitute
            # the sidecar into the traversal-indexed slot; fall back.
            return None
        edge_scenic_score_by_rank = sidecar.values.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

    spec = _CompactGraphSpec(
        node_count=int(graph.node_count),
        edge_count=int(graph.edge_count),
        traversal_count=int(graph.traversal_count),
        forward_indptr=fwd_indptr,
        forward_indices=fwd_indices,
        reverse_indptr=rev_indptr,
        reverse_indices=rev_indices,
        reverse_positions=rev_positions,
        trav_travel_time=trav_travel_time,
        trav_highway_mask=trav_highway_mask,
        trav_scenic_score=trav_scenic_score,
        trav_scenic_byway_mask=trav_byway_mask,
        trav_edge_rank=trav_edge_rank,
        trav_reverse=trav_reverse,
        edge_id_strings=edge_id_strings,
        edge_id_offsets=edge_id_offsets,
    )

    weights = getattr(cost_function, "weights", None)
    native_deadline_seconds = -1.0
    if deadline_seconds is not None:
        native_deadline_seconds = float(deadline_seconds)
        if (
            not math.isfinite(native_deadline_seconds)
            or native_deadline_seconds < 0.0
        ):
            raise ValueError("compiled search deadline must be finite and non-negative")
    cost_spec = _CostSpec(
        scenic_weight=float(getattr(cost_function, "scenic_weight", 0.0)),
        strict_highways=1 if getattr(cost_function, "strict_highways", False) else 0,
        highway_preference=float(getattr(cost_function, "highway_preference", 0.0)),
        travel_weight=float(getattr(weights, "travel_time", 1.0) if weights else 1.0),
        scenic_reward=float(getattr(weights, "scenic_reward", 0.0) if weights else 0.0),
        highway_penalty=float(getattr(weights, "highway_penalty", 0.0) if weights else 0.0),
        scenic_byway_bonus=float(getattr(weights, "scenic_byway_bonus", 0.0) if weights else 0.0),
        lagrangian_multiplier=float(lagrangian_multiplier),
        cost_limit=float(cost_limit) if cost_limit is not None else 0.0,
    )

    node_index = topology.node_index
    fwd_seed_indices: list[int] = []
    fwd_nodes = (ctypes.c_int32 * len(forward_seeds))()
    fwd_costs = (ctypes.c_double * len(forward_seeds))()
    fwd_rank_primary = (ctypes.c_int32 * len(forward_seeds))()
    fwd_rank_secondary = (ctypes.c_int32 * len(forward_seeds))()
    for seed_index, (node_id, cost, rank, _edge) in enumerate(forward_seeds):
        node_rank = node_index.get(str(node_id))
        if node_rank is not None:
            compact_index = len(fwd_seed_indices)
            fwd_nodes[compact_index] = int(node_rank)
            fwd_costs[compact_index] = float(cost)
            rank_tuple = tuple(rank)
            fwd_rank_primary[compact_index] = (
                int(rank_tuple[0]) if len(rank_tuple) >= 2 else 0
            )
            fwd_rank_secondary[compact_index] = (
                int(rank_tuple[1]) if len(rank_tuple) >= 2 else 0
            )
            fwd_seed_indices.append(seed_index)

    rev_seed_indices: list[int] = []
    rev_nodes = (ctypes.c_int32 * len(reverse_seeds))()
    rev_costs = (ctypes.c_double * len(reverse_seeds))()
    rev_rank_primary = (ctypes.c_int32 * len(reverse_seeds))()
    rev_rank_secondary = (ctypes.c_int32 * len(reverse_seeds))()
    for seed_index, (node_id, cost, rank, _edge) in enumerate(reverse_seeds):
        node_rank = node_index.get(str(node_id))
        if node_rank is not None:
            compact_index = len(rev_seed_indices)
            rev_nodes[compact_index] = int(node_rank)
            rev_costs[compact_index] = float(cost)
            rank_tuple = tuple(rank)
            rev_rank_primary[compact_index] = (
                int(rank_tuple[0]) if len(rank_tuple) >= 2 else 0
            )
            rev_rank_secondary[compact_index] = (
                int(rank_tuple[1]) if len(rank_tuple) >= 2 else 0
            )
            rev_seed_indices.append(seed_index)

    if not fwd_seed_indices or not rev_seed_indices:
        return None

    out_pos_ptr = ctypes.POINTER(ctypes.c_int64)()
    out_cost = ctypes.c_double(0.0)
    out_fwd_idx = ctypes.c_int32(-1)
    out_rev_idx = ctypes.c_int32(-1)

    if edge_scenic_score_by_rank is not None:
        res = lib.compact_bidirectional_search_alloc_ranked_edge_scores(
            ctypes.byref(spec),
            ctypes.byref(cost_spec),
            edge_scenic_score_by_rank,
            fwd_nodes,
            fwd_costs,
            fwd_rank_primary,
            fwd_rank_secondary,
            len(fwd_seed_indices),
            rev_nodes,
            rev_costs,
            rev_rank_primary,
            rev_rank_secondary,
            len(rev_seed_indices),
            ctypes.byref(out_pos_ptr),
            ctypes.byref(out_cost),
            ctypes.byref(out_fwd_idx),
            ctypes.byref(out_rev_idx),
            native_deadline_seconds,
        )
    else:
        res = lib.compact_bidirectional_search_alloc_ranked(
            ctypes.byref(spec),
            ctypes.byref(cost_spec),
            fwd_nodes,
            fwd_costs,
            fwd_rank_primary,
            fwd_rank_secondary,
            len(fwd_seed_indices),
            rev_nodes,
            rev_costs,
            rev_rank_primary,
            rev_rank_secondary,
            len(rev_seed_indices),
            ctypes.byref(out_pos_ptr),
            ctypes.byref(out_cost),
            ctypes.byref(out_fwd_idx),
            ctypes.byref(out_rev_idx),
            native_deadline_seconds,
        )

    if res == -2:
        raise RoutingTimeout("compiled compact search deadline exceeded")
    if res < 0 or out_fwd_idx.value < 0 or out_rev_idx.value < 0:
        return None

    try:
        positions = [int(out_pos_ptr[i]) for i in range(res)]
    finally:
        lib.compact_free_positions(out_pos_ptr)
    return (
        positions,
        float(out_cost.value),
        fwd_seed_indices[out_fwd_idx.value],
        rev_seed_indices[out_rev_idx.value],
    )
