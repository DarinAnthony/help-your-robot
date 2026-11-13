"""Explicit inference-only normalization derived from immutable checkpoint assets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np


def require_effective_normalization_stats(
    norm_stats: dict[str, Any],
) -> dict[str, Any]:
    """Validate the post-selection table consumed by inference transforms."""

    missing = {"state", "actions"}.difference(norm_stats)
    if missing:
        raise ValueError(
            "effective checkpoint normalization is missing "
            + ", ".join(sorted(missing))
            + " statistics"
        )
    return norm_stats


def average_per_embodiment_stats(
    norm_stats: dict[str, Any], embodiment_ids: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Average each statistics vector equally across the requested embodiments.

    The checkpoint's pooled ``state``/``actions`` entries are deliberately ignored.
    Every robot contributes one table regardless of its episode or frame count.
    """

    ids = tuple(int(value) for value in embodiment_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("average normalization requires non-empty unique embodiment IDs")
    averaged: dict[str, Any] = {}
    digest_payload: dict[str, Any] = {"embodiment_ids": list(ids), "fields": {}}
    for field in ("state", "actions"):
        tables = []
        for embodiment_id in ids:
            key = f"{field}__embodiment_{embodiment_id}"
            if key not in norm_stats:
                raise ValueError(f"missing checkpoint normalization table {key}")
            tables.append(norm_stats[key])
        values = {
            name: np.mean(
                np.stack([np.asarray(getattr(table, name), np.float64) for table in tables]),
                axis=0,
            )
            for name in ("mean", "std", "q01", "q99")
        }
        if np.any(values["q99"] < values["q01"]):
            raise ValueError(f"averaged {field} q99 falls below q01")
        averaged[field] = type(tables[0])(**values)
        digest_payload["fields"][field] = {
            name: values[name].tolist() for name in ("mean", "std", "q01", "q99")
        }
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "mode": "checkpoint_per_embodiment_mean",
        "embodiment_ids": list(ids),
        "robot_weighting": "equal",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "derived_stats": digest_payload["fields"],
    }
    return averaged, provenance


def select_identical_per_embodiment_stats(
    norm_stats: dict[str, Any],
    embodiment_ids: Sequence[int],
    *,
    group_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one frame-pooled group table duplicated under several embodiment IDs."""

    ids = tuple(int(value) for value in embodiment_ids)
    name = str(group_name).strip()
    if not name:
        raise ValueError("normalization group name must be nonempty")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("group normalization requires non-empty unique embodiment IDs")
    selected: dict[str, Any] = {}
    digest_payload: dict[str, Any] = {
        "group_name": name,
        "embodiment_ids": list(ids),
        "fields": {},
    }
    for field in ("state", "actions"):
        tables = []
        for embodiment_id in ids:
            key = f"{field}__embodiment_{embodiment_id}"
            if key not in norm_stats:
                raise ValueError(f"missing checkpoint normalization table {key}")
            tables.append(norm_stats[key])
        values = {}
        for statistic in ("mean", "std", "q01", "q99"):
            reference = np.asarray(getattr(tables[0], statistic), np.float64)
            for embodiment_id, table in zip(ids[1:], tables[1:], strict=True):
                candidate = np.asarray(getattr(table, statistic), np.float64)
                if not np.array_equal(reference, candidate):
                    raise ValueError(
                        f"{field} {statistic} table for embodiment {embodiment_id} "
                        f"is not identical within normalization group {name!r}"
                    )
            values[statistic] = reference
        if np.any(values["q99"] < values["q01"]):
            raise ValueError(f"{name} {field} q99 falls below q01")
        selected[field] = type(tables[0])(**values)
        digest_payload["fields"][field] = {
            statistic: values[statistic].tolist()
            for statistic in ("mean", "std", "q01", "q99")
        }
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    return selected, {
        "mode": "checkpoint_normalization_group",
        "group_name": name,
        "embodiment_ids": list(ids),
        "source_weighting": "frame_pooled_within_group",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "derived_stats": digest_payload["fields"],
    }
