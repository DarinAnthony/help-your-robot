"""Immutable dataset normalization artifacts exported with Pi checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from etils import epath


SCHEMA_VERSION = 1
MANIFEST_NAME = "normalization_manifest.json"
BUNDLE_NAME = "normalization_bundle.json"
STATS_NAME = "norm_stats.json"


def _json_value(value: Any) -> Any:
    """Canonicalize tuples/numpy-free records exactly as they appear on disk."""
    return json.loads(json.dumps(value, sort_keys=True))


def sha256(path: epath.Path | str) -> str:
    return hashlib.sha256(epath.Path(path).read_bytes()).hexdigest()


def write_dataset_manifest(
    stats_dir: epath.Path | str,
    *,
    dataset_id: str,
    split: str | dict[str, Any],
    frame_count: int,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    directory = epath.Path(stats_dir)
    stats_path = directory / STATS_NAME
    if not stats_path.exists():
        raise FileNotFoundError(f"normalization statistics are absent: {stats_path}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "split": _json_value(split),
        "frame_count": int(frame_count),
        "normalization": _json_value(normalization),
        "stats_sha256": sha256(stats_path),
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _same_position_statistics(source: dict[str, Any], consumer: dict[str, Any]) -> bool:
    """The two opt-in action modes fit the same complete per-position tables."""
    if not isinstance(source, dict) or not isinstance(consumer, dict):
        return False
    modes = {
        "GroupedChunkPositionStandardNormQuantileState": False,
        "GroupedChunkPositionQuantileNorm": True,
    }
    if source.get("type") not in modes or consumer.get("type") not in modes:
        return False
    if not source.get("groups") or not consumer.get("groups"):
        return False
    if any(record.get("use_quantiles") is not modes[record["type"]] for record in (source, consumer)):
        return False
    return {
        key: value for key, value in source.items() if key not in {"type", "use_quantiles"}
    } == {
        key: value for key, value in consumer.items() if key not in {"type", "use_quantiles"}
    }


def _require_complete_position_tables(stats_path: epath.Path, normalization: dict[str, Any]) -> None:
    payload = json.loads(stats_path.read_text())
    tables = payload.get("norm_stats", payload)
    names = ["actions"] + [
        f"actions__embodiment_{embodiment_id}"
        for group in normalization["groups"] for embodiment_id in group["embodiment_ids"]
    ]
    shape = None
    for name in names:
        for field in ("mean", "std", "q01", "q99"):
            values = tables.get(name, {}).get(field)
            if not isinstance(values, list) or not values or not isinstance(values[0], list) or not values[0]:
                raise ValueError(f"shared per-position statistics require complete (H,D) tables: {name}/{field}")
            current_shape = (len(values), len(values[0]))
            if shape is None:
                shape = current_shape
            if current_shape != shape or any(
                not isinstance(row, list) or len(row) != shape[1]
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)
                for row in values
            ):
                raise ValueError(f"shared per-position statistics have invalid shape/values: {name}/{field}")


def resolve_training_bundle(
    stats_dir: epath.Path | str,
    *,
    asset_id: str,
    dataset_id: str,
    split: str | dict[str, Any],
    normalization: dict[str, Any],
    require_manifest: bool,
    expected_stats_sha256: str | None = None,
) -> dict[str, Any]:
    directory = epath.Path(stats_dir)
    stats_path = directory / STATS_NAME
    if not stats_path.exists():
        raise FileNotFoundError(f"training normalization statistics are absent: {stats_path}")
    actual_sha = sha256(stats_path)
    if expected_stats_sha256 is not None and actual_sha != expected_stats_sha256:
        raise ValueError(
            "configured normalization statistics hash does not match the artifact: "
            f"{actual_sha} != {expected_stats_sha256}"
        )
    manifest_path = directory / MANIFEST_NAME
    statistics_source_normalization = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "split": _json_value(split),
            "normalization": _json_value(normalization),
            "stats_sha256": actual_sha,
        }
        mismatches = {
            key: {"manifest": manifest.get(key), "expected": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if (
            set(mismatches) == {"normalization"}
            and require_manifest
            and expected_stats_sha256 is not None
            and _same_position_statistics(manifest["normalization"], expected["normalization"])
        ):
            # Preserve the immutable fitting manifest and payload. A checkpoint
            # must describe its own consuming mode, not the mode that fitted the
            # shared mean/std/q01/q99 tables.
            _require_complete_position_tables(stats_path, expected["normalization"])
            statistics_source_normalization = manifest["normalization"]
            mismatches = {}
        if mismatches:
            raise ValueError(f"normalization artifact mismatch: {mismatches}")
    elif require_manifest:
        raise FileNotFoundError(
            f"immutable normalization manifest is absent: {manifest_path}; "
            "build the dataset artifact explicitly before training"
        )
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "split": split,
            "frame_count": None,
            "normalization": _json_value(normalization),
            "stats_sha256": actual_sha,
            "legacy_unmanifested_source": True,
        }
    bundle = {
        **manifest,
        "asset_id": asset_id,
        "source_stats_uri": str(stats_path),
    }
    if statistics_source_normalization is not None:
        bundle["normalization"] = _json_value(normalization)
        bundle["statistics_source_normalization"] = statistics_source_normalization
    return bundle


def write_checkpoint_bundle(directory: epath.Path | str, bundle: dict[str, Any]) -> None:
    path = epath.Path(directory) / BUNDLE_NAME
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True))


def validate_checkpoint_bundle(
    directory: epath.Path | str,
    *,
    asset_id: str,
    normalization: dict[str, Any],
) -> dict[str, Any] | None:
    directory = epath.Path(directory)
    bundle_path = directory / BUNDLE_NAME
    if not bundle_path.exists():
        return None
    bundle = json.loads(bundle_path.read_text())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "normalization": _json_value(normalization),
        "stats_sha256": sha256(directory / STATS_NAME),
    }
    mismatches = {
        key: {"bundle": bundle.get(key), "expected": value}
        for key, value in expected.items()
        if bundle.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint normalization bundle mismatch: {mismatches}")
    return bundle
