"""Resolve normalization assets saved inside a Pi0.5 checkpoint."""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint_asset_id(assets_dir: str | Path, preferred: str | None) -> str:
    """Return the saved norm-stat asset ID, allowing one unambiguous legacy alias.

    Some TPU and GPU runs used byte-identical normalization tables under different legacy
    directory names. Checkpoint assets are authoritative at inference time, so prefer the ID named
    by the registered config and otherwise accept exactly one saved ``norm_stats.json`` directory.
    Multiple alternatives remain an error rather than silently selecting the wrong statistics.
    """
    root = Path(assets_dir)
    if preferred and (root / preferred / "norm_stats.json").is_file():
        return preferred

    candidates = sorted(path.parent.name for path in root.glob("*/norm_stats.json"))
    if len(candidates) == 1:
        return candidates[0]
    expected = preferred or "<unset>"
    raise ValueError(
        f"checkpoint norm asset {expected!r} is absent under {root}; "
        f"expected one unambiguous fallback, found {candidates}"
    )
