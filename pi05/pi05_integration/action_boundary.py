"""Action-space projection at the Pi policy-server boundary."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def unnormalize_left_packed_actions(
    normalized_actions: np.ndarray,
    action_masks: np.ndarray,
    embodiment_ids: Sequence[int],
    *,
    stats_for_embodiment: Callable[[int], Any],
    use_quantiles: bool,
) -> np.ndarray:
    """Project to valid wire coordinates, then unnormalize only those coordinates.

    The model samples all 32 coordinates, but normalization statistics belong to the physical
    action representation.  Each row is therefore sliced to its left-packed valid prefix before
    any normalization statistics are read.  Invalid wire coordinates are constructed as zeros
    after unnormalization rather than being passed through an artificial padded-coordinate table.
    A ``(D,)`` table applies to every chunk position; a ``(H, D)`` table (per-chunk-position
    normalization) holds one row per position and must match the chunk horizon.
    """

    actions = np.asarray(normalized_actions, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=bool)
    if actions.ndim != 3:
        raise ValueError("normalized_actions must have shape [B,H,D]")
    if masks.ndim != 2 or masks.shape[0] != actions.shape[0]:
        raise ValueError("action_masks must have shape [B,wire_dim]")
    if len(embodiment_ids) != actions.shape[0]:
        raise ValueError("embodiment_ids must contain one value per batch row")
    if actions.shape[-1] < masks.shape[-1]:
        raise ValueError("model action width must cover the wire action width")

    output = np.zeros((*actions.shape[:2], masks.shape[-1]), dtype=np.float32)
    for index, embodiment_id in enumerate(embodiment_ids):
        valid_dim = int(masks[index].sum())
        if not np.array_equal(
            masks[index], np.arange(masks.shape[-1]) < valid_dim
        ):
            raise ValueError("action mask must be a left-packed valid prefix")
        if valid_dim == 0:
            continue

        stats = stats_for_embodiment(int(embodiment_id))
        values = actions[index, :, :valid_dim]
        if use_quantiles:
            q01 = np.asarray(stats.q01, dtype=np.float32)
            q99 = np.asarray(stats.q99, dtype=np.float32)
            _check_chunk_table(q01, q99, horizon=actions.shape[1])
            if q01.shape[-1] < valid_dim or q99.shape[-1] < valid_dim:
                raise ValueError("action quantile statistics do not cover the valid prefix")
            output[index, :, :valid_dim] = (
                (values + 1.0) / 2.0 * (q99[..., :valid_dim] - q01[..., :valid_dim] + 1e-6)
                + q01[..., :valid_dim]
            )
        else:
            mean = np.asarray(stats.mean, dtype=np.float32)
            std = np.asarray(stats.std, dtype=np.float32)
            _check_chunk_table(mean, std, horizon=actions.shape[1])
            if mean.shape[-1] < valid_dim or std.shape[-1] < valid_dim:
                raise ValueError("action standardization statistics do not cover the valid prefix")
            output[index, :, :valid_dim] = (
                values * (std[..., :valid_dim] + 1e-6) + mean[..., :valid_dim]
            )
    return output


def _check_chunk_table(*tables: np.ndarray, horizon: int) -> None:
    """Fail closed unless the action tables are all ``(D,)`` or all ``(horizon, D)``."""
    shapes = {table.shape[:-1] for table in tables}
    if len(shapes) != 1 or next(iter(shapes)) not in ((), (horizon,)):
        raise ValueError(f"action statistics of shapes {sorted(table.shape for table in tables)} do not fit a "
                         f"{horizon}-step chunk")


def unnormalize_base_actions(
    normalized_actions: np.ndarray,
    base_masks: np.ndarray,
    stats: Any,
    *,
    use_quantiles: bool,
) -> np.ndarray:
    """Holonomic-base command of a ``mobile_base_slots_v1``/``v2`` checkpoint, ``(B, H, 3)``.

    Reads model slots 10-12 (``pi05_integration/mobile_base.py``) and unnormalizes them with the
    global ``base_actions`` table; a row whose base mask is false (a fixed arm) gets zeros, as in
    training, where those slots are zero targets."""
    from pi05_integration.mobile_base import BASE_SLOTS

    actions = np.asarray(normalized_actions, dtype=np.float32)
    masks = np.asarray(base_masks, dtype=bool)
    if actions.ndim != 3 or actions.shape[-1] <= BASE_SLOTS[-1]:
        raise ValueError("normalized_actions must have shape [B,H,D] with D covering the base slots")
    if masks.shape != (actions.shape[0], len(BASE_SLOTS)):
        raise ValueError("base_masks must have shape [B,3]")
    if np.any(masks.any(axis=-1) != masks.all(axis=-1)):
        raise ValueError("a base mask marks all three coordinates or none")
    values = actions[..., BASE_SLOTS[0] : BASE_SLOTS[-1] + 1]
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float32)
        q99 = np.asarray(stats.q99, dtype=np.float32)
        restored = (values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        restored = values * (np.asarray(stats.std, dtype=np.float32) + 1e-6) + np.asarray(stats.mean, np.float32)
    return np.where(masks[:, None, :], restored, 0.0).astype(np.float32)
