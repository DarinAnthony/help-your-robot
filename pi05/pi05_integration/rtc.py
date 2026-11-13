"""Pure utilities for guidance Real-Time Chunking (RTC).

The actual vector-Jacobian-product guidance lives in the vendored openpi JAX model.  This module
owns the backend-independent prefix alignment and weight schedule so the evaluator, server and tests
agree on exactly which actions are committed, softly guided, or free.

Notation follows Black et al., "Real-Time Execution of Action Chunking Flow Policies":

* ``horizon`` is H, the model prediction horizon.
* ``execution_horizon`` is K (called s in the paper).
* ``delay_steps`` is d, the number of actions committed while inference is outstanding.
* the previous plan constrains indices ``[0, H-K)`` of the new plan; indices ``[0, d)`` are fully
  committed and the remainder decays smoothly to zero.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    execution_horizon: int = 16
    delay_steps: int = 1
    prefix_horizon: int | None = None
    max_guidance_weight: float = 5.0
    schedule: str = "exponential"
    velocity_rescale: bool = True

    def validate(self, prediction_horizon: int) -> None:
        if not self.enabled:
            return
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if not 0 <= self.delay_steps <= self.execution_horizon:
            raise ValueError("RTC requires 0 <= delay_steps <= execution_horizon")
        if self.execution_horizon > prediction_horizon - self.delay_steps:
            raise ValueError(
                "RTC requires execution_horizon <= prediction_horizon - delay_steps; "
                f"got K={self.execution_horizon}, H={prediction_horizon}, d={self.delay_steps}"
            )
        overlap_horizon = prediction_horizon - self.execution_horizon
        if self.prefix_horizon is not None and not (
            self.delay_steps <= self.prefix_horizon <= overlap_horizon
        ):
            raise ValueError(
                "RTC prefix_horizon must satisfy delay_steps <= prefix_horizon <= H-K; "
                f"got G={self.prefix_horizon}, d={self.delay_steps}, H-K={overlap_horizon}"
            )
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
        if self.schedule not in {"exponential", "linear"}:
            raise ValueError(f"unsupported RTC schedule: {self.schedule}")


def make_prefix_weights(
    horizon: int,
    *,
    prefix_change_start: int,
    prefix_horizon: int,
    schedule: str = "exponential",
    dtype=np.float32,
) -> np.ndarray:
    """Return the RTC soft mask W with shape ``[H]``.

    The committed interval is exactly one, the free interval is exactly zero, and the interval in
    between decreases monotonically.  The exponential schedule is normalized to share the same
    endpoints as the linear schedule.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= prefix_change_start <= prefix_horizon <= horizon:
        raise ValueError(
            "expected 0 <= prefix_change_start <= prefix_horizon <= horizon, got "
            f"{prefix_change_start}, {prefix_horizon}, {horizon}"
        )
    if schedule not in {"exponential", "linear"}:
        raise ValueError(f"unsupported RTC schedule: {schedule}")

    weights = np.zeros(horizon, dtype=dtype)
    weights[:prefix_change_start] = 1.0
    ramp = prefix_horizon - prefix_change_start
    if ramp <= 0:
        return weights

    # Include the value 1 at the left edge and approach (but do not duplicate) zero at the right
    # edge; prefix_horizon and later are set exactly to zero above.
    progress = np.arange(ramp, dtype=np.float64) / float(ramp)
    if schedule == "linear":
        values = 1.0 - progress
    else:
        # Normalized exp ramp: p=0 -> 1, p=1 -> 0.
        values = (np.exp(1.0 - progress) - 1.0) / (np.e - 1.0)
    weights[prefix_change_start:prefix_horizon] = values.astype(dtype)
    return weights


def align_action_prefix(
    previous_chunk: np.ndarray,
    *,
    executed_steps: int,
    horizon: int | None = None,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Shift a previous chunk to the current observation and pad it to H.

    Returns ``(prefix, valid_horizon)``.  ``valid_horizon`` is the number of previous-plan actions
    that still align with the new chunk; callers use it as ``prefix_horizon`` for the soft mask.
    """
    chunk = np.asarray(previous_chunk)
    if chunk.ndim != 2:
        raise ValueError(f"previous_chunk must have shape [H,D], got {chunk.shape}")
    if not 0 <= executed_steps <= len(chunk):
        raise ValueError(f"executed_steps {executed_steps} outside [0,{len(chunk)}]")
    horizon = len(chunk) if horizon is None else int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    tail = chunk[executed_steps:]
    valid = min(len(tail), horizon)
    out = np.full((horizon, chunk.shape[-1]), pad_value, dtype=chunk.dtype)
    out[:valid] = tail[:valid]
    return out, valid


def stable_query_seed(global_seed: int, cell_id: str, repeat: int, query_index: int) -> int:
    """Derive a request-local uint32 seed independent of worker/server arrival order."""
    payload = f"{int(global_seed)}\0{cell_id}\0{int(repeat)}\0{int(query_index)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def stable_sampling_seed(
    global_seed: int,
    cell_id: str,
    repeat: int,
    query_index: int,
    *,
    mode: str = "independent_query",
) -> int:
    """Choose independent-query or rollout-correlated flow noise reproducibly.

    ``fixed_rollout`` deliberately reuses the same initial flow noise at every replan in one
    rollout. The observation still changes, so the returned action chunk can adapt, while the
    latent draw no longer invites query-to-query mode switching. Different cells and repeats keep
    independent draws. This is an inference ablation, not an RTC/action-prefix constraint.
    """
    if mode == "independent_query":
        effective_query = query_index
    elif mode == "fixed_rollout":
        effective_query = 0
    else:
        raise ValueError(f"unknown flow noise mode {mode!r}")
    return stable_query_seed(global_seed, cell_id, repeat, effective_query)
