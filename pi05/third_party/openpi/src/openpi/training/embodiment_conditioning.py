"""Compact, deterministic text descriptions of robot embodiments.

The descriptor is deliberately independent of model code so the RLDS training
loader and remote inference client can produce identical conditioning strings.
It summarizes the controller contract and the morphology parameters that vary
in the generated robot bank.  Raw URDF/XML is intentionally not placed in a
language prompt: it is long, tokenizer-inefficient, and includes irrelevant
mesh/file details.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import re
from typing import Any, Literal

from pi05_integration.input_config import prompt_conditioning_from_legacy

ConditioningMode = Literal["none", "id", "text", "id_text"]
SUPPORTED_CONDITIONING_MODES = ("none", "id", "text", "id_text")

_ARM_DOF = {
    "franka_droid": 7,
    "ur5e_2f85": 6,
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _params(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = _as_text(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _natural_key(name: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _is_gripper(name: str) -> bool:
    name = name.lower()
    return any(part in name for part in ("gripper", "finger", "pad", "follower", "spring"))


def _finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: float) -> str:
    # Three significant decimal places are enough to distinguish the generated
    # variants without spending most of pi0.5's prompt budget on number tokens.
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _geometric_mean(values: Any) -> float | None:
    if not isinstance(values, list | tuple):
        return _finite_float(values)
    finite = [_finite_float(v) for v in values]
    positive = [v for v in finite if v is not None and v > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def _ordered_arm_names(params: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    for field in ("body_scales", "mass_factors", "joint_factors"):
        values = params.get(field, {})
        if isinstance(values, Mapping):
            names.update(str(name) for name in values if not _is_gripper(str(name)))
    return sorted(names, key=_natural_key)


def _series(values: Mapping[str, Any], names: list[str], *, geometric: bool = False) -> str:
    result: list[str] = []
    for name in names:
        value = values.get(name)
        number = _geometric_mean(value) if geometric else _finite_float(value)
        if number is not None:
            result.append(_fmt(number))
    return " ".join(result)


def _mean_series(values: Mapping[str, Any], names: list[str]) -> float | None:
    result = [_finite_float(values.get(name)) for name in names]
    finite = [value for value in result if value is not None]
    return sum(finite) / len(finite) if finite else None


def _mean_joint_factor(params: Mapping[str, Any], key: str) -> float | None:
    joint_factors = params.get("joint_factors", {})
    if not isinstance(joint_factors, Mapping):
        return None
    values = []
    for name, factors in joint_factors.items():
        if _is_gripper(str(name)) or not isinstance(factors, Mapping):
            continue
        value = _finite_float(factors.get(key))
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def make_embodiment_descriptor(
    *,
    robot: Any = "",
    action_space: Any = "joint",
    embodiment_params: Any = None,
    embodiment_id: Any = -1,
    variant: Any = "",
    action_dim: int | None = None,
) -> str:
    """Return a compact morphology/control descriptor for a prompt.

    The structured descriptor intentionally omits ``variant`` and
    ``embodiment_id``.  That prevents the recommended text mode from collapsing
    into an ID lookup and lets held-out variants be described at inference time.
    Those values are accepted because callers often have one metadata mapping;
    ``embodiment_id`` is used separately by :func:`condition_prompt` in ``id``
    mode.
    """
    del embodiment_id, variant
    params = _params(embodiment_params)
    family = _as_text(robot).strip() or _as_text(params.get("robot")).strip() or "unknown"
    family = family.replace("_", " ")
    action_space_text = _as_text(action_space).strip().lower() or "joint"
    inferred_dof = action_dim - 1 if action_dim is not None and action_space_text == "joint" else None
    dof = _ARM_DOF.get(family.replace(" ", "_"), inferred_dof)

    if action_space_text in {"ee", "eef", "xyz"}:
        control = "absolute end effector pose and gripper"
    else:
        control = "delta joints and absolute gripper"

    parts = [f"robot {family}"]
    if dof is not None and dof > 0:
        parts.append(f"arm dof {dof}")
    parts.append(f"actions {control}")

    arm_names = _ordered_arm_names(params)
    body_scales = params.get("body_scales", {})
    if arm_names and isinstance(body_scales, Mapping):
        values = _series(body_scales, arm_names, geometric=True)
        if values:
            parts.append(f"link scales {values}")

    gripper_scales = []
    if isinstance(body_scales, Mapping):
        for name, value in body_scales.items():
            if _is_gripper(str(name)):
                number = _geometric_mean(value)
                if number is not None:
                    gripper_scales.append(number)
    if gripper_scales:
        parts.append(f"gripper scale {_fmt(sum(gripper_scales) / len(gripper_scales))}")

    mass_factors = params.get("mass_factors", {})
    if arm_names and isinstance(mass_factors, Mapping):
        value = _mean_series(mass_factors, arm_names)
        if value is not None:
            parts.append(f"mean arm mass factor {_fmt(value)}")

    joint_means = []
    for key in ("stiffness", "damping", "force", "armature"):
        value = _mean_joint_factor(params, key)
        if value is not None:
            joint_means.append(f"{key} {_fmt(value)}")
    if joint_means:
        parts.append("mean actuator " + " ".join(joint_means))

    gripper_length = _finite_float(params.get("gripper_length"))
    if gripper_length is not None:
        parts.append(f"gripper length {_fmt(gripper_length)}")

    return "; ".join(parts)


def condition_prompt(
    prompt: Any,
    *,
    mode: ConditioningMode = "none",
    robot: Any = "",
    action_space: Any = "joint",
    embodiment_params: Any = None,
    embodiment_id: Any = -1,
    variant: Any = "",
    action_dim: int | None = None,
    descriptor: Any = "",
) -> str:
    """Compatibility adapter for callers that still carry the historical string mode."""
    conditioning = prompt_conditioning_from_legacy(mode)
    return conditioning.condition(
        prompt,
        robot=robot,
        action_space=action_space,
        embodiment_params=embodiment_params,
        embodiment_id=embodiment_id,
        variant=variant,
        action_dim=action_dim,
        descriptor=descriptor,
        descriptor_builder=make_embodiment_descriptor,
    )
