"""Typed Pi0.5 input and conditioning specifications.

These objects are deliberately dependency-free.  The repository experiment registry, the
vendored OpenPI trainer, and the simulator-side inference client can therefore share one
serializable contract without importing JAX or TensorFlow.

The classes describe five independent routes by which embodiment information can affect a
policy.  Keeping the routes separate is important: prompt tokens, discretized state tokens,
data-space normalization, backbone tokens, and AdaRMS modulation are not interchangeable
mechanisms.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

LegacyPromptMode = Literal["none", "id", "text", "id_text", "variant", "control_rate"]
ActionLossCoordinates = Literal["model", "physical"]

_STUDY_VARIANTS = ("v013", "v018", "v066", "v130", "v234", "v236", "v343", "v461", "v588")


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


def _identity(embodiment_id: Any, variant: Any) -> int | str:
    try:
        if hasattr(embodiment_id, "item"):
            embodiment_id = embodiment_id.item()
        identity = int(embodiment_id)
    except (TypeError, ValueError):
        identity = -1
    return identity if identity >= 0 else (_as_text(variant).strip() or "unknown")


class PromptConditioning(abc.ABC):
    """Construct the actual task text that is tokenized by Pi0.5."""

    @property
    @abc.abstractmethod
    def legacy_mode(self) -> LegacyPromptMode:
        """Stable wire value used by the existing RLDS and inference boundaries."""

    @abc.abstractmethod
    def condition(
        self,
        prompt: Any,
        *,
        robot: Any = "",
        action_space: Any = "joint",
        embodiment_params: Any = None,
        embodiment_id: Any = -1,
        variant: Any = "",
        action_dim: int | None = None,
        descriptor: Any = "",
        descriptor_builder: Callable[..., str] | None = None,
    ) -> str:
        """Return the exact conditioned prompt."""

    @property
    def uses_description_bundle(self) -> bool:
        return False

    @property
    def fail_on_token_overflow(self) -> bool:
        return False

    @property
    def requires_verbatim_conditioned_prompt(self) -> bool:
        """Whether inference must preserve the conditioned prompt's exact casing."""

        return False

    @property
    def uses_control_period(self) -> bool:
        """Whether ``condition`` takes ``control_period_s``: the RLDS loader passes ``episode_metadata/policy_dt_s``
        and the server the request's ``action_dt_s``. Others never receive the keyword."""

        return False


@dataclasses.dataclass(frozen=True)
class NoPromptConditioning(PromptConditioning):
    @property
    def legacy_mode(self) -> LegacyPromptMode:
        return "none"

    def condition(self, prompt: Any, **_: Any) -> str:
        return _as_text(prompt)


@dataclasses.dataclass(frozen=True)
class EmbodimentIdPrompt(PromptConditioning):
    @property
    def legacy_mode(self) -> LegacyPromptMode:
        return "id"

    def condition(
        self, prompt: Any, *, embodiment_id: Any = -1, variant: Any = "", **_: Any
    ) -> str:
        return f"{_as_text(prompt)}; embodiment id {_identity(embodiment_id, variant)}"


@dataclasses.dataclass(frozen=True)
class EmbodimentVariantPrompt(PromptConditioning):
    """Append one readable, stable study-variant name such as ``franka v066``."""

    allowed_variants: tuple[str, ...] = _STUDY_VARIANTS
    # Optional stable family name. This is deliberately independent of dataset-local
    # embodiment indices so overlapping robots receive identical text across mixtures.
    robot_family: str = ""

    @property
    def legacy_mode(self) -> LegacyPromptMode:
        return "variant"

    def condition(self, prompt: Any, *, variant: Any = "", **_: Any) -> str:
        raw_variant = _as_text(variant).strip()
        match = re.search(r"(?:^|_)v0*(\d+)$", raw_variant)
        if match is None:
            raise ValueError(f"cannot canonicalize embodiment variant {raw_variant!r}")
        canonical = f"v{int(match.group(1)):03d}"
        if canonical not in self.allowed_variants:
            raise ValueError(
                f"unknown training embodiment variant {canonical!r}; "
                f"allowed={self.allowed_variants}"
            )
        family = self.robot_family.strip()
        if ";" in family:
            raise ValueError(f"invalid robot family {self.robot_family!r}")
        embodiment_name = f"{family} {canonical}" if family else canonical
        text = _as_text(prompt)
        existing = re.search(r"; embodiment ([^;]+)$", text)
        if existing:
            if existing.group(1) != embodiment_name:
                raise ValueError(
                    f"prompt already names embodiment {existing.group(1)!r}, "
                    f"not {embodiment_name!r}"
                )
            return text
        return f"{text}; embodiment {embodiment_name}"


@dataclasses.dataclass(frozen=True)
class ControlRatePrompt(PromptConditioning):
    """Append the control rate as text, ``<prompt>; freq: 15 hz`` (the rounded 1 / control period). It needs the
    period: ``episode_metadata/policy_dt_s`` in training and the request's ``action_dt_s`` when serving. Lowercase
    because the server lowercases conditioned prompts; the task text keeps the twin recipe's handling."""

    @property
    def legacy_mode(self) -> LegacyPromptMode:
        return "control_rate"

    @property
    def uses_control_period(self) -> bool:
        return True

    def condition(self, prompt: Any, *, control_period_s: Any = None, **_: Any) -> str:
        period = float(control_period_s) if control_period_s is not None else math.nan
        if not math.isfinite(period) or period <= 0:
            raise ValueError(f"control-rate prompt needs a finite positive control period, got {control_period_s!r}")
        return f"{_as_text(prompt)}; freq: {round(1.0 / period)} hz"


@dataclasses.dataclass(frozen=True)
class MorphologyTextPrompt(PromptConditioning):
    """Append a compact morphology descriptor, optionally with a seen-ID hint."""

    include_id: bool = False

    @property
    def legacy_mode(self) -> LegacyPromptMode:
        return "id_text" if self.include_id else "text"

    def condition(
        self,
        prompt: Any,
        *,
        robot: Any = "",
        action_space: Any = "joint",
        embodiment_params: Any = None,
        embodiment_id: Any = -1,
        variant: Any = "",
        action_dim: int | None = None,
        descriptor: Any = "",
        descriptor_builder: Callable[..., str] | None = None,
    ) -> str:
        descriptor_text = _as_text(descriptor).strip()
        if not descriptor_text:
            if descriptor_builder is None:
                raise ValueError("morphology text conditioning requires a descriptor builder")
            descriptor_text = descriptor_builder(
                robot=robot,
                action_space=action_space,
                embodiment_params=embodiment_params,
                embodiment_id=embodiment_id,
                variant=variant,
                action_dim=action_dim,
            )
        prefix = _as_text(prompt)
        if self.include_id:
            prefix += f"; embodiment id {_identity(embodiment_id, variant)}"
        return f"{prefix}; embodiment: {descriptor_text}"


@dataclasses.dataclass(frozen=True)
class Urma4BundleTextPrompt(MorphologyTextPrompt):
    """Serialize the exact hash-bound URMA4 row into the language prompt."""

    bundle_dir: str = ""
    bundle_sha256: str = ""
    max_token_len: int = 512

    def __post_init__(self) -> None:
        if not self.bundle_dir:
            raise ValueError("URMA4 bundle-text conditioning requires a bundle directory")
        if not re.fullmatch(r"[0-9a-f]{64}", self.bundle_sha256):
            raise ValueError("URMA4 bundle-text conditioning requires a lowercase SHA-256 identity")
        if self.max_token_len <= 0:
            raise ValueError("URMA4 bundle-text max_token_len must be positive")

    @property
    def uses_description_bundle(self) -> bool:
        return True

    @property
    def fail_on_token_overflow(self) -> bool:
        return True

    def resolved(self, bundle_dir: str, bundle_sha256: str) -> "Urma4BundleTextPrompt":
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
            raise ValueError("resolved URMA4 bundle requires a lowercase SHA-256 identity")
        return dataclasses.replace(
            self, bundle_dir=bundle_dir, bundle_sha256=bundle_sha256
        )

    def prepare_bundle(self, bundle_module: Any) -> tuple[str, str]:
        return bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=None,
            expected_build_manifest_sha256=None,
            expected_schema="urma4_mjcf_controls_global_v1",
        )

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str]:
        bundle_dir, bundle_sha256 = self.prepare_bundle(bundle_module)
        return bundle_module.checkpoint_bundle_source(
            bundle_dir,
            bundle_sha256,
            expected_schema="urma4_mjcf_controls_global_v1",
        )

    def create_descriptor_builder(self, bundle_module: Any) -> Callable[..., str]:
        bundle_dir, _ = self.prepare_bundle(bundle_module)
        index = bundle_module.load_bundle_text_index(bundle_dir)

        def build(*, embodiment_id: Any = -1, **_: Any) -> str:
            if hasattr(embodiment_id, "item"):
                embodiment_id = embodiment_id.item()
            try:
                identity = int(embodiment_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"URMA4 bundle-text embodiment ID must be integral, got {embodiment_id!r}"
                ) from exc
            return index.serialize(identity)

        return build

    def condition(self, prompt: Any, *, descriptor: Any = "", **kwargs: Any) -> str:
        if _as_text(descriptor).strip():
            raise ValueError(
                "URMA4 bundle-text conditioning rejects request/dataset descriptors; "
                "the hash-bound bundle is authoritative"
            )
        return super().condition(prompt, descriptor="", **kwargs)


@dataclasses.dataclass(frozen=True)
class TextBundlePrompt(MorphologyTextPrompt):
    """Append one exact hash-bound entry from a versioned text bundle."""

    bundle_dir: str = ""
    bundle_sha256: str = ""
    max_token_len: int = 256
    expected_schema: str = "embodiment_text_bundle_v1"
    format_id: str = ""

    def __post_init__(self) -> None:
        if self.include_id:
            raise ValueError("text-bundle prompting does not support an ID hint")
        if not self.bundle_dir:
            raise ValueError("text-bundle conditioning requires a bundle directory")
        if not re.fullmatch(r"[0-9a-f]{64}", self.bundle_sha256):
            raise ValueError("text-bundle conditioning requires a lowercase SHA-256 identity")
        if self.max_token_len <= 0:
            raise ValueError("text-bundle max_token_len must be positive")
        if self.expected_schema != "embodiment_text_bundle_v1":
            raise ValueError(f"unsupported text-bundle schema: {self.expected_schema!r}")
        if not self.format_id:
            raise ValueError("text-bundle conditioning requires a format ID")

    @property
    def legacy_mode(self) -> LegacyPromptMode:
        return "text"

    @property
    def uses_description_bundle(self) -> bool:
        return True

    @property
    def fail_on_token_overflow(self) -> bool:
        return True

    @property
    def requires_verbatim_conditioned_prompt(self) -> bool:
        # Bundle-entry hashes cover the exact text, including capitalization.
        return True

    def resolved(self, bundle_dir: str, bundle_sha256: str) -> TextBundlePrompt:
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
            raise ValueError("resolved text bundle requires a lowercase SHA-256 identity")
        return dataclasses.replace(
            self, bundle_dir=bundle_dir, bundle_sha256=bundle_sha256
        )

    def prepare_bundle(self, bundle_module: Any) -> tuple[str, str]:
        return bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=None,
            expected_build_manifest_sha256=None,
            expected_schema=self.expected_schema,
        )

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str]:
        bundle_dir, bundle_sha256 = self.prepare_bundle(bundle_module)
        return bundle_module.checkpoint_bundle_source(
            bundle_dir,
            bundle_sha256,
            expected_schema=self.expected_schema,
        )

    def create_descriptor_builder(self, bundle_module: Any) -> Callable[..., str]:
        bundle_dir, _ = self.prepare_bundle(bundle_module)
        index = bundle_module.load_text_bundle_index(
            bundle_dir,
            expected_schema=self.expected_schema,
            expected_format_id=self.format_id,
        )

        def build(*, embodiment_id: Any = -1, **_: Any) -> str:
            if hasattr(embodiment_id, "item"):
                embodiment_id = embodiment_id.item()
            try:
                identity = int(embodiment_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"text-bundle embodiment ID must be integral, got {embodiment_id!r}"
                ) from exc
            return index.serialize(identity)

        return build

    def condition(self, prompt: Any, *, descriptor: Any = "", **kwargs: Any) -> str:
        if _as_text(descriptor).strip():
            raise ValueError(
                "text-bundle conditioning rejects request/dataset descriptors; "
                "the hash-bound bundle is authoritative"
            )
        # RLDS task language is canonical lowercase. Apply that normalization here so
        # training and serving agree even when an evaluation client sends title case,
        # while preserving the exact case-sensitive bundle entry appended below.
        return super().condition(_as_text(prompt).lower(), descriptor="", **kwargs)


def prompt_conditioning_from_legacy(mode: str) -> PromptConditioning:
    """Translate an old string value without changing its formatting semantics."""
    match mode:
        case "none":
            return NoPromptConditioning()
        case "id":
            return EmbodimentIdPrompt()
        case "text":
            return MorphologyTextPrompt()
        case "id_text":
            return MorphologyTextPrompt(include_id=True)
        case "variant":
            return EmbodimentVariantPrompt()
        case "control_rate":
            return ControlRatePrompt()
        case _:
            raise ValueError(
                f"Unknown embodiment conditioning mode {mode!r}; expected one of "
                "('none', 'id', 'text', 'id_text', 'variant', 'control_rate')"
            )


class StateEncoding(abc.ABC):
    """Describe embodiment features appended before Pi0.5 state tokenization."""

    @property
    @abc.abstractmethod
    def embodiment_ids(self) -> tuple[int, ...]:
        pass

    @abc.abstractmethod
    def create_transform(self, transforms: Any) -> Any | None:
        """Materialize the backend transform, or ``None`` for a no-op."""


@dataclasses.dataclass(frozen=True)
class NoStateEncoding(StateEncoding):
    @property
    def embodiment_ids(self) -> tuple[int, ...]:
        return ()

    def create_transform(self, transforms: Any) -> None:
        del transforms
        return None


@dataclasses.dataclass(frozen=True)
class EmbodimentOneHotState(StateEncoding):
    ids: tuple[int, ...]

    def __init__(self, ids: Sequence[int]):
        values = tuple(int(value) for value in ids)
        if not values or len(values) != len(set(values)):
            raise ValueError("embodiment IDs must be a non-empty unique sequence")
        object.__setattr__(self, "ids", values)

    @property
    def embodiment_ids(self) -> tuple[int, ...]:
        return self.ids

    def create_transform(self, transforms: Any) -> Any:
        return transforms.AppendEmbodimentOneHot(self.ids)


class StateSerialization(abc.ABC):
    """Choose which normalized proprioceptive coordinates enter the text prefix."""

    @property
    @abc.abstractmethod
    def requires_mask(self) -> bool:
        pass

    @abc.abstractmethod
    def create_transform(self, transforms: Any) -> Any | None:
        pass


@dataclasses.dataclass(frozen=True)
class LegacyFixedLayoutState(StateSerialization):
    """Reproduce checkpoints that serialized every fixed-width state slot."""

    @property
    def requires_mask(self) -> bool:
        return False

    def create_transform(self, transforms: Any) -> None:
        del transforms
        return None


@dataclasses.dataclass(frozen=True)
class CompactLeftPackedState(StateSerialization):
    """Serialize only the valid left-packed normalized state coordinates."""

    @property
    def requires_mask(self) -> bool:
        return True

    def create_transform(self, transforms: Any) -> Any:
        return transforms.CompactLeftPackedState()


class Normalization(abc.ABC):
    """Select the reversible state/action normalization table."""

    use_quantiles: bool = True

    @property
    @abc.abstractmethod
    def per_embodiment(self) -> bool:
        pass

    @property
    def embodiment_ids(self) -> tuple[int, ...]:
        return ()

    @property
    def state_quantiles(self) -> bool:
        """Whether state uses q01/q99 (else mean/std): the actions' mode unless a subclass splits them."""
        return self.use_quantiles

    @property
    def per_chunk_position(self) -> bool:
        """Whether the action tables are ``(H, D)``, one row per chunk position (else ``(D,)``)."""
        return False

    @property
    def component_weighted(self) -> bool:
        """Whether a mixture's statistics weight each dataset component by its sampling weight (else by its
        frames: the statistics scan reads every frame of every component once)."""
        return False

    @abc.abstractmethod
    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        pass

    @abc.abstractmethod
    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        pass


@dataclasses.dataclass(frozen=True)
class GlobalQuantileNorm(Normalization):
    @property
    def per_embodiment(self) -> bool:
        return False

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        if not norm_stats:
            # transform_dataset passes {} only for repo_id="fake" or skip_norm_stats: identity.
            return transforms.Normalize(None, use_quantiles=use_quantiles)
        return transforms.Normalize(
            {key: norm_stats[key] for key in ("state", "actions")},
            use_quantiles=use_quantiles,
        )

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        if not norm_stats:
            # No statistics at all (repo_id="fake"): nothing was normalized, so nothing to undo.
            return transforms.Unnormalize(None, use_quantiles=use_quantiles)
        selected = {
            key: norm_stats[key]
            for key in ("state", "actions")
            if key in norm_stats
        }
        if not selected:
            raise ValueError("global unnormalization requires state or action statistics")
        return transforms.Unnormalize(
            selected,
            use_quantiles=use_quantiles,
        )


@dataclasses.dataclass(frozen=True)
class GlobalStandardNorm(Normalization):
    """Per-dimension mean/std normalization for state and action."""

    use_quantiles: bool = False

    @property
    def per_embodiment(self) -> bool:
        return False

    def create_normalizer(
        self, transforms: Any, norm_stats: Any, *, use_quantiles: bool
    ) -> Any:
        del use_quantiles
        return transforms.Normalize(norm_stats, use_quantiles=False)

    def create_unnormalizer(
        self, transforms: Any, norm_stats: Any, *, use_quantiles: bool
    ) -> Any:
        del use_quantiles
        return transforms.Unnormalize(norm_stats, use_quantiles=False)


@dataclasses.dataclass(frozen=True)
class PerEmbodimentQuantileNorm(Normalization):
    ids: tuple[int, ...]

    def __init__(self, ids: Sequence[int]):
        values = tuple(int(value) for value in ids)
        if not values or len(values) != len(set(values)):
            raise ValueError("per-embodiment normalization requires unique IDs")
        object.__setattr__(self, "ids", values)

    @property
    def per_embodiment(self) -> bool:
        return True

    @property
    def embodiment_ids(self) -> tuple[int, ...]:
        return self.ids

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        return transforms.NormalizeByEmbodiment(norm_stats, self.ids, use_quantiles=use_quantiles)

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        return transforms.UnnormalizeByEmbodiment(norm_stats, self.ids, use_quantiles=use_quantiles)


@dataclasses.dataclass(frozen=True)
class PerEmbodimentStandardNorm(PerEmbodimentQuantileNorm):
    """Route reversible per-dimension mean/std tables by embodiment ID."""

    use_quantiles: bool = False

    def create_normalizer(
        self, transforms: Any, norm_stats: Any, *, use_quantiles: bool
    ) -> Any:
        del use_quantiles
        return transforms.NormalizeByEmbodiment(
            norm_stats, self.ids, use_quantiles=False
        )

    def create_unnormalizer(
        self, transforms: Any, norm_stats: Any, *, use_quantiles: bool
    ) -> Any:
        del use_quantiles
        return transforms.UnnormalizeByEmbodiment(
            norm_stats, self.ids, use_quantiles=False
        )


class ActionClipping(abc.ABC):
    """Select a training-only transform applied after action normalization."""

    @abc.abstractmethod
    def create_transform(self, transforms: Any) -> Any | None:
        pass


@dataclasses.dataclass(frozen=True)
class NoActionClipping(ActionClipping):
    def create_transform(self, transforms: Any) -> None:
        del transforms
        return None


@dataclasses.dataclass(frozen=True)
class NormalizedActionClip(ActionClipping):
    limit: float = 5.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("normalized action clipping limit must be finite and positive")

    def create_transform(self, transforms: Any) -> Any:
        return transforms.ClipActions(self.limit)


@dataclasses.dataclass(frozen=True)
class NormalizationGroup:
    """One pooled normalization table shared by a frozen set of embodiments."""

    name: str
    embodiment_ids: tuple[int, ...]

    def __init__(self, name: str, embodiment_ids: Sequence[int]):
        normalized_name = str(name).strip()
        values = tuple(int(value) for value in embodiment_ids)
        if not normalized_name:
            raise ValueError("normalization group name must be nonempty")
        if not values or len(values) != len(set(values)):
            raise ValueError(
                "normalization group embodiment IDs must be a non-empty unique sequence"
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "embodiment_ids", values)


@dataclasses.dataclass(frozen=True)
class GroupedQuantileNorm(Normalization):
    """Pool statistics within named groups while routing by embodiment ID.

    The checkpoint stores ordinary per-embodiment flat keys so the existing reversible
    normalization transforms remain unchanged. Every embodiment in a group receives the same
    pooled table. Group membership affects preprocessing and postprocessing only; it does not add
    a prompt, state feature, or model-side conditioning route.
    """

    groups: tuple[NormalizationGroup, ...]

    def __init__(self, groups: Sequence[NormalizationGroup]):
        values = tuple(groups)
        if not values:
            raise ValueError("grouped normalization requires at least one group")
        names = tuple(group.name for group in values)
        if len(names) != len(set(names)):
            raise ValueError("normalization group names must be unique")
        ids = tuple(
            embodiment_id
            for group in values
            for embodiment_id in group.embodiment_ids
        )
        if len(ids) != len(set(ids)):
            raise ValueError("normalization groups must have disjoint embodiment IDs")
        object.__setattr__(self, "groups", values)

    @property
    def per_embodiment(self) -> bool:
        return True

    @property
    def embodiment_ids(self) -> tuple[int, ...]:
        return tuple(
            embodiment_id
            for group in self.groups
            for embodiment_id in group.embodiment_ids
        )

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        return transforms.NormalizeByEmbodiment(
            norm_stats, self.embodiment_ids, use_quantiles=use_quantiles
        )

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        return transforms.UnnormalizeByEmbodiment(
            norm_stats, self.embodiment_ids, use_quantiles=use_quantiles
        )


@dataclasses.dataclass(frozen=True)
class GroupedStandardNorm(GroupedQuantileNorm):
    """Pooled per-dimension mean/std within named groups, routed by embodiment ID.

    The standard-normalization twin of ``GroupedQuantileNorm`` (as ``PerEmbodimentStandardNorm`` is
    of ``PerEmbodimentQuantileNorm``): every embodiment of a group reads the group's pooled table
    from its ordinary flat per-embodiment keys. XEMB-1K and OER-9 group arms by joint count
    (``groups_by_key`` with ``oer_runtime.dof_class``: dof5/dof6/dof7) or by robot class, so a
    left-packed slot means the same joint role within a group. Opt-in; nothing else changes.
    """

    use_quantiles: bool = dataclasses.field(default=False, init=False)

    def __init__(self, groups: Sequence[NormalizationGroup]):
        super().__init__(groups)
        object.__setattr__(self, "use_quantiles", False)

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        return transforms.NormalizeByEmbodiment(norm_stats, self.embodiment_ids, use_quantiles=False)

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        return transforms.UnnormalizeByEmbodiment(norm_stats, self.embodiment_ids, use_quantiles=False)


class GroupedStandardNormQuantileState(GroupedStandardNorm):
    """``GroupedStandardNorm`` for actions; the group's q01/q99 mapped to -1/+1 for state.

    Pi0.5 reads state only as prompt tokens over [-1, 1); under mean/std a quarter of the OER and twin state values
    fall outside and collapse into the end tokens, under q01/q99 2%. Actions, their clip and the physical-coordinate
    loss are unchanged, and the asset is the same (it stores q01/q99 beside mean/std). Opt-in; nothing else changes.
    """

    @property
    def state_quantiles(self) -> bool:
        return True

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        return transforms.NormalizeByEmbodiment(
            norm_stats, self.embodiment_ids, use_quantiles=False, state_quantiles=True
        )

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        return transforms.UnnormalizeByEmbodiment(
            norm_stats, self.embodiment_ids, use_quantiles=False, state_quantiles=True
        )


@dataclasses.dataclass(frozen=True)
class GroupedChunkPositionStandardNorm(GroupedStandardNorm):
    """``GroupedStandardNorm`` with one action mean/std per chunk position k and dimension: ``(H, D)`` action
    tables, ``(D,)`` state tables.

    For query-anchored labels a_k = q_cmd(t+k) - q(t) the spread grows with k, so one table pooled over positions
    over-scales the near targets and clips the far ones. The statistics scan masks the positions past the episode
    end exactly as the loss does (``compute_norm_stats.py``). ``mixture_weighting="components"`` weights each
    dataset component by its sampling weight instead of its frame count, so a mixture of control rates is
    normalized under the distribution it trains on. The tables reuse the flat per-embodiment keys and the
    ordinary transforms, which broadcast a ``(H, D)`` table over a ``(H, D)`` chunk. Opt-in; nothing else changes.
    """

    mixture_weighting: Literal["frames", "components"] = "frames"

    def __init__(
        self, groups: Sequence[NormalizationGroup], mixture_weighting: Literal["frames", "components"] = "frames"
    ):
        super().__init__(groups)
        if mixture_weighting not in ("frames", "components"):
            raise ValueError(f"mixture_weighting must be 'frames' or 'components', got {mixture_weighting!r}")
        object.__setattr__(self, "mixture_weighting", mixture_weighting)

    @property
    def per_chunk_position(self) -> bool:
        return True

    @property
    def component_weighted(self) -> bool:
        return self.mixture_weighting == "components"

    def _check_tables(self, transforms: Any, norm_stats: Any) -> None:
        """Refuse a pooled asset (or a per-position state table) under this scheme: one horizon for every
        action table, one row per dimension for every state table."""
        horizons = set()
        for embodiment_id in self.embodiment_ids:
            for field, ndim in (("actions", 2), ("state", 1)):
                key = transforms.embodiment_norm_key(field, embodiment_id)
                if key not in norm_stats:
                    continue  # NormalizeByEmbodiment names the missing table.
                stats = norm_stats[key]
                shapes = {getattr(stats, name).shape for name in ("mean", "std")}
                if len(shapes) != 1 or len(next(iter(shapes))) != ndim:
                    raise ValueError(
                        f"{type(self).__name__} needs {ndim}-D {field} tables; {key} has mean/std shapes "
                        f"{sorted(shapes)}"
                    )
                if field == "actions":
                    horizons.add(next(iter(shapes))[0])
        if len(horizons) > 1:
            raise ValueError(f"per-position action tables disagree on the chunk horizon: {sorted(horizons)}")

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        self._check_tables(transforms, norm_stats)
        return super().create_normalizer(transforms, norm_stats, use_quantiles=use_quantiles)

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        self._check_tables(transforms, norm_stats)
        return super().create_unnormalizer(transforms, norm_stats, use_quantiles=use_quantiles)


class GroupedChunkPositionStandardNormQuantileState(GroupedChunkPositionStandardNorm):
    """``GroupedChunkPositionStandardNorm`` for actions; the group's q01/q99 mapped to -1/+1 for state, as in
    ``GroupedStandardNormQuantileState``. Same asset (the scan stores q01/q99 beside mean/std). Opt-in."""

    @property
    def state_quantiles(self) -> bool:
        return True

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        self._check_tables(transforms, norm_stats)
        return transforms.NormalizeByEmbodiment(
            norm_stats, self.embodiment_ids, use_quantiles=False, state_quantiles=True
        )

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        self._check_tables(transforms, norm_stats)
        return transforms.UnnormalizeByEmbodiment(
            norm_stats, self.embodiment_ids, use_quantiles=False, state_quantiles=True
        )


class GroupedChunkPositionQuantileNorm(GroupedChunkPositionStandardNorm):
    """openpi's quantile mode with one action q01/q99 per chunk position k and dimension: every ``(H, D)`` action
    row and the ``(D,)`` state map q01/q99 to -1/+1. A normalized-action clip, when configured, applies after it as
    in every other mode. Same asset layout as ``GroupedChunkPositionStandardNorm``. Opt-in; nothing else changes."""

    def __init__(
        self, groups: Sequence[NormalizationGroup], mixture_weighting: Literal["frames", "components"] = "frames"
    ):
        super().__init__(groups, mixture_weighting)
        object.__setattr__(self, "use_quantiles", True)

    def create_normalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        self._check_tables(transforms, norm_stats)
        return transforms.NormalizeByEmbodiment(norm_stats, self.embodiment_ids, use_quantiles=True)

    def create_unnormalizer(self, transforms: Any, norm_stats: Any, *, use_quantiles: bool) -> Any:
        del use_quantiles
        self._check_tables(transforms, norm_stats)
        return transforms.UnnormalizeByEmbodiment(norm_stats, self.embodiment_ids, use_quantiles=True)


def groups_by_key(key_by_embodiment: dict[int, str]) -> tuple[NormalizationGroup, ...]:
    """Normalization groups of the embodiments that share a key (e.g. ``{id: "dof6"}`` or
    ``{id: robot_class}``), in sorted key order with sorted IDs."""
    members: dict[str, list[int]] = {}
    for embodiment_id, key in key_by_embodiment.items():
        members.setdefault(str(key), []).append(int(embodiment_id))
    if not members:
        raise ValueError("grouped normalization needs at least one embodiment")
    return tuple(NormalizationGroup(key, sorted(ids)) for key, ids in sorted(members.items()))


def pooled_group_standard_stats(
    moments_by_embodiment: dict[int, dict[str, tuple[int, Any, Any]]],
    groups: Sequence[NormalizationGroup],
    *,
    std_floor: float = 1e-6,
) -> dict[str, dict[str, list[float]]]:
    """Flat mean/std tables of a ``GroupedStandardNorm`` from per-embodiment moments.

    ``moments_by_embodiment[id][field] = (count, mean, variance)`` per dimension for ``field`` in
    ``state``/``actions``. Each group's moments are pooled exactly (law of total variance); every
    member gets the group table under ``<field>__embodiment_<id>`` (``transforms.embodiment_norm_key``),
    and the ordinary global ``state``/``actions`` keys pool all embodiments."""
    import numpy as np

    def pool(entries):
        counts = np.asarray([count for count, _, _ in entries], dtype=np.float64)
        if np.any(counts <= 0):
            raise ValueError("pooled moments need positive counts")
        means = np.stack([np.asarray(mean, dtype=np.float64) for _, mean, _ in entries])
        variances = np.stack([np.asarray(variance, dtype=np.float64) for _, _, variance in entries])
        weights = counts[:, None] / counts.sum()
        mean = (weights * means).sum(axis=0)
        variance = (weights * (variances + (means - mean) ** 2)).sum(axis=0)
        return {"mean": mean.tolist(), "std": np.maximum(np.sqrt(variance), std_floor).tolist()}

    grouped = {embodiment_id for group in groups for embodiment_id in group.embodiment_ids}
    missing = grouped - set(moments_by_embodiment)
    if missing:
        raise ValueError(f"no moments for grouped embodiments {sorted(missing)}")
    stats: dict[str, dict[str, list[float]]] = {}
    for field in ("state", "actions"):
        stats[field] = pool([moments_by_embodiment[e][field] for e in sorted(grouped)])
        for group in groups:
            table = pool([moments_by_embodiment[e][field] for e in group.embodiment_ids])
            for embodiment_id in group.embodiment_ids:
                stats[f"{field}__embodiment_{int(embodiment_id)}"] = table
    return stats


class ActionExpertConditioning(abc.ABC):
    """Describe the additional condition supplied to Pi0.5 action-expert AdaRMS."""

    @property
    @abc.abstractmethod
    def embodiment_vocab_size(self) -> int:
        pass

    @property
    def condition_width_multiplier(self) -> int:
        """Width of the AdaRMS condition relative to the action-expert width."""
        return 1

    def condition_width(self, action_expert_width: int) -> int:
        """Resolve the absolute AdaRMS condition width for one action expert."""
        return action_expert_width * self.condition_width_multiplier

    @property
    def concatenate_embodiment_embedding(self) -> bool:
        return False

    @property
    def uses_embodiment_description(self) -> bool:
        return False

    @property
    def uses_robotokens(self) -> bool:
        """Whether this route requires a static RoboToken bundle."""
        return False

    def create_description_transform(self, bundle_module: Any) -> Any | None:
        del bundle_module
        return None

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str] | None:
        del bundle_module
        return None


class EmbodimentEmbeddingExtractor(abc.ABC):
    """Backend-neutral contract for a substitutable embodiment encoder."""

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def structured_input_shape(self) -> tuple[int, int, int]:
        """Return (max_controls, description_dim, control_state_dim)."""

    @property
    @abc.abstractmethod
    def processor_schema(self) -> str:
        """Versioned description schema required by this extractor."""

    @property
    def output_normalization(self) -> Literal["none", "layer_norm"]:
        """Normalization applied at the extractor/fusion interface."""
        return "none"

    @property
    def global_description_shape(self) -> tuple[int, ...] | None:
        """Shape of separately encoded static-global context, when present."""
        return None

    @property
    def geometry_input_shape(self) -> tuple[int, int, int] | None:
        """Return (max_controls, points_per_control, point_feature_dim), if used."""
        return None

    @abc.abstractmethod
    def create_input_transform(self, bundle_module: Any) -> Any:
        """Create the host-side transform that supplies this extractor's inputs."""

    @abc.abstractmethod
    def checkpoint_artifact_source(self, bundle_module: Any) -> dict[str, str]:
        """Return the immutable extractor artifact copied into every checkpoint."""


@dataclasses.dataclass(frozen=True)
class Urma2EmbodimentExtractor(EmbodimentEmbeddingExtractor):
    """Three-head, scale-2 URMA joint router ending at its global action latent."""

    bundle_dir: str
    bundle_sha256: str | None = None
    build_manifest: str | None = None
    build_manifest_sha256: str | None = None
    description_dim: int = 35
    max_controls: int = 10
    routing_dim: int = 128
    state_feature_dim: int = 8
    embedding_dim: int = 256
    num_heads: int = 3

    def __post_init__(self) -> None:
        if not self.bundle_dir:
            raise ValueError("URMA2 extraction requires a description bundle directory")
        for value, label in (
            (self.bundle_sha256, "description bundle"),
            (self.build_manifest_sha256, "build manifest"),
        ):
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"URMA2 {label} requires a lowercase SHA-256 identity")
        if (self.build_manifest is None) != (self.build_manifest_sha256 is None):
            raise ValueError("URMA2 build manifest path and SHA-256 must be configured together")
        if self.bundle_sha256 is None and self.build_manifest is None:
            raise ValueError("URMA2 extraction requires a bundle hash or pinned build manifest")
        if (
            self.num_heads,
            self.routing_dim,
            self.state_feature_dim,
            self.embedding_dim,
        ) != (3, 128, 8, 256):
            raise ValueError(
                "the three-head Urma2EmbodimentExtractor requires num_heads=3, routing_dim=128, "
                "state_feature_dim=8, and embedding_dim=256"
            )
        if self.description_dim <= 0 or self.max_controls <= 0:
            raise ValueError("URMA2 description dimensions must be positive")

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    @property
    def processor_schema(self) -> str:
        return "urma2_urdf_controls_v1"

    @property
    def structured_input_shape(self) -> tuple[int, int, int]:
        return self.max_controls, self.description_dim, 1

    def create_input_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.AttachEmbodimentDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=(
                None
                if self.global_description_shape is None
                else self.global_description_shape[0]
            ),
        )

    def checkpoint_artifact_source(self, bundle_module: Any) -> dict[str, str]:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.checkpoint_bundle_source(
            bundle_dir,
            bundle_sha256,
            expected_schema=self.processor_schema,
        )


@dataclasses.dataclass(frozen=True)
class Urma3EmbodimentExtractor(Urma2EmbodimentExtractor):
    """URMA routing over the expanded final-MJCF physical/control description."""

    description_dim: int = 67
    normalize_output: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.normalize_output:
            raise ValueError(
                "URMA3 requires output LayerNorm at the timestep-fusion boundary"
            )

    @property
    def processor_schema(self) -> str:
        return "urma3_mjcf_controls_v1"

    @property
    def output_normalization(self) -> Literal["layer_norm"]:
        return "layer_norm"


@dataclasses.dataclass(frozen=True)
class Urma4EmbodimentExtractor(Urma3EmbodimentExtractor):
    """URMA3 controls plus a separately encoded static-global morphology vector."""

    global_description_dim: int = 52
    global_embedding_dim: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.global_description_dim <= 0 or self.global_embedding_dim <= 0:
            raise ValueError("URMA4 global dimensions must be positive")

    @property
    def processor_schema(self) -> str:
        return "urma4_mjcf_controls_global_v1"

    @property
    def global_description_shape(self) -> tuple[int]:
        return (self.global_description_dim,)


@dataclasses.dataclass(frozen=True)
class Urma4NewNormEmbodimentExtractor(Urma4EmbodimentExtractor):
    """URMA4 fields with complete train-only control/global/state normalization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.description_dim != 67 or self.global_description_dim != 52:
            raise ValueError(
                "URMA4-newnorm preserves exactly 67 control and 52 global fields"
            )

    @property
    def processor_schema(self) -> str:
        return "urma4_fields_complete_standard_v1"

    def create_input_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.AttachUrma4NewNormDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=self.global_description_dim,
        )


@dataclasses.dataclass(frozen=True)
class Urma4PhysicalStateEmbodimentExtractor(Urma4NewNormEmbodimentExtractor):
    """URMA4-newnorm attached to physical q/aperture before numeric normalization.

    This distinct named extractor preserves historical newnorm recipes, whose
    model-stage attachment reads the already-normalized numeric state.
    """

    @property
    def attach_before_numeric_normalization(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class Urma4AbsolutePhysicalStateEmbodimentExtractor(Urma4EmbodimentExtractor):
    """Absolute URMA4 (``urma4_absolute_standard_v1``, docs/urma_embodiment_conditioning.md §12).

    Attached to the physical q/aperture before numeric normalization (like
    ``Urma4PhysicalStateEmbodimentExtractor``). The control and global widths are read from the schema
    (``molmo_spaces.embodiment.description_v4_absolute``) rather than hard-coded; an explicit width must agree.
    """

    description_dim: int | None = None
    global_description_dim: int | None = None
    # The joint frame the bundle describes the robots in (pi05_integration/joint_frame.py): "stored" (each robot's own
    # joints, every bundle before the canonical one) or "canonical_mdh_tool_v1" (descriptions compiled from the
    # canonical model, control-state statistics in phi). Must equal the bundle's and Pi05InputConfig.joint_frame.
    joint_frame: str = "stored"

    def __post_init__(self) -> None:
        from molmo_spaces.embodiment import description_v4_absolute as schema

        for name, width in (("description_dim", schema.DESCRIPTION_DIM),
                            ("global_description_dim", schema.GLOBAL_DESCRIPTION_DIM)):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, width)
            elif value != width:
                raise ValueError(f"URMA4-absolute {name} is the schema width {width}, not {value}")
        super().__post_init__()

    @property
    def processor_schema(self) -> str:
        return "urma4_absolute_standard_v1"

    @property
    def attach_before_numeric_normalization(self) -> bool:
        return True

    def create_input_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        import json
        from pathlib import Path

        from molmo_spaces.embodiment.bundle import BUNDLE_MANIFEST

        build = json.loads((Path(bundle_dir) / BUNDLE_MANIFEST).read_text()).get("build") or {}
        bundle_frame = (build.get("joint_frame") or {}).get("version", "stored")
        if bundle_frame != self.joint_frame:
            raise ValueError(f"URMA4-absolute bundle {bundle_dir} is in joint frame {bundle_frame!r}, "
                             f"the extractor expects {self.joint_frame!r}")
        return bundle_module.AttachUrma4AbsoluteDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=self.global_description_dim,
        )


@dataclasses.dataclass(frozen=True)
class Urma4DynamicEmbodimentExtractor(Urma4EmbodimentExtractor):
    """URMA4 plus a masked, causal per-control MuJoCo state.

    This is a distinct extractor and parameterization.  In particular it does
    not reuse the historical response-history token and cannot silently restore
    a legacy URMA4 checkpoint whose routing-head input width is one.
    """

    dynamic_stats_path: str = ""
    dynamic_stats_sha256: str | None = None
    dynamic_field_dim: int = 8
    dynamic_profile: Literal["static", "minimal", "full"] = "full"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.dynamic_stats_path:
            raise ValueError("URMA4-dynamic requires a normalization-statistics path")
        if self.dynamic_stats_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.dynamic_stats_sha256
        ):
            raise ValueError("URMA4-dynamic requires a lowercase SHA-256 statistics identity")
        if self.dynamic_field_dim != 8:
            raise ValueError("URMA4-dynamic v1 has exactly eight causal fields")
        if self.dynamic_profile not in ("static", "minimal", "full"):
            raise ValueError("URMA4-dynamic profile must be static, minimal, or full")

    @property
    def structured_input_shape(self) -> tuple[int, int, int]:
        # Each normalized field is paired with its own availability mask.
        return self.max_controls, self.description_dim, 2 * self.dynamic_field_dim

    def create_input_transform(self, bundle_module: Any) -> Any:
        if self.dynamic_stats_sha256 is None:
            raise ValueError(
                "URMA4-dynamic statistics are unpublished; materialize and pin the "
                "normalization SHA-256 before launch"
            )
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.AttachDynamicEmbodimentDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=self.global_description_dim,
            dynamic_stats_path=self.dynamic_stats_path,
            expected_dynamic_stats_sha256=self.dynamic_stats_sha256,
            dynamic_profile=self.dynamic_profile,
        )

    def checkpoint_artifact_source(self, bundle_module: Any) -> dict[str, str]:
        if self.dynamic_stats_sha256 is None:
            raise ValueError(
                "URMA4-dynamic statistics are unpublished; no checkpoint bundle can be built"
            )
        source = super().checkpoint_artifact_source(bundle_module)
        dynamic_stats_path, dynamic_stats_sha256 = (
            bundle_module.validate_dynamic_normalization(
                self.dynamic_stats_path, self.dynamic_stats_sha256
            )
        )
        return {
            **source,
            "dynamic_stats_source": dynamic_stats_path,
            "dynamic_stats_sha256": dynamic_stats_sha256,
        }


@dataclasses.dataclass(frozen=True)
class Urma5EmbodimentExtractor(Urma3EmbodimentExtractor):
    """URMA5 scalar conditioning with one immutable complete normalizer."""

    description_dim: int = 89
    global_description_dim: int = 17
    global_embedding_dim: int = 128
    dynamic_field_dim: int = 7

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.description_dim != 89 or self.global_description_dim != 17:
            raise ValueError("URMA5 has fixed 89-control/17-global static dimensions")
        if self.dynamic_field_dim != 7:
            raise ValueError(
                "URMA5 has four deployable control values plus current anchor xyz"
            )

    @property
    def processor_schema(self) -> str:
        return "urma5_mjcf_controls_dynamic_v2"

    @property
    def global_description_shape(self) -> tuple[int]:
        return (self.global_description_dim,)

    @property
    def structured_input_shape(self) -> tuple[int, int, int]:
        return self.max_controls, self.description_dim, self.dynamic_field_dim

    def create_input_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.AttachUrma5EmbodimentDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=self.global_description_dim,
        )


@dataclasses.dataclass(frozen=True)
class Urma6EmbodimentExtractor(Urma5EmbodimentExtractor):
    """All URMA5 fields plus shared PointNet collision-shape features."""

    points_per_control: int = 256
    point_feature_dim: int = 9
    geometry_embedding_dim: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.points_per_control,
            self.point_feature_dim,
            self.geometry_embedding_dim,
        ) != (256, 9, 128):
            raise ValueError("URMA6 requires 256x9 collision points and PointNet128")

    @property
    def processor_schema(self) -> str:
        return "urma6_mjcf_controls_dynamic_pointnet_v2"

    @property
    def geometry_input_shape(self) -> tuple[int, int, int]:
        return self.max_controls, self.points_per_control, self.point_feature_dim

    def create_input_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.AttachUrma6EmbodimentDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=self.global_description_dim,
        )


@dataclasses.dataclass(frozen=True)
class Urma51EmbodimentExtractor(Urma5EmbodimentExtractor):
    """URMA5 plus nine arm-only analytical dynamics and applicability bits."""

    analytic_field_dim: int = 9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.analytic_field_dim != 9:
            raise ValueError("URMA5.1 has exactly nine analytical dynamic fields")

    @property
    def structured_input_shape(self) -> tuple[int, int, int]:
        return (
            self.max_controls,
            self.description_dim,
            self.dynamic_field_dim + 2 * self.analytic_field_dim,
        )

    def create_input_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_description_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
            expected_schema=self.processor_schema,
        )
        return bundle_module.AttachUrma51EmbodimentDescription(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_controls=self.max_controls,
            description_dim=self.description_dim,
            expected_schema=self.processor_schema,
            global_description_dim=self.global_description_dim,
        )


@dataclasses.dataclass(frozen=True)
class TimestepOnly(ActionExpertConditioning):
    @property
    def embodiment_vocab_size(self) -> int:
        return 0


@dataclasses.dataclass(frozen=True)
class LearnedEmbodimentAdaRMS(ActionExpertConditioning):
    vocab_size: int
    zero_init: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("AdaRMS embodiment vocabulary must be positive")
        if not self.zero_init:
            raise ValueError("only checkpoint-compatible zero initialization is supported")

    @property
    def embodiment_vocab_size(self) -> int:
        return self.vocab_size


@dataclasses.dataclass(frozen=True)
class LearnedEmbodimentConcatAdaRMS(ActionExpertConditioning):
    """Concatenate a learned stable-ID embedding with the flow-time condition."""

    vocab_size: int
    zero_init: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("AdaRMS embodiment vocabulary must be positive")
        if not self.zero_init:
            raise ValueError("only checkpoint-compatible zero initialization is supported")

    @property
    def embodiment_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def condition_width_multiplier(self) -> int:
        return 2

    @property
    def concatenate_embodiment_embedding(self) -> bool:
        return True


UrmaFusionMode = Literal["residual_mlp", "add", "concat", "concat_project"]


@dataclasses.dataclass(frozen=True)
class ExtractedEmbodimentAdaRMS(ActionExpertConditioning):
    """Fuse a pluggable embodiment extractor into Pi0.5's AdaRMS condition."""

    extractor: EmbodimentEmbeddingExtractor
    fusion: UrmaFusionMode = "residual_mlp"

    def __post_init__(self) -> None:
        if not isinstance(self.extractor, EmbodimentEmbeddingExtractor):
            raise TypeError("extractor must implement EmbodimentEmbeddingExtractor")
        if self.fusion not in {"residual_mlp", "add", "concat", "concat_project"}:
            raise ValueError(f"unsupported embodiment AdaRMS fusion: {self.fusion!r}")

    @property
    def embodiment_vocab_size(self) -> int:
        return 0

    @property
    def uses_embodiment_description(self) -> bool:
        return True

    def condition_width(self, action_expert_width: int) -> int:
        if self.fusion == "concat":
            return action_expert_width + self.extractor.output_dim
        return action_expert_width

    def create_description_transform(self, bundle_module: Any) -> Any:
        return self.extractor.create_input_transform(bundle_module)

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str]:
        return self.extractor.checkpoint_artifact_source(bundle_module)


@dataclasses.dataclass(frozen=True)
class ExtractedEmbodimentConcatProjectAdaRMS(ExtractedEmbodimentAdaRMS):
    """Project URMA to expert width, concatenate with time, then project once to AdaRMS."""

    projection_dim: int = 1024
    fusion: Literal["concat_project"] = dataclasses.field(
        default="concat_project", init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.extractor, EmbodimentEmbeddingExtractor):
            raise TypeError("extractor must implement EmbodimentEmbeddingExtractor")
        if self.projection_dim <= 0:
            raise ValueError("URMA concat-project conditioning projection_dim must be positive")


class BackboneConditioning(abc.ABC):
    """Describe an optional non-text modality supplied to the VLM prefix."""

    @property
    def uses_embodiment_description(self) -> bool:
        return False

    @property
    def uses_robotokens(self) -> bool:
        return False

    @property
    def embodiment_vocab_size(self) -> int:
        return 0

    @property
    def uses_embodiment_codebook(self) -> bool:
        return False

    def create_description_transform(self, bundle_module: Any) -> Any | None:
        del bundle_module
        return None

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str] | None:
        del bundle_module
        return None


@dataclasses.dataclass(frozen=True)
class NoBackboneConditioning(BackboneConditioning):
    """Keep the PaliGemma prefix limited to its existing image and language tokens."""


@dataclasses.dataclass(frozen=True)
class LearnedEmbodimentBackboneToken(BackboneConditioning):
    """Use one learned stable-ID embedding as a PaliGemma prefix token."""

    vocab_size: int | None = None
    token_width: int = 2048
    zero_init: bool = True
    placement: Literal["after_prompt"] = "after_prompt"
    raw_id_codebook: tuple[int, ...] | None = None
    oov_index: int | None = None

    def __post_init__(self) -> None:
        if self.raw_id_codebook is None:
            if self.vocab_size is None or self.vocab_size <= 0:
                raise ValueError("backbone-token embodiment vocabulary must be positive")
            if self.oov_index is not None:
                raise ValueError("legacy vocab_size mode does not define an OOV index")
        else:
            codebook = tuple(int(value) for value in self.raw_id_codebook)
            if not codebook or len(codebook) != len(set(codebook)):
                raise ValueError("raw ID codebook must be a non-empty unique sequence")
            expected_vocab_size = len(codebook) + 1
            if self.vocab_size is not None and self.vocab_size != expected_vocab_size:
                raise ValueError(
                    "codebook vocabulary must contain one row per raw ID plus OOV"
                )
            if self.oov_index is not None and self.oov_index != len(codebook):
                raise ValueError("OOV index must immediately follow compact raw-ID rows")
            object.__setattr__(self, "raw_id_codebook", codebook)
            object.__setattr__(self, "vocab_size", expected_vocab_size)
            object.__setattr__(self, "oov_index", len(codebook))
        if self.token_width <= 0:
            raise ValueError("backbone token width must be positive")
        if not self.zero_init:
            raise ValueError("only zero-initialized backbone-token embeddings are supported")
        if self.placement != "after_prompt":
            raise ValueError(f"unsupported backbone-token placement: {self.placement!r}")

    @property
    def embodiment_vocab_size(self) -> int:
        return int(self.vocab_size)

    @property
    def uses_embodiment_codebook(self) -> bool:
        return self.raw_id_codebook is not None

    def create_description_transform(self, bundle_module: Any) -> Any | None:
        if self.raw_id_codebook is None:
            return None
        return bundle_module.RemapEmbodimentId(
            self.raw_id_codebook,
            int(self.oov_index),
        )


@dataclasses.dataclass(frozen=True)
class ExtractedEmbodimentBackboneToken(BackboneConditioning):
    """Project a structured embodiment embedding into one PaliGemma prefix token."""

    extractor: EmbodimentEmbeddingExtractor
    token_width: int = 2048
    zero_init: bool = True
    placement: Literal["after_prompt"] = "after_prompt"

    def __post_init__(self) -> None:
        if not isinstance(self.extractor, EmbodimentEmbeddingExtractor):
            raise TypeError("extractor must implement EmbodimentEmbeddingExtractor")
        if self.token_width <= 0:
            raise ValueError("backbone token width must be positive")
        if not self.zero_init:
            raise ValueError("only zero-initialized backbone-token projection is supported")
        if self.placement != "after_prompt":
            raise ValueError(f"unsupported backbone-token placement: {self.placement!r}")

    @property
    def uses_embodiment_description(self) -> bool:
        return True

    def create_description_transform(self, bundle_module: Any) -> Any:
        return self.extractor.create_input_transform(bundle_module)

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str]:
        return self.extractor.checkpoint_artifact_source(bundle_module)


@dataclasses.dataclass(frozen=True)
class FlatUrma4BackboneToken(BackboneConditioning):
    """Place every normalized URMA4 input directly in one backbone token.

    This is the literal aggregation-bottleneck control: it has no learned URMA
    encoder, pooling, normalization, or projection.  The fixed block order is
    control descriptions, description availability, live control state, control
    mask, global description, and global availability, followed by right zeros.
    """

    extractor: Urma4NewNormEmbodimentExtractor | Urma4AbsolutePhysicalStateEmbodimentExtractor
    token_width: int = 2048
    placement: Literal["after_prompt"] = "after_prompt"

    def __post_init__(self) -> None:
        if type(self.extractor) not in (
            Urma4NewNormEmbodimentExtractor,
            Urma4PhysicalStateEmbodimentExtractor,
            Urma4AbsolutePhysicalStateEmbodimentExtractor,
        ):
            raise TypeError(
                "flat URMA4 conditioning requires Urma4NewNormEmbodimentExtractor, "
                "Urma4PhysicalStateEmbodimentExtractor or Urma4AbsolutePhysicalStateEmbodimentExtractor"
            )
        if self.token_width < self.flattened_width:
            raise ValueError(
                "flat URMA4 inputs do not fit in one backbone token: "
                f"{self.flattened_width} > {self.token_width}"
            )
        if self.placement != "after_prompt":
            raise ValueError(f"unsupported backbone-token placement: {self.placement!r}")

    @property
    def flattened_width(self) -> int:
        controls, description_dim, state_dim = self.extractor.structured_input_shape
        global_dim = self.extractor.global_description_shape[0]
        return (
            controls * description_dim
            + controls * description_dim
            + controls * state_dim
            + controls
            + global_dim
            + global_dim
        )

    @property
    def uses_embodiment_description(self) -> bool:
        return True

    def create_description_transform(self, bundle_module: Any) -> Any:
        return self.extractor.create_input_transform(bundle_module)

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str]:
        return self.extractor.checkpoint_artifact_source(bundle_module)


@dataclasses.dataclass(frozen=True)
class RobotTokenBackboneSequence(BackboneConditioning):
    """Append a complete masked Transformer-Transformer robot-token sequence.

    The first four token types retain T2's link, dynamic-joint-connection,
    fixed-joint-connection, and actuator schemas. Additional typed tokens preserve
    control-relevant MJCF information that T2 cannot represent (collision meshes,
    tendons, equality constraints, and policy-control correspondence).
    """

    bundle_dir: str
    bundle_sha256: str | None = None
    build_manifest: str | None = None
    build_manifest_sha256: str | None = None
    max_tokens: int = 192
    feature_dim: int = 32
    reference_namespaces: int = 5
    token_types: int = 8
    type_capacities: tuple[int, ...] = (32, 32, 40, 16, 32, 8, 16, 16)
    token_width: int = 2048
    placement: Literal["after_prompt"] = "after_prompt"

    def __post_init__(self) -> None:
        if not self.bundle_dir:
            raise ValueError("RoboToken conditioning requires a bundle directory")
        if self.bundle_sha256 is None and self.build_manifest is None:
            raise ValueError("RoboToken conditioning requires a bundle hash or build manifest")
        if (self.build_manifest is None) != (self.build_manifest_sha256 is None):
            raise ValueError("RoboToken build manifest path and SHA-256 must be paired")
        for value, label in (
            (self.bundle_sha256, "bundle"),
            (self.build_manifest_sha256, "build manifest"),
        ):
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"RoboToken {label} requires a lowercase SHA-256 identity")
        if (self.max_tokens, self.feature_dim, self.reference_namespaces, self.token_types) != (
            192,
            32,
            5,
            8,
        ):
            raise ValueError("RoboToken v1 requires shapes (192, 32, 5) and 8 token types")
        if self.type_capacities != (32, 32, 40, 16, 32, 8, 16, 16):
            raise ValueError("RoboToken v1 type-block capacities changed")
        if self.token_width <= 0:
            raise ValueError("RoboToken prefix width must be positive")
        if self.placement != "after_prompt":
            raise ValueError(f"unsupported RoboToken placement: {self.placement!r}")

    @property
    def uses_robotokens(self) -> bool:
        return True

    def create_description_transform(self, bundle_module: Any) -> Any:
        bundle_dir, bundle_sha256 = bundle_module.prepare_robotoken_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
        )
        return bundle_module.AttachRobotTokens(
            bundle_dir=bundle_dir,
            expected_sha256=bundle_sha256,
            max_tokens=self.max_tokens,
            feature_dim=self.feature_dim,
            reference_namespaces=self.reference_namespaces,
        )

    def checkpoint_bundle_source(self, bundle_module: Any) -> dict[str, str]:
        bundle_dir, bundle_sha256 = bundle_module.prepare_robotoken_bundle(
            self.bundle_dir,
            expected_sha256=self.bundle_sha256,
            build_manifest=self.build_manifest,
            expected_build_manifest_sha256=self.build_manifest_sha256,
        )
        return bundle_module.checkpoint_bundle_source(
            bundle_dir,
            bundle_sha256,
            expected_schema="t2_robotokens_plus_mjcf_sidecars_v1",
        )


class ResponseHistoryConditioning(abc.ABC):
    """Describe the optional causal measured-response PaliGemma token."""

    @property
    def enabled(self) -> bool:
        return False

    def create_normalizer(self, transforms: Any, norm_stats: Any) -> Any | None:
        del transforms, norm_stats
        return None


@dataclasses.dataclass(frozen=True)
class NoResponseHistoryConditioning(ResponseHistoryConditioning):
    """Disable causal response-history input (checkpoint-compatible default)."""


@dataclasses.dataclass(frozen=True)
class CausalResponseHistoryBackboneToken(ResponseHistoryConditioning):
    """Project q[t-8:t], u[t-8:t-1], and temporal masks into one prefix token."""

    state_steps: int = 9
    action_steps: int = 8
    coordinate_dim: int = 7
    token_width: int = 2048
    zero_init: bool = True
    placement: Literal["after_prompt"] = "after_prompt"
    stats_sha256: str | None = None

    def __post_init__(self) -> None:
        if (self.state_steps, self.action_steps, self.coordinate_dim) != (9, 8, 7):
            raise ValueError("response-history v1 has the fixed 9x7 state / 8x7 action contract")
        if self.token_width <= 0:
            raise ValueError("response-history token width must be positive")
        if not self.zero_init:
            raise ValueError("only zero-initialized response-history projection is supported")
        if self.placement != "after_prompt":
            raise ValueError(f"unsupported response-history token placement: {self.placement!r}")
        if self.stats_sha256 is not None and (
            len(self.stats_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.stats_sha256)
        ):
            raise ValueError("response-history stats_sha256 must be lowercase SHA-256")

    @property
    def enabled(self) -> bool:
        return True

    @property
    def flattened_width(self) -> int:
        return (
            self.state_steps * self.coordinate_dim
            + self.action_steps * self.coordinate_dim
            + self.state_steps
            + self.action_steps
        )

    def create_normalizer(self, transforms: Any, norm_stats: Any) -> Any:
        required = (
            "response_state_history_gaussian",
            "response_action_history_gaussian",
        )
        missing = [key for key in required if key not in norm_stats]
        if missing:
            raise ValueError(f"response-history Gaussian stats are missing: {missing}")
        return transforms.NormalizeResponseHistory(
            state_stats=norm_stats[required[0]],
            action_stats=norm_stats[required[1]],
        )


class MobileBaseInput(abc.ABC):
    """Whether a holonomic base's three coordinates enter the model (``pi05_integration/mobile_base.py``)."""

    @property
    def enabled(self) -> bool:
        return False

    @property
    def contract(self) -> str | None:
        """The ``mobile_base.CONTRACTS`` name (it picks the RLDS fields read), or None when disabled."""
        return None

    def create_normalizer(
        self, transforms: Any, norm_stats: Any, *, use_quantiles: bool, state_quantiles: bool | None = None
    ) -> Any | None:
        """``state_quantiles`` (the recipe's ``Normalization.state_quantiles``) picks ``base_state``'s mode; None
        keeps ``use_quantiles`` for both base keys (every recipe whose scheme uses one mode)."""
        del transforms, norm_stats, use_quantiles, state_quantiles
        return None


@dataclasses.dataclass(frozen=True)
class NoMobileBase(MobileBaseInput):
    """Fixed-arm contract of every recipe before October 25: the loader emits no base fields."""


@dataclasses.dataclass(frozen=True)
class MobileBaseSlotsV1(MobileBaseInput):
    """``mobile_base_slots_v1`` (superseded by v2): base action and base state in model slots 10-12.

    Reads RLDS ``mobile_base_feature_v1``: base command minus measured pose in the current base frame,
    and the base pose in the episode's first base frame (a component without the fields gets zeros
    with false masks; a ``mobile_base_feature_v2`` component is refused).
    The arm vector, its masks, gripper handling and the ``state``/``actions`` statistics are unchanged;
    the base gets its own global ``base_state``/``base_actions`` statistics over mobile frames only. A
    fixed arm's model inputs and targets are exactly the arm-only ones (``transforms.AppendMobileBase``)."""

    @property
    def enabled(self) -> bool:
        return True

    @property
    def contract(self) -> str:
        from pi05_integration.mobile_base import CONTRACT_V1

        return CONTRACT_V1

    def create_normalizer(
        self, transforms: Any, norm_stats: Any, *, use_quantiles: bool, state_quantiles: bool | None = None
    ) -> Any:
        from pi05_integration.mobile_base import BASE_NORM_KEYS

        if not norm_stats:
            # Same identity path as GlobalQuantileNorm for repo_id="fake" or skip_norm_stats.
            return transforms.Normalize(None, use_quantiles=use_quantiles)
        missing = [key for key in BASE_NORM_KEYS if key not in norm_stats]
        if missing:
            raise ValueError(f"mobile-base normalization stats are missing: {missing}")
        if state_quantiles is None or state_quantiles == use_quantiles:
            return transforms.Normalize(
                {key: norm_stats[key] for key in BASE_NORM_KEYS}, use_quantiles=use_quantiles
            )
        # A scheme with split modes (e.g. Gaussian actions, q01/q99 state): the base state takes the arm state's
        # mode, so it stays inside Pi0.5's [-1, 1] state-token range; the base actions keep the actions' mode.
        return transforms.CompositeTransform([
            transforms.Normalize({"base_state": norm_stats["base_state"]}, use_quantiles=state_quantiles),
            transforms.Normalize({"base_actions": norm_stats["base_actions"]}, use_quantiles=use_quantiles),
        ])


@dataclasses.dataclass(frozen=True)
class MobileBaseSlotsV2(MobileBaseSlotsV1):
    """``mobile_base_slots_v2``: body-frame base velocities in model slots 10-12, after normalization.

    Reads RLDS ``mobile_base_feature_v2``: the action is the commanded base velocity and the state the
    measured base velocity, both ``[v_forward m/s, v_left m/s, yaw_rate rad/s]`` in the body frame (no
    heading or pose relative to the episode start, so no full-turn discontinuity). Everything else is
    v1's: slot placement, global base statistics over mobile frames, fixed arms unchanged, and a
    component without the fields (or with v1's) handled as in v1.

    ``dt_s`` is the control period the velocities were recorded at (RLDS
    ``episode_metadata/base_velocity_dt_s``): a velocity means a different displacement at another
    period, so the loader refuses a mobile episode recorded at another period and the server refuses
    a client stepping at another one."""

    dt_s: float

    def __post_init__(self):
        if not self.dt_s > 0:
            raise ValueError(f"mobile_base_slots_v2 needs a positive dt_s, got {self.dt_s!r}")

    @property
    def contract(self) -> str:
        from pi05_integration.mobile_base import CONTRACT_V2

        return CONTRACT_V2


@dataclasses.dataclass(frozen=True)
class MobileBaseSlotsV2AnyRate(MobileBaseSlotsV1):
    """``mobile_base_slots_v2`` for data whose robots run at different control periods (opt-in, 2026-11-04).

    Same RLDS fields, slots, statistics and fixed-arm guarantee as ``MobileBaseSlotsV2``, but no single ``dt_s``: a
    population collection draws one control rate per robot (xsrc40: 5-50 Hz), so every episode keeps its own
    ``episode_metadata/base_velocity_dt_s``. The velocities are SI per second at that period, so they mean the same
    motion at every rate; the loader and the server skip the period check (``getattr(mobile_base, "dt_s", None)`` is
    None) and the client integrates a commanded velocity over its own step. A displacement-per-step reading of the
    chunk (the H-step horizon spans H periods) still differs by rate; nothing here conditions on the rate."""

    @property
    def contract(self) -> str:
        from pi05_integration.mobile_base import CONTRACT_V2

        return CONTRACT_V2


class ArmAction(abc.ABC):
    """The arm action labels the loader builds (``pi05_integration/arm_action.py`` defines the forms)."""

    @property
    @abc.abstractmethod
    def contract(self) -> str:
        """The ``arm_action.CONTRACTS`` name; the loader and the client key on it."""

    @property
    def default(self) -> bool:
        from pi05_integration.arm_action import PER_STEP_JOINT_DELTA

        return self.contract == PER_STEP_JOINT_DELTA

    @property
    def query_anchored(self) -> bool:
        """Chunk targets against the query-step state: labels exist only for chunks, so the
        normalization statistics pool every valid chunk position instead of frames."""
        from pi05_integration.arm_action import QUERY_ANCHORED

        return self.contract in QUERY_ANCHORED


@dataclasses.dataclass(frozen=True)
class PerStepJointDelta(ArmAction):
    """Every recipe before October 26: the RLDS joint deltas, each against its own step's joints."""

    @property
    def contract(self) -> str:
        from pi05_integration.arm_action import PER_STEP_JOINT_DELTA

        return PER_STEP_JOINT_DELTA


@dataclasses.dataclass(frozen=True)
class QueryAnchoredJoint(ArmAction):
    """Arm joint targets of the chunk minus the joints measured at its query; reads any joint build."""

    @property
    def contract(self) -> str:
        from pi05_integration.arm_action import QUERY_ANCHORED_JOINT

        return QUERY_ANCHORED_JOINT


@dataclasses.dataclass(frozen=True)
class QueryAnchoredBaseTcp(ArmAction):
    """Commanded base-frame TCP goals against the TCP measured at the query; joint state unchanged.
    Reads ``arm_reference_feature_v1``; the client solves IK for each executed goal."""

    @property
    def contract(self) -> str:
        from pi05_integration.arm_action import QUERY_ANCHORED_BASE_TCP

        return QUERY_ANCHORED_BASE_TCP


@dataclasses.dataclass(frozen=True)
class QueryAnchoredJointAndBaseTcp(ArmAction):
    """Both query-anchored forms in one 20-slot chunk, the measured TCP appended to the joint state.
    Reads ``arm_reference_feature_v1`` of a left-packed joint layout; the client executes one head."""

    @property
    def contract(self) -> str:
        from pi05_integration.arm_action import QUERY_ANCHORED_JOINT_AND_BASE_TCP

        return QUERY_ANCHORED_JOINT_AND_BASE_TCP


class ImageGeometry(abc.ABC):
    """How the loader presents stored sim frames (``pi05_integration/image_geometry.py``; review T5)."""

    contract: str = "stretch"

    @property
    def enabled(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class StoredImageGeometry(ImageGeometry):
    """Checkpoint-compatible default: the frames as the builds stored them (sim renders stretched to squares)."""


@dataclasses.dataclass(frozen=True)
class LetterboxStretched(ImageGeometry):
    """Every sim frame un-stretched to its source aspect and zero-padded: the letterbox (``resize_with_pad``) of a
    ``source_width`` x ``source_height`` render, the Pi server's convention. Evaluation clients set the policy config's
    ``image_geometry`` to the same contract."""

    source_width: int = 848
    source_height: int = 480
    contract: str = "letterbox_stretched_v1"

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("letterbox source size must be positive")

    @property
    def enabled(self) -> bool:
        return True


class GripperValue(abc.ABC):
    """The gripper value the checkpoint is trained on (``pi05_integration/gripper_value.py``; review T8)."""

    contract: str = "stored"

    @property
    def enabled(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class StoredGripperValue(GripperValue):
    """Checkpoint-compatible default: the gripper slots as the builds stored them."""


@dataclasses.dataclass(frozen=True)
class GripperClosedness(GripperValue):
    """``gripper_closedness_v1``: state and action 0 = open, 1 = closed. With ``endpoints_table`` (and its sha256) the
    loader converts a ``physical_aperture_m`` build's gripper state at load, per robot (by its artifact hash); without
    it the builds already carry the value (``--gripper-proprio-source gripper_closedness_v1``). Evaluation clients set
    ``proprio_gripper_source="gripper_closedness_v1"``, which names the contract to the server."""

    endpoints_table: str = ""
    endpoints_table_sha256: str = ""
    contract: str = "gripper_closedness_v1"

    def __post_init__(self) -> None:
        if bool(self.endpoints_table) != bool(self.endpoints_table_sha256):
            raise ValueError("endpoints_table and endpoints_table_sha256 go together")

    @property
    def enabled(self) -> bool:
        return True


class JointFrame(abc.ABC):
    """The arm-joint frame the checkpoint is trained in (``pi05_integration/joint_frame.py``; review T2/item 4)."""

    contract: str = "stored"

    @property
    def enabled(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class StoredJointFrame(JointFrame):
    """Checkpoint-compatible default: each robot's arm joints in its own model's convention, as the builds stored them."""


@dataclasses.dataclass(frozen=True)
class CanonicalJointFrame(JointFrame):
    """``canonical_mdh_tool_v1``: the loader maps every sim episode's arm state to phi = theta + s * q and its
    joint-difference actions to s * a, with its robot's (s, theta) from the sha-pinned joint-frame table (by its
    artifact hash). Evaluation clients set ``joint_frame="canonical_mdh_tool_v1"`` and send their robot's (s, theta);
    the server maps in and back."""

    table: str
    table_sha256: str
    contract: str = "canonical_mdh_tool_v1"

    def __post_init__(self) -> None:
        if not self.table or not self.table_sha256:
            raise ValueError("CanonicalJointFrame needs its table and table_sha256")

    @property
    def enabled(self) -> bool:
        return True


class ActionTimeConditioning(abc.ABC):
    """Whether the Pi0.5 action expert sees the time offset of each chunk entry (``Pi0.embed_suffix``).

    Enabled, every sample carries ``action_dt_s`` (float32 seconds between chunk entries): the loader reads it from
    ``episode_metadata/policy_dt_s`` (``mlspaces_to_rlds.py --policy-dt-metadata``) and serving requests send the
    executed control period. The data config and ``Pi0Config.action_time_conditioning`` must agree."""

    @property
    def enabled(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class NoActionTime(ActionTimeConditioning):
    """Checkpoint-compatible default: no ``action_dt_s`` field and no parameter."""


@dataclasses.dataclass(frozen=True)
class ActionTokenTimeSeconds(ActionTimeConditioning):
    """One raw-seconds channel tau_k = (k + 1) * action_dt_s per chunk entry k (its label is the target reached at
    t + (k + 1) dt), through a zero-initialized bias-free ``action_time_proj`` (1 -> action expert width) added to
    the action tokens: the released function is unchanged at step 0. Pi0.5 only."""

    @property
    def enabled(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class ActionTokenTimeSecondsGPNoise(ActionTokenTimeSeconds):
    """The time channel plus a function-space flow-matching source: per action dimension the noise is one draw of a
    zero-mean, unit-variance Gaussian process over the same tau_k = (k + 1) * action_dt_s, squared-exponential kernel
    exp(-(tau_i - tau_j)^2 / (2 length_scale_s^2)) with ``jitter`` added to the diagonal (renormalized to unit
    variance). So every control rate samples the same random function of time, where white noise adds energy at
    finer rates. ``Pi0`` colors standard normal draws (training, sampling, and a server's seeded noise) with the
    Cholesky factor of the kernel matrix."""

    length_scale_s: float = 0.1
    jitter: float = 1e-3

    def __post_init__(self) -> None:
        if not (math.isfinite(self.length_scale_s) and self.length_scale_s > 0):
            raise ValueError(f"length_scale_s must be finite and positive, got {self.length_scale_s!r}")
        if not (math.isfinite(self.jitter) and self.jitter > 0):
            raise ValueError(f"jitter must be finite and positive, got {self.jitter!r}")


@dataclasses.dataclass(frozen=True)
class ActionTokenTimeSecondsMaternNoise(ActionTokenTimeSecondsGPNoise):
    """``ActionTokenTimeSecondsGPNoise`` with the Matern kernel of smoothness ``nu`` (0.5, 1.5 or 2.5; lower is
    rougher) instead of the squared exponential (its infinitely smooth limit). Its spectrum is flat below about
    1 / (2 pi length_scale_s) Hz and falls as a power law above it, so the source can stay rougher than the action
    labels at every rate while sharing their low-frequency correlation."""

    nu: float = 1.5
    jitter: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be 0.5, 1.5 or 2.5, got {self.nu!r}")


@dataclasses.dataclass(frozen=True)
class Pi05InputConfig:
    """Composable, serializable input contract for one Pi0.5 recipe.

    ``action_loss_coordinates="model"`` is the canonical padded-action loss: supervise the
    full fixed-width model action, including zero padding.  This matches original Pi training and
    is the desirable default.  ``"physical"`` exists only for named valid-coordinate ablations.
    """

    prompt: PromptConditioning = dataclasses.field(default_factory=NoPromptConditioning)
    # Preserve old checkpoint behavior by default. New corrected recipes opt in under
    # a new config name because state tokenization is a scientific contract.
    state_serialization: StateSerialization = dataclasses.field(
        default_factory=LegacyFixedLayoutState
    )
    state: StateEncoding = dataclasses.field(default_factory=NoStateEncoding)
    normalization: Normalization = dataclasses.field(default_factory=GlobalQuantileNorm)
    action_clipping: ActionClipping = dataclasses.field(default_factory=NoActionClipping)
    response_history: ResponseHistoryConditioning = dataclasses.field(
        default_factory=NoResponseHistoryConditioning
    )
    # ``model`` reproduces released Pi0.5 training by supervising every padded model
    # coordinate and is the recommended default. ``physical`` retains the dataset's
    # left-packed semantic action mask for an explicit scientific ablation.
    action_loss_coordinates: ActionLossCoordinates = "model"
    # Holonomic-base slots (opt-in, new named configs only). The default emits no base fields, so
    # every earlier recipe, normalization asset and checkpoint keeps its exact batches.
    mobile_base: MobileBaseInput = dataclasses.field(default_factory=NoMobileBase)
    # Arm action form (opt-in, new named configs only). The default keeps every earlier recipe's labels.
    arm_action: ArmAction = dataclasses.field(default_factory=PerStepJointDelta)
    # Chunk-entry time offsets for the action expert (opt-in, new named configs only); mirrored on Pi0Config.
    action_time: ActionTimeConditioning = dataclasses.field(default_factory=NoActionTime)
    # Sim-frame geometry (opt-in, new named configs only; review T5). The default keeps the stored stretched frames.
    image_geometry: ImageGeometry = dataclasses.field(default_factory=StoredImageGeometry)
    # Gripper value (opt-in, new named configs only; review T8). The default keeps the stored gripper slots.
    gripper_value: GripperValue = dataclasses.field(default_factory=StoredGripperValue)
    # Arm-joint frame (opt-in, new named configs only; review T2/item 4). The default keeps each robot's own joints.
    joint_frame: JointFrame = dataclasses.field(default_factory=StoredJointFrame)

    def __post_init__(self) -> None:
        if not isinstance(self.image_geometry, ImageGeometry):
            raise TypeError("image_geometry must implement ImageGeometry")
        if not isinstance(self.gripper_value, GripperValue):
            raise TypeError("gripper_value must implement GripperValue")
        if not isinstance(self.joint_frame, JointFrame):
            raise TypeError("joint_frame must implement JointFrame")
        if not isinstance(self.mobile_base, MobileBaseInput):
            raise TypeError("mobile_base must implement MobileBaseInput")
        if not isinstance(self.arm_action, ArmAction):
            raise TypeError("arm_action must implement ArmAction")
        if not isinstance(self.action_time, ActionTimeConditioning):
            raise TypeError("action_time must implement ActionTimeConditioning")
        if self.mobile_base.enabled and not (self.arm_action.default or isinstance(self.arm_action, QueryAnchoredJoint)):
            # The BASE TCP forms are untested with a base: the base slots sit at 10-12, where the joint + BASE TCP
            # chunk continues. Query-anchored joints keep the 10-wide arm block, so the base slots are unchanged
            # (2026-11-04; tests/test_pi05_mobile_base_slots.py).
            raise ValueError("mobile-base slots support only the per-step joint-delta and query-anchored joint arm actions")
        if self.mobile_base.enabled and not isinstance(self.state_serialization, LegacyFixedLayoutState):
            # The base state follows the fixed 10-wide arm state; compact serialization would move it.
            raise ValueError("mobile-base slots require the fixed-layout (10-wide) state serialization")
        if not isinstance(self.state_serialization, StateSerialization):
            raise TypeError("state_serialization must implement StateSerialization")
        if not isinstance(self.response_history, ResponseHistoryConditioning):
            raise TypeError("response_history must implement ResponseHistoryConditioning")
        if self.joint_frame.enabled:
            from pi05_integration.joint_frame import JOINT_DIFFERENCE_FORMS

            if self.arm_action.contract not in JOINT_DIFFERENCE_FORMS:
                raise ValueError(f"the canonical joint frame supports the arm action forms {JOINT_DIFFERENCE_FORMS}")
            if self.response_history.enabled:
                # The response histories carry each robot's own joints.
                raise ValueError("the canonical joint frame does not support response-history conditioning")
        norm_ids = self.normalization.embodiment_ids
        state_ids = self.state.embodiment_ids
        if norm_ids and state_ids and norm_ids != state_ids:
            raise ValueError(
                f"state and normalization embodiment codebooks differ: {state_ids} != {norm_ids}"
            )
        if self.action_loss_coordinates not in ("model", "physical"):
            raise ValueError(
                "action_loss_coordinates must be 'model' or 'physical', got "
                f"{self.action_loss_coordinates!r}"
            )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe record that retains concrete strategy class names."""
        return {
            "prompt": {"type": type(self.prompt).__name__, **dataclasses.asdict(self.prompt)},
            "state_serialization": {
                "type": type(self.state_serialization).__name__,
                **dataclasses.asdict(self.state_serialization),
            },
            "state": {"type": type(self.state).__name__, **dataclasses.asdict(self.state)},
            "normalization": {
                "type": type(self.normalization).__name__,
                **dataclasses.asdict(self.normalization),
            },
            "action_clipping": {
                "type": type(self.action_clipping).__name__,
                **dataclasses.asdict(self.action_clipping),
            },
            "response_history": {
                "type": type(self.response_history).__name__,
                **dataclasses.asdict(self.response_history),
            },
            "action_loss_coordinates": self.action_loss_coordinates,
            # Recorded only when enabled, so earlier recipes keep their exact provenance records.
            **(
                {"mobile_base": {"type": type(self.mobile_base).__name__, **dataclasses.asdict(self.mobile_base)}}
                if self.mobile_base.enabled
                else {}
            ),
            # Recorded only when not the default, for the same reason.
            **(
                {"arm_action": {"type": type(self.arm_action).__name__, "contract": self.arm_action.contract}}
                if not self.arm_action.default
                else {}
            ),
            **(
                {"action_time": {"type": type(self.action_time).__name__, **dataclasses.asdict(self.action_time)}}
                if self.action_time.enabled
                else {}
            ),
            **(
                {"image_geometry": {"type": type(self.image_geometry).__name__,
                                    **dataclasses.asdict(self.image_geometry)}}
                if self.image_geometry.enabled
                else {}
            ),
            **(
                {"gripper_value": {"type": type(self.gripper_value).__name__, **dataclasses.asdict(self.gripper_value)}}
                if self.gripper_value.enabled
                else {}
            ),
            **(
                {"joint_frame": {"type": type(self.joint_frame).__name__, **dataclasses.asdict(self.joint_frame)}}
                if self.joint_frame.enabled
                else {}
            ),
        }


def action_expert_conditioning_record(
    conditioning: ActionExpertConditioning,
) -> dict[str, Any]:
    record = {"type": type(conditioning).__name__, **dataclasses.asdict(conditioning)}
    if isinstance(conditioning, ExtractedEmbodimentAdaRMS):
        record["extractor"] = _extractor_record(conditioning.extractor)
    return record


def _extractor_record(extractor: Any) -> dict[str, Any]:
    """The extractor's fields; ``joint_frame`` only when not the default, so earlier records are unchanged."""
    record = {"type": type(extractor).__name__, **dataclasses.asdict(extractor)}
    if record.get("joint_frame") == "stored":
        del record["joint_frame"]
    return record


def backbone_conditioning_record(
    conditioning: BackboneConditioning,
) -> dict[str, Any]:
    record = {"type": type(conditioning).__name__, **dataclasses.asdict(conditioning)}
    if isinstance(
        conditioning,
        (ExtractedEmbodimentBackboneToken, FlatUrma4BackboneToken),
    ):
        record["extractor"] = _extractor_record(conditioning.extractor)
    return record


@dataclasses.dataclass(frozen=True)
class DataVisualizationConfig:
    """Periodic visualization of actual train-mode inputs and sampled chunks."""

    interval: int = 500
    sample_count: int = 4

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError("visualization interval must be non-negative")
        if self.sample_count <= 0:
            raise ValueError("visualization sample count must be positive")
