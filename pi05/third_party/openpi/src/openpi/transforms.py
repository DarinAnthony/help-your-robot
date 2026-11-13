from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class NormalizeResponseHistory(DataTransformFn):
    """Gaussian-normalize valid q/u history rows and keep padding exactly zero."""

    state_stats: NormStats
    action_stats: NormStats

    def __call__(self, data: DataDict) -> DataDict:
        from molmo_spaces.response_history import gaussian_normalize_history

        pairs = (
            ("response_state_history", "response_state_history_mask", self.state_stats),
            ("response_action_history", "response_action_history_mask", self.action_stats),
        )
        for value_key, mask_key, stats in pairs:
            if value_key not in data or mask_key not in data:
                raise ValueError(f"causal response history requires {value_key} and {mask_key}")
            data[value_key] = gaussian_normalize_history(
                data[value_key],
                data[mask_key],
                mean=stats.mean,
                std=stats.std,
            )
        return data


@dataclasses.dataclass(frozen=True)
class ClipActions(DataTransformFn):
    """Winsorize normalized training actions without changing inference outputs."""

    limit: float

    def __post_init__(self):
        if not np.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("action clipping limit must be finite and positive")

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        return {**data, "actions": np.clip(data["actions"], -self.limit, self.limit)}


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def embodiment_norm_key(field: str, embodiment_id: int) -> str:
    """Stable flat asset key for one field's seen-embodiment statistics."""

    return f"{field}__embodiment_{int(embodiment_id)}"


@dataclasses.dataclass(frozen=True)
class NormalizeByEmbodiment(DataTransformFn):
    """Normalize state/actions using the table selected by explicit embodiment ID.

    The flat key convention keeps the asset file backward compatible: it still contains the
    ordinary global ``state``/``actions`` entries, while opt-in configs select the additional
    per-embodiment entries.  Scalar IDs cover training and ordinary inference; a vector of IDs
    covers the accelerator-batched policy path.  ``state_quantiles`` (opt-in) picks the state's
    mode independently of the actions'; ``None`` keeps one mode for both.
    """

    norm_stats: at.PyTree[NormStats]
    embodiment_ids: Sequence[int]
    use_quantiles: bool = False
    state_quantiles: bool | None = None

    def __post_init__(self):
        ids = tuple(int(value) for value in self.embodiment_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("embodiment_ids must be a non-empty unique sequence")
        object.__setattr__(self, "embodiment_ids", ids)
        for embodiment_id in ids:
            for field in ("state", "actions"):
                key = embodiment_norm_key(field, embodiment_id)
                if key not in self.norm_stats:
                    raise ValueError(f"missing per-embodiment normalization statistics: {key}")

    def _selector(self, embodiment_id: int) -> dict[str, NormStats]:
        if embodiment_id not in self.embodiment_ids:
            raise ValueError(
                f"embodiment_id {embodiment_id} is outside normalization codebook "
                f"{self.embodiment_ids}"
            )
        return {
            field: self.norm_stats[embodiment_norm_key(field, embodiment_id)]
            for field in ("state", "actions")
        }

    def _quantiles(self, field: str) -> bool:
        if field == "state" and self.state_quantiles is not None:
            return self.state_quantiles
        return self.use_quantiles

    def _apply(self, data: DataDict, transform: type, action: str) -> DataDict:
        if "embodiment_id" not in data:
            raise ValueError(f"per-embodiment {action} requires embodiment_id")
        ids = np.asarray(data["embodiment_id"])
        if ids.ndim == 0:
            selected = self._selector(int(ids.item()))
            if self._quantiles("state") == self.use_quantiles:
                return transform(selected, use_quantiles=self.use_quantiles)(data)
            output = dict(data)
            for field in ("state", "actions"):
                if field in output:
                    output[field] = transform(
                        {field: selected[field]}, use_quantiles=self._quantiles(field)
                    )({field: output[field]})[field]
            return output

        output = dict(data)
        for field in ("state", "actions"):
            if field not in output:
                continue
            values = np.asarray(output[field]).copy()
            if values.shape[0] != ids.shape[0]:
                raise ValueError(
                    f"batched {field} and embodiment_id disagree: {values.shape}, {ids.shape}"
                )
            for index, embodiment_id in enumerate(ids):
                item = transform(
                    {field: self._selector(int(embodiment_id))[field]},
                    use_quantiles=self._quantiles(field),
                )({field: values[index]})
                values[index] = item[field]
            output[field] = values
        return output

    def __call__(self, data: DataDict) -> DataDict:
        return self._apply(data, Normalize, "normalization")


@dataclasses.dataclass(frozen=True)
class UnnormalizeByEmbodiment(NormalizeByEmbodiment):
    """Inverse of :class:`NormalizeByEmbodiment` for sampled action batches."""

    def __call__(self, data: DataDict) -> DataDict:
        return self._apply(data, Unnormalize, "unnormalization")


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    fail_on_overflow: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            if hasattr(prompt, "item"):
                prompt = prompt.item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

        tokens, token_masks = self.tokenizer.tokenize(
            prompt, state, fail_on_overflow=self.fail_on_overflow
        )
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


def _left_packed_width(mask: np.ndarray, *, name: str) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {mask.shape}")
    width = int(np.sum(mask))
    if not np.array_equal(mask, np.arange(mask.size) < width):
        raise ValueError(f"{name} must be left-packed")
    if width == 0:
        raise ValueError(f"{name} must select at least one coordinate")
    return width


@dataclasses.dataclass(frozen=True)
class CompactLeftPackedState(DataTransformFn):
    """Keep only valid normalized state coordinates before text tokenization."""

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data or "state_mask" not in data:
            raise ValueError("compact state serialization requires state and state_mask")
        state = np.asarray(data["state"])
        mask = np.asarray(data["state_mask"], dtype=bool)
        if state.ndim != 1 or state.shape != mask.shape:
            raise ValueError(
                "compact state serialization requires matching one-dimensional "
                f"state/state_mask, got {state.shape} and {mask.shape}"
            )
        width = _left_packed_width(mask, name="state_mask")
        if not np.all(np.isfinite(state[:width])):
            raise ValueError("compact state serialization received non-finite valid state")
        return {**data, "state": state[:width]}


@dataclasses.dataclass(frozen=True)
class ZeroInvalidActions(DataTransformFn):
    """Reset absent left-packed clean action slots after normalization."""

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        if "action_semantic_mask" not in data:
            raise ValueError("semantic action zeroing requires action_semantic_mask")
        actions = np.asarray(data["actions"])
        mask = np.asarray(data["action_semantic_mask"], dtype=bool)
        if mask.shape != actions.shape[-1:]:
            raise ValueError(
                "action_semantic_mask must match the action dimension, got "
                f"{mask.shape} and {actions.shape}"
            )
        width = _left_packed_width(mask, name="action_semantic_mask")
        result = actions.copy()
        result[..., width:] = 0
        return {**data, "actions": result}


@dataclasses.dataclass(frozen=True)
class AppendMobileBase(DataTransformFn):
    """Write normalized holonomic-base coordinates into the model slots after the arm vector.

    ``mobile_base_slots_v1``/``v2`` (``pi05_integration/mobile_base.py``). Consumes ``base_state``,
    ``base_state_mask``, ``base_actions`` and ``base_action_mask`` (already normalized) and removes
    them. Runs after ``ZeroInvalidActions`` and before tokenization and padding:

    * actions ``(H, arm_width)`` -> ``(H, arm_width + 3)``: the base slots hold the base action of a
      mobile sample and zero for a fixed arm (what padding puts there today), so every sample keeps
      one action width and a fixed arm's padded model target is unchanged;
    * state: the base state is appended only for a mobile sample. A fixed arm's state, and so its
      Pi0.5 state text, is exactly the arm-only one; padding then fills the same slots with zero;
    * ``action_semantic_mask`` gets the base mask (constant width, metrics only); a physical-coordinate
      ``action_loss_mask`` gets the base mask restricted to valid timesteps; a model-coordinate one
      (width 1) is unchanged. ``state_mask`` keeps its arm-only meaning (nothing reads it afterwards).
    """

    arm_width: int = 10
    # Training-only winsorization of the normalized base action, matching ``ClipActions`` on the arm.
    clip_limit: float | None = None

    def __call__(self, data: DataDict) -> DataDict:
        missing = [key for key in ("base_state", "base_state_mask") if key not in data]
        if missing:
            raise ValueError(f"mobile-base input is missing {missing}")
        data = dict(data)
        base_state = np.asarray(data.pop("base_state"), np.float32)
        base_state_mask = np.asarray(data.pop("base_state_mask"), dtype=bool)
        if base_state.shape != (3,) or base_state_mask.shape != (3,):
            raise ValueError("base_state and base_state_mask must have shape (3,)")
        if base_state_mask.any() != base_state_mask.all():
            raise ValueError("base_state_mask must mark all three base coordinates or none")
        state = np.asarray(data["state"])
        if state.shape[-1] != self.arm_width:
            raise ValueError(f"mobile-base slots follow a {self.arm_width}-wide state, got {state.shape}")
        if base_state_mask.all():
            data["state"] = np.concatenate([state, base_state.astype(state.dtype)], axis=-1)

        if "base_actions" in data or "actions" in data:
            if not {"actions", "base_actions", "base_action_mask"} <= set(data):
                raise ValueError("mobile-base training input needs actions, base_actions and base_action_mask")
            actions = np.asarray(data["actions"])
            base_actions = np.asarray(data.pop("base_actions"), actions.dtype)
            base_mask = np.asarray(data.pop("base_action_mask"), dtype=bool)
            if base_mask.shape != (3,) or base_actions.shape != (*actions.shape[:-1], 3):
                raise ValueError("base_actions must be (..., 3) beside actions and base_action_mask (3,)")
            if not np.array_equal(base_mask, base_state_mask):
                raise ValueError("base_action_mask and base_state_mask disagree")
            if actions.shape[-1] != self.arm_width:
                raise ValueError(f"mobile-base slots follow {self.arm_width}-wide actions, got {actions.shape}")
            if self.clip_limit is not None:
                base_actions = np.clip(base_actions, -self.clip_limit, self.clip_limit)
            data["actions"] = np.concatenate([actions, np.where(base_mask, base_actions, 0)], axis=-1)
            if "action_semantic_mask" in data:
                data["action_semantic_mask"] = np.concatenate(
                    [np.asarray(data["action_semantic_mask"], dtype=bool), base_mask], axis=-1
                )
            if "action_loss_mask" in data:
                loss_mask = np.asarray(data["action_loss_mask"], dtype=bool)
                if loss_mask.shape[-1] != 1:
                    timestep = np.any(loss_mask, axis=-1, keepdims=True)
                    data["action_loss_mask"] = np.concatenate([loss_mask, timestep & base_mask], axis=-1)
        else:
            data.pop("base_action_mask", None)
        return data


@dataclasses.dataclass(frozen=True)
class DropKeys(DataTransformFn):
    """Remove transport-only metadata that has no numeric model leaf (e.g. string identities)."""

    keys: tuple[str, ...]

    def __call__(self, data: DataDict) -> DataDict:
        return {key: value for key, value in data.items() if key not in self.keys}


@dataclasses.dataclass(frozen=True)
class AppendEmbodimentOneHot(DataTransformFn):
    """Append a stable seen-embodiment code to normalized Pi0.5 state.

    Pi0.5 serializes normalized state into its language prefix. Appending the
    code after normalization therefore gives the model an explicit, direct
    identity signal without changing model parameters or corrupting physical
    proprioception statistics. IDs may be non-contiguous; their order in
    ``embodiment_ids`` defines the fixed codebook.
    """

    embodiment_ids: Sequence[int]

    def __post_init__(self):
        ids = tuple(int(value) for value in self.embodiment_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("embodiment_ids must be a non-empty unique sequence")
        object.__setattr__(self, "embodiment_ids", ids)

    def __call__(self, data: DataDict) -> DataDict:
        if "embodiment_id" not in data:
            raise ValueError("explicit embodiment state conditioning requires embodiment_id")
        embodiment_id = int(np.asarray(data["embodiment_id"]).item())
        try:
            index = self.embodiment_ids.index(embodiment_id)
        except ValueError as error:
            raise ValueError(
                f"embodiment_id {embodiment_id} is outside configured codebook "
                f"{self.embodiment_ids}"
            ) from error
        code = np.full(len(self.embodiment_ids), -1.0, dtype=np.float32)
        code[index] = 1.0
        data["state"] = np.concatenate([np.asarray(data["state"], np.float32), code], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            # RLDS/tf.data yields prompts as bytes (or numpy bytes scalars), not str. Mirror
            # TokenizePrompt's handling so this transform also works on the molmospaces
            # streaming pipeline, not just the LeRobot path it was originally written for.
            if hasattr(prompt, "item"):
                prompt = prompt.item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        if "action_loss_mask" in data:
            action_loss_mask = np.asarray(data["action_loss_mask"], dtype=bool)
            if action_loss_mask.shape[-1] == 1:
                data["action_loss_mask"] = np.broadcast_to(
                    action_loss_mask, data["actions"].shape
                ).copy()
            else:
                data["action_loss_mask"] = pad_to_dim(
                    action_loss_mask, self.model_action_dim, axis=-1, value=False
                )
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
