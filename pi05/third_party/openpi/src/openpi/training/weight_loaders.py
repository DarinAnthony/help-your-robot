import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class UnspecifiedWeightLoader(WeightLoader):
    """TrainConfig's default: refuses to train rather than silently starting from random weights.

    A config that forgets its weight_loader used to get NoOpWeightLoader and train the whole
    model, VLM included, from scratch with nothing in the logs to say so. Name the init instead:
    NoOpWeightLoader() for deliberate random init, PaliGemmaWeightLoader() for the VLM backbone,
    or CheckpointWeightLoader(...) to fine-tune.
    """

    def load(self, params: at.Params) -> at.Params:
        raise ValueError(
            "This TrainConfig names no weight_loader. Set weight_loader=NoOpWeightLoader() for "
            "deliberate random init, PaliGemmaWeightLoader(), or CheckpointWeightLoader(<params>)."
        )


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Released checkpoints do not contain opt-in adapter parameters. Keep the
    # historical LoRA-only behavior by default; experiments introducing another
    # zero-initialized adapter must opt in explicitly.
    missing_regex: str = ".*lora.*"
    # Concatenated AdaRMS doubles only the condition-input axis. Expand each
    # released timestep-conditioning kernel by copying it into the new
    # embodiment branch, preserving the released function when the learned
    # embodiment lookup is zero-initialized.
    adarms_condition_width_multiplier: int = 1
    adarms_condition_extra_width: int = 0

    def __post_init__(self) -> None:
        if self.adarms_condition_width_multiplier <= 0:
            raise ValueError("AdaRMS condition-width multiplier must be positive")
        if self.adarms_condition_extra_width < 0:
            raise ValueError("AdaRMS condition extra width must be non-negative")
        if self.adarms_condition_width_multiplier > 1 and self.adarms_condition_extra_width:
            raise ValueError(
                "AdaRMS condition multiplier and extra width are mutually exclusive"
            )

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.adarms_condition_width_multiplier > 1:
            loaded_params = _expand_adarms_conditioning_kernels(
                loaded_params,
                params,
                multiplier=self.adarms_condition_width_multiplier,
            )
        elif self.adarms_condition_extra_width:
            loaded_params = _expand_adarms_conditioning_kernels_extra(
                loaded_params,
                params,
                extra_width=self.adarms_condition_extra_width,
            )
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        _check_paligemma_coverage(loaded_params, params)
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _check_paligemma_coverage(loaded_params: at.Params, params: at.Params) -> None:
    """Fail loudly instead of silently training a partly random VLM.

    _merge_params drops loaded tensors whose names match no model parameter and, with
    missing_regex=".*", fills every unmatched parameter from the random init. A renamed module
    would therefore start from scratch without any error. Every checkpoint tensor must land on a
    parameter, and every PaliGemma parameter must be loaded except the action expert's, which the
    pi0 Gemma marks with a ``_1`` path segment under ``PaliGemma/llm/`` (``q_einsum_1``, ``mlp_1``,
    ``final_norm_1``, ...). The ``llm`` restriction matters: SigLIP has its own ``_1`` modules
    (``LayerNorm_1``, ``MlpBlock_0/Dense_1``, about a third of its weights) that must be loaded.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    unused = sorted(k for k in flat_loaded if k not in flat_ref)
    if unused:
        raise ValueError(f"{len(unused)} PaliGemma checkpoint tensors match no model parameter: {unused[:10]}")
    not_loaded = sorted(
        k for k in flat_ref if k.startswith("PaliGemma/") and k not in flat_loaded and not _is_action_expert(k)
    )
    if not_loaded:
        raise ValueError(f"{len(not_loaded)} PaliGemma parameters would stay randomly initialized: {not_loaded[:10]}")
    logger.info(
        "PaliGemma init: loaded %d tensors; %d action-expert PaliGemma parameters and all non-PaliGemma "
        "parameters keep their random init",
        len(flat_loaded),
        sum(1 for k in flat_ref if k.startswith("PaliGemma/") and k not in flat_loaded),
    )


def _is_action_expert(path: str) -> bool:
    """pi0's second Gemma (the action expert): a ``_1``-suffixed module inside PaliGemma/llm."""
    return path.startswith("PaliGemma/llm/") and any(part.endswith("_1") for part in path.split("/"))


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            if v.shape != flat_ref[k].shape:
                raise ValueError(f"checkpoint parameter {k!r} has shape {v.shape}, expected {flat_ref[k].shape}")
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


_ADARMS_CONDITION_KERNELS = {
    "PaliGemma/llm/final_norm_1/Dense_0/kernel",
    "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel",
    "PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel",
}


def _expand_adarms_conditioning_kernels(
    loaded_params: at.Params,
    params: at.Params,
    *,
    multiplier: int,
) -> at.Params:
    """Expand only action-expert AdaRMS condition kernels along their input axis."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    for key in _ADARMS_CONDITION_KERNELS:
        if key not in flat_loaded or key not in flat_ref:
            continue
        loaded = flat_loaded[key]
        reference = flat_ref[key]
        expected_shape = (*loaded.shape[:-2], loaded.shape[-2] * multiplier, loaded.shape[-1])
        if reference.shape != expected_shape:
            raise ValueError(
                f"cannot expand AdaRMS kernel {key!r} from {loaded.shape} to {reference.shape} "
                f"with multiplier {multiplier}"
            )
        flat_loaded[key] = np.concatenate([loaded] * multiplier, axis=-2)
    return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")


def _expand_adarms_conditioning_kernels_extra(
    loaded_params: at.Params,
    params: at.Params,
    *,
    extra_width: int,
) -> at.Params:
    """Append base AdaRMS columns for a narrower, initially zero-gated condition."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    for key in _ADARMS_CONDITION_KERNELS:
        if key not in flat_loaded or key not in flat_ref:
            continue
        loaded = flat_loaded[key]
        reference = flat_ref[key]
        if extra_width > loaded.shape[-2]:
            raise ValueError(
                f"cannot append {extra_width} AdaRMS columns from kernel {key!r} "
                f"with input width {loaded.shape[-2]}"
            )
        expected_shape = (
            *loaded.shape[:-2],
            loaded.shape[-2] + extra_width,
            loaded.shape[-1],
        )
        if reference.shape != expected_shape:
            raise ValueError(
                f"cannot expand AdaRMS kernel {key!r} from {loaded.shape} to "
                f"{reference.shape} with extra width {extra_width}"
            )
        flat_loaded[key] = np.concatenate(
            [loaded, loaded[..., :extra_width, :]], axis=-2
        )
    return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
