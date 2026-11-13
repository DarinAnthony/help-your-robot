import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms
from pi05_integration import normalization_bundle as _normalization_bundle


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    normalization_record = data_config.input_config.to_record()["normalization"]
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    bundle = _normalization_bundle.validate_checkpoint_bundle(
        checkpoint_dir / "assets" / data_config.asset_id,
        asset_id=data_config.asset_id,
        normalization=normalization_record,
    )
    if bundle is not None and norm_stats is not None:
        raise ValueError(
            "This checkpoint exports an immutable normalization processor; "
            "evaluation-time norm_stats overrides are forbidden."
        )
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        if bundle is None:
            logging.warning(
                "Checkpoint predates normalization bundles; using its checkpoint-local "
                "norm_stats.json compatibility path."
            )

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    normalizer = data_config.input_config.normalization.create_normalizer(
        transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    response_history_normalizer = data_config.input_config.response_history.create_normalizer(
        transforms, norm_stats
    )
    mobile_base_normalizer = data_config.input_config.mobile_base.create_normalizer(
        transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        state_quantiles=data_config.input_config.normalization.state_quantiles,
    )
    unnormalizer = data_config.input_config.normalization.create_unnormalizer(
        transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            normalizer,
            *([] if response_history_normalizer is None else [response_history_normalizer]),
            *([] if mobile_base_normalizer is None else [mobile_base_normalizer]),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            unnormalizer,
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
