"""pi0.5 (openpi) inference server speaking the octo_eval_server websocket protocol,
so the EXISTING MuJoCo eval clients (scripts/datagen/eval_replay.py etc., mlspaces env)
can drive a pi0.5 policy with zero client changes.

Protocol (msgpack-numpy over websocket), identical to octo_integration/octo_eval_server.py:
  client -> obs dict:
      single: image_primary (H,W,3)u8, image_wrist (h,w,3)u8, proprio (10,)f32, language str
      batch:  image_primary (B,H,W,3), image_wrist (B,h,w,3), proprio (B,10), language list[str]
      + embodiment_id, embodiment_variant, optional embodiment_action_dim (=n_arm+1),
        optional embodiment_action_mask (non-contiguous cross-family slots), optional model tag
  server -> {"actions": (action_horizon, 10) f32}  (single)  or (B, action_horizon, 10) (batch)
      UNNORMALIZED, [delta_arm | grip] left-packed to width 10 (the wire contract).
  Mobile-base checkpoints (``Pi05InputConfig.mobile_base``, pi05_integration/mobile_base.py) also read
  optional ``base_state`` (3,)/(B,3) f32 and ``base_state_mask`` (3,)/(B,3) bool (absent = fixed arm),
  ``base_contract`` (the client's ``mobile_base.CONTRACTS`` name; required with base_state by a v2
  checkpoint, and a mismatch is refused), ``base_dt_s`` (the client's control period; required by a v2
  checkpoint, refused unless equal to its ``dt_s``) and, with RTC, ``base_action_prefix`` (action_horizon, 3) /
  (B, action_horizon, 3) beside ``action_prefix``: the pending chunk's unnormalized base actions, guided
  in model slots 10-12 with the same weights. They also return ``base_actions`` (action_horizon, 3) in
  the contract's units (v2: body-frame velocity [m/s, m/s, rad/s]; v1: [forward m, left m, yaw rad]
  relative to the current base pose), zero for a fixed-arm row. Other checkpoints read and return none
  of these.
  ``arm_action`` (the client's ``arm_action.CONTRACTS`` name; absent = ``per_step_joint_delta``) must equal
  the checkpoint's ``Pi05InputConfig.arm_action``: a query-anchored chunk means other targets
  (pi05_integration/arm_action.py). The joint + BASE TCP form uses a 20-wide proprio and returns
  (action_horizon, 20) chunks.

Run from this repository with `.venv/pi05-gpu/bin/python` or
`.venv/pi05-tpu/bin/python`, created by `scripts/setup/pi05_env.sh`. The simulator and
evaluation client remain in the main MolmoSpaces environment; only this policy server
runs in the Pi environment.

Key correctness note: openpi's LiberoOutputs slices actions[..., :7], which would DROP the
Franka gripper (Franka = 7 arm joints + gripper = 8 valid dims, gripper at index 7). We retain the
full normalized model output, select each embodiment's valid left-packed prefix, and unnormalize
only that prefix. The server then returns a zero-filled width-10 wire tensor.
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

_BOOT_START = time.perf_counter()
print("[pi05.server] importing OpenPI runtime", flush=True)
ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "third_party" / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages" / "openpi-client" / "src"))

import msgpack_numpy  # noqa: E402
import numpy as np  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402
import websockets  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared import download as _download  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training.embodiment_conditioning import make_embodiment_descriptor  # noqa: E402

from pi05_integration.action_boundary import unnormalize_base_actions  # noqa: E402
from pi05_integration.action_boundary import unnormalize_left_packed_actions  # noqa: E402
from pi05_integration.checkpoint_assets import resolve_checkpoint_asset_id  # noqa: E402
from pi05_integration import embodiment_bundle as embodiment_bundle_module  # noqa: E402
from pi05_integration.embodiment_bundle import (  # noqa: E402
    resolve_checkpoint_bundle,
    resolve_transfer_bundle,
)
from pi05_integration.dynamic_batching import DynamicInferenceBatcher  # noqa: E402
from pi05_integration.normalization_override import (  # noqa: E402
    average_per_embodiment_stats,
    require_effective_normalization_stats,
    select_identical_per_embodiment_stats,
)
from pi05_integration.prompt import canonicalize_molmospaces_prompt  # noqa: E402
from pi05_integration.runtime import log_runtime  # noqa: E402

print(
    f"[pi05.server] imported OpenPI runtime in {time.perf_counter() - _BOOT_START:.2f}s",
    flush=True,
)

# Some OpenPI dependencies install a root handler during import. ``force=True`` ensures our INFO
# phase markers remain visible even in that case; missing markers previously made a slow restore
# look like a silent checkpoint-loader hang.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
log = logging.getLogger("pi05_server")


def _expected_bundle_schema(conditioning, prompt_conditioning, extractor):
    """Select the checkpoint bundle schema for one mutually exclusive route."""

    return (
        extractor.processor_schema
        if extractor is not None
        else "t2_robotokens_plus_mjcf_sidecars_v1"
        if conditioning.uses_robotokens
        else getattr(
            prompt_conditioning,
            "expected_schema",
            "urma4_mjcf_controls_global_v1",
        )
    )


def _policy_prompt_text(prompt, prompt_conditioning) -> str:
    """Apply the legacy lowercase rule unless a prompt owns exact bundle text."""

    text = str(prompt)
    return (
        text
        if prompt_conditioning.requires_verbatim_conditioned_prompt
        else text.lower()
    )


def _resolve_left_packed_action_mask(obs, *, batch_size, wire_dim, default_valid_dim, batched):
    valid_dims = np.asarray(obs.get("embodiment_action_dim", default_valid_dim))
    if valid_dims.ndim == 0:
        valid_dims = np.full(batch_size, int(valid_dims))
    if valid_dims.shape != (batch_size,):
        raise ValueError("embodiment_action_dim must be scalar or have shape [B]")
    mask = np.arange(wire_dim)[None, :] < valid_dims[:, None]
    if "embodiment_action_mask" in obs:
        explicit = np.asarray(obs["embodiment_action_mask"], dtype=bool)
        if not batched:
            explicit = explicit[None]
        if explicit.shape != (batch_size, wire_dim):
            raise ValueError("embodiment_action_mask must have shape [B,D] or [D]")
        if not np.array_equal(explicit, mask):
            raise ValueError("embodiment_action_mask must be left-packed and match embodiment_action_dim")
    return mask


def _default_action_valid_dim(spread, action_horizon: int) -> int:
    """The valid action prefix of a request without a mask: up to the last dimension whose statistics vary at any
    chunk position. A per-chunk-position ``(H, D)`` table must hold one row per position of the model."""
    spread = np.asarray(spread)
    if spread.ndim == 2 and spread.shape[0] != action_horizon:
        raise ValueError(f"per-position action statistics have {spread.shape[0]} rows for a {action_horizon}-step model")
    spread = spread.reshape(-1, spread.shape[-1]).max(axis=0)
    nonconstant = np.flatnonzero(spread > 1e-8)
    return int(nonconstant[-1] + 1) if len(nonconstant) else len(spread)


class Pi05Server:
    def __init__(
        self,
        config_name,
        checkpoint_dir,
        *,
        num_steps=10,
        max_guidance_weight=5.0,
        velocity_rescale=True,
        normalization_mode="checkpoint",
        embodiment_description_bundle=None,
        embodiment_description_sha256=None,
        normalization_embodiment_ids=(),
        normalization_group_name="",
    ):
        phase_start = time.perf_counter()
        log.info("Resolving pi05 config: %s", config_name)
        cfg = _config.get_config(config_name)
        log.info("Resolved pi05 config in %.2fs", time.perf_counter() - phase_start)
        log.info("Restoring pi05 policy: config=%s ckpt=%s", config_name, checkpoint_dir)
        ckpt = _download.maybe_download(str(checkpoint_dir))
        checkpoint_assets = Path(str(ckpt)) / "assets"
        preferred_asset_id = cfg.data.assets.asset_id
        checkpoint_asset_id = resolve_checkpoint_asset_id(
            checkpoint_assets, preferred_asset_id
        )
        if checkpoint_asset_id != preferred_asset_id:
            log.warning(
                "Using checkpoint-saved norm asset %s instead of registered alias %s",
                checkpoint_asset_id,
                preferred_asset_id,
            )
        cfg = dataclasses.replace(
            cfg,
            data=dataclasses.replace(
                cfg.data,
                assets=dataclasses.replace(
                    cfg.data.assets,
                    assets_dir=str(checkpoint_assets),
                    asset_id=checkpoint_asset_id,
                ),
            ),
        )
        description_route = (
            "backbone_conditioning"
            if (
                cfg.model.backbone_conditioning.uses_embodiment_description
                or cfg.model.backbone_conditioning.uses_robotokens
            )
            else "action_expert_conditioning"
        )
        conditioning = getattr(cfg.model, description_route)
        self.embodiment_description_provenance = {"mode": "none"}
        prompt_conditioning = cfg.data.input_config.prompt
        uses_prompt_bundle = prompt_conditioning.uses_description_bundle
        if uses_prompt_bundle and (
            conditioning.uses_embodiment_description or conditioning.uses_robotokens
        ):
            raise ValueError(
                "one policy cannot independently configure both URMA4 bundle text and "
                "structured embodiment bundles"
            )
        if (
            conditioning.uses_embodiment_description
            or conditioning.uses_robotokens
            or uses_prompt_bundle
        ):
            extractor = conditioning.extractor if conditioning.uses_embodiment_description else None
            expected_bundle_sha256 = (
                extractor.bundle_sha256
                if extractor is not None
                else conditioning.bundle_sha256
                if conditioning.uses_robotokens
                else prompt_conditioning.bundle_sha256
            )
            expected_schema = _expected_bundle_schema(
                conditioning, prompt_conditioning, extractor
            )
            bundle_dir, bundle_sha256 = resolve_checkpoint_bundle(
                checkpoint_assets
                / checkpoint_asset_id
                / "embodiment_descriptions",
                expected_bundle_sha256,
                expected_schema=expected_schema,
            )
            checkpoint_bundle_dir = bundle_dir
            checkpoint_bundle_sha256 = bundle_sha256
            if (embodiment_description_bundle is None) != (
                embodiment_description_sha256 is None
            ):
                raise ValueError(
                    "embodiment description bundle and sha256 must be provided together"
                )
            if embodiment_description_bundle is not None:
                bundle_dir, bundle_sha256 = resolve_transfer_bundle(
                    checkpoint_bundle_dir,
                    checkpoint_bundle_sha256,
                    embodiment_description_bundle,
                    embodiment_description_sha256,
                )
                self.embodiment_description_provenance = {
                    "mode": "explicit_transfer_bundle",
                    "path": bundle_dir,
                    "npz_sha256": bundle_sha256,
                    "checkpoint_bundle_path": checkpoint_bundle_dir,
                    "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
                }
                log.info(
                    "Using explicit transfer embodiment bundle: %s sha256=%s",
                    bundle_dir,
                    bundle_sha256,
                )
            else:
                self.embodiment_description_provenance = {
                    "mode": "checkpoint",
                    "path": checkpoint_bundle_dir,
                    "npz_sha256": checkpoint_bundle_sha256,
                }
            if uses_prompt_bundle:
                cfg = dataclasses.replace(
                    cfg,
                    data=dataclasses.replace(
                        cfg.data,
                        input_config=dataclasses.replace(
                            cfg.data.input_config,
                            prompt=prompt_conditioning.resolved(
                                bundle_dir, bundle_sha256
                            ),
                        ),
                    ),
                )
            else:
                resolved_conditioning = (
                    dataclasses.replace(
                        conditioning,
                        bundle_dir=bundle_dir,
                        bundle_sha256=bundle_sha256,
                        build_manifest=None,
                        build_manifest_sha256=None,
                    )
                    if conditioning.uses_robotokens
                    else dataclasses.replace(
                        conditioning,
                        extractor=dataclasses.replace(
                            extractor,
                            bundle_dir=bundle_dir,
                            bundle_sha256=bundle_sha256,
                            build_manifest=None,
                            build_manifest_sha256=None,
                            **(
                                {
                                    "dynamic_stats_path": str(
                                        Path(checkpoint_bundle_dir)
                                        / embodiment_bundle_module.DYNAMIC_NORMALIZATION
                                    )
                                }
                                if type(extractor).__name__
                                == "Urma4DynamicEmbodimentExtractor"
                                else {}
                            ),
                        ),
                    )
                )
                cfg = dataclasses.replace(
                    cfg,
                    model=dataclasses.replace(
                        cfg.model, **{description_route: resolved_conditioning}
                    ),
                )
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        norm_stats = _checkpoints.load_norm_stats(
            checkpoint_assets, checkpoint_asset_id
        )
        if normalization_mode == "checkpoint_per_embodiment_mean":
            selected_normalization_ids = tuple(
                int(value) for value in normalization_embodiment_ids
            ) or data_config.input_config.normalization.embodiment_ids
            effective_norm_stats, self.normalization_provenance = (
                average_per_embodiment_stats(
                    norm_stats, selected_normalization_ids
                )
            )
        elif normalization_mode == "checkpoint_normalization_group":
            effective_norm_stats, self.normalization_provenance = (
                select_identical_per_embodiment_stats(
                    norm_stats,
                    normalization_embodiment_ids,
                    group_name=normalization_group_name,
                )
            )
        elif normalization_mode == "checkpoint":
            effective_norm_stats = norm_stats
            self.normalization_provenance = {"mode": "checkpoint"}
        else:
            raise ValueError(f"unknown normalization mode {normalization_mode!r}")
        _sk = {"num_steps": int(num_steps)}
        log.info("flow sampler num_steps=%s", _sk["num_steps"])
        restore_start = time.perf_counter()
        self.policy = _policy_config.create_trained_policy(
            # Policy construction owns checkpoint-local normalization. Passing the
            # already-loaded table here would be an evaluation-time override and is
            # intentionally rejected by immutable normalization bundles.
            cfg, ckpt, sample_kwargs=_sk, norm_stats=None
        )
        log.info("Restored pi05 policy in %.2fs", time.perf_counter() - restore_start)
        self.action_horizon = cfg.model.action_horizon
        self.model_action_dim = cfg.model.action_dim
        self.max_guidance_weight = float(max_guidance_weight)
        self.velocity_rescale = bool(velocity_rescale)

        # Keep the model output normalized here. The request-specific valid action prefix is
        # selected before unnormalization in infer(), so padded coordinates never require stats.
        assets_start = time.perf_counter()
        self.prompt_conditioning = data_config.input_config.prompt
        self.prompt_descriptor_builder = (
            self.prompt_conditioning.create_descriptor_builder(
                embodiment_bundle_module
            )
            if self.prompt_conditioning.uses_description_bundle
            else make_embodiment_descriptor
        )
        self.state_serialization = data_config.input_config.state_serialization
        self.response_history_conditioning = data_config.input_config.response_history
        self.embodiment_conditioning = getattr(data_config, "embodiment_conditioning", "none")
        self.action_expert_conditioning = type(
            cfg.model.action_expert_conditioning
        ).__name__
        extractor = getattr(cfg.model.action_expert_conditioning, "extractor", None)
        self.action_expert_extractor = (
            type(extractor).__name__ if extractor is not None else "none"
        )
        description_extractor = getattr(
            getattr(cfg.model, description_route), "extractor", None
        )
        self.description_extractor = (
            type(description_extractor).__name__
            if description_extractor is not None
            else "none"
        )
        self.dynamic_urma_mode = {
            "Urma4DynamicEmbodimentExtractor": "urma4",
            "Urma5EmbodimentExtractor": "urma5",
            "Urma6EmbodimentExtractor": "urma5",
            "Urma51EmbodimentExtractor": "urma51",
        }.get(self.description_extractor)
        self.dynamic_urma_enabled = self.dynamic_urma_mode is not None
        self.embodiment_normalization = bool(
            getattr(data_config, "embodiment_normalization", False)
            and normalization_mode == "checkpoint"
        )
        self.embodiment_state_ids = tuple(
            int(value) for value in getattr(data_config, "embodiment_state_ids", ())
        )
        self.embodiment_normalization_ids = tuple(
            int(value)
            for value in getattr(data_config, "embodiment_normalization_ids", ())
        )
        self.norm_stats = require_effective_normalization_stats(effective_norm_stats)
        self.action_norm_stats = self.norm_stats["actions"]
        self.mobile_base = data_config.input_config.mobile_base
        self.arm_action = data_config.input_config.arm_action.contract
        self.action_time = data_config.input_config.action_time
        self.image_geometry = getattr(data_config.input_config, "image_geometry", None)
        self.gripper_value = getattr(data_config.input_config, "gripper_value", None)
        self.joint_frame = getattr(data_config.input_config, "joint_frame", None)
        if getattr(self.joint_frame, "enabled", False) and self.dynamic_urma_enabled:
            raise ValueError("the canonical joint frame does not map dynamic URMA fields")
        if self.mobile_base.enabled and "base_actions" not in self.norm_stats:
            raise ValueError("mobile-base checkpoint normalization has no base_actions statistics")
        self.use_quantile_norm = data_config.use_quantile_norm
        bundle_path = checkpoint_assets / checkpoint_asset_id / "normalization_bundle.json"
        # Runs trained by calling openpi's train.py directly (e.g. the 2026-10-10 TPU arms) save
        # only norm_stats.json; create_trained_policy then loads those checkpoint-local stats, and
        # the config's own input contract names the normalization they were trained with.
        bundle = (
            json.loads(bundle_path.read_text())
            if bundle_path.exists()
            else {"normalization": data_config.input_config.to_record()["normalization"]}
        )
        expected_history_stats_sha256 = getattr(
            self.response_history_conditioning, "stats_sha256", None
        )
        if (
            self.response_history_conditioning.enabled
            and expected_history_stats_sha256 is not None
            and bundle.get("stats_sha256") != expected_history_stats_sha256
        ):
            raise ValueError(
                "response-history checkpoint stats hash does not match the configured "
                f"frozen asset: {bundle.get('stats_sha256')} != "
                f"{expected_history_stats_sha256}"
            )
        if (
            normalization_mode
            in (
                "checkpoint_per_embodiment_mean",
                "checkpoint_normalization_group",
            )
            or bundle["normalization"]["type"] == "GlobalQuantileNorm"
        ):
            self.policy._input_transform = _transforms.compose([
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                # Inputs carry state only; a config may normalize it in a mode of its own.
                _transforms.Normalize(
                    effective_norm_stats,
                    use_quantiles=(
                        normalization.state_quantiles
                        if (normalization := data_config.input_config.normalization).state_quantiles
                        != normalization.use_quantiles
                        else data_config.use_quantile_norm
                    ),
                ),
                *(
                    [self.response_history_conditioning.create_normalizer(
                        _transforms, effective_norm_stats
                    )]
                    if self.response_history_conditioning.enabled
                    else []
                ),
                *(
                    [self.mobile_base.create_normalizer(
                        _transforms,
                        effective_norm_stats,
                        use_quantiles=data_config.use_quantile_norm,
                        state_quantiles=data_config.input_config.normalization.state_quantiles,
                    )]
                    if self.mobile_base.enabled
                    else []
                ),
                *data_config.model_transforms.inputs,
            ])
        spread = (
            self.action_norm_stats.q99 - self.action_norm_stats.q01
            if self.use_quantile_norm
            else self.action_norm_stats.std
        )
        self.default_action_valid_dim = _default_action_valid_dim(spread, self.action_horizon)
        self.policy._output_transform = _transforms.compose([
            *data_config.model_transforms.outputs,
        ])
        # serialize jax dispatch across websocket handler threads (shared rng in Policy)
        self._lock = threading.Lock()
        log.info(
            "pi05 policy ready in %.2fs (assets=%.2fs, prompt_conditioning=%s, "
            "action_expert_conditioning=%s, action_expert_extractor=%s, "
            "embodiment_normalization=%s, normalization_mode=%s).",
            time.perf_counter() - phase_start,
            time.perf_counter() - assets_start,
            self.embodiment_conditioning,
            self.action_expert_conditioning,
            self.action_expert_extractor,
            self.embodiment_normalization,
            normalization_mode,
        )

    def _mobile_base_enabled(self) -> bool:
        # Focused tests build partial servers without __init__; those keep the fixed-arm contract.
        return bool(getattr(getattr(self, "mobile_base", None), "enabled", False))

    def _make_input(
        self,
        image,
        wrist,
        state,
        prompt,
        embodiment_id=-1,
        dataset_component_id=None,
        embodiment_descriptor="",
        embodiment_variant="",
        state_mask=None,
        response_state_history=None,
        response_state_history_mask=None,
        response_action_history=None,
        response_action_history_mask=None,
        dynamic_urma_values=None,
        dynamic_urma_mask=None,
        dynamic_urma5_values=None,
        dynamic_urma51_values=None,
        dynamic_urma51_analytic_mask=None,
        base_state=None,
        base_state_mask=None,
        action_dt_s=None,
    ):
        # create_trained_policy does NOT apply the data config's RepackTransform, so feed the
        # post-repack "observation/*" keys that LiberoInputs consumes directly.
        prompt = canonicalize_molmospaces_prompt(prompt)
        prompt = self.prompt_conditioning.condition(
            prompt,
            embodiment_id=embodiment_id,
            descriptor=embodiment_descriptor,
            variant=embodiment_variant,
            descriptor_builder=self.prompt_descriptor_builder,
            **({"control_period_s": action_dt_s} if self.prompt_conditioning.uses_control_period else {}),
        )
        result = {
            "observation/image": np.asarray(image),            # (H,W,3) uint8
            "observation/wrist_image": np.asarray(wrist),      # (h,w,3) uint8
            "observation/state": np.asarray(state, np.float32),  # (10,)
            "prompt": _policy_prompt_text(prompt, self.prompt_conditioning),
            "embodiment_id": np.asarray(embodiment_id, np.int32),
        }
        if dataset_component_id is not None:
            result["dataset_component_id"] = np.asarray(dataset_component_id)
        if self.state_serialization.requires_mask:
            if state_mask is None:
                raise ValueError("compact-state checkpoint requires embodiment_state_mask")
            state_mask = np.asarray(state_mask, dtype=bool)
            if state_mask.shape != result["observation/state"].shape:
                raise ValueError("embodiment_state_mask must match proprio shape")
            result["state_mask"] = state_mask
        if self.response_history_conditioning.enabled:
            response = {
                "response_state_history": response_state_history,
                "response_state_history_mask": response_state_history_mask,
                "response_action_history": response_action_history,
                "response_action_history_mask": response_action_history_mask,
            }
            missing = [key for key, value in response.items() if value is None]
            if missing:
                raise ValueError(f"causal response-history request is missing: {missing}")
            result.update(
                {
                    key: np.asarray(value, dtype=bool if key.endswith("_mask") else np.float32)
                    for key, value in response.items()
                }
            )
        if self.dynamic_urma_mode == "urma4":
            from molmo_spaces.dynamic_urma import MASK_KEY, VALUES_KEY, validate_dynamic_fields

            if dynamic_urma_values is None or dynamic_urma_mask is None:
                raise ValueError("URMA4-dynamic request is missing live causal fields")
            values, mask = validate_dynamic_fields(dynamic_urma_values, dynamic_urma_mask)
            result.update({VALUES_KEY: values, MASK_KEY: mask})
        elif self.dynamic_urma_mode == "urma5":
            if dynamic_urma5_values is None:
                raise ValueError("URMA5 request is missing live causal fields")
            result["dynamic_urma5_values"] = np.asarray(
                dynamic_urma5_values, dtype=np.float32
            )
        elif self.dynamic_urma_mode == "urma51":
            if dynamic_urma51_values is None or dynamic_urma51_analytic_mask is None:
                raise ValueError("URMA5.1 request is missing live causal fields")
            result.update(
                dynamic_urma51_values=np.asarray(
                    dynamic_urma51_values, dtype=np.float32
                ),
                dynamic_urma51_analytic_mask=np.asarray(
                    dynamic_urma51_analytic_mask, dtype=bool
                ),
            )
        if self._mobile_base_enabled():
            # Absent base fields mean a fixed arm: zeros with false masks, exactly as in training.
            result["base_state"] = np.zeros(3, np.float32) if base_state is None else np.asarray(base_state, np.float32)
            result["base_state_mask"] = (
                np.zeros(3, bool) if base_state_mask is None else np.asarray(base_state_mask, dtype=bool)
            )
            if result["base_state"].shape != (3,) or result["base_state_mask"].shape != (3,):
                raise ValueError("base_state and base_state_mask must have shape (3,)")
        # Servers assembled without __init__ (focused tests) and checkpoints without the time channel ignore it.
        if getattr(getattr(self, "action_time", None), "enabled", False):
            if action_dt_s is None:
                raise ValueError("a time-conditioned checkpoint needs the client's action_dt_s (control period, s)")
            dt = np.asarray(action_dt_s, np.float32)
            if dt.shape != () or not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"action_dt_s must be one finite positive period in seconds, got {action_dt_s!r}")
            result["action_dt_s"] = dt
        return result

    def _action_stats_for_embodiment(self, embodiment_id):
        if not self.embodiment_normalization:
            return self.action_norm_stats
        embodiment_id = int(embodiment_id)
        if embodiment_id not in self.embodiment_normalization_ids:
            raise ValueError(
                f"embodiment_id {embodiment_id} is outside normalization codebook "
                f"{self.embodiment_normalization_ids}"
            )
        return self.norm_stats[
            _transforms.embodiment_norm_key("actions", embodiment_id)
        ]

    def _check_arm_action(self, obs, batch_size):
        """Refuse a client whose arm action form (``arm_action``; absent = per-step joint delta) differs."""
        from pi05_integration.arm_action import PER_STEP_JOINT_DELTA

        sent = obs.get("arm_action", PER_STEP_JOINT_DELTA)
        sent = [sent] if isinstance(sent, str | bytes | bytearray) else list(sent)
        sent = [value.decode() if isinstance(value, bytes | bytearray) else str(value) for value in sent]
        # Servers assembled without __init__ (focused tests) serve the default form.
        expected = getattr(self, "arm_action", PER_STEP_JOINT_DELTA)
        if len(sent) not in (1, batch_size) or any(value != expected for value in sent):
            raise ValueError(f"client arm_action {sorted(set(sent))} does not match this {expected} checkpoint")

    def _joint_frame_vectors(self, obs, batch_size, action_mask):
        """(sign, zero), each (B, 10), of the client's robots for a canonical-joint-frame checkpoint, else None
        (``pi05_integration/joint_frame.py``). A client in another frame is refused, as is a frame whose arm slots are
        not the client's valid action slots."""
        from pi05_integration.joint_frame import SLOTS, STORED

        expected = getattr(getattr(self, "joint_frame", None), "contract", STORED)
        sent = obs.get("joint_frame")
        if isinstance(sent, bytes | bytearray):
            sent = sent.decode()
        if (sent is not None and sent != expected) or (sent is None and expected != STORED):
            raise ValueError(f"client joint frame {sent!r}; this checkpoint was trained in {expected!r}")
        if expected == STORED:
            return None
        vectors = []
        for key in ("joint_frame_sign", "joint_frame_zero"):
            if key not in obs:
                raise ValueError(f"a {expected} checkpoint needs the client's {key}")
            value = np.asarray(obs[key], np.float32)
            value = value[None] if value.ndim == 1 else value
            if value.shape not in ((1, SLOTS), (batch_size, SLOTS)) or not np.all(np.isfinite(value)):
                raise ValueError(f"{key} must be finite with shape [{SLOTS}] or [B,{SLOTS}], got {value.shape}")
            vectors.append(np.broadcast_to(value, (batch_size, SLOTS)))
        sign, zero = vectors
        if not np.all(np.isin(sign, (-1.0, 0.0, 1.0))) or np.any((sign != 0) & ~action_mask[:, :SLOTS]):
            raise ValueError("joint_frame_sign must be +-1 on the client's arm slots and 0 elsewhere")
        return sign, zero

    def _check_base_contract(self, obs, batch_size):
        """Refuse a client whose base semantics (``base_contract``) differ from this checkpoint's."""
        contract = self.mobile_base.contract
        sent = obs.get("base_contract")
        if sent is None:
            if contract != "mobile_base_slots_v1":
                raise ValueError(f"a {contract} checkpoint needs the client's base_contract with base_state")
            return
        sent = [sent] if isinstance(sent, str | bytes | bytearray) else list(sent)
        sent = [value.decode() if isinstance(value, bytes | bytearray) else str(value) for value in sent]
        if len(sent) not in (1, batch_size) or any(value != contract for value in sent):
            raise ValueError(f"client base_contract {sorted(set(sent))} does not match this {contract} checkpoint")
        dt_s = getattr(self.mobile_base, "dt_s", None)
        if dt_s is not None:
            # A base velocity means another displacement at another control period.
            if obs.get("base_dt_s") is None:
                raise ValueError(f"a {contract} checkpoint needs the client's base_dt_s with base_state")
            sent_dt = np.asarray(obs["base_dt_s"], np.float64).reshape(-1)
            if sent_dt.size not in (1, batch_size) or not np.allclose(sent_dt, dt_s, rtol=1e-5, atol=0.0):
                raise ValueError(f"client base_dt_s {sorted(set(sent_dt.tolist()))} does not match this checkpoint's dt_s={dt_s}")

    def _add_base_prefix(self, obs, prefix, model_action_mask, inputs, batched):
        """Write the normalized ``base_action_prefix`` into the base slots of the normalized RTC prefix.

        Returns ``(prefix, guided_slots)``: ``guided_slots`` is the arm's model action mask plus the base
        slots of every mobile row. A fixed-arm row's base slots stay zero and unguided; a mobile row
        without ``base_action_prefix`` is refused (its base would silently go unguided)."""
        from pi05_integration.mobile_base import BASE_DIM, BASE_SLOTS

        base_mask = np.stack([
            np.asarray(item.get("base_state_mask", np.zeros(BASE_DIM, bool)), bool) for item in inputs
        ])
        mobile_rows = base_mask.all(axis=-1)
        guided = np.asarray(model_action_mask, bool).copy()
        if not mobile_rows.any():
            return prefix, guided
        if "base_action_prefix" not in obs:
            raise ValueError("RTC with a mobile base needs base_action_prefix beside action_prefix")
        base_prefix = np.asarray(obs["base_action_prefix"], np.float32)
        if not batched:
            base_prefix = base_prefix[None]
        if base_prefix.shape != (*prefix.shape[:2], BASE_DIM):
            raise ValueError(
                f"base_action_prefix must have shape [B,{self.action_horizon},{BASE_DIM}], got {base_prefix.shape}"
            )
        stats = self.norm_stats["base_actions"]
        if self.use_quantile_norm:
            q01, q99 = np.asarray(stats.q01, np.float32), np.asarray(stats.q99, np.float32)
            normalized = (base_prefix - q01) / (q99 - q01 + 1e-6) * 2 - 1
        else:
            normalized = (base_prefix - np.asarray(stats.mean, np.float32)) / (
                np.asarray(stats.std, np.float32) + 1e-6
            )
        prefix = np.asarray(prefix, np.float32).copy()
        slots = slice(BASE_SLOTS[0], BASE_SLOTS[-1] + 1)
        prefix[..., slots] = np.where(mobile_rows[:, None, None], normalized, 0.0)
        guided[:, slots] = mobile_rows[:, None]
        return prefix, guided

    def _normalize_prefix(self, action_prefix, embodiment_ids):
        prefix = np.asarray(action_prefix, np.float32)
        data_dim = prefix.shape[-1]
        normalized = np.empty_like(prefix)
        for index, embodiment_id in enumerate(embodiment_ids):
            stats = self._action_stats_for_embodiment(embodiment_id)
            # [..., :n]: a per-chunk-position (H, D) table normalizes each prefix step with its own row.
            if self.use_quantile_norm:
                normalized[index] = (
                    (prefix[index] - stats.q01[..., :data_dim])
                    / (stats.q99[..., :data_dim] - stats.q01[..., :data_dim] + 1e-6)
                    * 2
                    - 1
                )
            else:
                normalized[index] = (
                    prefix[index] - stats.mean[..., :data_dim]
                ) / (stats.std[..., :data_dim] + 1e-6)
        if data_dim < self.model_action_dim:
            normalized = np.pad(normalized, (*([(0, 0)] * (normalized.ndim - 1)), (0, self.model_action_dim - data_dim)))
        return normalized[..., : self.model_action_dim]

    def infer(self, obs):
        # The client's frames must be in the checkpoint's geometry (pi05_integration/image_geometry.py, review T5): a
        # client that names one must match, and a letterbox checkpoint refuses a client that names none.
        expected = getattr(getattr(self, "image_geometry", None), "contract", "stretch")
        sent = obs.get("image_geometry")
        if (sent is not None and sent != expected) or (sent is None and expected != "stretch"):
            raise ValueError(f"client image geometry {sent!r}; this checkpoint was trained on {expected!r}")
        # The gripper value likewise (pi05_integration/gripper_value.py, review T8): a closedness checkpoint refuses a
        # client that sends another gripper value, and a client that names closedness needs a closedness checkpoint.
        expected = getattr(getattr(self, "gripper_value", None), "contract", "stored")
        sent = obs.get("gripper_value")
        if (sent is not None and sent != expected) or (sent is None and expected != "stored"):
            raise ValueError(f"client gripper value {sent!r}; this checkpoint was trained on {expected!r}")
        img_p = np.asarray(obs["image_primary"])
        # Ablation: blank the main (base/exo) camera so the policy runs on wrist-only input.
        if os.environ.get("PI05_MASK_MAIN_IMAGE") == "1":
            img_p = np.zeros_like(img_p)
        batched = img_p.ndim == 4
        langs = obs["language"]
        if not isinstance(langs, list | tuple):
            langs = [langs]
        langs = [value.decode() if isinstance(value, bytes | bytearray) else value for value in langs]
        img_w = np.asarray(obs["image_wrist"])
        proprio = np.asarray(obs["proprio"], np.float32)
        if not batched:
            img_p = img_p[None]
            img_w = img_w[None]
            proprio = proprio.reshape(1, -1)
        B = img_p.shape[0]
        def batch_values(key, default):
            value = obs.get(key, default)
            array = np.asarray(value)
            if array.ndim == 0:
                return [array.item()] * B
            return list(array)

        wire_action_mask = _resolve_left_packed_action_mask(
            obs,
            batch_size=B,
            wire_dim=proprio.shape[-1],
            default_valid_dim=self.default_action_valid_dim,
            batched=batched,
        )
        # The canonical joint frame (review T2/item 4): the client's joints in, the predicted arm actions back out.
        joint_frame = self._joint_frame_vectors(obs, B, wire_action_mask)
        if joint_frame is not None:
            from pi05_integration.joint_frame import apply_action as joint_frame_action
            from pi05_integration.joint_frame import apply_state as joint_frame_state

            proprio = joint_frame_state(proprio, *joint_frame)
        model_action_mask = np.pad(
            wire_action_mask,
            ((0, 0), (0, max(0, self.model_action_dim - wire_action_mask.shape[1]))),
        )[:, : self.model_action_dim]

        embodiment_ids = batch_values("embodiment_id", -1)
        action_dts = batch_values("action_dt_s", None)
        dataset_component_ids = batch_values("dataset_component_id", None)
        dataset_component_ids = [
            value.decode() if isinstance(value, bytes | bytearray) else value
            for value in dataset_component_ids
        ]
        descriptors = batch_values("embodiment_descriptor", "")
        descriptors = [
            value.decode() if isinstance(value, bytes | bytearray) else str(value) for value in descriptors
        ]
        variants = batch_values("embodiment_variant", "")
        variants = [
            value.decode() if isinstance(value, bytes | bytearray) else str(value) for value in variants
        ]
        if "embodiment_state_mask" in obs:
            state_masks = np.asarray(obs["embodiment_state_mask"], dtype=bool)
            if not batched:
                state_masks = state_masks[None]
            if state_masks.shape != proprio.shape:
                raise ValueError("embodiment_state_mask must match proprio shape")
        else:
            state_masks = wire_action_mask
        self._check_arm_action(obs, B)
        base_inputs = [{}] * B
        if self._mobile_base_enabled() and "base_state" in obs:
            self._check_base_contract(obs, B)
            base_state = np.asarray(obs["base_state"], np.float32).reshape(B, 3)
            base_state_mask = np.asarray(
                obs.get("base_state_mask", np.ones_like(base_state, bool)), dtype=bool
            ).reshape(B, 3)
            base_inputs = [
                {"base_state": base_state[i], "base_state_mask": base_state_mask[i]} for i in range(B)
            ]
        inputs = [
            self._make_input(
                img_p[i], img_w[i], proprio[i], langs[i % len(langs)],
                embodiment_ids[i % len(embodiment_ids)],
                dataset_component_ids[i % len(dataset_component_ids)],
                descriptors[i % len(descriptors)],
                variants[i % len(variants)],
                state_mask=state_masks[i],
                **(
                    {
                        key: np.asarray(obs[key])[i] if batched else np.asarray(obs[key])
                        for key in (
                            "response_state_history",
                            "response_state_history_mask",
                            "response_action_history",
                            "response_action_history_mask",
                        )
                    }
                    if self.response_history_conditioning.enabled
                    else {}
                ),
                **(
                    {
                        key: np.asarray(obs[key])[i] if batched else np.asarray(obs[key])
                        for key in {
                            "urma4": ("dynamic_urma_values", "dynamic_urma_mask"),
                            "urma5": ("dynamic_urma5_values",),
                            "urma51": (
                                "dynamic_urma51_values",
                                "dynamic_urma51_analytic_mask",
                            ),
                        }[self.dynamic_urma_mode]
                    }
                    if self.dynamic_urma_enabled
                    else {}
                ),
                **base_inputs[i],
                action_dt_s=action_dts[i % len(action_dts)],
            )
            for i in range(B)
        ]

        sample_seeds = batch_values("sample_seed", None)
        noise = None
        if all(seed is not None for seed in sample_seeds):
            noise = np.stack([
                np.random.RandomState(int(seed)).randn(self.action_horizon, self.model_action_dim).astype(np.float32)
                for seed in sample_seeds
            ])

        sample_kwargs = {}
        if "action_prefix" in obs:
            prefix = np.asarray(obs["action_prefix"], np.float32)
            if not batched:
                prefix = prefix[None]
            if prefix.shape[:2] != (B, self.action_horizon):
                raise ValueError(
                    f"action_prefix must have shape [B,{self.action_horizon},D], got {prefix.shape}"
                )
            if joint_frame is not None:
                prefix = joint_frame_action(prefix, joint_frame[0][:, None, :])
            prefix = self._normalize_prefix(prefix, embodiment_ids)

            if obs.get("action_prefix_weights") is None:
                raise ValueError("action_prefix_weights is required with action_prefix")
            weights = np.asarray(obs["action_prefix_weights"], np.float32)
            if not batched and weights.ndim in (1, 2):
                weights = weights[None]
            if weights.ndim == 2:  # [B,H] -> [B,H,1]
                weights = weights[..., None]
            weights = np.broadcast_to(weights, prefix.shape).copy()
            guided_slots = model_action_mask
            if self._mobile_base_enabled():
                # The base slots are guided like the arm: the pending chunk's base actions, normalized
                # with the base table into slots 10-12, under the same per-timestep weights.
                prefix, guided_slots = self._add_base_prefix(obs, prefix, model_action_mask, inputs, batched)

            sample_kwargs.update(
                action_prefix=prefix,
                action_prefix_weights=weights * guided_slots[:, None, :],
                max_guidance_weight=self.max_guidance_weight,
                rtc_velocity_rescale=self.velocity_rescale,
            )

        with self._lock:
            out = self.policy.infer_batch(inputs, noise=noise, sample_kwargs=sample_kwargs)
        # Sampling remains unconstrained over the full model action space. At the policy boundary,
        # retain the normalized valid prefix first and unnormalize only those physical values.
        actions = unnormalize_left_packed_actions(
            np.asarray(out["actions"], np.float32),
            wire_action_mask,
            embodiment_ids,
            stats_for_embodiment=self._action_stats_for_embodiment,
            use_quantiles=self.use_quantile_norm,
        )
        if joint_frame is not None:
            actions = joint_frame_action(actions, joint_frame[0][:, None, :])
        if not self._mobile_base_enabled():
            return {"actions": actions if batched else actions[0]}
        base_actions = unnormalize_base_actions(
            np.asarray(out["actions"], np.float32),
            np.stack([np.asarray(item["base_state_mask"], bool) for item in inputs]),
            self.norm_stats["base_actions"],
            use_quantiles=self.use_quantile_norm,
        )
        return {
            "actions": actions if batched else actions[0],
            "base_actions": base_actions if batched else base_actions[0],
        }


async def _serve(server, host, port, *, inference_batch_size=1, inference_batch_wait_ms=2.0):
    batcher = DynamicInferenceBatcher(
        server,
        max_batch_size=inference_batch_size,
        wait_ms=inference_batch_wait_ms,
        logger=log,
    )
    batcher_task = asyncio.create_task(batcher.run())

    async def handler(ws):
        async for raw in ws:
            try:
                obs = msgpack_numpy.unpackb(raw, raw=False)
                out = await batcher.infer(obs)
                await ws.send(msgpack_numpy.packb(out))
            except Exception as e:  # keep the connection alive across episodes
                log.exception("inference error: %s", e)
                await ws.send(msgpack_numpy.packb({"error": str(e)}))

    log.info(
        "pi05 server listening on ws://%s:%d (dynamic_batch_size=%d wait_ms=%.3f)",
        host,
        port,
        inference_batch_size,
        inference_batch_wait_ms,
    )
    try:
        async with websockets.serve(handler, host, port, max_size=None):
            await asyncio.Future()
    finally:
        batcher_task.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_fr_ur5_rlds")
    ap.add_argument("--checkpoint", required=True, help="checkpoint STEP dir (…/<exp>/<step>)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--max-guidance-weight", type=float, default=5.0)
    ap.add_argument("--no-velocity-rescale", action="store_true")
    ap.add_argument(
        "--normalization-mode",
        choices=(
            "checkpoint",
            "checkpoint_per_embodiment_mean",
            "checkpoint_normalization_group",
        ),
        default="checkpoint",
    )
    ap.add_argument(
        "--normalization-embodiment-ids",
        default="",
        help=(
            "Comma-separated checkpoint normalization IDs used by "
            "checkpoint_per_embodiment_mean or checkpoint_normalization_group; "
            "empty selects the complete codebook only for the mean mode."
        ),
    )
    ap.add_argument(
        "--normalization-group-name",
        default="",
        help="Provenance name required by checkpoint_normalization_group.",
    )
    ap.add_argument("--runtime-info-out", default=None)
    ap.add_argument("--embodiment-description-bundle", default=None)
    ap.add_argument("--embodiment-description-sha256", default=None)
    ap.add_argument("--inference-batch-size", type=int, default=1)
    ap.add_argument("--inference-batch-wait-ms", type=float, default=2.0)
    args = ap.parse_args()
    if args.inference_batch_size < 1:
        ap.error("--inference-batch-size must be positive")
    if args.inference_batch_wait_ms < 0:
        ap.error("--inference-batch-wait-ms must be non-negative")
    runtime_info = log_runtime()
    server = Pi05Server(
        args.config,
        args.checkpoint,
        num_steps=args.num_steps,
        max_guidance_weight=args.max_guidance_weight,
        velocity_rescale=not args.no_velocity_rescale,
        normalization_mode=args.normalization_mode,
        embodiment_description_bundle=args.embodiment_description_bundle,
        embodiment_description_sha256=args.embodiment_description_sha256,
        normalization_embodiment_ids=tuple(
            int(value)
            for value in args.normalization_embodiment_ids.split(",")
            if value.strip()
        ),
        normalization_group_name=args.normalization_group_name,
    )
    runtime_info["normalization"] = server.normalization_provenance
    runtime_info["embodiment_description"] = server.embodiment_description_provenance
    if args.runtime_info_out:
        Path(args.runtime_info_out).write_text(json.dumps(runtime_info, indent=2))
    asyncio.run(
        _serve(
            server,
            args.host,
            args.port,
            inference_batch_size=args.inference_batch_size,
            inference_batch_wait_ms=args.inference_batch_wait_ms,
        )
    )


if __name__ == "__main__":
    main()
