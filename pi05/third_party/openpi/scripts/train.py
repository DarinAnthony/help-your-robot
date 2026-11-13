import concurrent.futures
import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform
import tempfile
import time
from collections.abc import Callable
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
import numpy as np
import optax
from pi05_integration import data_processing as _data_processing
from pi05_integration import input_config as _pi_input_config
from pi05_integration import runtime as _pi_runtime
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.dataset_fingerprint as _dataset_fingerprint
import openpi.training.dataset_mixture as _dataset_mixture
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.train_metrics as _train_metrics
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _resume_verification_payload(
    train_state, *, checkpoint_dir: str, expected_step: int | None = None,
    minimum_step: int | None = None,
) -> dict:
    if (expected_step is None) == (minimum_step is None):
        raise ValueError("specify exactly one expected_step or minimum_step")
    if (expected_step if expected_step is not None else minimum_step) <= 0:
        raise ValueError("resume verification step must be positive")
    restored_step = int(train_state.step)
    if expected_step is not None and restored_step != expected_step:
        raise ValueError(
            f"restored step {restored_step} does not match expected {expected_step}"
        )
    if minimum_step is not None and restored_step < minimum_step:
        raise ValueError(
            f"restored step {restored_step} is below minimum {minimum_step}"
        )
    optimizer_arrays = [
        leaf
        for leaf in jax.tree_util.tree_leaves(train_state.opt_state)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    ]
    if not optimizer_arrays:
        raise ValueError("restored optimizer state contains no arrays")
    jax.block_until_ready(optimizer_arrays)
    optimizer_elements = sum(int(np.prod(array.shape)) for array in optimizer_arrays)
    if optimizer_elements <= 0:
        raise ValueError("restored optimizer state contains no elements")
    return {
        "status": "ok",
        "checkpoint_dir": checkpoint_dir,
        "restored_step": restored_step,
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "optimizer_array_count": len(optimizer_arrays),
        "optimizer_element_count": optimizer_elements,
    }


def _resolve_v3_replay(
    sampling: _data_processing.DeterministicReplayV3,
    *,
    recorded: dict,
    completed_updates: int,
    process_count: int,
    local_batch_size: int,
    sampler_seed: int,
    tf_version: str,
    tfds_version: str,
    allow_topology_restart: bool,
    dataset_changed: bool = False,
) -> tuple[int, int, list[dict]]:
    """Resolve active seed/cursor and persist explicit distributional restarts.

    An accepted dataset change also restarts the stream: the recorded position describes the old
    files, so replaying it over the new ones would cost the replay and reproduce nothing.
    """
    topology_restart = sampling.validate_resume_topology(
        recorded=recorded,
        process_count=process_count,
        local_batch_size=local_batch_size,
        sampler_seed=sampler_seed,
        allow_topology_restart=allow_topology_restart,
    )
    version_mismatches = {
        key: (recorded.get(key), current)
        for key, current in (("tf_version", tf_version), ("tfds_version", tfds_version))
        if recorded.get(key) != current
    }
    if version_mismatches and not allow_topology_restart:
        raise ValueError(
            "DeterministicReplayV3 exact resume requires unchanged TF/TFDS versions: "
            + ", ".join(
                f"{key}={old!r} -> {new!r}"
                for key, (old, new) in version_mismatches.items()
            )
        )
    restarts = list(recorded.get("input_restarts", []))
    if topology_restart or version_mismatches or dataset_changed:
        derived_seed = sampler_seed ^ completed_updates
        restarts.append(
            {
                "completed_updates": completed_updates,
                "reason": (
                    "explicit_topology_or_software_distributional_restart"
                    if topology_restart or version_mismatches
                    else "explicit_dataset_change_distributional_restart"
                ),
                "dataset_changed": dataset_changed,
                "recorded_process_count": recorded.get("input_process_count"),
                "process_count": process_count,
                "recorded_local_batch_size": recorded.get("input_local_batch_size"),
                "local_batch_size": local_batch_size,
                "recorded_sampler_seed": recorded.get("input_sampler_seed"),
                "sampler_seed": sampler_seed,
                "recorded_tf_version": recorded.get("tf_version"),
                "tf_version": tf_version,
                "recorded_tfds_version": recorded.get("tfds_version"),
                "tfds_version": tfds_version,
                "derived_sampler_seed": derived_seed,
            }
        )
        return derived_seed, 0, restarts
    if restarts:
        last = restarts[-1]
        return (
            int(last["derived_sampler_seed"]),
            completed_updates - int(last["completed_updates"]),
            restarts,
        )
    return sampler_seed, completed_updates, restarts



def _physical_action_mask(action_mask: at.Array, semantic_mask: at.Array) -> at.Array:
    """Timestep validity restricted to the physical action coordinates.

    ``action_mask`` is the (b, ah, ad) loss mask. Under the model-coordinate loss it marks every
    model dimension of a valid timestep, zero padding included; ``semantic_mask`` (b, sd) or
    (b, ah, sd), sd <= ad, marks the dimensions that carry real joints and gripper.
    """
    semantic = jnp.asarray(semantic_mask, dtype=bool)
    if semantic.ndim == 2:
        semantic = semantic[:, None, :]
    pad = action_mask.shape[-1] - semantic.shape[-1]
    semantic = jnp.pad(semantic, [(0, 0), (0, 0), (0, pad)])
    return jnp.logical_and(jnp.asarray(action_mask, dtype=bool), semantic)


def _mean_over_val_batches(stacked_val: dict) -> dict:
    """Average validation metrics over the stacked batch axis ONLY.

    Each entry is (val_batches, *metric_shape). Reducing every axis (plain ``jnp.mean``) collapsed
    the per-horizon and per-dimension vectors to one number, so ``val/sample_mae_by_horizon/*`` and
    ``val/sample_mae_by_dimension/*`` were never logged as breakdowns.
    """
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), stacked_val)


def _scalar_metrics(payload: dict) -> dict:
    """The JSON-serializable scalar entries of a W&B payload (tables, HTML and images dropped)."""
    out = {}
    for key, value in payload.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)) or (
            isinstance(value, (np.ndarray, jax.Array)) and np.ndim(value) == 0
        ):
            out[key] = float(value)
    return out


class _LocalMetricsLog:
    """Append every logged scalar to <checkpoint_dir>/metrics.jsonl on the primary process.

    W&B history can be lost (an online run truncated at a job boundary, an offline run never
    synced); this file sits next to the checkpoints it describes and survives either. Remote
    (gs://) checkpoint directories cannot be appended to, so the log is skipped there.
    """

    def __init__(self, checkpoint_dir, *, enabled: bool):
        self._path = None
        if enabled and not str(checkpoint_dir).startswith(("gs://", "/bigstore/", "/gcs/")):
            self._path = pathlib.Path(str(checkpoint_dir)) / "metrics.jsonl"

    def __call__(self, payload: dict, *, step: int) -> None:
        wandb.log(payload, step=step)
        if self._path is None:
            return
        scalars = _scalar_metrics(payload)
        if scalars:
            with self._path.open("a") as f:
                f.write(json.dumps({"step": int(step), "time": time.time(), **scalars}) + "\n")

def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled or jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not str(ckpt_dir).startswith(("gs://", "/bigstore/", "/gcs/")) and not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    # dataclasses.asdict renders every field-less loader (NoOp and PaliGemma alike) as {}, which
    # hides whether a run started from random or pretrained weights, so record the type
    # explicitly. Weights are initialized once, at the fresh start, so that is the only time it
    # is recorded (weight_loader.json next to wandb_id.txt). A resume reads that record back
    # instead of describing the *current* config, which may have changed loaders since: a
    # random-init "_run1" lineage resumed after its config moved to PaliGemma init must not be
    # relabelled as PaliGemma.
    init_record_path = ckpt_dir / "weight_loader.json"
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        # A checkpoint can be durable even if the original process died before W&B finished
        # initializing its run. Keep the checkpoint's stable run ID, while allowing W&B to
        # initialize that ID when the backend does not know it yet.
        wandb.init(
            id=run_id,
            resume="allow",
            project=config.project_name,
            name=config.wandb_name,
        )
        if init_record_path.exists():
            recorded = json.loads(init_record_path.read_text())
            wandb.config.update({"weight_loader": recorded}, allow_val_change=True)
            current = type(getattr(config, "weight_loader", None)).__name__
            if recorded.get("type") != current:
                logging.warning(
                    "Resuming a run initialized with %s; the config now names %s, which a resume does not use.",
                    recorded.get("type"),
                    current,
                )
        else:
            logging.warning("No %s: this run started before its weight init was recorded.", init_record_path)
    else:
        weight_loader_record = {
            "type": type(config.weight_loader).__name__,
            **(dataclasses.asdict(config.weight_loader) if dataclasses.is_dataclass(config.weight_loader) else {}),
        }
        init_record_path.write_text(json.dumps(weight_loader_record))
        wandb_config = dataclasses.asdict(config)
        wandb_config["weight_loader"] = weight_loader_record
        if hasattr(config.data, "input_config"):
            wandb_config["data"]["input_config"] = config.data.input_config.to_record()
        if hasattr(config.model, "action_expert_conditioning"):
            wandb_config["model"]["action_expert_conditioning"] = _pi_input_config.action_expert_conditioning_record(
                config.model.action_expert_conditioning
            )
        if hasattr(config.model, "backbone_conditioning"):
            wandb_config["model"]["backbone_conditioning"] = (
                _pi_input_config.backbone_conditioning_record(
                    config.model.backbone_conditioning
                )
            )
        wandb.init(
            name=config.wandb_name,
            config=wandb_config,
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    # Match Octo's dashboard organization and put every asynchronously logged metric on the
    # completed-step axis rather than W&B's implicit call counter. A step is one microbatch; with
    # gradient accumulation, training/optimizer_updates gives the applied-update count.
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _publish_input_contract(contract_path, input_contract, *, is_primary, synchronize):
    """Publish only after every rank validated the previous lineage contract."""
    synchronize()
    if not is_primary:
        return
    payload = json.dumps(input_contract, indent=2)
    if "://" in str(contract_path):
        # Object-store writes publish complete objects; POSIX writes need rename.
        contract_path.write_text(payload)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(contract_path.parent),
            prefix=".input_contract-", delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(contract_path))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _primary_phase_marker(checkpoint_dir: epath.Path, phase: str) -> epath.Path | None:
    """Return a launch-scoped marker for slow primary-only multihost work."""
    if jax.process_count() == 1:
        return None
    launch_id = os.environ.get("MOLMOSPACES_MULTIHOST_LAUNCH_ID")
    if launch_id is None:
        request_id = os.environ.get("BOAI_TPU_REQUEST_ID")
        attempt = os.environ.get("BOAI_TPU_ATTEMPT")
        if request_id and attempt:
            launch_id = f"{request_id}-attempt-{attempt}"
    if launch_id is None:
        return None
    if not launch_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in launch_id
    ):
        raise ValueError(
            "MOLMOSPACES_MULTIHOST_LAUNCH_ID and BoAI TPU request identities may contain "
            "only letters, digits, '.', '_', and '-'"
        )
    return checkpoint_dir / ".multihost_rendezvous" / launch_id / f"{phase}.complete"


def _wait_for_primary_phase(marker: epath.Path, phase: str) -> None:
    """Wait outside JAX collectives while process zero performs slow metadata work."""
    timeout_seconds = int(
        os.environ.get("MOLMOSPACES_PRIMARY_PHASE_TIMEOUT_SECONDS", "7200")
    )
    if timeout_seconds <= 0:
        raise ValueError("MOLMOSPACES_PRIMARY_PHASE_TIMEOUT_SECONDS must be positive")
    deadline = time.monotonic() + timeout_seconds
    logging.info("Waiting for primary-only %s phase at %s", phase, marker)
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"process zero did not finish {phase} within {timeout_seconds} seconds"
            )
        time.sleep(5)
    logging.info("Primary-only %s phase completed", phase)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh
) -> tuple[training_utils.TrainState, Any, Callable[[], training_utils.TrainState]]:
    """Return the abstract train state, its sharding, and a function that loads the initial state.

    The abstract state is available at once; loading the base weights is deferred to the returned
    function, which a resumed run never calls.
    """
    tx = _optimizer.create_optimizer(
        config.optimizer,
        config.lr_schedule,
        weight_decay_mask=None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            # Strongly typed, as every later and every restored step is, so the train step compiles once.
            step=jnp.zeros((), jnp.int32),
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    def initialize() -> training_utils.TrainState:
        partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # Initialize the train state and mix in the partial params.
        return jax.jit(
            init,
            donate_argnums=(1,),  # donate the partial params buffer.
            in_shardings=replicated_sharding,
            out_shardings=state_sharding,
        )(init_rng, partial_params)

    return train_state_shape, state_sharding, initialize


def _leaf_signatures(tree) -> list[tuple]:
    """What jit keys a compiled executable on, per leaf: shape, dtype, weak type and sharding."""
    return [(leaf.shape, leaf.dtype, leaf.weak_type, leaf.sharding) for leaf in jax.tree.leaves(tree)]


def optimizer_update_applied(opt_state: Any) -> bool | jax.Array:
    """Whether the optimizer step that produced ``opt_state`` actually changed the params.

    With gradient accumulation (optax.MultiSteps) only one call in k applies an update; the other
    k-1 only accumulate gradients and leave the params untouched. MultiSteps resets ``mini_step``
    to 0 on the call that applies, so this reads the optimizer's own counter -- it survives resume
    and does not assume ``state.step`` is aligned with the accumulation window.
    """
    if isinstance(opt_state, optax.MultiStepsState):
        return opt_state.mini_step == 0
    return True


def ema_update(ema_params: Any, new_params: Any, decay: float, applied: bool | jax.Array) -> Any:
    """One EMA step per applied optimizer update; a no-op on accumulate-only microbatches.

    Averaging on every microbatch would fold the unchanged params in k times per update, turning
    a configured decay d into d**k per update (0.999 -> 0.996 at k=4, a ~250- instead of
    ~1000-update horizon) and making the horizon depend on gradient_accumulation_steps.
    """
    return jax.tree.map(
        lambda old, new: jnp.where(applied, decay * old + (1 - decay) * new, old), ema_params, new_params
    )


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    is_ki = getattr(config.model, "enable_langact_training", False)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if is_ki:
            chunked_loss, extras = model.compute_loss(rng, observation, actions, train=True, return_extras=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            extras = {}
        if observation.action_loss_mask is None:
            timestep_mask = jnp.ones_like(chunked_loss)
        else:
            timestep_mask = jnp.any(observation.action_loss_mask, axis=-1).astype(chunked_loss.dtype)
        per_sample_loss = jnp.sum(chunked_loss * timestep_mask, axis=-1) / jnp.maximum(
            jnp.sum(timestep_mask, axis=-1), 1.0
        )
        loss = jnp.sum(chunked_loss * timestep_mask) / jnp.maximum(jnp.sum(timestep_mask), 1.0)
        # action_loss is [b, ah] like chunked_loss; lang_loss is already [b] (one CE per episode).
        ki_info = {}
        if "action_loss" in extras:
            action_loss = extras["action_loss"]
            ki_info["_per_sample_action_loss"] = jnp.sum(action_loss * timestep_mask, axis=-1) / jnp.maximum(
                jnp.sum(timestep_mask, axis=-1), 1.0
            )
        if "lang_loss" in extras:
            ki_info["_per_sample_lang_loss"] = extras["lang_loss"]
            # The language half is one CE per sample, so it is added *after* the masked mean over
            # chunk steps. Folding it into the [b, ah] array first would weight each sample by how
            # many of its chunk steps are valid, which down-weights every chunk that runs past the
            # end of its episode. See Pi0.compute_loss.
            lang_weight = getattr(config.model, "language_loss_weight", 1.0)
            per_sample_loss = per_sample_loss + lang_weight * extras["lang_loss"]
            loss = loss + lang_weight * jnp.mean(extras["lang_loss"])
        return loss, (per_sample_loss, ki_info)

    # state.step counts applied optimizer updates. Under gradient accumulation the k microbatches
    # of one update share that count, so fold in the position within the update too.
    mini_step = state.opt_state.mini_step if isinstance(state.opt_state, optax.MultiStepsState) else 0
    train_rng = jax.random.fold_in(jax.random.fold_in(rng, state.step), mini_step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)

    (loss, (per_sample_loss, ki_info)), grads = grad_fn(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    applied = optimizer_update_applied(new_opt_state)
    # One step == one applied optimizer update, so checkpoints, intervals, the lr schedule and
    # num_train_steps all share one unit. Accumulate-only microbatches leave the count alone.
    new_state = dataclasses.replace(
        state, step=state.step + jnp.asarray(applied, jnp.int32), params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=ema_update(state.ema_params, new_params, state.ema_decay, applied),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    if isinstance(state.opt_state, optax.MultiStepsState):
        # The gradient the optimizer sees: MultiSteps' running mean over this update's
        # microbatches so far, including this one. On the microbatch that applies (the one whose
        # info the loop logs) it is exactly what gets clipped and stepped on, not one
        # microbatch's gradient.
        acc_grads = state.opt_state.acc_grads
        applied_grads = jax.tree.map(lambda g, acc: acc + (g - acc) / (mini_step + 1), grads, acc_grads)
    else:
        applied_grads = grads
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(applied_grads),
        "param_norm": optax.global_norm(kernel_params),
        "_per_sample_loss": per_sample_loss,
        "_embodiment_id": (
            observation.embodiment_id
            if observation.embodiment_id is not None
            else jnp.full((actions.shape[0],), -1, dtype=jnp.int32)
        ),
        "_task_id": (
            observation.task_id
            if observation.task_id is not None
            else jnp.full((actions.shape[0],), -1, dtype=jnp.int32)
        ),
        **ki_info,
    }
    if isinstance(
        config.model.backbone_conditioning,
        _pi_input_config.ExtractedEmbodimentBackboneToken,
    ):
        info.update(
            {
                "urma_encoder_grad_norm": optax.global_norm(
                    grads.filter(nnx_utils.PathRegex(".*embodiment_encoder.*"))
                ),
                "urma_token_projector_grad_norm": optax.global_norm(
                    grads.filter(
                        nnx_utils.PathRegex(".*embodiment_token_(norm|proj).*")
                    )
                ),
            }
        )
    return new_state, info


@at.typecheck
def backbone_conditioning_metrics_step(
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, _ = batch
    token = model.embed_embodiment_backbone_token(observation)
    return {
        "urma_token_norm": jnp.mean(jnp.linalg.norm(token, axis=-1)),
    }


@at.typecheck
def sampled_action_metrics_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Sample a full action chunk and return masked action-space MSE per example."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    predictions = model.sample_actions(rng, observation)
    action_mask = (
        jnp.ones_like(actions, dtype=jnp.bool_)
        if observation.action_loss_mask is None
        else observation.action_loss_mask
    )
    squared_error = jnp.square(predictions - actions)
    absolute_error = jnp.abs(predictions - actions)
    weight = action_mask.astype(squared_error.dtype)
    per_sample = jnp.sum(squared_error * weight, axis=(1, 2)) / jnp.maximum(jnp.sum(weight, axis=(1, 2)), 1.0)
    per_sample_mae = jnp.sum(absolute_error * weight, axis=(1, 2)) / jnp.maximum(jnp.sum(weight, axis=(1, 2)), 1.0)
    return {
        "sample_loss": per_sample,
        "sample_mae": per_sample_mae,
        "predictions": predictions,
        "embodiment_id": (
            observation.embodiment_id
            if observation.embodiment_id is not None
            else jnp.full((actions.shape[0],), -1, dtype=jnp.int32)
        ),
        "task_id": (
            observation.task_id
            if observation.task_id is not None
            else jnp.full((actions.shape[0],), -1, dtype=jnp.int32)
        ),
    }


@at.typecheck
def data_visualization_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[_model.Observation, at.Array]:
    """Return actual train-mode images and predictions sampled from those same inputs."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, _ = batch
    augmentation_rng, sample_rng = jax.random.split(rng)
    observation = _model.preprocess_observation(
        augmentation_rng,
        observation,
        train=True,
        augmentation_mode=getattr(config.model, "augmentation_mode", "pi05_native"),
    )
    return observation, model.sample_actions(sample_rng, observation)


@at.typecheck
def validation_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Compute held-out flow loss and closed-form sample fidelity on one batch."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    loss_rng, sample_rng = jax.random.split(rng)
    is_ki = getattr(config.model, "enable_langact_training", False)
    if is_ki:
        chunked_loss, val_extras = model.compute_loss(loss_rng, observation, actions, train=False, return_extras=True)
    else:
        chunked_loss = model.compute_loss(loss_rng, observation, actions, train=False)
        val_extras = {}
    predictions = model.sample_actions(sample_rng, observation)

    if observation.action_loss_mask is None:
        action_mask = jnp.ones_like(actions, dtype=jnp.bool_)
    else:
        action_mask = observation.action_loss_mask
    timestep_mask = jnp.any(action_mask, axis=-1)
    timestep_weight = timestep_mask.astype(chunked_loss.dtype)
    loss = jnp.sum(chunked_loss * timestep_weight) / jnp.maximum(jnp.sum(timestep_weight), 1.0)
    per_sample_loss = jnp.sum(chunked_loss * timestep_weight, axis=-1) / jnp.maximum(
        jnp.sum(timestep_weight, axis=-1), 1.0
    )
    if "lang_loss" in val_extras:
        # Same reduction as train_step, so val/loss stays comparable with training/loss. Note
        # chunked_loss (and therefore loss_by_horizon below) is the action half only: a
        # per-horizon breakdown of a per-episode language loss would be meaningless anyway.
        val_lang_weight = getattr(config.model, "language_loss_weight", 1.0)
        per_sample_loss = per_sample_loss + val_lang_weight * val_extras["lang_loss"]
        loss = loss + val_lang_weight * jnp.mean(val_extras["lang_loss"])

    absolute_error = jnp.abs(predictions - actions)
    squared_error = jnp.square(predictions - actions)
    dimension_weight = action_mask.astype(absolute_error.dtype)
    sample_mae = jnp.sum(absolute_error * dimension_weight) / jnp.maximum(jnp.sum(dimension_weight), 1.0)
    sample_loss = jnp.sum(squared_error * dimension_weight) / jnp.maximum(jnp.sum(dimension_weight), 1.0)
    per_sample_sample_loss = jnp.sum(squared_error * dimension_weight, axis=(1, 2)) / jnp.maximum(
        jnp.sum(dimension_weight, axis=(1, 2)), 1.0
    )
    per_sample_mae = jnp.sum(absolute_error * dimension_weight, axis=(1, 2)) / jnp.maximum(
        jnp.sum(dimension_weight, axis=(1, 2)), 1.0
    )
    # sample_mae above averages over whatever the loss mask covers; with model-coordinate loss
    # that includes the zero padding dims, which dilutes it. This one averages real joints and
    # gripper only, comparable with runs trained on the physical-coordinate mask.
    physical_info = {}
    if observation.action_semantic_mask is not None:
        physical_weight = _physical_action_mask(action_mask, observation.action_semantic_mask).astype(
            absolute_error.dtype
        )
        physical_info["val/sample_mae_physical"] = jnp.sum(absolute_error * physical_weight) / jnp.maximum(
            jnp.sum(physical_weight), 1.0
        )
        physical_info["_per_sample_mae_physical"] = jnp.sum(
            absolute_error * physical_weight, axis=(1, 2)
        ) / jnp.maximum(jnp.sum(physical_weight, axis=(1, 2)), 1.0)
    loss_by_horizon = jnp.sum(chunked_loss * timestep_weight, axis=0) / jnp.maximum(
        jnp.sum(timestep_weight, axis=0), 1.0
    )
    sample_mae_by_horizon = jnp.sum(absolute_error * dimension_weight, axis=(0, 2)) / jnp.maximum(
        jnp.sum(dimension_weight, axis=(0, 2)), 1.0
    )
    sample_mae_by_dimension = jnp.sum(absolute_error * dimension_weight, axis=(0, 1)) / jnp.maximum(
        jnp.sum(dimension_weight, axis=(0, 1)), 1.0
    )
    val_ki_info = {}
    if "action_loss" in val_extras:
        val_ki_info["_per_sample_action_loss"] = jnp.sum(
            val_extras["action_loss"] * timestep_weight, axis=-1
        ) / jnp.maximum(jnp.sum(timestep_weight, axis=-1), 1.0)
    if "lang_loss" in val_extras:
        val_ki_info["_per_sample_lang_loss"] = val_extras["lang_loss"]
    return {
        "val/loss": loss,
        "val/sample_mae": sample_mae,
        "val/sample_loss": sample_loss,
        "val/loss_by_horizon": loss_by_horizon,
        "val/sample_mae_by_horizon": sample_mae_by_horizon,
        "val/sample_mae_by_dimension": sample_mae_by_dimension,
        **val_ki_info,
        **physical_info,
        "_per_sample_loss": per_sample_loss,
        "_per_sample_sample_loss": per_sample_sample_loss,
        "_per_sample_mae": per_sample_mae,
        "_embodiment_id": (
            observation.embodiment_id
            if observation.embodiment_id is not None
            else jnp.full((actions.shape[0],), -1, dtype=jnp.int32)
        ),
        "_task_id": (
            observation.task_id
            if observation.task_id is not None
            else jnp.full((actions.shape[0],), -1, dtype=jnp.int32)
        ),
    }


def main(
    config: _config.TrainConfig,
    *,
    verify_resume_only: bool = False,
    expected_resume_step: int | None = None,
    minimum_resume_step: int | None = None,
):
    if verify_resume_only and not config.resume:
        raise ValueError("verify_resume_only requires a resume config")
    if verify_resume_only and ((expected_resume_step is None) == (minimum_resume_step is None)):
        raise ValueError("verify_resume_only requires exactly one expected_resume_step or minimum_resume_step")

    if os.environ.get("MOLMOSPACES_JAX_DISTRIBUTED") == "1":
        _pi_runtime.initialize_distributed("gpu")
    init_logging()
    is_primary = jax.process_index() == 0
    logging.info(
        "Running on %s: process=%d/%d local_devices=%d global_devices=%d",
        platform.node(),
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resume_tier, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        keep_every_n_completed_updates=config.keep_interval,
        overwrite=config.overwrite,
        resume=config.resume,
        freeze_filter=config.freeze_filter,
        read_only=verify_resume_only,
        unified=config.unified_checkpoints,
    )
    # By default compiled executables are cached beside the checkpoints, on storage that
    # outlives the machine, so a resume on the same accelerator type and topology skips
    # compilation wherever it runs. It is set only now, because a fresh run's checkpoint directory
    # must not exist before it is opened; JAX fixes the cache directory at its first
    # compilation, so its cache state is reset to take the new one.
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            str(epath.Path("~/.cache/jax").expanduser())
            if verify_resume_only
            else str(epath.Path(config.checkpoint_dir) / _checkpoints.COMPILATION_CACHE_DIRNAME),
        ),
    )
    compilation_cache.reset_cache()
    if (
        not verify_resume_only
        and is_primary
        and config.policy_metadata
        and "resolved_training_experiment" in config.policy_metadata
    ):
        resolved_path = config.checkpoint_dir / "resolved_training_config.json"
        resolved_path.write_text(json.dumps(config.policy_metadata["resolved_training_experiment"], indent=2))
        logging.info("Wrote resolved training config to %s", resolved_path)
    if not verify_resume_only:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and is_primary)

    data_config = config.data.create(config.assets_dirs, config.model)
    if (
        config.gradient_accumulation_steps > 1
        and data_config.rlds_data_dir is not None
        and isinstance(data_config.rlds_input_sampling, _data_processing.DeterministicReplayV3)
    ):
        # V3 resumes by skipping completed_updates * local_batch_size frames, which assumes one
        # batch per update.
        raise ValueError("DeterministicReplayV3 input does not support gradient_accumulation_steps > 1")
    import tensorflow as tf
    import tensorflow_datasets as tfds

    latest_checkpoint_step, restore_tier = (
        _checkpoints.select_restore_point(checkpoint_manager, resume_tier)
        if resuming
        else (None, None)
    )
    if resuming and latest_checkpoint_step is None:
        logging.info("No restorable checkpoint in either tier; starting fresh.")
        resuming = False
    if resuming:
        logging.info("Resuming from %s checkpoint %s", restore_tier, latest_checkpoint_step)
    input_completed_updates = (
        int(latest_checkpoint_step) + 1 if latest_checkpoint_step is not None else 0
    )
    input_local_batch_size = config.batch_size // jax.process_count()
    configured_sampler_seed = (
        config.seed
        if data_config.rlds_sampler_seed is None
        else data_config.rlds_sampler_seed
    )
    active_sampler_seed = configured_sampler_seed
    replay_completed_updates = input_completed_updates
    input_restarts = []
    action_expert_conditioning = getattr(config.model, "action_expert_conditioning", _pi_input_config.TimestepOnly())
    backbone_conditioning = getattr(
        config.model,
        "backbone_conditioning",
        _pi_input_config.NoBackboneConditioning(),
    )
    input_contract = {
        "schema_version": 1,
        "config_name": config.name,
        "model_type": config.model.model_type.value,
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
        "max_token_len": config.model.max_token_len,
        "input_config": data_config.input_config.to_record(),
        "action_expert_conditioning": _pi_input_config.action_expert_conditioning_record(action_expert_conditioning),
        "backbone_conditioning": _pi_input_config.backbone_conditioning_record(backbone_conditioning),
        "behavior_prompt_overlays": [
            overlay.to_record() for overlay in data_config.behavior_prompt_overlays
        ],
        "episode_selector": (
            None
            if data_config.episode_selector is None
            else data_config.episode_selector.to_record()
        ),
        "episode_split": (
            None
            if data_config.episode_split is None
            else data_config.episode_split.to_record()
        ),
        "normalization_selection": (
            None
            if data_config.normalization_bundle is None
            else data_config.normalization_bundle.get("split")
        ),
        "normalization_asset_id": (
            data_config.asset_id
            if data_config.asset_id is not None
            else data_config.repo_id
        ),
        "normalization_stats_sha256": (
            None
            if data_config.normalization_bundle is None
            else data_config.normalization_bundle.get("stats_sha256")
        ),
        "rlds_input_sampling": data_config.rlds_input_sampling.to_record(),
        "input_process_count": jax.process_count(),
        "input_local_batch_size": input_local_batch_size,
        "input_sampler_seed": configured_sampler_seed,
        "input_restarts": input_restarts,
        "tf_version": tf.__version__,
        "tfds_version": tfds.__version__,
        "dataset_fingerprint": (
            _dataset_fingerprint.dataset_fingerprint(data_config)
            if data_config.rlds_data_dir is not None and data_config.datasets
            else None
        ),
        # Accepted dataset changes, each at the completed update it took effect (rlds_input_dataset_restart).
        "dataset_changes": [],
        "compatibility": {
            "embodiment_conditioning": data_config.embodiment_conditioning,
            "embodiment_state_ids": list(data_config.embodiment_state_ids),
            "embodiment_normalization": data_config.embodiment_normalization,
            "embodiment_vocab_size": getattr(config.model, "embodiment_vocab_size", 0),
        },
    }
    contract_path = config.checkpoint_dir / "input_contract.json"
    if resuming:
        # A resume must continue the lineage's input distribution. Checkpoints written before
        # the contract was recorded were all trained with the legacy unsharded traversal.
        recorded_contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
        _data_processing.validate_resumed_input_sampling(
            recorded=_data_processing.recorded_input_sampling(recorded_contract),
            current=data_config.rlds_input_sampling,
        )
        dataset_differences = (
            []
            if input_contract["dataset_fingerprint"] is None
            else _dataset_fingerprint.check_resumed_datasets(
                (recorded_contract or {}).get("dataset_fingerprint"),
                input_contract["dataset_fingerprint"],
                allow_change=config.rlds_input_dataset_restart,
            )
        )
        input_contract["dataset_changes"] = list((recorded_contract or {}).get("dataset_changes", [])) + (
            [{"completed_updates": input_completed_updates, "differences": dataset_differences}]
            if dataset_differences
            else []
        )
        if isinstance(
            data_config.rlds_input_sampling,
            _data_processing.DeterministicReplayV3,
        ):
            invariant_keys = (
                "episode_selector",
                "episode_split",
                "normalization_selection",
                "normalization_asset_id",
                "normalization_stats_sha256",
            )
            invariant_mismatches = {
                key: {
                    "recorded": (recorded_contract or {}).get(key),
                    "current": input_contract[key],
                }
                for key in invariant_keys
                if (recorded_contract or {}).get(key) != input_contract[key]
            }
            if invariant_mismatches:
                raise ValueError(
                    "DeterministicReplayV3 resume would change the training-view or "
                    f"normalization invariant: {invariant_mismatches}"
                )
        if isinstance(
            data_config.rlds_input_sampling,
            _data_processing.DeterministicReplayV3,
        ):
            (
                active_sampler_seed,
                replay_completed_updates,
                input_restarts,
            ) = _resolve_v3_replay(
                data_config.rlds_input_sampling,
                recorded=recorded_contract or {},
                completed_updates=input_completed_updates,
                process_count=jax.process_count(),
                local_batch_size=input_local_batch_size,
                sampler_seed=configured_sampler_seed,
                tf_version=tf.__version__,
                tfds_version=tfds.__version__,
                allow_topology_restart=config.rlds_input_topology_restart,
                dataset_changed=bool(dataset_differences),
            )
            input_contract["input_restarts"] = input_restarts
        recorded_processes = (recorded_contract or {}).get("input_process_count")
        if (
            not isinstance(data_config.rlds_input_sampling, _data_processing.DeterministicReplayV3)
            and recorded_processes is not None
            and recorded_processes != jax.process_count()
        ):
            logging.warning(
                "Resuming with %d input processes; the lineage previously used %d. Under a "
                "process-sharded contract this repartitions the file cohorts (recorded as a "
                "distributional restart, not exact iterator continuity).",
                jax.process_count(),
                recorded_processes,
            )
    if not verify_resume_only:
        _publish_input_contract(
            contract_path, input_contract, is_primary=is_primary,
            synchronize=(
                lambda: multihost_utils.sync_global_devices("input_contract_validated")
                if jax.process_count() > 1 else None
            ),
        )
        if is_primary:
            logging.info("Wrote typed input contract beside checkpoints")
    mixture_audit_marker = _primary_phase_marker(
        config.checkpoint_dir, "dataset_mixture_audit"
    )
    if resuming:
        # The run's first start audited the mixture and logged it to W&B at step 0, which a
        # resumed run cannot log to again. Repeating the scan (it decodes up to
        # mixture_report_max_episodes episodes per dataset) would only delay every restart.
        logging.info("Dataset mixture audit skipped on resume; the run's first start reported it")
    elif (
        is_primary
        and not verify_resume_only
        and data_config.rlds_data_dir is not None
        and data_config.datasets
    ):
        training_view = None
        if isinstance(
            data_config.rlds_input_sampling,
            _data_processing.DeterministicReplayV3,
        ):
            # This is part of V3's launch contract, not a best-effort W&B audit.
            # Keep it outside the reporting try/except so an inexact denominator
            # cannot silently start optimizer training.
            training_view = _dataset_mixture.training_view_frames(
                data_config, process_count=jax.process_count()
            )
        try:
            mixture_rows = _dataset_mixture.scan_rlds_mixture(
                data_config, max_episodes=config.mixture_report_max_episodes
            )
            # The exact training view reads every record of every dataset. For DROID that is the
            # full ~92k-episode, image-bearing build at every launch, and the count would still be
            # wrong: the success, pick-and-place and idle filters apply later, in the loader.
            # Report the sampled mixture alone for DROID mixtures (V3 already returned above).
            includes_droid = any(d.name == "droid" for d in data_config.datasets)
            if training_view is None and not includes_droid:
                training_view = _dataset_mixture.training_view_frames(
                    data_config, process_count=jax.process_count()
                )
            repeat_budget = None
            if training_view is None:
                logging.info("Mixture audit: exact training view skipped for a DROID mixture")
            else:
                mixture_rows.append(
                    {
                        "group": "training view",
                        "value": "selected optimizer input",
                        "episodes": training_view["training_episodes"],
                        "frames": training_view["training_frames"],
                        "sampled": False,
                        "sample_size": training_view["training_episodes"],
                    }
                )
                mixture_rows.extend(
                    {
                        "group": "process cohort",
                        "value": f"process {index}",
                        "episodes": 0,
                        "frames": frames,
                        "sampled": False,
                        "sample_size": training_view["training_episodes"],
                    }
                    for index, frames in enumerate(training_view["cohort_frames"])
                )
                repeat_budget = _dataset_mixture.nominal_repeat_budget(
                    mixture_rows,
                    global_batch_size=config.batch_size,
                    training_frames=training_view["training_frames"],
                )
                if repeat_budget is not None:
                    repeat_budget = {
                        **repeat_budget,
                        "training_frames": training_view["training_frames"],
                        "training_episodes": training_view["training_episodes"],
                        "process_count": training_view["process_count"],
                        "local_batch_size": input_local_batch_size,
                        "cohort_frames_min": training_view["cohort_frames_min"],
                        "cohort_frames_max": training_view["cohort_frames_max"],
                    }
            mixture_rows = _dataset_mixture.add_repeat_columns(mixture_rows, repeat_budget)
            mixture_columns = [
                "group",
                "value",
                "episodes",
                "frames",
                "sampled",
                "sample_size",
                "batch size",
                "steps to repeat",
            ]
            wandb.log(
                {
                    "dataset/mixture": wandb.Table(
                        columns=mixture_columns,
                        data=[
                            [row[column] for column in mixture_columns]
                            for row in mixture_rows
                        ],
                    )
                },
                step=0,
            )
            if repeat_budget is not None:
                repeat_columns = list(repeat_budget)
                wandb.log(
                    {
                        "dataset/repeat_budget": wandb.Table(
                            columns=repeat_columns,
                            data=[[repeat_budget[column] for column in repeat_columns]],
                        ),
                    },
                    step=0,
                )
                logging.info("Nominal dataset repeat budget: %s", repeat_budget)
            logging.info("Dataset mixture audit:\n%s", _dataset_mixture.format_markdown(mixture_rows))
        except Exception:
            logging.exception("Dataset mixture audit failed; continuing training")
        finally:
            if mixture_audit_marker is not None:
                mixture_audit_marker.parent.mkdir(parents=True, exist_ok=True)
                mixture_audit_marker.write_text("complete\n")
    elif (
        not is_primary
        and not verify_resume_only
        and data_config.rlds_data_dir is not None
        and data_config.datasets
        and mixture_audit_marker is not None
    ):
        _wait_for_primary_phase(mixture_audit_marker, "dataset mixture audit")

    # Building and positioning the input stream (split selection, then the V3 replay on tf.data
    # CPU threads), restoring the checkpoint (storage I/O) and compiling the train step (XLA) need
    # only one another's shapes, never their values, so they run concurrently: a restart costs the
    # slowest of the three rather than their sum.
    startup_started = time.perf_counter()
    startup = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="startup")

    def finished_at(fn, *args):
        result = fn(*args)
        return result, time.perf_counter() - startup_started

    def start_input():
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            completed_updates=replay_completed_updates,
            input_sampling_seed=active_sampler_seed,
        )
        data_iter = iter(data_loader)
        return data_loader, data_iter, next(data_iter)

    started_input = startup.submit(finished_at, start_input)

    abstract_state, train_state_sharding, initialize_train_state = init_train_state(config, init_rng, mesh)
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    compiled = None
    if not verify_resume_only:
        # Initialized and restored states alike are committed to these shardings and never weakly
        # typed. Compiling for exactly that signature lets the loop's first call reuse this
        # executable, and every start of the run lowers the same program for the persistent cache.
        compile_state = jax.tree.map(
            lambda leaf, leaf_sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf_sharding),
            abstract_state,
            train_state_sharding,
        )

        def compile_train_step():
            # The loop runs the train step under set_mesh, which turns on the activation sharding
            # constraints. jit's cache hands the loop whatever is lowered here, so lowering without
            # the mesh would train with an unconstrained program (about 9x slower per update on a
            # v4-32).
            with sharding.set_mesh(mesh):
                lowered = ptrain_step.lower(train_rng, compile_state, started_input.result()[0][2])
            return lowered.compile()

        compiled = startup.submit(finished_at, compile_train_step)
    # The first batch launches device programs (Observation.from_dict converts the images on the
    # sharded arrays), and so do initializing the state and the restore's cross-process syncs.
    # Every process must launch programs in the same order; on a multi-host TPU two threads
    # launching at once interleave differently per host and the runtime aborts ("an unexpected peer
    # shows up in the launch group with a different launch id"). With more than one process the
    # state is therefore created only after the input thread has launched its programs.
    if jax.process_count() > 1:
        started_input.result()
    train_state = abstract_state if resuming else initialize_train_state()
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        restore_started = time.perf_counter()
        if restore_tier == "resume":
            train_state = resume_tier.restore(
                train_state, latest_checkpoint_step, state_sharding=train_state_sharding
            )
        else:
            train_state = _checkpoints.restore_state(
                checkpoint_manager,
                train_state,
                None,  # restore_state ignores the loader, which is still being built
                latest_checkpoint_step,
                state_sharding=train_state_sharding,
            )
        jax.block_until_ready(train_state)
        _checkpoints.check_restored_description_schema(
            checkpoint_manager, data_config, latest_checkpoint_step
        )
        logging.info(
            "Restored %s checkpoint %d in %.1fs",
            restore_tier,
            latest_checkpoint_step,
            time.perf_counter() - restore_started,
        )
    if compiled is not None and _leaf_signatures(train_state) != _leaf_signatures(compile_state):
        logging.warning(
            "The train step compiled at startup does not match the train state's shapes, dtypes, "
            "weak types and shardings; the first update compiles it again"
        )
    state_ready_s = time.perf_counter() - startup_started
    (data_loader, data_iter, batch), input_ready_s = started_input.result()
    logging.info(
        "Initialized training input at completed_updates=%d using %s",
        replay_completed_updates,
        data_config.rlds_input_sampling.to_record(),
    )
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    if not verify_resume_only and not config.unified_checkpoints:
        resume_tier.prepare(
            train_state, keep_step=latest_checkpoint_step if restore_tier == "resume" else None
        )
    elif not verify_resume_only and resume_tier is not None and restore_tier != "resume":
        # Unified checkpoints: the run directory holds a checkpoint at least as new as the resume
        # tier an earlier start wrote, so that tier only holds storage.
        resume_tier.clear()

    if isinstance(
        data_config.rlds_input_sampling,
        (
            _data_processing.ProcessShardedFileShuffleV1,
            _data_processing.DeterministicReplayV3,
        ),
    ):
        data_config.rlds_input_sampling.validate_restored_completed_updates(
            restored_completed_updates=int(train_state.step),
            seeded_completed_updates=input_completed_updates,
        )

    if verify_resume_only:
        payload = _resume_verification_payload(
            train_state,
            expected_step=expected_resume_step,
            minimum_step=minimum_resume_step,
            checkpoint_dir=str(config.checkpoint_dir),
        )
        logging.info("Read-only resume verification passed: %s", payload)
        print(f"[verify-resume] {json.dumps(payload, sort_keys=True)}", flush=True)
        startup.shutdown()
        checkpoint_manager.close()
        return
    _, compiled_s = compiled.result()
    startup.shutdown()
    logging.info(
        "Startup overlapped: train state ready at %.1fs, input positioned at %.1fs, "
        "train step compiled at %.1fs",
        state_ready_s,
        input_ready_s,
        compiled_s,
    )

    psampled_action_metrics_step = jax.jit(
        sampled_action_metrics_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    pbackbone_conditioning_metrics_step = (
        jax.jit(
            backbone_conditioning_metrics_step,
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        if isinstance(
            config.model.backbone_conditioning,
            (
                _pi_input_config.ExtractedEmbodimentBackboneToken,
                _pi_input_config.FlatUrma4BackboneToken,
            ),
        )
        else None
    )

    pdata_visualization_step = jax.jit(
        functools.partial(data_visualization_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(data_sharding, replicated_sharding),
    )

    token_describer = None
    for transform in data_config.model_transforms.inputs:
        tokenizer = getattr(transform, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "describe_tokens"):
            token_describer = tokenizer.describe_tokens
            break
    physical_state_dim = (
        0
        if data_config.norm_stats is None or "state" not in data_config.norm_stats
        else len(np.asarray(data_config.norm_stats["state"].mean))
    )
    visualization_norm_stats = (
        data_config.norm_stats
        if data_config.embodiment_normalization or data_config.norm_stats is None
        else {"actions": data_config.norm_stats["actions"]}
    )
    action_unnormalizer = data_config.input_config.normalization.create_unnormalizer(
        _transforms,
        visualization_norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )

    val_iter = None
    pvalidation_step = None
    if config.val_interval > 0 and data_config.rlds_data_dir is not None:
        # Validation always measures the empirical held-out distribution. Training-only
        # frame resampling is an explicit processing module and must never leak here.
        # Refuses, rather than silently reading a slice of the training data, when the
        # config enables validation without defining any held-out data.
        val_data_config = dataclasses.replace(
            data_config,
            rlds_split=_data_processing.validation_source_split(
                episode_split=data_config.episode_split,
                rlds_split=data_config.rlds_split,
            ),
            episode_split_role="validation",
            post_close_sampling_weight=None,
        )
        val_loader = _data_loader.create_rlds_data_loader(
            val_data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=data_sharding,
            # A fixed seeded frame shuffle makes a bounded validation scan cover all
            # embodiments instead of walking several long episodes from only the first IDs.
            shuffle=True,
            skip_norm_stats=False,
            seed=config.seed + 1,
            deterministic=True,
            shuffle_buffer_size=4096,
            num_parallel_reads=1,
            num_parallel_calls=1,
        )
        val_iter = iter(val_loader)
        pvalidation_step = jax.jit(
            functools.partial(validation_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not is_primary,
    )

    def grouped_payload(prefix, metric, values, task_ids, embodiment_ids):
        grouped = _train_metrics.grouped_means(values, task_ids, embodiment_ids)
        payload = {f"{prefix}/{group}/{metric}": value for group, value in grouped.items() if group != "_aggregate"}
        if "_aggregate" in grouped:
            payload[f"{prefix}/{metric}"] = grouped["_aggregate"]
        return payload

    def source_payload(prefix, metric, values, embodiment_ids):
        """Real DROID vs sim breakdown (droid_real/sim), independent of the task/embodiment
        grouping above."""
        grouped = _train_metrics.source_grouped_means(values, embodiment_ids)
        return {f"{prefix}/{group}/{metric}": value for group, value in grouped.items() if group != "_aggregate"}

    # Every loop step is one optimizer update: with gradient accumulation it feeds k microbatches
    # (each its own jitted call, so only one microbatch is ever on-device) and the last one
    # applies. num_train_steps, every *_interval, checkpoint numbers and the lr schedule all
    # count updates; training/microbatches gives the sample-level x-axis.
    log_metrics = _LocalMetricsLog(config.checkpoint_dir, enabled=is_primary)
    accumulation_steps = max(1, config.gradient_accumulation_steps)
    lr_fn = config.lr_schedule.create()
    infos = []
    update_grad_norms = []
    # Host input accounting: ``data_wait`` is the wall time ``next(data_iter)`` blocks the
    # training thread (tf.data production plus host-to-device placement); ``step_wall`` is
    # the wall time between consecutive dispatches. Their ratio says how input-bound a run
    # is without any accelerator profiler.
    perf_data_wait = 0.0
    perf_step_wall = 0.0
    perf_steps = 0
    perf_last_dispatch = time.perf_counter()
    perf_resume_save_blocking = 0.0
    perf_resume_saves = perf_resume_saves_skipped = 0
    preemption = _checkpoints.PreemptionNotice()
    preempted = False
    for step in pbar:
        for microbatch in range(accumulation_steps):
            if microbatch:
                fetch_started = time.perf_counter()
                batch = next(data_iter)
                perf_data_wait += time.perf_counter() - fetch_started
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            # Only the applying (last) microbatch's grad_norm is the norm of the update's gradient.
            grad_norm = info.pop("grad_norm")
            infos.append(info)
        update_grad_norms.append(grad_norm)
        now = time.perf_counter()
        perf_step_wall += now - perf_last_dispatch
        perf_last_dispatch = now
        perf_steps += 1
        if step % config.log_interval == 0:
            stacked_infos = dict(common_utils.stack_forest(infos))
            per_sample_loss = jax.device_get(stacked_infos.pop("_per_sample_loss"))
            embodiment_ids = jax.device_get(stacked_infos.pop("_embodiment_id"))
            task_ids = jax.device_get(stacked_infos.pop("_task_id"))
            per_sample_action_loss = jax.device_get(stacked_infos.pop("_per_sample_action_loss", None))
            per_sample_lang_loss = jax.device_get(stacked_infos.pop("_per_sample_lang_loss", None))
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info = {"grad_norm": float(np.mean(jax.device_get(update_grad_norms))), **reduced_info}
            update_grad_norms = []
            if pbackbone_conditioning_metrics_step is not None:
                reduced_info.update(
                    jax.device_get(
                        pbackbone_conditioning_metrics_step(train_state, batch)
                    )
                )
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            completed_step = step + 1
            training_payload = {
                **reduced_info,
                **{f"training/{key}": value for key, value in reduced_info.items()},
                "train/step": completed_step,
                "training/optimizer_updates": completed_step,
                "training/microbatches": completed_step * accumulation_steps,
                # lr the next update uses (optax evaluates the schedule at the count of updates
                # already applied).
                "training/learning_rate": float(lr_fn(completed_step)),
            }
            if perf_steps:
                training_payload.update(
                    {
                        "perf/data_wait_s": perf_data_wait / perf_steps,
                        "perf/step_wall_s": perf_step_wall / perf_steps,
                        "perf/data_wait_fraction": (
                            perf_data_wait / perf_step_wall if perf_step_wall > 0 else 0.0
                        ),
                    }
                )
                pbar.write(
                    f"Step {step}: perf data_wait={perf_data_wait / perf_steps:.3f}s "
                    f"step_wall={perf_step_wall / perf_steps:.3f}s"
                )
                perf_data_wait = perf_step_wall = 0.0
                perf_steps = 0
            if perf_resume_saves or perf_resume_saves_skipped:
                training_payload.update(
                    {
                        "perf/resume_save_blocking_s": (
                            perf_resume_save_blocking / perf_resume_saves if perf_resume_saves else 0.0
                        ),
                        "perf/resume_saves": perf_resume_saves,
                        "perf/resume_saves_skipped": perf_resume_saves_skipped,
                    }
                )
                perf_resume_save_blocking = 0.0
                perf_resume_saves = perf_resume_saves_skipped = 0
            # Preserve Octo's task/embodiment hierarchy. The aggregate key is intentionally
            # overwritten by the exact globally masked scalar above.
            grouped_loss = grouped_payload("training", "loss", per_sample_loss, task_ids, embodiment_ids)
            grouped_loss.pop("training/loss", None)
            training_payload.update(grouped_loss)
            # Real (DROID) vs sim (realrig9k) co-training breakdown, keyed off embodiment_id.
            training_payload.update(source_payload("training", "loss", per_sample_loss, embodiment_ids))
            # KI dual-objective: action expert (flow matching) vs language head (FAST-token CE).
            if per_sample_action_loss is not None:
                training_payload["training/ki/action_loss"] = float(np.mean(per_sample_action_loss))
                training_payload.update(
                    source_payload("training/ki", "action_loss", per_sample_action_loss, embodiment_ids)
                )
            if per_sample_lang_loss is not None:
                training_payload["training/ki/lang_loss"] = float(np.mean(per_sample_lang_loss))
                training_payload.update(
                    source_payload("training/ki", "lang_loss", per_sample_lang_loss, embodiment_ids)
                )
            log_metrics(training_payload, step=completed_step)
            infos = []

        sample_due = config.train_sample_interval > 0 and (step + 1) % config.train_sample_interval == 0
        visualization_due = config.visualization_interval > 0 and (step + 1) % config.visualization_interval == 0
        if sample_due or visualization_due:
            sample_rng = jax.random.fold_in(train_rng, 10_000_000 + step + 1)
            sampled = jax.device_get(psampled_action_metrics_step(sample_rng, train_state, batch))
            sample_payload = grouped_payload(
                "training",
                "sample_loss",
                sampled["sample_loss"],
                sampled["task_id"],
                sampled["embodiment_id"],
            )
            sample_payload.update(
                grouped_payload("training", "sample_mae", sampled["sample_mae"], sampled["task_id"], sampled["embodiment_id"])
            )
            sample_payload.update(source_payload("training", "sample_loss", sampled["sample_loss"], sampled["embodiment_id"]))
            sample_payload.update(source_payload("training", "sample_mae", sampled["sample_mae"], sampled["embodiment_id"]))
            sample_payload["train/step"] = step + 1
            log_metrics(sample_payload, step=step + 1)

        if visualization_due:
            visualization_rng = jax.random.fold_in(train_rng, 20_000_000 + step + 1)
            visual_observation, visual_predictions = _pi_runtime.global_device_get(
                pdata_visualization_step(visualization_rng, train_state, batch)
            )
            normalized_targets = np.asarray(_pi_runtime.global_device_get(batch[1]))
            if is_primary:
                visualization_ids = (
                    np.asarray(visual_observation.embodiment_id)
                    if visual_observation.embodiment_id is not None
                    else np.full(normalized_targets.shape[0], -1, np.int32)
                )
                physical_targets = action_unnormalizer(
                    {
                        "actions": normalized_targets,
                        "embodiment_id": visualization_ids,
                    }
                )["actions"]
                physical_predictions = action_unnormalizer(
                    {
                        "actions": np.asarray(visual_predictions),
                        "embodiment_id": visualization_ids,
                    }
                )["actions"]
                panel = _train_metrics.data_visualization_html(
                    visual_observation,
                    physical_targets,
                    physical_predictions,
                    count=config.visualization_samples,
                    physical_state_dim=physical_state_dim,
                    state_conditioning_dim=len(data_config.embodiment_state_ids),
                    prompt_conditioning=type(data_config.input_config.prompt).__name__,
                    state_encoding=type(data_config.input_config.state).__name__,
                    normalization=type(data_config.input_config.normalization).__name__,
                    action_expert_conditioning=type(action_expert_conditioning).__name__,
                    token_describer=token_describer,
                )
                wandb.log(
                    {
                        "data_visualization/training_samples": wandb.Html(panel),
                        "train/step": step + 1,
                    },
                    step=step + 1,
                )

        if val_iter is not None and (step + 1) % config.val_interval == 0:
            val_infos = []
            for val_index in range(config.val_batches):
                val_rng = jax.random.fold_in(train_rng, (step + 1) * config.val_batches + val_index)
                val_infos.append(pvalidation_step(val_rng, train_state, next(val_iter)))
            stacked_val = dict(common_utils.stack_forest(val_infos))
            val_per_sample_loss = jax.device_get(stacked_val.pop("_per_sample_loss"))
            val_per_sample_sample_loss = jax.device_get(stacked_val.pop("_per_sample_sample_loss"))
            val_per_sample_mae = jax.device_get(stacked_val.pop("_per_sample_mae"))
            val_per_sample_mae_physical = jax.device_get(stacked_val.pop("_per_sample_mae_physical", None))
            val_per_sample_action_loss = jax.device_get(stacked_val.pop("_per_sample_action_loss", None))
            val_per_sample_lang_loss = jax.device_get(stacked_val.pop("_per_sample_lang_loss", None))
            val_embodiment_ids = jax.device_get(stacked_val.pop("_embodiment_id"))
            val_task_ids = jax.device_get(stacked_val.pop("_task_id"))
            val_info = jax.device_get(_mean_over_val_batches(stacked_val))
            flat_val = {
                key if np.asarray(value).ndim == 0 else f"{key}/{index}": scalar
                for key, value in val_info.items()
                for index, scalar in (
                    [(0, float(value))] if np.asarray(value).ndim == 0 else enumerate(np.asarray(value).tolist())
                )
            }
            # Remove the artificial /0 suffix from scalar metrics.
            scalar_keys = {"val/loss/0", "val/sample_mae/0", "val/sample_loss/0"}
            flat_val = {key.removesuffix("/0") if key in scalar_keys else key: value for key, value in flat_val.items()}
            validation_payload = {
                **flat_val,
                **{
                    f"validation/{key.removeprefix('val/')}": value
                    for key, value in flat_val.items()
                    if key.startswith("val/")
                },
                "train/step": step + 1,
            }
            grouped_val_loss = grouped_payload(
                "validation", "loss", val_per_sample_loss, val_task_ids, val_embodiment_ids
            )
            grouped_val_sample = grouped_payload(
                "validation",
                "sample_loss",
                val_per_sample_sample_loss,
                val_task_ids,
                val_embodiment_ids,
            )
            grouped_val_mae = grouped_payload(
                "validation", "sample_mae", val_per_sample_mae, val_task_ids, val_embodiment_ids
            )
            grouped_val_loss.pop("validation/loss", None)
            grouped_val_sample.pop("validation/sample_loss", None)
            grouped_val_mae.pop("validation/sample_mae", None)
            validation_payload.update(grouped_val_loss)
            validation_payload.update(grouped_val_sample)
            validation_payload.update(grouped_val_mae)
            # Real (DROID) vs sim (realrig9k) co-training breakdown, keyed off embodiment_id.
            validation_payload.update(source_payload("validation", "loss", val_per_sample_loss, val_embodiment_ids))
            validation_payload.update(
                source_payload("validation", "sample_loss", val_per_sample_sample_loss, val_embodiment_ids)
            )
            validation_payload.update(
                source_payload("validation", "sample_mae", val_per_sample_mae, val_embodiment_ids)
            )
            if val_per_sample_mae_physical is not None:
                validation_payload.update(
                    source_payload(
                        "validation", "sample_mae_physical", val_per_sample_mae_physical, val_embodiment_ids
                    )
                )
            # KI dual-objective: action expert (flow matching) vs language head (FAST-token CE).
            if val_per_sample_action_loss is not None:
                validation_payload["validation/ki/action_loss"] = float(np.mean(val_per_sample_action_loss))
                validation_payload.update(
                    source_payload("validation/ki", "action_loss", val_per_sample_action_loss, val_embodiment_ids)
                )
            if val_per_sample_lang_loss is not None:
                validation_payload["validation/ki/lang_loss"] = float(np.mean(val_per_sample_lang_loss))
                validation_payload.update(
                    source_payload("validation/ki", "lang_loss", val_per_sample_lang_loss, val_embodiment_ids)
                )
            if is_primary:
                logging.info(
                    "Step %d validation: loss=%.4f, sample_loss=%.4f, sample_mae=%.4f",
                    step + 1,
                    validation_payload["val/loss"],
                    validation_payload["val/sample_loss"],
                    validation_payload["val/sample_mae"],
                )
            log_metrics(validation_payload, step=step + 1)
        fetch_started = time.perf_counter()
        batch = next(data_iter)
        perf_data_wait += time.perf_counter() - fetch_started

        # Config cadence is expressed in completed optimizer updates. ``step`` is the
        # zero-based loop/checkpoint index, so 4k updates intentionally lands in directory 3999.
        # Every process must poll the preemption notice on every step (collective agreement).
        preempted = preemption.reached(step)
        tier = _checkpoints.checkpoint_tier(
            step,
            save_interval=config.save_interval,
            keep_interval=config.keep_interval,
            num_train_steps=config.num_train_steps,
        )
        # Unified checkpoints send the resume and preemption saves to the run directory as full
        # checkpoints; Orbax drops an off-grid one once the next save commits. A preemption save
        # there waits for an earlier save inside Orbax.
        if tier == "milestone" or (preempted and config.unified_checkpoints):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
        elif preempted:
            resume_tier.wait_until_finished()
            resume_tier.save(train_state, step)
        elif tier == "resume":
            if (
                _checkpoints.save_ready(checkpoint_manager)
                if config.unified_checkpoints
                else resume_tier.ready()
            ):
                # Time the save alone; the update still running on the device is not save cost.
                jax.block_until_ready(train_state)
                save_started = time.perf_counter()
                if config.unified_checkpoints:
                    _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
                else:
                    resume_tier.save(train_state, step)
                blocked = time.perf_counter() - save_started
                perf_resume_save_blocking += blocked
                perf_resume_saves += 1
                logging.info("Started resume checkpoint %d; training blocked %.2fs", step, blocked)
            else:
                perf_resume_saves_skipped += 1
                logging.info("Skipped resume checkpoint %d: the previous save is still writing", step)
        if preempted:
            logging.warning("Stopping after saving checkpoint %d for a preemption notice", step)
            break

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if resume_tier is not None:
        resume_tier.wait_until_finished()
    if preempted:
        raise SystemExit(143)
    if resume_tier is not None and checkpoint_manager.latest_step() == config.num_train_steps - 1:
        # The final milestone supersedes every resume step, which would otherwise only hold storage.
        resume_tier.clear()


if __name__ == "__main__":
    main(_config.cli())
