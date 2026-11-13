from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
import pathlib
import signal
import time
from typing import Protocol
import zlib

from etils import epath
import flax.nnx as nnx
import jax
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils
from pi05_integration import embodiment_bundle as _embodiment_bundle
from pi05_integration import normalization_bundle as _normalization_bundle


# Frequent preemption-recovery checkpoints live beside, not inside, the milestone steps so that
# evaluators, staging and recovery tooling keep seeing exactly the configured milestone directories.
# With ``unified=True`` (opt-in, ``TrainConfig.unified_checkpoints``) every save is instead a full
# checkpoint in the run directory, and this tier is only read (and cleared) for runs that wrote it.
RESUME_DIRNAME = "resume"
FROZEN_PARAMS_DIRNAME = "resume_frozen_params"
# The trainer's default JAX persistent compilation cache inside a run's checkpoint directory.
COMPILATION_CACHE_DIRNAME = "xla_cache"


def _is_completed_update_milestone(step: int, *, interval: int) -> bool:
    """Return whether a zero-based checkpoint step completes an interval milestone."""

    return (step + 1) % interval == 0


def checkpoint_tier(
    step: int, *, save_interval: int, keep_interval: int | None, num_train_steps: int
) -> str | None:
    """Return where the state after zero-based ``step`` is saved: "milestone", "resume" or None.

    The final step and steps on the keep grid are milestones; other ``save_interval`` steps go
    to the newest-only resume tier. Without a keep grid (legacy ``keep_period`` retention) every
    save is a milestone.
    """
    if step == num_train_steps - 1 or (
        keep_interval is not None and _is_completed_update_milestone(step, interval=keep_interval)
    ):
        return "milestone"
    if _is_completed_update_milestone(step, interval=save_interval):
        return "milestone" if keep_interval is None else "resume"
    return None


def _revalidate(path: epath.Path) -> None:
    """List the deepest directory on ``path`` this host sees, then each one below it in turn.

    An NFS client keeps a failed lookup (every process resolved the root before process zero created
    it) until the parent's attributes expire, up to a minute; listing the parent revalidates it.
    """
    directories = [path, *path.parents]
    for depth, directory in enumerate(directories):
        try:
            os.listdir(directory)
        except OSError:
            continue
        for below in reversed(directories[:depth]):
            try:
                os.listdir(below)
            except OSError:
                return
        return


def _require_shared_directory(checkpoint_dir: epath.Path, timeout_s: float = 120.0) -> None:
    """Fail on every process unless all of them see the checkpoint root process zero created.

    Orbax has each process write its own shards of a step into one directory, so every process must
    see the same directory: a disk local to each host would split a checkpoint across hosts. Process
    zero writes a random token there, and every other process must read it back.
    """
    probe = pathlib.Path(checkpoint_dir, ".shared_storage_probe")
    token, write_error = 0, None
    if jax.process_index() == 0:
        token = int.from_bytes(os.urandom(3), "little") + 1
        try:
            probe.write_text(str(token))
        except OSError as error:
            token, write_error = -1, error
    # The broadcast is collective, so process zero reports a failed write instead of raising alone.
    token = int(multihost_utils.broadcast_one_to_all(np.asarray(token, np.int32)))
    if token < 0:
        raise OSError(f"process 0 could not write {probe}: {write_error or 'see process 0'}")
    seen = jax.process_index() == 0
    deadline = time.monotonic() + timeout_s
    while not seen:
        _revalidate(epath.Path(checkpoint_dir))
        try:
            seen = probe.read_text() == str(token)
        except OSError:
            pass
        if not seen:
            if time.monotonic() > deadline:
                break
            time.sleep(1.0)
    seen_by = np.asarray(multihost_utils.process_allgather(np.asarray(seen))).reshape(-1)
    if jax.process_index() == 0:
        try:
            probe.unlink()
        except OSError as error:
            logging.warning("Could not remove %s: %s", probe, error)
    if not seen_by.all():
        raise ValueError(
            "Multi-process checkpointing needs a checkpoint directory that every process shares (gs://, "
            "or a network filesystem mounted at the same path on every host); processes "
            f"{np.flatnonzero(~seen_by).tolist()} could not read what process 0 wrote to {checkpoint_dir} "
            f"within {timeout_s:.0f} s. A shared NFS mount with nocto or long attribute caching can also "
            "delay this."
        )


def _require_consistent_steps(managers: list[ocp.CheckpointManager], attempts: int = 60) -> None:
    """Fail unless every process lists the same steps in each manager.

    Orbax restores and deletes a step collectively, so a process whose filesystem cache misses a step
    another host wrote would pick a different restore point or wait forever in a delete. Each attempt
    is collective, so every process gives up at the same one.
    """
    for attempt in range(attempts):
        if attempt:
            time.sleep(2.0)
            for manager in managers:
                _revalidate(epath.Path(manager.directory))
                manager.reload()
        digest = zlib.crc32(repr([tuple(manager.all_steps()) for manager in managers]).encode()) & 0x7FFFFFFF
        digests = np.asarray(multihost_utils.process_allgather(np.asarray(digest, np.int32))).reshape(-1)
        if (digests == digests[0]).all():
            return
    raise RuntimeError(
        f"Processes still list different checkpoint steps under {[str(m.directory) for m in managers]} "
        f"after {attempts} attempts; the checkpoint directory is not consistent across hosts."
    )


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    keep_every_n_completed_updates: int | None = None,
    overwrite: bool,
    resume: bool,
    freeze_filter: nnx.filterlib.Filter = nnx.Nothing,
    read_only: bool = False,
    unified: bool = False,
) -> tuple[ocp.CheckpointManager, "ResumeTier | None", bool]:
    """Open the milestone manager and the resume tier; ``read_only`` creates no directories.

    With ``unified`` every save goes to the run directory's manager: no resume tier is created,
    one an earlier start wrote is returned only so it can be restored from and cleared (``None``
    when there is none), and step directories whose save never committed are removed at startup.
    """
    if keep_period is not None and keep_every_n_completed_updates is not None:
        raise ValueError("keep_period and keep_every_n_completed_updates are mutually exclusive")
    if keep_every_n_completed_updates is not None and keep_every_n_completed_updates <= 0:
        raise ValueError("keep_every_n_completed_updates must be positive")
    checkpoint_dir = epath.Path(checkpoint_dir)
    is_remote = "://" in str(checkpoint_dir)
    if not is_remote:  # .resolve() corrupts gs:// (// -> / + cwd prefix)
        checkpoint_dir = checkpoint_dir.resolve()
    # Resolve remote-root existence once on the primary process. Querying GCS on every
    # process races with process zero creating the root below: a later worker can observe
    # the newly-created directory and incorrectly reject a fresh multihost launch.
    checkpoint_exists = checkpoint_dir.exists() if jax.process_index() == 0 else False
    if jax.process_count() > 1:
        checkpoint_exists = bool(
            multihost_utils.broadcast_one_to_all(np.asarray(checkpoint_exists))
        )
    resuming = checkpoint_exists and resume
    if checkpoint_exists and not overwrite and not resume:
        raise FileExistsError(
            f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
            "to indicate how to handle it."
        )

    # Every process participates in Orbax save/restore, but only process zero may
    # create or destroy the shared root. The barrier makes the root visible before
    # any worker constructs its manager.
    if jax.process_index() == 0:
        if checkpoint_exists and overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif not checkpoint_exists:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        elif unified and not read_only:
            _remove_interrupted_saves(checkpoint_dir, gcs=str(checkpoint_dir).startswith("gs://"))
        if not read_only and not unified:
            (checkpoint_dir / RESUME_DIRNAME).mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / FROZEN_PARAMS_DIRNAME).mkdir(parents=True, exist_ok=True)
    if jax.process_count() > 1:
        if is_remote or read_only:
            multihost_utils.sync_global_devices("checkpoint_directory_ready")
        else:
            _require_shared_directory(checkpoint_dir)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            should_keep_fn=(
                None
                if keep_every_n_completed_updates is None
                else lambda step: _is_completed_update_milestone(
                    step, interval=keep_every_n_completed_updates
                )
            ),
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    has_resume_tier = True
    if unified:
        has_resume_tier = (checkpoint_dir / RESUME_DIRNAME).exists() if jax.process_index() == 0 else False
        if jax.process_count() > 1:
            has_resume_tier = bool(multihost_utils.broadcast_one_to_all(np.asarray(has_resume_tier)))
    resume_tier = (
        ResumeTier(checkpoint_dir, freeze_filter=freeze_filter, read_only=read_only) if has_resume_tier else None
    )
    if jax.process_count() > 1:
        _require_consistent_steps([mngr, *(resume_tier.managers if resume_tier is not None else ())])

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if (
        resuming
        and tuple(mngr.all_steps()) in [(), (0,)]
        and (resume_tier is None or resume_tier.latest_step() is None)
    ):
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resume_tier, resuming


def _remove_interrupted_saves(checkpoint_dir: epath.Path, *, gcs: bool) -> None:
    """Delete step directories whose save never committed, such as a write a preemption killed.

    Used only for unified checkpoints, whose frequent saves land in the run directory. Locally an
    unfinished step is an Orbax temporary directory. On GCS, where Orbax writes in place, it is a
    step directory without ``commit_success.txt`` that is newer than every committed step: saves
    run in step order, so only the newest can be a killed write. An older unmarked step is left
    alone (Orbax ignores it): it is more likely a complete checkpoint copied from local storage,
    where Orbax never writes the marker. Orbax's own ``cleanup_tmp_directories`` cannot be used: on
    GCS it deletes every directory without the marker, including ``xla_cache`` and the resume
    tier. A leftover step directory would also collide with the next save of that step.
    """
    steps = []
    for path in checkpoint_dir.iterdir():
        step, _, suffix = path.name.partition(".")
        if step.isdigit() and path.is_dir():
            committed = not suffix and (not gcs or (path / "commit_success.txt").exists())
            steps.append((int(step), suffix, committed, path))
    newest_committed = max((step for step, _, committed, _ in steps if committed), default=-1)
    for step, suffix, committed, path in steps:
        if "orbax-checkpoint-tmp" in suffix or (gcs and not suffix and not committed and step > newest_committed):
            logging.info("Removing interrupted checkpoint save %s", path)
            path.rmtree()
        elif gcs and not suffix and not committed:
            logging.warning(
                "Leaving %s: it has no commit_success.txt but is older than committed step %d, so it is not an "
                "interrupted save of this run (a copied checkpoint needs the marker to be restorable)",
                path,
                newest_committed,
            )


def select_restore_point(
    milestones: ocp.CheckpointManager, resume_tier: "ResumeTier | None"
) -> tuple[int | None, str]:
    """Return the newest restorable checkpoint step and the tier that holds it.

    A resume-tier step wins only when it is strictly newer than the newest milestone, so a
    milestone written at the same update is always preferred. A resume step's format is read only
    when it would win: one no newer than the milestones may be half deleted by an interrupted
    ``clear()``. (With unified checkpoints, "milestone" means any step in the run directory.)
    """
    milestone_step = milestones.latest_step()
    resume_step = resume_tier.latest_step() if resume_tier is not None else None
    if (
        resume_step is not None
        and (milestone_step is None or resume_step > milestone_step)
        and resume_tier.latest_restorable_step() is not None
    ):
        return resume_step, "resume"
    return milestone_step, "milestone"


def save_ready(checkpoint_manager: ocp.CheckpointManager) -> bool:
    """Whether every process can start a save on ``checkpoint_manager`` without waiting for an earlier one."""
    busy = checkpoint_manager.is_saving_in_progress()
    if jax.process_count() > 1:
        busy = bool(np.any(multihost_utils.process_allgather(np.asarray(busy))))
    return not busy


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if data_config.asset_id is not None:
            asset_dir = directory / data_config.asset_id
            if norm_stats is not None:
                if data_config.normalization_bundle is not None:
                    # Copy the verified dataset artifact byte-for-byte so its recorded hash
                    # remains the identity of the processor used for training and serving.
                    source = epath.Path(data_config.normalization_bundle["source_stats_uri"])
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    (asset_dir / _normalization_bundle.STATS_NAME).write_bytes(
                        source.read_bytes()
                    )
                    _normalization_bundle.write_checkpoint_bundle(
                        asset_dir, data_config.normalization_bundle
                    )
                else:
                    _normalize.save(asset_dir, norm_stats)
            if data_config.embodiment_description_bundle is not None:
                _embodiment_bundle.copy_checkpoint_bundle(
                    asset_dir / "embodiment_descriptions",
                    data_config.embodiment_description_bundle,
                )
        elif data_config.embodiment_description_bundle is not None:
            raise ValueError("an embodiment description bundle requires a checkpoint asset_id")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def check_restored_description_schema(
    checkpoint_manager: ocp.CheckpointManager,
    data_config,
    step: int | None = None,
) -> None:
    """Refuse to resume a checkpoint trained on another embodiment description schema.

    Reads the milestone checkpoint's embodiment assets (the resume tier stores no assets; every milestone of one run
    holds the same copy) and compares their recorded schema with the configured description bundle.
    """
    source = data_config.embodiment_description_bundle
    milestone = step if step is not None and step in set(checkpoint_manager.all_steps()) else (
        checkpoint_manager.latest_step())
    if source is None or milestone is None or data_config.asset_id is None:
        return
    _embodiment_bundle.check_checkpoint_description_schema(
        epath.Path(checkpoint_manager.directory) / str(milestone) / "assets" / data_config.asset_id
        / "embodiment_descriptions",
        source.get("schema"),
    )


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
    *,
    state_sharding=None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        if state_sharding is None:
            train_state_sharding, params_sharding = train_state, params
        else:
            train_state_sharding, params_sharding = _split_params(state_sharding)
        params = {"params": params}
        params_sharding = {"params": params_sharding}
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                train_state=_pytree_restore_args(train_state, train_state_sharding),
                params=_pytree_restore_args(params, params_sharding),
            ),
        )
    return _merge_params(restored["train_state"], restored["params"])


def _pytree_restore_args(item, target_sharding) -> ocp.args.PyTreeRestore:
    """Restore a tree into its initialized shape, dtype, and device sharding.

    Passing only ``items`` leaves dtype inference to Orbax. That fails when a
    single-GPU checkpoint is resharded such that a multi-host TPU process has no
    addressable shard for an array. Explicit target metadata lets every process
    reconstruct the same global array, including hosts with zero local shards.
    """

    def restore_arg(value, sharding):
        if isinstance(value, jax.Array | jax.ShapeDtypeStruct):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=value.dtype,
                sharding=sharding,
                global_shape=value.shape,
            )
        return ocp.RestoreArgs(
            restore_type=type(value),
            dtype=getattr(value, "dtype", None),
        )

    return ocp.args.PyTreeRestore(
        item=item,
        restore_args=jax.tree.map(restore_arg, item, target_sharding),
    )


def _resume_tier_options(barrier_prefix: str, *, read_only: bool) -> ocp.CheckpointManagerOptions:
    # A distinct barrier prefix keeps this manager's multi-host barriers from colliding with the
    # milestone manager's, which runs concurrently in the same processes. A preemption can kill a
    # write midway; cleaning its temporary directory on startup keeps those from accumulating.
    return ocp.CheckpointManagerOptions(
        max_to_keep=1,
        create=False,
        cleanup_tmp_directories=not read_only,
        read_only=read_only,
        async_options=ocp.AsyncOptions(timeout_secs=7200),
        multiprocessing_options=ocp.options.MultiprocessingOptions(
            barrier_sync_key_prefix=barrier_prefix
        ),
    )


class ResumeTier:
    """Frequent, newest-only checkpoints that bound the work a preemption loses.

    Steps live in ``<checkpoint_dir>/resume`` under the same zero-based update index as the
    milestone directories, and only the newest committed step is kept. Saves are asynchronous;
    callers check :meth:`ready` and skip a save rather than wait for the previous one, so a
    resume save never stalls training on storage.

    When EMA is off and the config freezes parameters, a step stores only what can change: the
    non-frozen parameters, the optimizer state and the step. The frozen parameters are written
    once per run directory to ``<checkpoint_dir>/resume_frozen_params`` and merged back on
    restore; a step whose frozen parameters were never committed is not restorable.
    """

    TRAINABLE_ONLY = "trainable_params_v1"
    FULL = "full_state_v1"

    def __init__(
        self, checkpoint_dir: epath.Path | str, *, freeze_filter: nnx.filterlib.Filter, read_only: bool = False
    ):
        checkpoint_dir = epath.Path(checkpoint_dir)
        self._frozen_filter = nnx.All(nnx.Param, freeze_filter)
        self._manager = ocp.CheckpointManager(
            checkpoint_dir / RESUME_DIRNAME,
            item_handlers={
                "train_state": ocp.PyTreeCheckpointHandler(),
                "params": ocp.PyTreeCheckpointHandler(),
                "resume_format": ocp.JsonCheckpointHandler(),
            },
            options=_resume_tier_options("resume_tier", read_only=read_only),
        )
        self._frozen = ocp.CheckpointManager(
            checkpoint_dir / FROZEN_PARAMS_DIRNAME,
            item_handlers={"params": ocp.PyTreeCheckpointHandler()},
            options=_resume_tier_options("resume_frozen_params", read_only=read_only),
        )

    @property
    def managers(self) -> tuple[ocp.CheckpointManager, ocp.CheckpointManager]:
        return (self._manager, self._frozen)

    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    def step_format(self, step: int) -> str:
        restored = self._manager.restore(
            step, args=ocp.args.Composite(resume_format=ocp.args.JsonRestore())
        )
        return restored["resume_format"]["format"]

    def latest_restorable_step(self) -> int | None:
        step = self._manager.latest_step()
        if step is None:
            return None
        if self.step_format(step) == self.TRAINABLE_ONLY and self._frozen.latest_step() is None:
            logging.warning(
                "Resume step %d needs frozen parameters that were never committed; ignoring it.", step
            )
            return None
        return step

    def _can_store_trainable_only(self, state: training_utils.TrainState) -> bool:
        return state.ema_params is None and bool(
            jax.tree.leaves(state.params.filter(self._frozen_filter))
        )

    def prepare(self, state: training_utils.TrainState, *, keep_step: int | None) -> None:
        """Drop every resume step except ``keep_step`` and start the one-time frozen write.

        A step newer than the restored one (unrestorable, or from an abandoned attempt) would
        otherwise make a later save of the same step fail; an older one is superseded.
        """
        for step in self._manager.all_steps():
            if step != keep_step:
                self._manager.delete(step)
        if self._can_store_trainable_only(state) and self._frozen.latest_step() is None:
            with at.disable_typechecking():
                self._frozen.save(
                    0,
                    args=ocp.args.Composite(
                        params=ocp.args.PyTreeSave({"params": state.params.filter(self._frozen_filter)})
                    ),
                )

    def ready(self) -> bool:
        """Whether every process can start a save without waiting for an earlier one."""
        busy = self._manager.is_saving_in_progress() or self._frozen.is_saving_in_progress()
        if jax.process_count() > 1:
            busy = bool(np.any(multihost_utils.process_allgather(np.asarray(busy))))
        return not busy

    def save(self, state: training_utils.TrainState, step: int) -> None:
        with at.disable_typechecking():
            # ``ready()`` (or the preemption path's wait) guarantees the frozen write has
            # finished before a save, so a trainable-only step is always restorable.
            if self._can_store_trainable_only(state) and self._frozen.latest_step() is not None:
                train_state = dataclasses.replace(state, params={})
                params = state.params.filter(nnx.Not(self._frozen_filter))
                step_format = self.TRAINABLE_ONLY
            else:
                train_state, params = _split_params(state)
                step_format = self.FULL
            self._manager.save(
                step,
                args=ocp.args.Composite(
                    train_state=ocp.args.PyTreeSave(train_state),
                    params=ocp.args.PyTreeSave({"params": params}),
                    resume_format=ocp.args.JsonSave({"format": step_format}),
                ),
            )

    def restore(
        self, state: training_utils.TrainState, step: int, *, state_sharding
    ) -> training_utils.TrainState:
        step_format = self.step_format(step)
        with at.disable_typechecking():
            if step_format == self.FULL:
                train_state, params = _split_params(state)
                train_state_sharding, params_sharding = _split_params(state_sharding)
            else:
                train_state = dataclasses.replace(state, params={})
                train_state_sharding = dataclasses.replace(state_sharding, params={})
                params = state.params.filter(nnx.Not(self._frozen_filter))
                params_sharding = state_sharding.params.filter(nnx.Not(self._frozen_filter))
            restored = self._manager.restore(
                step,
                args=ocp.args.Composite(
                    train_state=_pytree_restore_args(train_state, train_state_sharding),
                    params=_pytree_restore_args({"params": params}, {"params": params_sharding}),
                ),
            )
            if step_format == self.FULL:
                return _merge_params(restored["train_state"], restored["params"])
            frozen = self._frozen.restore(
                0,
                args=ocp.args.Composite(
                    params=_pytree_restore_args(
                        {"params": state.params.filter(self._frozen_filter)},
                        {"params": state_sharding.params.filter(self._frozen_filter)},
                    )
                ),
            )["params"]["params"]
            merged = nnx.State.merge(frozen, restored["params"]["params"])
            if jax.tree.structure(merged) != jax.tree.structure(state.params):
                raise ValueError("restored resume-tier parameters do not match the configured tree")
            return dataclasses.replace(restored["train_state"], params=merged)

    def wait_until_finished(self) -> None:
        self._manager.wait_until_finished()
        self._frozen.wait_until_finished()

    def clear(self) -> None:
        """Delete every resume step and the frozen parameters, once a newer milestone holds the run."""
        for step in self._manager.all_steps():
            self._manager.delete(step)
        for step in self._frozen.all_steps():
            self._frozen.delete(step)


class PreemptionNotice:
    """A collective request to checkpoint now and stop, raised by SIGTERM.

    Under ``jax.distributed`` the coordination service propagates a SIGTERM received by any
    process and all processes agree on one step, so every process must call :meth:`reached` on
    every step. A run without ``jax.distributed`` (one GPU process) handles SIGTERM itself, and
    also SIGUSR1 for an advance warning sent to the trainer process.
    """

    def __init__(self) -> None:
        self._distributed = jax.distributed.is_initialized()
        self.signal_name: str | None = None
        if not self._distributed:
            for sig in (signal.SIGTERM, signal.SIGUSR1):
                signal.signal(sig, self._record)

    def _record(self, signum, frame) -> None:
        del frame
        self.signal_name = signal.Signals(signum).name
        logging.warning("Received %s; saving a resume checkpoint after this update.", self.signal_name)

    def reached(self, step: int) -> bool:
        if self._distributed:
            return multihost_utils.reached_preemption_sync_point(step)
        return self.signal_name is not None


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
