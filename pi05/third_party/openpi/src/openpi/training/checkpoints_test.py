import dataclasses
import os
import shutil
import signal
from types import SimpleNamespace
import zlib

from etils import epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
from openpi.training import checkpoints
from openpi.training import utils as training_utils


def test_completed_update_milestone_uses_one_based_update_count():
    assert not checkpoints._is_completed_update_milestone(999, interval=4_000)
    assert not checkpoints._is_completed_update_milestone(1_999, interval=4_000)
    assert not checkpoints._is_completed_update_milestone(2_999, interval=4_000)
    assert checkpoints._is_completed_update_milestone(3_999, interval=4_000)
    assert checkpoints._is_completed_update_milestone(7_999, interval=4_000)


def test_checkpoint_tier_keeps_grid_and_final_step_as_milestones():
    def tiers(*, keep_interval):
        return {
            step: tier
            for step in range(10_000)
            if (
                tier := checkpoints.checkpoint_tier(
                    step, save_interval=1_000, keep_interval=keep_interval, num_train_steps=10_000
                )
            )
        }

    assert tiers(keep_interval=4_000) == {
        **{step: "resume" for step in (999, 1_999, 2_999, 4_999, 5_999, 6_999, 8_999)},
        3_999: "milestone",
        7_999: "milestone",
        9_999: "milestone",
    }
    # A keep grid that save_interval does not divide is still saved.
    assert checkpoints.checkpoint_tier(
        2_499, save_interval=1_000, keep_interval=2_500, num_train_steps=10_000
    ) == "milestone"
    # Legacy keep_period configs have no keep grid: every save stays a milestone.
    assert set(tiers(keep_interval=None)) == {999 + 1_000 * k for k in range(10)}
    assert set(tiers(keep_interval=None).values()) == {"milestone"}


FREEZE = nnx_utils.PathRegex(".*backbone.*")


class _Tiny(nnx.Module):
    def __init__(self, rngs):
        self.backbone = nnx.Linear(3, 4, rngs=rngs)
        self.head = nnx.Linear(4, 2, rngs=rngs)


def _state(*, step=7, ema=False):
    model = _Tiny(nnx.Rngs(0))
    params = nnx_utils.state_map(
        nnx.state(model), FREEZE, lambda p: p.replace(p.value.astype(jnp.bfloat16))
    )
    tx = optax.adam(1e-3)
    trainable = params.filter(nnx.All(nnx.Param, nnx.Not(FREEZE)))
    # Non-initial optimizer moments, so a restore that silently re-initializes them is caught.
    opt_state = jax.tree.map(lambda x: x + 0.5, tx.init(trainable))
    return training_utils.TrainState(
        step=jnp.asarray(step),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=opt_state,
        tx=tx,
        ema_decay=0.99 if ema else None,
        ema_params=jax.tree.map(lambda x: x * 2, params) if ema else None,
    )


def _open(root, *, resume, unified=False, keep_every=None, read_only=False):
    return checkpoints.initialize_checkpoint_dir(
        root,
        keep_period=None,
        keep_every_n_completed_updates=keep_every,
        overwrite=False,
        resume=resume,
        freeze_filter=FREEZE,
        read_only=read_only,
        unified=unified,
    )


def _restore_target(state):
    shape = jax.eval_shape(lambda: state)
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return shape, jax.tree.map(lambda _: device, shape)


def _assert_same(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.dtype == e.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(e))


def _save_milestone(mngr, state, step):
    data_config = SimpleNamespace(asset_id=None, norm_stats=None, embodiment_description_bundle=None)
    checkpoints.save_state(mngr, state, SimpleNamespace(data_config=lambda: data_config), step)
    mngr.wait_until_finished()


def test_resume_tier_stores_frozen_params_once_and_round_trips(tmp_path):
    _, tier, _ = _open(tmp_path / "run", resume=False)
    state = _state()
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    assert tier.ready()
    tier.save(state, 41)
    tier.wait_until_finished()
    assert tier.step_format(41) == checkpoints.ResumeTier.TRAINABLE_ONLY
    step_bytes = sum(f.stat().st_size for f in (tmp_path / "run" / "resume" / "41").rglob("*") if f.is_file())
    frozen_bytes = sum(
        f.stat().st_size for f in (tmp_path / "run" / "resume_frozen_params").rglob("*") if f.is_file()
    )
    assert step_bytes > 0 and frozen_bytes > 0

    mngr, tier, resuming = _open(tmp_path / "run", resume=True)
    assert resuming
    assert checkpoints.select_restore_point(mngr, tier) == (41, "resume")
    shape, sharding = _restore_target(state)
    _assert_same(tier.restore(shape, 41, state_sharding=sharding), state)


def test_resume_tier_keeps_full_state_with_ema(tmp_path):
    _, tier, _ = _open(tmp_path / "run", resume=False)
    state = _state(ema=True)
    tier.prepare(state, keep_step=None)
    tier.save(state, 9)
    tier.wait_until_finished()
    assert tier.step_format(9) == checkpoints.ResumeTier.FULL
    shape, sharding = _restore_target(state)
    _assert_same(tier.restore(shape, 9, state_sharding=sharding), state)


def test_newer_resume_step_wins_and_equal_milestone_is_preferred(tmp_path):
    mngr, tier, _ = _open(tmp_path / "run", resume=False)
    state = _state()
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    _save_milestone(mngr, state, 99)
    tier.save(state, 149)
    tier.wait_until_finished()
    assert checkpoints.select_restore_point(mngr, tier) == (149, "resume")
    _save_milestone(mngr, state, 149)
    assert checkpoints.select_restore_point(mngr, tier) == (149, "milestone")
    # Resuming from that milestone retires the superseded resume step.
    tier.prepare(state, keep_step=None)
    assert tier.latest_step() is None


def test_resume_tier_clear_removes_steps_and_frozen_params(tmp_path):
    mngr, tier, _ = _open(tmp_path / "run", resume=False)
    state = _state()
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    tier.save(state, 8_999)
    tier.wait_until_finished()
    _save_milestone(mngr, state, 9_999)
    tier.clear()
    assert tier.latest_step() is None
    assert not [p for p in (tmp_path / "run" / "resume").iterdir() if p.name.isdigit()]
    assert not [p for p in (tmp_path / "run" / "resume_frozen_params").iterdir() if p.name.isdigit()]
    mngr, tier, resuming = _open(tmp_path / "run", resume=True)
    assert resuming
    assert checkpoints.select_restore_point(mngr, tier) == (9_999, "milestone")


def test_trainable_only_step_without_frozen_params_is_not_restorable(tmp_path):
    mngr, tier, _ = _open(tmp_path / "run", resume=False)
    state = _state()
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    tier.save(state, 49)
    tier.wait_until_finished()
    (tmp_path / "run" / "resume_frozen_params" / "0").rename(tmp_path / "frozen_moved")
    mngr, tier, resuming = _open(tmp_path / "run", resume=True)
    # No milestone either: the run starts fresh, and the stale step is dropped before saving.
    assert resuming
    assert checkpoints.select_restore_point(mngr, tier) == (None, "milestone")
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    tier.save(state, 49)
    tier.wait_until_finished()
    assert tier.latest_restorable_step() == 49


def test_resume_step_no_newer_than_the_milestones_is_not_read(tmp_path):
    # An interrupted clear() can leave a resume step without its format item; it must not block resume.
    mngr, tier, _ = _open(tmp_path / "run", resume=False)
    state = _state()
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    tier.save(state, 49)
    tier.wait_until_finished()
    _save_milestone(mngr, state, 99)
    shutil.rmtree(tmp_path / "run" / "resume" / "49" / "resume_format")
    mngr, tier, resuming = _open(tmp_path / "run", resume=True)
    assert resuming and tier.latest_step() == 49
    assert checkpoints.select_restore_point(mngr, tier) == (99, "milestone")


def _save_with_assets(mngr, state, step, *, asset_id):
    data_config = SimpleNamespace(
        asset_id=asset_id,
        norm_stats={"state": _normalize.NormStats(mean=np.zeros(2), std=np.ones(2))},
        normalization_bundle=None,
        embodiment_description_bundle=None,
    )
    checkpoints.save_state(mngr, state, SimpleNamespace(data_config=lambda: data_config), step)
    mngr.wait_until_finished()


def test_unified_saves_are_full_evaluable_checkpoints_and_only_milestones_accumulate(tmp_path):
    root = tmp_path / "run"
    mngr, tier, _ = _open(root, resume=False, unified=True, keep_every=100)
    assert tier is None
    assert not (root / "resume").exists() and not (root / "resume_frozen_params").exists()
    state = _state()
    for step in (49, 99, 149, 199, 249):
        _save_with_assets(mngr, dataclasses.replace(state, step=jnp.asarray(step)), step, asset_id="robot")
        # Off-grid saves carry the same assets as milestones, so evaluation can read either.
        assert (root / str(step) / "assets" / "robot" / "norm_stats.json").is_file()
    # Milestones (100 and 200 completed updates) stay; an off-grid save lasts until the next one commits.
    assert list(mngr.all_steps()) == [99, 199, 249]

    mngr, tier, resuming = _open(root, resume=True, unified=True, keep_every=100)
    assert resuming and tier is None
    assert checkpoints.select_restore_point(mngr, tier) == (249, "milestone")
    expected = dataclasses.replace(state, step=jnp.asarray(249))
    shape, sharding = _restore_target(expected)
    _assert_same(checkpoints.restore_state(mngr, shape, None, 249, state_sharding=sharding), expected)


def test_unified_restores_then_clears_a_resume_tier_an_earlier_start_wrote(tmp_path):
    root = tmp_path / "run"
    mngr, tier, _ = _open(root, resume=False)
    state = _state()
    tier.prepare(state, keep_step=None)
    tier.wait_until_finished()
    _save_milestone(mngr, state, 99)
    tier.save(state, 149)
    tier.wait_until_finished()

    mngr, tier, resuming = _open(root, resume=True, unified=True)
    assert resuming and tier is not None
    assert checkpoints.select_restore_point(mngr, tier) == (149, "resume")
    shape, sharding = _restore_target(state)
    _assert_same(tier.restore(shape, 149, state_sharding=sharding), state)
    # A checkpoint in the run directory at the same update is preferred, and retires the tier.
    _save_milestone(mngr, state, 149)
    assert checkpoints.select_restore_point(mngr, tier) == (149, "milestone")
    tier.clear()
    assert tier.latest_step() is None


def test_save_ready_is_false_while_any_process_is_still_writing(monkeypatch):
    class Manager:
        def __init__(self, busy):
            self.busy = busy

        def is_saving_in_progress(self):
            return self.busy

    assert checkpoints.save_ready(Manager(False))
    assert not checkpoints.save_ready(Manager(True))
    # This process is idle, but the other one is still writing.
    _as_process(monkeypatch, 0, others_saw=True)
    assert not checkpoints.save_ready(Manager(False))


def test_unified_removes_interrupted_saves_unless_read_only(tmp_path):
    root = tmp_path / "run"
    mngr, _, _ = _open(root, resume=False, unified=True)
    _save_milestone(mngr, _state(), 3)
    interrupted = root / "7.orbax-checkpoint-tmp-1760000000000000"
    interrupted.mkdir()
    (root / "xla_cache").mkdir()
    _open(root, resume=True, unified=True, read_only=True)
    assert interrupted.exists()
    # The default layout leaves the run directory alone.
    _open(root, resume=True)
    assert interrupted.exists()
    _open(root, resume=True, unified=True)
    assert not interrupted.exists()
    assert (root / "3").is_dir() and (root / "xla_cache").is_dir()


def test_on_gcs_only_unmarked_steps_newer_than_every_committed_step_are_interrupted(tmp_path):
    # 12 lacks the marker but predates committed 13 (e.g. copied from local storage): kept.
    for name in ("12", "13", "14", "xla_cache", "resume", ".multihost_rendezvous"):
        (tmp_path / name).mkdir()
    (tmp_path / "13" / "commit_success.txt").write_text("")
    checkpoints._remove_interrupted_saves(epath.Path(tmp_path), gcs=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".multihost_rendezvous", "12", "13", "resume", "xla_cache"]


def test_preemption_notice_without_distributed_handles_sigusr1():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        notice = checkpoints.PreemptionNotice()
        assert not notice.reached(0)
        os.kill(os.getpid(), signal.SIGUSR1)
        assert notice.reached(1)
        assert notice.signal_name == "SIGUSR1"
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_resume_tier_removes_interrupted_writes_unless_read_only(tmp_path):
    _open(tmp_path / "run", resume=False)
    interrupted = tmp_path / "run" / "resume" / "7.orbax-checkpoint-tmp-1760000000000000"
    interrupted.mkdir()
    checkpoints.initialize_checkpoint_dir(
        tmp_path / "run", keep_period=None, overwrite=False, resume=True, freeze_filter=FREEZE, read_only=True
    )
    assert interrupted.exists()
    _open(tmp_path / "run", resume=True)
    assert not interrupted.exists()


def _as_process(monkeypatch, index, *, others_saw=True):
    """Run as process ``index`` of two; the other process reports ``others_saw``."""
    monkeypatch.setattr(jax, "process_index", lambda: index)
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(checkpoints.multihost_utils, "broadcast_one_to_all", lambda value: value)

    def process_allgather(value):
        other = np.asarray(others_saw) if value.dtype == bool else value
        return np.stack([other, value] if index else [value, other])

    monkeypatch.setattr(checkpoints.multihost_utils, "process_allgather", process_allgather)


def test_shared_directory_probe_waits_until_process_zero_token_is_visible(tmp_path, monkeypatch):
    import threading

    _as_process(monkeypatch, 1)
    probe = tmp_path / ".shared_storage_probe"
    monkeypatch.setattr(checkpoints.os, "urandom", lambda n: b"\x07" * n)
    token = int.from_bytes(b"\x07" * 3, "little") + 1
    monkeypatch.setattr(
        checkpoints.multihost_utils, "broadcast_one_to_all", lambda value: np.asarray(token, np.int32)
    )
    threading.Timer(0.3, lambda: probe.write_text(str(token))).start()
    checkpoints._require_shared_directory(epath.Path(tmp_path), timeout_s=10)


def test_shared_directory_probe_fails_on_every_process_when_storage_is_not_shared(tmp_path, monkeypatch):
    import pytest

    # Process zero wrote its token to its own disk; this process never sees it.
    _as_process(monkeypatch, 1)
    monkeypatch.setattr(checkpoints.multihost_utils, "broadcast_one_to_all", lambda value: np.asarray(5, np.int32))
    with pytest.raises(ValueError, match=r"every process shares.*processes \[1\]"):
        checkpoints._require_shared_directory(epath.Path(tmp_path / "run"), timeout_s=0.5)

    # Process zero fails too, once it hears that another process could not read the token.
    _as_process(monkeypatch, 0, others_saw=False)
    with pytest.raises(ValueError, match=r"processes \[1\]"):
        checkpoints._require_shared_directory(epath.Path(tmp_path), timeout_s=0.5)
    assert not (tmp_path / ".shared_storage_probe").exists()


def test_shared_directory_probe_reports_a_failed_write_on_every_process(tmp_path, monkeypatch):
    import pytest

    # Process zero cannot write the token (a full disk, say); the others hear it through the broadcast.
    _as_process(monkeypatch, 1)
    monkeypatch.setattr(checkpoints.multihost_utils, "broadcast_one_to_all", lambda value: np.asarray(-1, np.int32))
    with pytest.raises(OSError, match="process 0 could not write"):
        checkpoints._require_shared_directory(epath.Path(tmp_path), timeout_s=10)

    _as_process(monkeypatch, 0)
    with pytest.raises(OSError, match="process 0 could not write .*No such file"):
        checkpoints._require_shared_directory(epath.Path(tmp_path / "missing"), timeout_s=10)


def test_revalidate_lists_from_the_deepest_visible_directory(tmp_path, monkeypatch):
    listed = []
    real_listdir = checkpoints.os.listdir

    def listdir(path):
        listed.append(str(path))
        return real_listdir(path)

    monkeypatch.setattr(checkpoints.os, "listdir", listdir)
    (tmp_path / "a").mkdir()
    checkpoints._revalidate(epath.Path(tmp_path / "a" / "b" / "c"))
    assert listed == [str(tmp_path / "a" / "b" / "c"), str(tmp_path / "a" / "b"), str(tmp_path / "a"), str(tmp_path / "a" / "b")]


def test_consistent_steps_reloads_until_every_process_lists_the_same_steps(tmp_path, monkeypatch):
    import pytest

    class Manager:
        directory = tmp_path

        def __init__(self, listings):
            self.listings = listings

        def all_steps(self):
            return self.listings[0]

        def reload(self):
            self.listings = self.listings[1:] or self.listings

    # Process zero lists steps 999 and 1999; this process misses 1999 until its second reload.
    digest = zlib.crc32(repr([(999, 1999)]).encode()) & 0x7FFFFFFF
    monkeypatch.setattr(
        checkpoints.multihost_utils,
        "process_allgather",
        lambda value: np.stack([np.asarray(digest, np.int32), value]),
    )
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _: None)
    stale = Manager([[999], [999], [999, 1999]])
    checkpoints._require_consistent_steps([stale])
    assert stale.all_steps() == [999, 1999]

    with pytest.raises(RuntimeError, match="different checkpoint steps"):
        checkpoints._require_consistent_steps([Manager([[999]])], attempts=3)
