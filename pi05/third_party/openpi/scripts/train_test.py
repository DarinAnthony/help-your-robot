import dataclasses
import json
import math
import os
import pathlib
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"
# test_train resumes in the same process. Reloading the first run's executable from the
# persistent compilation cache corrupts the heap on the CPU backend ("corrupted double-linked
# list", abort); with the cache off both runs compile fresh and pass.
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

from pi05_integration import input_config as _input_config

import openpi.models.pi0_config as pi0_config
from openpi.training import config_utils as _config_utils
from openpi.training import weight_loaders

from . import train

# Main's recipe registry dropped the upstream "debug" config. Keep the tiny CPU recipe here,
# where only these tests use it, instead of re-registering a training config.
_DEBUG_CONFIG = _config_utils.TrainConfig(
    name="debug",
    data=_config_utils.FakeDataConfig(),
    batch_size=2,
    model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
    weight_loader=weight_loaders.NoOpWeightLoader(),
    save_interval=100,
    overwrite=True,
    exp_name="debug",
    num_train_steps=10,
    wandb_enabled=False,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        # EMA under gradient accumulation: the EMA must advance only on applied updates.
        {"ema_decay": 0.99, "gradient_accumulation_steps": 2},
        # The Pi0.5 action time channel (a bias-free projection: a None bias in the parameter tree) through EMA,
        # checkpointing and resume.
        {
            "ema_decay": 0.99,
            "model": pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="dummy",
                action_expert_variant="dummy",
                action_time_conditioning=_input_config.ActionTokenTimeSeconds(),
            ),
        },
        # Opt-in unified checkpoints, a save every update and every second one kept: the off-grid
        # saves land beside the milestones as full checkpoints.
        {"save_interval": 1, "keep_every_n_completed_updates": 2, "unified_checkpoints": True},
    ],
)
def test_train(tmp_path: pathlib.Path, overrides: dict):
    config = dataclasses.replace(
        _DEBUG_CONFIG,
        **overrides,
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # One step is one optimizer update, and every logged scalar also lands in metrics.jsonl.
    metrics_path = tmp_path / "checkpoint" / config.name / config.exp_name / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    last = [r for r in records if "training/optimizer_updates" in r][-1]
    accumulation = overrides.get("gradient_accumulation_steps", 1)
    assert last["training/optimizer_updates"] == 2
    assert last["training/microbatches"] == 2 * accumulation
    assert math.isfinite(last["grad_norm"])

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [r for r in records if "training/optimizer_updates" in r][-1]["training/optimizer_updates"] == 4
    if overrides.get("unified_checkpoints"):
        run_dir = metrics_path.parent
        assert not (run_dir / "resume").exists()
        # Steps 0 and 2 were resume saves, each deleted once the next save committed.
        assert sorted(int(p.name) for p in run_dir.iterdir() if p.name.isdigit()) == [1, 3]


def test_primary_phase_marker_uses_boai_attempt_identity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(train.jax, "process_count", lambda: 8)
    monkeypatch.setenv("BOAI_TPU_REQUEST_ID", "campaign-job")
    monkeypatch.setenv("BOAI_TPU_ATTEMPT", "3")

    marker = train._primary_phase_marker(  # noqa: SLF001
        train.epath.Path(tmp_path), "dataset_mixture_audit"
    )

    assert marker == train.epath.Path(
        tmp_path
        / ".multihost_rendezvous"
        / "campaign-job-attempt-3"
        / "dataset_mixture_audit.complete"
    )


def test_wait_for_primary_phase_accepts_completed_marker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    marker = train.epath.Path(tmp_path / "phase.complete")
    marker.write_text("complete\n")
    monkeypatch.setenv("MOLMOSPACES_PRIMARY_PHASE_TIMEOUT_SECONDS", "1")

    train._wait_for_primary_phase(marker, "test phase")  # noqa: SLF001


def test_init_wandb_allows_checkpoint_run_id_that_was_not_initialized(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint_dir = train.epath.Path(tmp_path / "checkpoint")
    checkpoint_dir.mkdir()
    (checkpoint_dir / "wandb_id.txt").write_text("stable-run-id\n")
    init_calls = []

    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(train.wandb, "define_metric", lambda *args, **kwargs: None)

    train.init_wandb(
        types.SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            project_name="test-project",
            wandb_name="explicit-test-name",
        ),
        resuming=True,
    )

    assert init_calls == [
        {
            "id": "stable-run-id",
            "resume": "allow",
            "project": "test-project",
            "name": "explicit-test-name",
        }
    ]


def test_init_wandb_resume_reports_the_recorded_init_not_the_current_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint_dir = train.epath.Path(tmp_path / "checkpoint")
    checkpoint_dir.mkdir()
    (checkpoint_dir / "wandb_id.txt").write_text("old-random-init-run\n")
    (checkpoint_dir / "weight_loader.json").write_text('{"type": "NoOpWeightLoader"}')
    updates = []
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(train.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "config", types.SimpleNamespace(update=lambda d, **kw: updates.append(d)))

    # The config has since moved this lineage's name to PaliGemma init.
    train.init_wandb(
        types.SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            project_name="test-project",
            wandb_name="run",
            weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        ),
        resuming=True,
    )

    assert updates == [{"weight_loader": {"type": "NoOpWeightLoader"}}]


def test_train_config_rejects_blank_explicit_wandb_name():
    config = dataclasses.replace(_DEBUG_CONFIG, exp_name="   ")

    with pytest.raises(ValueError, match="non-empty explicit W&B run name"):
        _ = config.checkpoint_dir


def test_validation_reduction_keeps_per_horizon_and_per_dimension_breakdowns():
    import jax.numpy as jnp
    import numpy as np

    # 3 validation batches; metric shapes as produced by validation_step.
    stacked = {
        "val/sample_mae": jnp.array([0.1, 0.2, 0.3]),
        "val/sample_mae_by_horizon": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "val/sample_mae_by_dimension": jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4),
    }
    out = train._mean_over_val_batches(stacked)
    assert np.ndim(out["val/sample_mae"]) == 0
    np.testing.assert_allclose(out["val/sample_mae"], 0.2, rtol=1e-6)
    np.testing.assert_allclose(out["val/sample_mae_by_horizon"], [3.0, 4.0])
    np.testing.assert_allclose(out["val/sample_mae_by_dimension"], [4.0, 5.0, 6.0, 7.0])


def test_physical_action_mask_drops_padding_dims_and_invalid_timesteps():
    import jax.numpy as jnp
    import numpy as np

    # 1 sample, 3 timesteps, model width 5; last timestep is a synthetic (invalid) future step.
    loss_mask = jnp.array([[[True] * 5, [True] * 5, [False] * 5]])
    semantic = jnp.array([[True, True, False]])  # 2 physical dims, 1 dataset padding dim
    out = np.asarray(train._physical_action_mask(loss_mask, semantic))
    expected = np.zeros((1, 3, 5), bool)
    expected[0, :2, :2] = True
    np.testing.assert_array_equal(out, expected)


def test_observation_from_dict_carries_action_semantic_mask():
    import numpy as np

    from openpi.models import model as _model

    data = {
        "image": {"base_0_rgb": np.zeros((1, 4, 4, 3), np.float32)},
        "image_mask": {"base_0_rgb": np.ones((1,), bool)},
        "state": np.zeros((1, 32), np.float32),
        "action_semantic_mask": np.array([[True] * 8 + [False] * 2]),
    }
    obs = _model.Observation.from_dict(data)
    assert obs.action_semantic_mask.shape == (1, 10)
    assert _model.Observation.from_dict({k: v for k, v in data.items() if k != "action_semantic_mask"}).action_semantic_mask is None
