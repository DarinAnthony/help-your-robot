"""pi0.5 LoRA fine-tuning configs for the help! tasks, for upstream openpi (TASK_SPEC.md §2-3, §7).

Written against Physical-Intelligence/openpi at ``OPENPI_COMMIT`` (the same LeRobot revision as
``scripts/to_lerobot.py``). mikulrai/openpi was considered for its LoRA-only checkpoints and validation loss, but it adds
a fourth image slot (``extra_0_rgb``) that is always embedded, masked or not: ~25-30% more compute per update for these
two-camera tasks. Instead the converter writes a held-out ``<repo>_val`` dataset (``scripts/openpi_run.py check``), and
the notebook keeps checkpoints on local disk and syncs the latest to Drive. Nothing here edits openpi: ``register()`` adds the configs to openpi's config table, after
which openpi's own ``scripts/compute_norm_stats.py`` and ``scripts/train.py`` run them by name, and its policy server
serves the checkpoints:

    import sys; sys.path.insert(0, "<help repo>")
    from helpsim import openpi_help; openpi_help.register()
    # then openpi's compute_norm_stats.main("help_shelf_pi05_lora"), train.main(get_config(...)), serve_policy

Dataset keys (``scripts/to_lerobot.py``) are repacked to the keys the sim client sends at inference:
``observation/exterior``, ``observation/wrist`` (uint8 HxWx3), ``observation/state``, ``prompt``.

Actions: the datasets store absolute joint targets; here the arm joints (and the drawer's base target) become deltas
from the state at the start of each chunk, with the gripper absolute, as openpi does for absolute-action datasets.
The output transform converts back, so the policy returns absolute targets.
"""

from __future__ import annotations

import dataclasses
import pathlib

import einops
import numpy as np

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
ACTION_DIM = {"shelf": 8, "drawer": 11}
# Arm joints delta, gripper absolute (shelf); then the base target delta (drawer).
DELTA_MASK = {"shelf": (7, -1), "drawer": (7, -1, 3)}
ACTION_HORIZON = 32


def _image(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


def _transforms():
    from openpi import transforms

    return transforms


@dataclasses.dataclass(frozen=True)
class HelpInputs:
    """Sim client / dataset dict -> pi0.5 inputs: the exterior camera as the base view, the wrist as the left wrist."""

    model_type: object

    def __call__(self, data: dict) -> dict:
        from openpi.models import model as _model

        base, wrist = _image(data["observation/exterior"]), _image(data["observation/wrist"])
        inputs = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_},
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class HelpOutputs:
    """pi0.5 outputs -> the task's action dimensions (the model pads to 32)."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


def _data_config_factory():
    from openpi.training import config as C

    @dataclasses.dataclass(frozen=True)
    class HelpDataConfig(C.DataConfigFactory):
        task: str = "shelf"

        def create(self, assets_dirs: pathlib.Path, model_config) -> C.DataConfig:
            T = _transforms()
            repack = T.Group(inputs=[T.RepackTransform({
                "observation/exterior": "observation.images.exterior",
                "observation/wrist": "observation.images.wrist",
                "observation/state": "observation.state",
                "actions": "actions",
                "prompt": "prompt",
            })])
            mask = T.make_bool_mask(*DELTA_MASK[self.task])
            data = T.Group(inputs=[HelpInputs(model_type=model_config.model_type)],
                           outputs=[HelpOutputs(action_dim=ACTION_DIM[self.task])])
            data = data.push(inputs=[T.DeltaActions(mask)], outputs=[T.AbsoluteActions(mask)])
            model = C.ModelTransformFactory()(model_config)
            return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
                                       repack_transforms=repack, data_transforms=data, model_transforms=model)

    return HelpDataConfig


def train_config(task: str, repo_id: str, steps: int = 10_000, batch_size: int = 32, name: str | None = None):
    """A pi0.5 LoRA fine-tune from pi05_base on one help! dataset (no EMA, as openpi's LoRA examples)."""
    from openpi.models import pi0_config
    from openpi.training import config as C
    from openpi.training import optimizer as O
    from openpi.training import weight_loaders as W

    def model():
        return pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, discrete_state_input=False,
                                    paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")

    return C.TrainConfig(
        name=name or f"help_{task}_pi05_lora",
        model=model(),
        data=_data_config_factory()(repo_id=repo_id, task=task, base_config=C.DataConfig(prompt_from_task=True)),
        weight_loader=W.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=model().get_freeze_filter(),
        ema_decay=None,
        # 500 warmup updates, fewer on short runs (optax needs decay_steps > warmup_steps).
        lr_schedule=O.CosineDecaySchedule(warmup_steps=min(500, steps // 10), peak_lr=5e-5,
                                          decay_steps=steps, decay_lr=5e-6),
        optimizer=O.AdamW(clip_gradient_norm=1.0),
        batch_size=batch_size,
        num_train_steps=steps,
        save_interval=1000,
        keep_period=5000,          # ~7 GB per pi0.5 checkpoint: the latest plus every 5000th
        wandb_enabled=False,
    )


def register(shelf_repo: str = "help/shelf_pi05", drawer_repo: str = "help/drawer_pi05", **kw) -> list[str]:
    """Add the help! configs to openpi's table so its scripts find them by name."""
    from openpi.training import config as C

    names = []
    for task, repo in (("shelf", shelf_repo), ("drawer", drawer_repo)):
        cfg = train_config(task, repo, **kw)
        C._CONFIGS_DICT[cfg.name] = cfg
        names.append(cfg.name)
    return names
