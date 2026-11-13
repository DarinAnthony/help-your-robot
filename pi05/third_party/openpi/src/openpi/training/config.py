"""Repository-owned native OpenPI experiment builders and name resolution."""

from collections.abc import Callable
import dataclasses
import difflib
from typing import Literal, TypeAlias

import flax.nnx as nnx
from molmo_spaces.paths import H5_DATA_DIR
from molmo_spaces.paths import PI_EMBODIMENT_ASSETS_DIR
from molmo_spaces.training import legacy_rlds_input_sampling as _legacy_sampling
from molmo_spaces.training import sep1_conditioning_contract as _sep1_contract
from molmo_spaces.training import sep6_camd8k_fullft_contract as _sep6_contract
from molmo_spaces.training import oer9_training_contract as _oer9_contract
from molmo_spaces.training import oct25_twin_gen3_pnp_contract as _oct25_twin_contract
from molmo_spaces.training import oct28_action_freq_contract as _oct28_freq_contract
from molmo_spaces.training import nov05_xsrc40_exclv2_training_contract as _nov05_xsrc40_contract
from molmo_spaces.training import nov06_xsrc40_exclv2_posquant_contract as _nov06_posquant_contract
from molmo_spaces.training import pop34fk_training_contract as _pop34fk_contract
from molmo_spaces.training import sep11_independent_oer8k_contract as _sep11_contract
from molmo_spaces.training import sep16_oer8k_center200_mixture_contract as _sep16_contract
from molmo_spaces.training import sep19_center200_urma51_noise_contract as _sep19_center_contract
from molmo_spaces.training import sep20_nativecentered_4k_contract as _sep20_contract
from molmo_spaces.training import sep21_native_mix6k_contract as _sep21_mix_contract
from molmo_spaces.training import sep22_dense2k_urma4_contract as _sep22_dense_contract
from molmo_spaces.training import sep23_clean2k_repeat100_mix_contract as _sep23_mix_contract
from molmo_spaces.training import sep24_repeat100_single_robot_contract as _sep24_single_contract
from molmo_spaces.training import sep14_reuse100_dense_contract as _sep14_dense_contract
from molmo_spaces.training import sep17_urma4_threeway_contract as _sep17_threeway_contract
from molmo_spaces.training import sep18_dense94_urma456_contract as _sep18_urma_contract
from molmo_spaces.training.oer_physical_validity import (
    oer_8095_exactrlds_union_clean_v2_ids,
    oer_8095_physical_valid_v2_ids,
    oer_8095_physical_valid_v2_tail_clean_v1_ids,
)
from molmo_spaces.training.oer_split_anchors import oer_hash8095_anchored_split
from molmo_spaces.training.r334_contract import R334_E9_TRAINING_EMBODIMENT_IDS
from molmo_spaces.training.r334_contract import R334_R81_TRAINING_EMBODIMENT_IDS
from molmo_spaces.training.remaining81_contract import REMAINING81_TRAINING_EMBODIMENT_IDS
from molmo_spaces.training.remaining81_contract import REMAINING81_TRAINING_VARIANTS
from pi05_integration import data_processing as _data_processing
from pi05_integration import input_config as _input_config
from pi05_integration import urma_config as _urma_config
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


from openpi.training.config_utils import (  # noqa: E402, F401, I001 - compatibility re-exports
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    FakeDataConfig,
    GroupFactory,
    LeRobotAlohaDataConfig,
    LeRobotDROIDDataConfig,
    LeRobotLiberoDataConfig,
    ModelTransformFactory,
    RLDSDroidDataConfig,
    RLDSMolmoSpacesDataConfig,
    SimpleDataConfig,
    TrainConfig,
)

_NINE_TRAINING_EMBODIMENT_IDS = (0, 1, 2, 10, 22, 23, 35, 42, 58)
_REMAINING81_RANDOM9_TRAINING_EMBODIMENT_IDS = (26, 31, 50, 52, 59, 65, 69, 87, 97)
_REMAINING81_RANDOM9_TRAINING_VARIANTS = (
    "v252",
    "v319",
    "v517",
    "v526",
    "v596",
    "v655",
    "v692",
    "v905",
    "v966",
)

_JULY14_N60_DATASET_REPO_ID = "molmospaces_9emb_n60_currentplanner_phasebalanced_perembnorm_terminal_exact"
_JULY14_N60_DATA_DIR_GPU = (
    "/iris/u/boai/data/"
    "rlds_20260714_smooth_9embodiments_h1_open_drawer_"
    "n60_currentplanner_seed277_terminal_action_exact"
)
_JULY14_N60_NORM_ASSETS = AssetsConfig(
    assets_dir=("gs://boai-iris-central2/pi05_norm_assets/n60_currentplanner_phasebalanced_perembnorm_v1"),
    asset_id=("molmospaces_9emb_n60_currentplanner_phasebalanced_perembnorm_id_terminal_exact"),
)


def july16_n60_h32_recipe(
    *,
    name: str,
    adaptation: Literal["lora", "full"],
    explicit_id: bool,
    embodiment_filter_ids: tuple[int, ...],
    batch_size: int,
    num_train_steps: int,
    peak_lr: float,
    decay_lr: float,
    fsdp_devices: int,
) -> TrainConfig:
    """Build one controlled cell in the 2026-07-16 N60 recipe study.

    All cells share H=32, the exact same data and per-embodiment normalization table,
    augmentation, optimizer, and checkpoint cadence. ``explicit_id=False`` removes both
    prompt-ID and one-hot-state routes but deliberately retains per-embodiment normalization.
    Filtering preserves the stable global embodiment IDs.
    """
    lora = adaptation == "lora"
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        discrete_state_input=True,
        augmentation_mode="shared_mild",
        max_token_len=224,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    input_config = _input_config.Pi05InputConfig(
        prompt=(_input_config.EmbodimentIdPrompt() if explicit_id else _input_config.NoPromptConditioning()),
        state=(
            _input_config.EmbodimentOneHotState(_NINE_TRAINING_EMBODIMENT_IDS)
            if explicit_id
            else _input_config.NoStateEncoding()
        ),
        normalization=_input_config.PerEmbodimentQuantileNorm(_NINE_TRAINING_EMBODIMENT_IDS),
    )
    return TrainConfig(
        name=name,
        model=model,
        data=RLDSMolmoSpacesDataConfig(
            repo_id=_JULY14_N60_DATASET_REPO_ID,
            assets=_JULY14_N60_NORM_ASSETS,
            rlds_data_dir=_JULY14_N60_DATA_DIR_GPU,
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
            input_config=input_config,
            post_close_sampling_weight=0.5,
            embodiment_filter_ids=embodiment_filter_ids,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir=(
            "gs://boai-iris-central2/pi05_ckpts"
            if name.startswith(("pi05_n60_r1_", "pi05_n60_r3_", "pi05_n60_r5_"))
            else "/iris/u/boai/pi05_ckpts_local"
        ),
        project_name="embodiment-scaling-manipulation",
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        keep_period=1,
        freeze_filter=model.get_freeze_filter() if lora else nnx.Nothing(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
            decay_lr=decay_lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=500,
        val_batches=8,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(interval=500, sample_count=9),
        save_interval=2_000,
        log_interval=50,
        num_workers=0,
    )


def july16_n60_h32_open_franka9emb_r1_lora() -> TrainConfig:
    return july16_n60_h32_recipe(
        name="pi05_n60_r1_exactid_lora_h32_b256_lr4e4_30k",
        adaptation="lora",
        explicit_id=True,
        embodiment_filter_ids=(),
        batch_size=256,
        num_train_steps=30_000,
        peak_lr=4e-4,
        decay_lr=2e-5,
        fsdp_devices=4,
    )


def july16_n60_h32_open_franka9emb_r2_fullft() -> TrainConfig:
    return july16_n60_h32_recipe(
        name="pi05_n60_r2_exactid_fullft_h32_b256_lr4e4_30k",
        adaptation="full",
        explicit_id=True,
        embodiment_filter_ids=(),
        batch_size=256,
        num_train_steps=30_000,
        peak_lr=4e-4,
        decay_lr=2e-5,
        fsdp_devices=4,
    )


def july16_n60_h32_open_franka2emb_r3_lora() -> TrainConfig:
    return july16_n60_h32_recipe(
        name="pi05_n60_r3_v013_v066_exactid_lora_h32_b64_lr1e4_20k",
        adaptation="lora",
        explicit_id=True,
        embodiment_filter_ids=(0, 2),
        batch_size=64,
        num_train_steps=20_000,
        peak_lr=1e-4,
        decay_lr=5e-6,
        fsdp_devices=4,
    )


def july16_n60_h32_open_frankav013_r4_lora() -> TrainConfig:
    return july16_n60_h32_recipe(
        name="pi05_n60_r4_v013_exactid_lora_h32_b64_lr1e4_20k",
        adaptation="lora",
        explicit_id=True,
        embodiment_filter_ids=(0,),
        batch_size=64,
        num_train_steps=20_000,
        peak_lr=1e-4,
        decay_lr=5e-6,
        fsdp_devices=1,
    )


def july16_n60_h32_open_franka2emb_r5_noid() -> TrainConfig:
    return july16_n60_h32_recipe(
        name="pi05_n60_r5_v013_v066_perembnorm_noexplicitid_lora_h32_b64_lr1e4_20k",
        adaptation="lora",
        explicit_id=False,
        embodiment_filter_ids=(0, 2),
        batch_size=64,
        num_train_steps=20_000,
        peak_lr=1e-4,
        decay_lr=5e-6,
        fsdp_devices=4,
    )


def july12_lowvarv2_h32_open_frankav013_lora_6k() -> TrainConfig:
    return TrainConfig(
        name="pi05_lowvar_v2_lora_h32",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="/iris/u/boai/data/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=6_000,
        batch_size=32,
        fsdp_devices=1,
        keep_period=500,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=1e-4, decay_steps=6_000, decay_lr=1e-4),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=500,
        log_interval=50,
        num_workers=0,
    )


def july13_lowvarv2_h32_open_frankav013_lora_20k() -> TrainConfig:
    return TrainConfig(
        name="pi05_lowvar_v2_lora_h32_20k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="gs://boai-iris-central2/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="gs://boai-iris-central2/pi05_ckpts",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=20_000,
        batch_size=128,
        fsdp_devices=4,
        # Only five checkpoints are emitted. Orbax applies keep_period to the raw zero-based
        # directory index (3999, 7999, ...), so keep every emitted checkpoint explicitly.
        keep_period=1,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=1e-4, decay_steps=20_000, decay_lr=1e-4),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=500,
        val_batches=2,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(interval=500, sample_count=4),
        save_interval=4_000,
        log_interval=50,
        num_workers=0,
    )


def july13_lowvarv2_h32_open_frankav013_fullft_20k() -> TrainConfig:
    return TrainConfig(
        # Controlled full-model counterpart to pi05_lowvar_v2_lora_h32_20k. Keep the
        # data, horizon, augmentation, optimizer schedule, logging, and checkpoint cadence
        # fixed; only adaptation and the single-GPU memory batch differ.
        name="pi05_lowvar_v2_fullft_h32_20k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="/iris/u/boai/data/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=20_000,
        # A prior full-model lowvar run established batch 32 on one H200. LoRA fits
        # batch 128, but retaining that batch while unfreezing 3.4B parameters does not.
        batch_size=32,
        fsdp_devices=1,
        keep_period=1,
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=1e-4, decay_steps=20_000, decay_lr=1e-4),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=500,
        val_batches=2,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(interval=500, sample_count=4),
        save_interval=4_000,
        log_interval=50,
        num_workers=0,
    )


def july11_lowvarv2_h50_open_frankav013_lora_lr5e4_20k() -> TrainConfig:
    return TrainConfig(
        name="july11_lowvarv2_h50_open_frankav013_lora_lr5e4_20k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="/iris/u/boai/data/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=20_000,
        batch_size=128,
        fsdp_devices=1,
        keep_period=2_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-4, decay_steps=20_000, decay_lr=1e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=2_000,
        log_interval=100,
        num_workers=0,
    )


def july11_lowvarv2_h32_open_frankav013_fullft_2k() -> TrainConfig:
    return TrainConfig(
        name="july11_lowvarv2_h32_open_frankav013_fullft_2k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32, discrete_state_input=True),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="/iris/u/boai/data/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=2_000,
        batch_size=32,
        fsdp_devices=1,
        keep_period=250,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=5e-5, decay_steps=2_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=250,
        log_interval=50,
        num_workers=0,
    )


def july11_lowvarv2_h50_open_frankav013_lora_4k() -> TrainConfig:
    return TrainConfig(
        name="july11_lowvarv2_h50_open_frankav013_lora_4k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="/iris/u/boai/data/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=4_000,
        batch_size=128,
        fsdp_devices=1,
        keep_period=500,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=1e-4, decay_steps=6_000, decay_lr=1e-4),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=500,
        log_interval=50,
        num_workers=0,
    )


def july14_9emb_n60_idtext_state_fullft_h32_40k_open_franka() -> TrainConfig:
    return TrainConfig(
        # Performance-oriented nine-seen-embodiment recipe. It carries identity
        # through both a compact prompt descriptor and a direct one-hot code in
        # Pi0.5's discrete-state prefix. The latter is appended after physical
        # state normalization, so morphology identity cannot corrupt proprio stats.
        name="pi05_9emb_n60_idtext_state_fullft_h32_40k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
            max_token_len=224,
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_9emb_n60_terminal_exact",
            rlds_data_dir=(
                "/iris/u/boai/data/"
                "rlds_20260713_smooth_9embodiments_h1_open_drawer_"
                "n60_mixedplanner_terminal_action_exact"
            ),
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
            input_config=_input_config.Pi05InputConfig(
                prompt=_input_config.MorphologyTextPrompt(include_id=True),
                state=_input_config.EmbodimentOneHotState(_NINE_TRAINING_EMBODIMENT_IDS),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=40_000,
        batch_size=64,
        fsdp_devices=2,
        keep_period=1,
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=1e-4, decay_steps=40_000, decay_lr=1e-4),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=500,
        val_batches=8,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(interval=500, sample_count=9),
        # Pi checkpoint selection is standardized at 4k-update resolution: 3999, 7999, ...
        save_interval=4_000,
        log_interval=50,
        num_workers=0,
    )


def july11_lowvarv2_h50_open_frankav013_fullft_2k() -> TrainConfig:
    return TrainConfig(
        name="july11_lowvarv2_h50_open_frankav013_fullft_2k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_lowvar_v2",
            rlds_data_dir="/iris/u/boai/data/rlds_sanity_lowvar_v2",
            datasets=(droid_rlds_dataset.RLDSDataset(name="molmospaces_emb_mix", version="1.0.0", weight=1.0),),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=2_000,
        batch_size=32,
        fsdp_devices=1,
        keep_period=250,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-5, decay_steps=6_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=250,
        log_interval=50,
        num_workers=0,
    )


_CONFIG_BUILDERS = (
    july16_n60_h32_open_franka9emb_r1_lora,
    july16_n60_h32_open_franka9emb_r2_fullft,
    july16_n60_h32_open_franka2emb_r3_lora,
    july16_n60_h32_open_frankav013_r4_lora,
    july16_n60_h32_open_franka2emb_r5_noid,
    july12_lowvarv2_h32_open_frankav013_lora_6k,
    july13_lowvarv2_h32_open_frankav013_lora_20k,
    july13_lowvarv2_h32_open_frankav013_fullft_20k,
    july11_lowvarv2_h50_open_frankav013_lora_lr5e4_20k,
    july11_lowvarv2_h32_open_frankav013_fullft_2k,
    july11_lowvarv2_h50_open_frankav013_lora_4k,
    july14_9emb_n60_idtext_state_fullft_h32_40k_open_franka,
    july11_lowvarv2_h50_open_frankav013_fullft_2k,
)


def _clean_pose_lora_native_config(
    name: str,
    *,
    repo_id: str = "molmospaces_clean_exactpose_v013_n100",
    rlds_data_dir: str = "/iris/u/boai/data/rlds_20260716_clean_exactpose_v013_n100",
    input_config: _input_config.Pi05InputConfig | None = None,
    action_expert_conditioning: _input_config.ActionExpertConditioning | None = None,
    backbone_conditioning: _input_config.BackboneConditioning | None = None,
    num_train_steps: int = 20_000,
    embodiment_filter_ids: tuple[int, ...] = (),
    dataset_version: str = "1.0.0",
    episode_selector: _data_processing.EpisodeSelector | None = None,
) -> TrainConfig:
    """Native shell for repository-owned clean-pose experiment variants."""
    resolved_action_expert_conditioning = (
        _input_config.TimestepOnly() if action_expert_conditioning is None else action_expert_conditioning
    )
    resolved_backbone_conditioning = (
        _input_config.NoBackboneConditioning()
        if backbone_conditioning is None
        else backbone_conditioning
    )
    resolved_input_config = input_config or _input_config.Pi05InputConfig()
    missing_regex = (
        r"(.*lora.*|embodiment_backbone_embed/.*)"
        if isinstance(
            resolved_backbone_conditioning,
            _input_config.LearnedEmbodimentBackboneToken,
        )
        else (
            r"(.*lora.*|embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)"
            if isinstance(
                resolved_backbone_conditioning,
                _input_config.ExtractedEmbodimentBackboneToken,
            )
            else (
                r"(.*lora.*|embodiment_encoder/.*|embodiment_fusion_(in|out)/.*|embodiment_fusion_gate)"
                if isinstance(
                    resolved_action_expert_conditioning,
                    _input_config.ExtractedEmbodimentAdaRMS,
                )
                else (
                    r"(.*lora.*|embodiment_embed/.*)"
                    if resolved_action_expert_conditioning.embodiment_vocab_size
                    else ".*lora.*"
                )
            )
        )
    )
    if resolved_input_config.response_history.enabled:
        missing_regex = rf"(?:{missing_regex}|response_history_(?:norm|proj)/.*)"
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        max_token_len=(
            resolved_input_config.prompt.max_token_len
            if resolved_input_config.prompt.fail_on_token_overflow
            else None
        ),
        discrete_state_input=True,
        augmentation_mode="shared_mild",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_expert_conditioning=resolved_action_expert_conditioning,
        backbone_conditioning=resolved_backbone_conditioning,
        response_history_conditioning=resolved_input_config.response_history,
    )
    return TrainConfig(
        name=name,
        model=model,
        data=RLDSMolmoSpacesDataConfig(
            repo_id=repo_id,
            rlds_data_dir=rlds_data_dir,
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version=dataset_version,
                    weight=1.0,
                ),
            ),
            input_config=resolved_input_config,
            embodiment_filter_ids=embodiment_filter_ids,
            episode_selector=episode_selector,
            rlds_split="train",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=missing_regex,
            adarms_condition_width_multiplier=(resolved_action_expert_conditioning.condition_width_multiplier),
            adarms_condition_extra_width=(
                resolved_action_expert_conditioning.extractor.output_dim
                if isinstance(
                    resolved_action_expert_conditioning,
                    _input_config.ExtractedEmbodimentAdaRMS,
                )
                and resolved_action_expert_conditioning.fusion == "concat"
                else 0
            ),
        ),
        checkpoint_base_dir="/iris/u/boai/pi05_ckpts_local",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=num_train_steps,
        batch_size=128,
        fsdp_devices=1,
        keep_period=1,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=1e-4,
            decay_steps=num_train_steps,
            decay_lr=1e-4,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=0,
        val_batches=0,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(interval=500, sample_count=4),
        save_interval=4_000,
        log_interval=50,
        num_workers=0,
    )


def _july18_fixcap_warmup_ema_base(
    name: str,
    *,
    repo_id: str,
    rlds_data_dir: str,
    input_config: _input_config.Pi05InputConfig,
    num_train_steps: int,
    action_expert_conditioning: _input_config.ActionExpertConditioning | None = None,
    backbone_conditioning: _input_config.BackboneConditioning | None = None,
    batch_size: int = 128,
    embodiment_filter_ids: tuple[int, ...] = (),
    dataset_version: str = "1.0.0",
    episode_selector: _data_processing.EpisodeSelector | None = None,
) -> TrainConfig:
    """Shared native recipe for the distinct July 18 warmup/EMA training round."""
    lr_scale = batch_size / 128
    base = _clean_pose_lora_native_config(
        name,
        repo_id=repo_id,
        rlds_data_dir=rlds_data_dir,
        input_config=input_config,
        action_expert_conditioning=action_expert_conditioning,
        backbone_conditioning=backbone_conditioning,
        num_train_steps=num_train_steps,
        embodiment_filter_ids=embodiment_filter_ids,
        dataset_version=dataset_version,
        episode_selector=episode_selector,
    )
    return dataclasses.replace(
        base,
        batch_size=batch_size,
        ema_decay=0.999,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5 * lr_scale,
            decay_steps=num_train_steps,
            decay_lr=1e-5 * lr_scale,
        ),
        optimizer=_optimizer.AdamW(
            weight_decay=0.0,
            clip_gradient_norm=1.0,
        ),
        data_visualization=_input_config.DataVisualizationConfig(interval=1_000, sample_count=4),
    )


def july16_cleanpose_h32_open_frankav013_10p10e() -> TrainConfig:
    return _clean_pose_lora_native_config("pi05_cleanpose_v013_10p10e_lora_h32_20k")


def july16_cleanpose_h32_open_frankav013_5p10e() -> TrainConfig:
    return _clean_pose_lora_native_config("pi05_cleanpose_v013_5p10e_lora_h32_20k")


def july22_cleanpose_h32_open_frankav013_5p5e() -> TrainConfig:
    return _clean_pose_lora_native_config("pi05_cleanpose_v013_5p5e_lora_h32_20k")


def july16_cleanpose_h32_open_frankav013_10p5e() -> TrainConfig:
    return _clean_pose_lora_native_config("pi05_cleanpose_v013_10p5e_lora_h32_20k")


def july18_fixcap_h32_open_frankav013() -> TrainConfig:
    return _clean_pose_lora_native_config(
        "pi05_fix_cap_pause_v013_10p10e_lora_h32_constlr20k",
        repo_id="fix_cap_pause_v013_10p10e_20260718",
        rlds_data_dir="/iris/u/boai/data/20260718_fix_cap_pause_v013_10p10e",
    )


def july18_fixcap_h32_open_frankav066() -> TrainConfig:
    return _clean_pose_lora_native_config(
        "pi05_fix_cap_pause_v066_10p10e_lora_h32_constlr20k",
        repo_id="fix_cap_pause_v066_10p10e_20260718",
        rlds_data_dir="/iris/u/boai/data/20260718_fix_cap_pause_v066_10p10e",
    )


def july18_fixcap_h32_open_franka2emb() -> TrainConfig:
    return _clean_pose_lora_native_config(
        "pi05_fix_cap_pause_v013_v066_varianttext_lora_h32_constlr40k",
        repo_id="fix_cap_pause_2emb_10p10e_20260718",
        rlds_data_dir="/iris/u/boai/data/20260718_fix_cap_pause_2emb_10p10e",
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm((0, 2)),
        ),
        num_train_steps=40_000,
    )


def july18_fixcap_warmupema_h32_open_frankav066_16k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july18_fixcap_warmupema_h32_open_frankav066_16k",
        repo_id="fix_cap_pause_v066_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_v066_10p10e"),
        input_config=_input_config.Pi05InputConfig(),
        num_train_steps=16_000,
    )


def july18_fixcap_warmupema_h32_open_franka2emb_16k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july18_fixcap_warmupema_h32_open_franka2emb_16k",
        repo_id="fix_cap_pause_2emb_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_2emb_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm((0, 2)),
        ),
        num_train_steps=16_000,
    )


def july19_fixcap_warmupema_h32_open_frankav066_20k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_warmupema_h32_open_frankav066_20k",
        repo_id="fix_cap_pause_v066_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_v066_10p10e"),
        input_config=_input_config.Pi05InputConfig(),
        num_train_steps=20_000,
    )


def july19_fixcap_warmupema_h32_open_franka2emb_20k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_warmupema_h32_open_franka2emb_20k",
        repo_id="fix_cap_pause_2emb_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_2emb_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm((0, 2)),
        ),
        num_train_steps=20_000,
    )


def july19_fixcap_warmupema_h32_open_franka2emb_b128_32k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_warmupema_h32_open_franka2emb_b128_32k",
        repo_id="fix_cap_pause_2emb_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_2emb_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm((0, 2)),
        ),
        num_train_steps=32_000,
        batch_size=128,
    )


def july19_fixcap_warmupema_h32_open_franka2emb_b256_20k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_warmupema_h32_open_franka2emb_b256_20k",
        repo_id="fix_cap_pause_2emb_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_2emb_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm((0, 2)),
        ),
        num_train_steps=20_000,
        batch_size=256,
    )


def july18_fixcap_warmupema_h32_open_franka9emb_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july18_fixcap_warmupema_h32_open_franka9emb_64k",
        repo_id="fix_cap_pause_9emb_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_9emb_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm(_NINE_TRAINING_EMBODIMENT_IDS),
        ),
        num_train_steps=64_000,
        batch_size=256,
    )


def july19_fixcap_warmupema_h32_open_franka_remaining81_b2048_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_warmupema_h32_open_franka_remaining81_b2048_64k",
        repo_id="fix_cap_pause_remaining81_10p10e_20260719",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_remaining81_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(
                allowed_variants=REMAINING81_TRAINING_VARIANTS,
            ),
            normalization=_input_config.PerEmbodimentQuantileNorm(REMAINING81_TRAINING_EMBODIMENT_IDS),
        ),
        num_train_steps=64_000,
        # The shared helper scales LR linearly with batch: this resolves to
        # 8e-4 peak / 1.6e-4 decay, exactly 8x the batch-256 E9 recipe.
        batch_size=2_048,
    )


def july20_fixcap_warmupema_h32_open_franka_remaining81_b1024_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july20_fixcap_warmupema_h32_open_franka_remaining81_b1024_64k",
        repo_id="fix_cap_pause_remaining81_10p10e_20260719",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_remaining81_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(
                allowed_variants=REMAINING81_TRAINING_VARIANTS,
            ),
            normalization=_input_config.PerEmbodimentQuantileNorm(REMAINING81_TRAINING_EMBODIMENT_IDS),
        ),
        num_train_steps=64_000,
        # Four times the batch-256 conditioned E9 recipe. The shared helper
        # resolves this to a 4e-4 peak LR and 8e-5 decay LR.
        batch_size=1_024,
    )


def july21_fixcap_remaining81_varianttext_heldout9_inference() -> TrainConfig:
    """Restore the Full81 text checkpoint while permitting held-out Original9 labels.

    This is inference-only: model structure, dataset identity, and the immutable
    Remaining81 normalization bundle match the training config exactly. The evaluation
    server explicitly derives the requested equal-robot average normalization from that
    bundle rather than altering or replacing the checkpoint asset.
    """

    return _july18_fixcap_warmup_ema_base(
        "july21_fixcap_remaining81_varianttext_heldout9_inference",
        repo_id="fix_cap_pause_remaining81_10p10e_20260719",
        rlds_data_dir="/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_remaining81_10p10e",
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(),
            normalization=_input_config.PerEmbodimentQuantileNorm(
                REMAINING81_TRAINING_EMBODIMENT_IDS
            ),
        ),
        num_train_steps=64_000,
        batch_size=1_024,
    )


def july20_fixcap_warmupema_h32_open_franka_remaining81_noembtext_b1024_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july20_fixcap_warmupema_h32_open_franka_remaining81_noembtext_b1024_64k",
        repo_id="fix_cap_pause_remaining81_10p10e_20260719",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_remaining81_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.PerEmbodimentQuantileNorm(REMAINING81_TRAINING_EMBODIMENT_IDS),
        ),
        num_train_steps=64_000,
        batch_size=1_024,
    )


def july21_fixcap_warmupema_h32_open_franka_remaining81_noembtext_globalnorm_b1024_64k() -> (
    TrainConfig
):
    """No-prompt Full81 control with pooled all-training-embodiment statistics."""

    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                "july21_fixcap_warmupema_h32_open_franka_remaining81_"
                "noembtext_globalnorm_b1024_64k"
            ),
            repo_id="fix_cap_pause_remaining81_10p10e_20260719",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260718_fix_cap_pause_remaining81_10p10e"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            num_train_steps=64_000,
            batch_size=1_024,
            dataset_version="1.1.0",
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def _july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm(
    batch_size: int,
) -> TrainConfig:
    """Near-example-matched 334-robot global-normalization scaling treatment."""
    name = (
        "july24_fixcap_warmupema_h32_open_franka334emb_5p5e_"
        f"noembtext_globalnorm_b{batch_size}_64k"
    )
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            name,
            repo_id="fix_cap_pause_r81_5p5e_v3paused253_5p5e_334emb_20260724",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260724_fixcap_r81_5p5e_v3paused253_5p5e_334emb"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            num_train_steps=64_000,
            batch_size=batch_size,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm_b1024_64k() -> (
    TrainConfig
):
    return _july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm(
        1_024
    )


def july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm_b2048_64k() -> (
    TrainConfig
):
    return _july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm(
        2_048
    )


def july26_fixcap_warmupema_h32_open_r334_exact8095_noembtext_globalnorm_b1024_64k() -> (
    TrainConfig
):
    """Exact-episode-budget R334 treatment matched to the Full81 optimizer contract."""

    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                "july26_fixcap_warmupema_h32_open_r334_exact8095_"
                "noembtext_globalnorm_b1024_64k"
            ),
            repo_id="fix_cap_r334_exact8095_outcomeblind_20260726",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260726_fixcap_r334_exact8095_outcomeblind"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            num_train_steps=64_000,
            batch_size=1_024,
            dataset_version="1.1.0",
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july29_fixcap_warmupema_h32_open_r334_exact8095_noembtext_banknorm_b1024_64k() -> (
    TrainConfig
):
    """Exact-budget R334 treatment with separate frame-pooled R81 and E9 tables."""

    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                "july29_fixcap_warmupema_h32_open_r334_exact8095_"
                "noembtext_banknorm_b1024_64k"
            ),
            repo_id="fix_cap_r334_exact8095_outcomeblind_20260726",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260726_fixcap_r334_exact8095_outcomeblind"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GroupedQuantileNorm(
                    (
                        _input_config.NormalizationGroup(
                            "r81", R334_R81_TRAINING_EMBODIMENT_IDS
                        ),
                        _input_config.NormalizationGroup(
                            "e9", R334_E9_TRAINING_EMBODIMENT_IDS
                        ),
                    )
                ),
            ),
            num_train_steps=64_000,
            batch_size=1_024,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july29_fixcap_warmupema_h32_open_r334_r81hom_exact8095_noembtext_globalnorm_b1024_64k() -> (
    TrainConfig
):
    """Homogeneous-R81-view exact-budget treatment matched to Full81."""

    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                "july29_fixcap_warmupema_h32_open_r334_r81hom_exact8095_"
                "noembtext_globalnorm_b1024_64k"
            ),
            repo_id="fix_cap_r334_r81hom_exact8095_outcomeblind_20260729",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260729_fixcap_r334_r81hom_exact8095_outcomeblind"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            num_train_steps=64_000,
            batch_size=1_024,
            dataset_version="1.1.0",
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def _july27_oer_scaling_noembtext_globalnorm(episode_count: int) -> TrainConfig:
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                f"july27_oer_hash{episode_count}_noembtext_"
                "globalnorm_b1024_64k"
            ),
            repo_id="oer_quality_hold_50k_20260727",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260727_oer_quality_hold_50k_v1"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            num_train_steps=64_000,
            batch_size=1_024,
            dataset_version="1.1.0",
            episode_selector=_data_processing.HashEpisodeSelector(
                episode_count=episode_count,
                seed=20260727,
            ),
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july27_oer_hash8095_noembtext_globalnorm_b1024_64k() -> TrainConfig:
    return _july27_oer_scaling_noembtext_globalnorm(8_095)


def july27_oer_hash25000_noembtext_globalnorm_b1024_64k() -> TrainConfig:
    return _july27_oer_scaling_noembtext_globalnorm(25_000)


def july27_oer_hash50000_noembtext_globalnorm_b1024_64k() -> TrainConfig:
    return _july27_oer_scaling_noembtext_globalnorm(50_000)


def _july28_oer_scaling_clip5_val5(episode_count: int) -> TrainConfig:
    base = _july27_oer_scaling_noembtext_globalnorm(episode_count)
    return dataclasses.replace(
        base,
        name=(
            f"july28_oer_hash{episode_count}_noembtext_"
            "globalnorm_clip5_val5_b1024_64k"
        ),
        data=dataclasses.replace(
            base.data,
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
            ),
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=20260728,
                require_one_episode_per_embodiment=True,
            ),
        ),
        val_interval=1_000,
        val_batches=2,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000,
            sample_count=4,
        ),
    )


def july28_oer_hash8095_noembtext_globalnorm_clip5_val5_b1024_64k() -> TrainConfig:
    return _july28_oer_scaling_clip5_val5(8_095)


def july28_oer_hash25000_noembtext_globalnorm_clip5_val5_b1024_64k() -> TrainConfig:
    return _july28_oer_scaling_clip5_val5(25_000)


def july28_oer_hash50000_noembtext_globalnorm_clip5_val5_b1024_64k() -> TrainConfig:
    return _july28_oer_scaling_clip5_val5(50_000)


def july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k() -> TrainConfig:
    base = _july28_oer_scaling_clip5_val5(8_095)
    return dataclasses.replace(
        base,
        name=(
            "july28_oer_hash8095_noembtext_"
            "stdnorm_clip5_val5_b1024_64k"
        ),
        data=dataclasses.replace(
            base.data,
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalStandardNorm(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
            ),
        ),
    )


def august30_oer_hash8095_noembtext_compactstate_stdnorm_clip5_val5_b1024_64k() -> TrainConfig:
    """Corrected compact-state and full-dimensional Pi0.5 objective recipe."""
    base = july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k()
    return dataclasses.replace(
        base,
        name=(
            "august30_oer_hash8095_noembtext_compactstate_"
            "stdnorm_clip5_val5_b1024_64k"
        ),
        seed=0,
        data=dataclasses.replace(
            base.data,
            input_config=dataclasses.replace(
                base.data.input_config,
                state_serialization=_input_config.CompactLeftPackedState(),
            ),
        ),
    )


def _august24_oer_phase0_bal6p5e_id261_nocond_lora(
    *, name: str, batch_size: int
) -> TrainConfig:
    if batch_size not in {512, 1_024}:
        raise ValueError("balanced 6x5 LoRA supports batch 512 or 1024")
    scaled = 1_024 // batch_size
    base = july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k()
    return dataclasses.replace(
        base,
        name=name,
        data=dataclasses.replace(
            base.data,
            repo_id=(
                "20260824_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_"
                "two_view_retainedhold_r1"
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1.0,
                    data_subdir=".",
                    joint_layout="cross_family_arm7_gripper7_v1",
                ),
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalStandardNorm(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
            ),
            episode_selector=None,
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=20260809,
                require_one_episode_per_embodiment=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params"
        ),
        num_train_steps=48_000 * scaled,
        batch_size=batch_size,
        fsdp_devices=64 if batch_size == 1_024 else 32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            initial_lr=1e-8 / scaled,
            warmup_steps=1_000 * scaled,
            peak_lr=8e-5 / scaled,
            decay_steps=48_000 * scaled,
            decay_lr=1e-5 / scaled,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000 * scaled,
        val_interval=1_000 * scaled,
        val_batches=2,
        log_interval=50 * scaled,
        train_sample_interval=1_000 * scaled,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000 * scaled,
            sample_count=4,
        ),
        seed=0,
    )


def august24_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_nocond_seed0_lora_stdnorm_clip5_b512_96k() -> TrainConfig:
    return _august24_oer_phase0_bal6p5e_id261_nocond_lora(
        name=(
            "august24_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_nocond_seed0_"
            "lora_stdnorm_clip5_b512_96k"
        ),
        batch_size=512,
    )


def august24_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_nocond_seed0_lora_stdnorm_clip5_b1024_48k() -> TrainConfig:
    return _august24_oer_phase0_bal6p5e_id261_nocond_lora(
        name=(
            "august24_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_nocond_seed0_"
            "lora_stdnorm_clip5_b1024_48k"
        ),
        batch_size=1_024,
    )


def july29_oer_hash8095_physvalidv2_noembtext_globalnorm_clip5_val5_b1024_64k() -> (
    TrainConfig
):
    base = _july28_oer_scaling_clip5_val5(8_095)
    return dataclasses.replace(
        base,
        name=(
            "july29_oer_hash8095_physvalidv2_noembtext_"
            "globalnorm_clip5_val5_b1024_64k"
        ),
        data=dataclasses.replace(
            base.data,
            embodiment_filter_ids=oer_8095_physical_valid_v2_ids(),
        ),
    )


def july29_oer_hash8095_physvalidv2_tailcleanv1_noembtext_globalnorm_clip5_val5_b1024_64k() -> (
    TrainConfig
):
    base = _july28_oer_scaling_clip5_val5(8_095)
    return dataclasses.replace(
        base,
        name=(
            "july29_oer_hash8095_physvalidv2_tailcleanv1_noembtext_"
            "globalnorm_clip5_val5_b1024_64k"
        ),
        data=dataclasses.replace(
            base.data,
            embodiment_filter_ids=(
                oer_8095_physical_valid_v2_tail_clean_v1_ids()
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_ckpts/"
            "july29_oer_hash8095_physvalidv2_noembtext_globalnorm_clip5_val5_"
            "b1024_64k/"
            "20260729_oer_hash8095_physvalidv2_quantile_clip5_val5_h32_"
            "b1024_v4-64/1999/params"
        ),
    )


def july28_oer_hash8095_noembtext_urma2_concatproj1024_globalnorm_clip5_val5_b1024_64k() -> (
    TrainConfig
):
    base = _july28_oer_scaling_clip5_val5(8_095)
    conditioning = _urma_config.franka_oer_hash8095_urma2_concat_project_adarms()
    return dataclasses.replace(
        base,
        name=(
            "july28_oer_hash8095_noembtext_urma2_concatproj1024_"
            "globalnorm_clip5_val5_b1024_64k"
        ),
        model=dataclasses.replace(
            base.model,
            action_expert_conditioning=conditioning,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=(
                r"(.*lora.*|embodiment_encoder/.*|"
                r"embodiment_fusion_(in|out)/.*)"
            ),
        ),
    )


def july28_oer_hash8095_noembtext_urma3_concatproj1024_globalnorm_clip5_val5_b1024_64k() -> (
    TrainConfig
):
    base = _july28_oer_scaling_clip5_val5(8_095)
    conditioning = _urma_config.franka_oer_hash8095_urma3_concat_project_adarms()
    return dataclasses.replace(
        base,
        name=(
            "july28_oer_hash8095_noembtext_urma3_concatproj1024_"
            "globalnorm_clip5_val5_b1024_64k"
        ),
        model=dataclasses.replace(
            base.model,
            action_expert_conditioning=conditioning,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=(
                r"(.*lora.*|embodiment_encoder/.*|"
                r"embodiment_fusion_(in|out)/.*)"
            ),
        ),
    )


def july30_oer_hash8095_quantile_clip5_val5_b1024_16k_continue() -> TrainConfig:
    """Named 16K continuation of the July 28 quantile checkpoint lineage."""
    return dataclasses.replace(
        july28_oer_hash8095_noembtext_globalnorm_clip5_val5_b1024_64k(),
        name="july30_oer_hash8095_quantile_clip5_val5_b1024_16k_continue",
        num_train_steps=16_000,
    )


def july30_oer_hash8095_standard_clip5_val5_b1024_16k_continue() -> TrainConfig:
    """Named 16K continuation of the July 28 standard checkpoint lineage."""
    return dataclasses.replace(
        july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k(),
        name="july30_oer_hash8095_standard_clip5_val5_b1024_16k_continue",
        num_train_steps=16_000,
    )


def july30_oer_hash8095_urma3_quantile_clip5_val5_b1024_16k_continue() -> (
    TrainConfig
):
    """Named 16K continuation of the July 28 URMA3 checkpoint lineage."""
    return dataclasses.replace(
        july28_oer_hash8095_noembtext_urma3_concatproj1024_globalnorm_clip5_val5_b1024_64k(),
        name=(
            "july30_oer_hash8095_urma3_quantile_clip5_val5_"
            "b1024_16k_continue"
        ),
        num_train_steps=16_000,
    )


def july30_oer_hash8095_exactrlds_unioncleanv2_noembtext_globalnorm_clip5_val5_b1024_16k_warmstart1999() -> (
    TrainConfig
):
    """Exact clean view with a params-only checkpoint-1,999 warm start."""
    base = _july28_oer_scaling_clip5_val5(8_095)
    return dataclasses.replace(
        base,
        name=(
            "july30_oer_hash8095_exactrlds_unioncleanv2_noembtext_"
            "globalnorm_clip5_val5_b1024_16k_warmstart1999"
        ),
        data=dataclasses.replace(
            base.data,
            embodiment_filter_ids=oer_8095_exactrlds_union_clean_v2_ids(),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_ckpts/"
            "july29_oer_hash8095_physvalidv2_noembtext_globalnorm_clip5_val5_"
            "b1024_64k/"
            "20260729_oer_hash8095_physvalidv2_quantile_clip5_val5_h32_"
            "b1024_v4-64/1999/params"
        ),
        num_train_steps=16_000,
    )


def july30_oer_hash8095_urma3_standard_clip5_val5_b1024_16k() -> (
    TrainConfig
):
    """URMA3 AdaRMS with pooled-global mean/std scaling and clip +/-5."""
    base = (
        july28_oer_hash8095_noembtext_urma3_concatproj1024_globalnorm_clip5_val5_b1024_64k()
    )
    return dataclasses.replace(
        base,
        name=(
            "july30_oer_hash8095_urma3_standard_clip5_val5_"
            "b1024_16k"
        ),
        data=dataclasses.replace(
            base.data,
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalStandardNorm(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
            ),
        ),
        num_train_steps=16_000,
    )


def _july30_oer_scale_standard_clip5_16k(episode_count: int) -> TrainConfig:
    if episode_count == 50_000:
        repo_id = "oer_quality_hold_50k_20260727"
        datasets = (
            droid_rlds_dataset.RLDSDataset(
                name="molmospaces_emb_mix",
                version="1.1.0",
                weight=1.0,
                data_subdir="20260727_oer_quality_hold_50k_v1",
            ),
        )
    elif episode_count == 100_000:
        repo_id = "oer_quality_hold_100k_union_20260727"
        datasets = (
            droid_rlds_dataset.RLDSDataset(
                name="molmospaces_emb_mix",
                version="1.1.0",
                weight=0.5,
                data_subdir="20260727_oer_quality_hold_50k_v1",
            ),
            droid_rlds_dataset.RLDSDataset(
                name="molmospaces_emb_mix",
                version="1.1.0",
                weight=0.5,
                data_subdir="20260727_oer_quality_hold_50k_v1_p2",
            ),
        )
    else:
        raise ValueError(f"unsupported OER scale population: {episode_count}")
    name = (
        f"july30_oer_full{episode_count}_standard_clip5_val5_"
        "b1024_16k"
    )
    base = _july18_fixcap_warmup_ema_base(
        name,
        repo_id=repo_id,
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=16_000,
        batch_size=1_024,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=datasets,
            episode_split=oer_hash8095_anchored_split(),
        ),
        save_interval=8_000,
        keep_period=None,
        keep_every_n_completed_updates=8_000,
        val_interval=1_000,
        val_batches=2,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000,
            sample_count=4,
        ),
    )


def july30_oer_full50000_standard_clip5_val5_b1024_16k() -> TrainConfig:
    return _july30_oer_scale_standard_clip5_16k(50_000)


def july30_oer_full100000_standard_clip5_val5_b1024_16k() -> TrainConfig:
    return _july30_oer_scale_standard_clip5_16k(100_000)


def july21_fixcap_warmupema_h32_open_franka_remaining81_adarms_only_b1024_64k() -> TrainConfig:
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            "july21_fixcap_warmupema_h32_open_franka_remaining81_adarms_only_b1024_64k",
            repo_id="fix_cap_pause_remaining81_10p10e_20260719",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260718_fix_cap_pause_remaining81_10p10e"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.PerEmbodimentQuantileNorm(
                    REMAINING81_TRAINING_EMBODIMENT_IDS
                ),
            ),
            action_expert_conditioning=_input_config.LearnedEmbodimentAdaRMS(vocab_size=100),
            num_train_steps=64_000,
            batch_size=1_024,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july19_fixcap_r81random1_warmupema_h32_open_frankav517_b128_20k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_r81random1_warmupema_h32_open_frankav517_b128_20k",
        repo_id="fix_cap_pause_remaining81_10p10e_20260719",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_remaining81_10p10e"),
        input_config=_input_config.Pi05InputConfig(),
        num_train_steps=20_000,
        batch_size=128,
        embodiment_filter_ids=(50,),
    )


def july19_fixcap_r81random9_warmupema_h32_open_franka9emb_b256_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july19_fixcap_r81random9_warmupema_h32_open_franka9emb_b256_64k",
        repo_id="fix_cap_pause_remaining81_10p10e_20260719",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_remaining81_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            prompt=_input_config.EmbodimentVariantPrompt(
                allowed_variants=_REMAINING81_RANDOM9_TRAINING_VARIANTS,
                robot_family="franka",
            ),
            normalization=_input_config.PerEmbodimentQuantileNorm(_REMAINING81_RANDOM9_TRAINING_EMBODIMENT_IDS),
        ),
        num_train_steps=64_000,
        batch_size=256,
        embodiment_filter_ids=_REMAINING81_RANDOM9_TRAINING_EMBODIMENT_IDS,
    )


def july18_fixcap_warmupema_h32_open_franka9emb_noembtext_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july18_fixcap_warmupema_h32_open_franka9emb_noembtext_64k",
        repo_id="fix_cap_pause_9emb_10p10e_20260718",
        rlds_data_dir=("/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_9emb_10p10e"),
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.PerEmbodimentQuantileNorm(_NINE_TRAINING_EMBODIMENT_IDS),
        ),
        num_train_steps=64_000,
        batch_size=256,
    )


def july20_fixcap_warmupema_h32_open_franka9emb_adarms_only_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july20_fixcap_warmupema_h32_open_franka9emb_adarms_only_64k",
        repo_id="fix_cap_pause_9emb_10p10e_20260718",
        rlds_data_dir="/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_9emb_10p10e",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.PerEmbodimentQuantileNorm(_NINE_TRAINING_EMBODIMENT_IDS),
        ),
        action_expert_conditioning=_input_config.LearnedEmbodimentAdaRMS(vocab_size=64),
        num_train_steps=64_000,
        batch_size=256,
    )


def july21_fixcap_warmupema_h32_open_franka9emb_concatadarms_only_64k() -> TrainConfig:
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            "july21_fixcap_warmupema_h32_open_franka9emb_concatadarms_only_64k",
            repo_id="fix_cap_pause_9emb_10p10e_20260718",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_9emb_10p10e"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.PerEmbodimentQuantileNorm(
                    _NINE_TRAINING_EMBODIMENT_IDS
                ),
            ),
            action_expert_conditioning=_input_config.LearnedEmbodimentConcatAdaRMS(vocab_size=64),
            num_train_steps=64_000,
            batch_size=256,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july21_fixcap_warmupema_h32_open_franka_remaining81_concatadarms_only_b1024_64k() -> (
    TrainConfig
):
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            "july21_fixcap_warmupema_h32_open_franka_remaining81_"
            "concatadarms_only_b1024_64k",
            repo_id="fix_cap_pause_remaining81_10p10e_20260719",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260718_fix_cap_pause_remaining81_10p10e"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.PerEmbodimentQuantileNorm(
                    REMAINING81_TRAINING_EMBODIMENT_IDS
                ),
            ),
            action_expert_conditioning=(
                _input_config.LearnedEmbodimentConcatAdaRMS(vocab_size=100)
            ),
            num_train_steps=64_000,
            batch_size=1_024,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july20_fixcap_warmupema_h32_open_franka9emb_urma2_only_64k() -> TrainConfig:
    return _july18_fixcap_warmup_ema_base(
        "july20_fixcap_warmupema_h32_open_franka9emb_urma2_only_64k",
        repo_id="fix_cap_pause_9emb_10p10e_20260718",
        rlds_data_dir="/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_9emb_10p10e",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.PerEmbodimentQuantileNorm(
                _NINE_TRAINING_EMBODIMENT_IDS
            ),
        ),
        action_expert_conditioning=_urma_config.franka9_urma2_adarms(),
        num_train_steps=64_000,
        batch_size=256,
    )


def _july22_fixcap_warmupema_h32_open_franka9emb_urma2(
    fusion: _input_config.UrmaFusionMode,
) -> TrainConfig:
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            f"july22_fixcap_warmupema_h32_open_franka9emb_urma2_{fusion}_64k",
            repo_id="fix_cap_pause_9emb_10p10e_20260718",
            rlds_data_dir="/iris/u/boai/artifact/rlds_data/20260718_fix_cap_pause_9emb_10p10e",
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.PerEmbodimentQuantileNorm(
                    _NINE_TRAINING_EMBODIMENT_IDS
                ),
            ),
            action_expert_conditioning=_urma_config.franka9_urma2_adarms(fusion),
            num_train_steps=64_000,
            batch_size=256,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july23_fixcap_warmupema_h32_open_franka_remaining81_urma2_concatproj1024_globalnorm_b1024_64k() -> (
    TrainConfig
):
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                "july23_fixcap_warmupema_h32_open_franka_remaining81_"
                "urma2_concatproj1024_globalnorm_b1024_64k"
            ),
            repo_id="fix_cap_pause_remaining81_10p10e_20260719",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260718_fix_cap_pause_remaining81_10p10e"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            action_expert_conditioning=(
                _urma_config.franka_remaining81_urma2_concat_project_adarms()
            ),
            num_train_steps=64_000,
            batch_size=1_024,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def july22_fixcap_warmupema_h32_open_franka9emb_urma2_add_64k() -> TrainConfig:
    return _july22_fixcap_warmupema_h32_open_franka9emb_urma2("add")


def july22_fixcap_warmupema_h32_open_franka9emb_urma2_concat_64k() -> TrainConfig:
    return _july22_fixcap_warmupema_h32_open_franka9emb_urma2("concat")


def july23_fixcap_warmupema_h32_open_franka_remaining81_urma2_backbonetoken_globalnorm_b1024_64k() -> (
    TrainConfig
):
    return dataclasses.replace(
        _july18_fixcap_warmup_ema_base(
            (
                "july23_fixcap_warmupema_h32_open_franka_remaining81_"
                "urma2_backbonetoken_globalnorm_b1024_64k"
            ),
            repo_id="fix_cap_pause_remaining81_10p10e_20260719",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260718_fix_cap_pause_remaining81_10p10e"
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalQuantileNorm(),
            ),
            backbone_conditioning=(
                _urma_config.franka_remaining81_urma2_backbone_token()
            ),
            num_train_steps=64_000,
            batch_size=1_024,
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def august23_id355_unique230_frankagauss_lora_b128_seed42_20k() -> TrainConfig:
    """Historical Franka optimizer with Gaussian normalization on ID355 unique230."""

    base = _clean_pose_lora_native_config(
        "august23_id355_unique230_frankagauss_lora_b128_seed42_20k",
        repo_id="20260823_1emb_id0355_bal15x10_plus_online80_n230_movingfirst_r1",
        rlds_data_dir=(
            "/iris/u/boai/artifact/rlds_data/"
            "20260823_1emb_id0355_bal15x10_plus_online80_n230_movingfirst_r1"
        ),
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NoActionClipping(),
        ),
        num_train_steps=20_000,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=tuple(
                dataclasses.replace(
                    dataset,
                    joint_layout="cross_family_arm7_gripper7_v1",
                )
                for dataset in base.data.datasets
            ),
        ),
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def august23_id355_unique230_currentgauss_lora_b1024_seed0_20k() -> TrainConfig:
    """Current ID355 optimizer with Gaussian/clip5 on the same unique230 corpus."""

    base = july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k()
    dataset_name = "molmospaces_emb_mix"
    return dataclasses.replace(
        base,
        name="august23_id355_unique230_currentgauss_lora_b1024_seed0_20k",
        data=dataclasses.replace(
            base.data,
            repo_id="20260823_1emb_id0355_bal15x10_plus_online80_n230_movingfirst_r1",
            rlds_data_dir=(
                "/iris/u/boai/artifact/rlds_data/"
                "20260823_1emb_id0355_bal15x10_plus_online80_n230_movingfirst_r1"
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name=dataset_name,
                    version="1.1.0",
                    weight=1.0,
                    joint_layout="cross_family_arm7_gripper7_v1",
                ),
            ),
            episode_selector=None,
            episode_split=None,
            rlds_split="train",
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            initial_lr=1e-8,
            warmup_steps=1_000,
            peak_lr=8e-5,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        save_interval=4_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000,
            sample_count=4,
        ),
    )


def _august15_oer_phase0_repeat1000x30_learnedid_native(
    *, name: str, batch_size: int, num_train_steps: int
) -> TrainConfig:
    """Native shell for the frozen x10-base plus incremental-x20 learned-ID union."""

    scaled = 1_024 // batch_size
    base = _july18_fixcap_warmup_ema_base(
        name,
        repo_id=(
            "20260815_oer_phase0_repeat1000x30_id999_ep29753_"
            "base10_incremental20_union_v1"
        ),
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        backbone_conditioning=_input_config.LearnedEmbodimentBackboneToken(
            vocab_size=1_000,
        ),
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1_271_674 / 3_854_915,
                    data_subdir=(
                        "20260811_oer_phase0_v6_runtimegc0_cataloggc0_first1000_"
                        "sourcepose_repeat_upto10_two_view_retainedhold_r1"
                    ),
                    joint_layout="cross_family_arm7_gripper7_v1",
                ),
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=2_583_241 / 3_854_915,
                    data_subdir=(
                        "20260814_oer_phase0_v6_runtimegc0_cataloggc0_first1000_"
                        "sourcepose_repeat_incremental_upto20_two_view_retainedhold_r1"
                    ),
                    joint_layout="cross_family_arm7_gripper7_v1",
                ),
            ),
            episode_split=_data_processing.FrozenEmbodimentHoldoutSplit(
                validation_embodiment_ids=(
                    18, 54, 80, 114, 200, 209, 218, 219, 228, 255, 270, 280,
                    295, 298, 325, 328, 375, 384, 385, 399, 410, 431, 436, 573,
                    577, 588, 589, 590, 599, 606, 609, 612, 619, 628, 638, 647,
                    693, 694, 784, 816, 836, 842, 864, 878, 890, 909, 933, 936,
                    963, 984,
                ),
                seed=20260809,
                represented_embodiment_count=999,
            ),
            rlds_split="train",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            initial_lr=1e-8,
            warmup_steps=1_000 * scaled,
            peak_lr=8e-5 / scaled,
            decay_steps=num_train_steps,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=1_000 * scaled,
        keep_period=None,
        keep_every_n_completed_updates=4_000 * scaled,
        val_interval=1_000 * scaled,
        val_batches=2,
        log_interval=50 * scaled,
        train_sample_interval=1_000 * scaled,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000 * scaled,
            sample_count=4,
        ),
        seed=20260809,
    )


def august15_oer_phase0_repeat1000x30_id999_ep29753_learnedid_token_stdnorm_clip5_b1024_48k() -> TrainConfig:
    return _august15_oer_phase0_repeat1000x30_learnedid_native(
        name=(
            "august15_oer_phase0_repeat1000x30_id999_ep29753_"
            "learnedid_token_stdnorm_clip5_b1024_48k"
        ),
        batch_size=1_024,
        num_train_steps=48_000,
    )


def august15_oer_phase0_repeat1000x30_id999_ep29753_learnedid_token_stdnorm_clip5_b512_96k() -> TrainConfig:
    return _august15_oer_phase0_repeat1000x30_learnedid_native(
        name=(
            "august15_oer_phase0_repeat1000x30_id999_ep29753_"
            "learnedid_token_stdnorm_clip5_b512_96k"
        ),
        batch_size=512,
        num_train_steps=96_000,
    )


def _august23_oer_phase0_repeat1000x30_nocond_seed0(
    name: str, *, full_ft: bool
) -> TrainConfig:
    """Native matched LoRA/full-FT pair over the immutable x30 union."""

    base = july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k()
    model = (
        dataclasses.replace(
            base.model,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        )
        if full_ft
        else base.model
    )
    return dataclasses.replace(
        base,
        name=name,
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=(
                "20260815_oer_phase0_repeat1000x30_id999_ep29753_"
                "base10_incremental20_union_v1"
            ),
            rlds_data_dir="/iris/u/boai/artifact/rlds_data",
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1_271_674 / 3_854_915,
                    data_subdir=(
                        "20260811_oer_phase0_v6_runtimegc0_cataloggc0_first1000_"
                        "sourcepose_repeat_upto10_two_view_retainedhold_r1"
                    ),
                    joint_layout="cross_family_arm7_gripper7_v1",
                ),
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=2_583_241 / 3_854_915,
                    data_subdir=(
                        "20260814_oer_phase0_v6_runtimegc0_cataloggc0_first1000_"
                        "sourcepose_repeat_incremental_upto20_two_view_retainedhold_r1"
                    ),
                    joint_layout="cross_family_arm7_gripper7_v1",
                ),
            ),
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.GlobalStandardNorm(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
            ),
            episode_selector=None,
            episode_split=_data_processing.FrozenEmbodimentHoldoutSplit(
                validation_embodiment_ids=(
                    18, 54, 80, 114, 200, 209, 218, 219, 228, 255, 270, 280,
                    295, 298, 325, 328, 375, 384, 385, 399, 410, 431, 436, 573,
                    577, 588, 589, 590, 599, 606, 609, 612, 619, 628, 638, 647,
                    693, 694, 784, 816, 836, 842, 864, 878, 890, 909, 933, 936,
                    963, 984,
                ),
                seed=20260809,
                represented_embodiment_count=999,
            ),
            rlds_split="train",
        ),
        weight_loader=(
            weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params",
                missing_regex=r"(?!)",
            )
            if full_ft
            else weight_loaders.CheckpointWeightLoader(
                "gs://boai-iris-central2/pi05_init/pi05_base/params"
            )
        ),
        freeze_filter=nnx.Nothing() if full_ft else model.get_freeze_filter(),
        num_train_steps=48_000,
        batch_size=1_024,
        fsdp_devices=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            initial_lr=1e-8,
            warmup_steps=1_000,
            peak_lr=8e-5,
            decay_steps=48_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
        val_interval=1_000,
        val_batches=2,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000,
            sample_count=4,
        ),
        seed=0,
    )


def august23_oer_phase0_repeat1000x30_id999_ep29753_nocond_seed0_lora_stdnorm_clip5_b1024_48k() -> TrainConfig:
    return _august23_oer_phase0_repeat1000x30_nocond_seed0(
        "august23_oer_phase0_repeat1000x30_id999_ep29753_nocond_seed0_"
        "lora_stdnorm_clip5_b1024_48k",
        full_ft=False,
    )


def august23_oer_phase0_repeat1000x30_id999_ep29753_nocond_seed0_full_stdnorm_clip5_b1024_48k() -> TrainConfig:
    return _august23_oer_phase0_repeat1000x30_nocond_seed0(
        "august23_oer_phase0_repeat1000x30_id999_ep29753_nocond_seed0_"
        "full_stdnorm_clip5_b1024_48k",
        full_ft=True,
    )


def _august24_id355_balanced150_franka_lora_20k(
    name: str, *, input_config: _input_config.Pi05InputConfig
) -> TrainConfig:
    base = _clean_pose_lora_native_config(
        name,
        repo_id="20260823_1emb_id0355_15p10e_balanced150_movingfirst_r1",
        rlds_data_dir=(
            "/iris/u/boai/artifact/rlds_data/"
            "20260823_1emb_id0355_15p10e_balanced150_movingfirst_r1"
        ),
        input_config=input_config,
        num_train_steps=20_000,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=tuple(
                dataclasses.replace(
                    dataset,
                    joint_layout="cross_family_arm7_gripper7_v1",
                )
                for dataset in base.data.datasets
            ),
        ),
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def august24_id355_balanced150_frankagauss_lora_b128_seed42_save1k_20k() -> TrainConfig:
    return _august24_id355_balanced150_franka_lora_20k(
        "august24_id355_balanced150_frankagauss_lora_b128_seed42_save1k_20k",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NoActionClipping(),
        ),
    )


def august24_id355_balanced150_frankaquant_lora_b128_seed42_save1k_20k() -> TrainConfig:
    return _august24_id355_balanced150_franka_lora_20k(
        "august24_id355_balanced150_frankaquant_lora_b128_seed42_save1k_20k",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalQuantileNorm(),
            action_clipping=_input_config.NoActionClipping(),
        ),
    )


def _august25_id355_balanced150_contiguous_franka_lora_20k(
    name: str, *, quantile: bool
) -> TrainConfig:
    base = _august24_id355_balanced150_franka_lora_20k(
        name,
        input_config=_input_config.Pi05InputConfig(
            normalization=(
                _input_config.GlobalQuantileNorm()
                if quantile
                else _input_config.GlobalStandardNorm()
            ),
            action_clipping=_input_config.NoActionClipping(),
        ),
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=tuple(
                dataclasses.replace(
                    dataset,
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                )
                for dataset in base.data.datasets
            ),
        ),
    )


def august25_id355_balanced150_contiguous_frankagauss_lora_b128_seed42_save1k_20k() -> TrainConfig:
    return _august25_id355_balanced150_contiguous_franka_lora_20k(
        "august25_id355_balanced150_contiguous_frankagauss_lora_b128_seed42_save1k_20k",
        quantile=False,
    )


def august25_id355_balanced150_contiguous_frankaquant_lora_b128_seed42_save1k_20k() -> TrainConfig:
    return _august25_id355_balanced150_contiguous_franka_lora_20k(
        "august25_id355_balanced150_contiguous_frankaquant_lora_b128_seed42_save1k_20k",
        quantile=True,
    )


def august27_id355_mixed430_contiguous_gaussclip5_lora_b128_seed42_lr1e5_60k() -> TrainConfig:
    """Low-LR Pi0.5 fit on slowdwell200 plus remapped unique230."""

    base = _clean_pose_lora_native_config(
        "august27_id355_mixed430_contiguous_gaussclip5_lora_b128_seed42_lr1e5_60k",
        repo_id="20260827_ur5e_id0355_slowdwell200_plus_unique230_n430_contiguous7_v1",
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=60_000,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=23_782 / 49_315,
                    data_subdir="20260826_ur5e_1emb_id0355_slowdwell20p10e_contiguous7_v1",
                    joint_layout="legacy_left_packed_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=25_533 / 49_315,
                    data_subdir="20260823_1emb_id0355_bal15x10_plus_online80_n230_movingfirst_r1",
                    joint_layout="cross_family_arm7_gripper7_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=1e-5,
            decay_steps=60_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=500, sample_count=4
        ),
    )


def _august29_id355_camd40p10e_lora_horizon(action_horizon: int) -> TrainConfig:
    """Resolve the exact Camera-D recipe with one declared horizon difference."""

    if action_horizon not in {32, 50}:
        raise ValueError("Camera-D horizon ablation supports exactly H32 and H50")
    name = (
        "august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_"
        f"seed42_h{action_horizon}_60k"
    )
    base = _clean_pose_lora_native_config(
        name,
        repo_id=(
            "20260827_ur5e_1emb_id0355_camD_slowdwell40p10e_"
            "contiguous7_oerasync_v2"
        ),
        rlds_data_dir=(
            "/iris/u/boai/artifact/rlds_data/"
            "20260827_ur5e_1emb_id0355_camD_slowdwell40p10e_"
            "contiguous7_oerasync_v2"
        ),
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=60_000,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, action_horizon=action_horizon),
        data=dataclasses.replace(
            base.data,
            datasets=tuple(
                dataclasses.replace(
                    dataset,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                )
                for dataset in base.data.datasets
            ),
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=1e-5,
            decay_steps=60_000,
            decay_lr=1e-5,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000, sample_count=4
        ),
    )


def september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1() -> TrainConfig:
    """No-conditioning LoRA baseline on the 4K discrete-axis bridge."""

    name = (
        "september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_"
        "nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1"
    )
    dataset_id = "20260909_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_v2"
    base = _clean_pose_lora_native_config(
        name,
        repo_id=dataset_id,
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=40_000,
        dataset_version="1.2.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    "20260909_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_"
                    "train95_stdnorm_v1"
                ),
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=dataset_id,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=42,
                require_one_episode_per_embodiment=True,
            ),
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=dataclasses.replace(
            base.weight_loader,
            params_path="gs://boai-iris-central2/pi05_init/pi05_base/params",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=2e-4,
            decay_steps=40_000,
            decay_lr=2e-4,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=8_000,
        val_interval=1_000,
        val_batches=4,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=500, sample_count=4
        ),
    )


def september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1() -> TrainConfig:
    """URMA4-prefix sibling of the 4K discrete-axis bridge baseline."""

    base = (
        september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1()
    )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=(
            _urma_config.ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_token(
                bundle_sha256=(
                    "d2a10ee6a50f2430a12f8c70bf396056dd0b60a376677afc5db173e531d118b4"
                )
            )
        ),
    )
    return dataclasses.replace(
        base,
        name=(
            "september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_"
            "urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1"
        ),
        model=model,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params",
            missing_regex=(
                r"(.*lora.*|embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)"
            ),
        ),
    )


def september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v2() -> TrainConfig:
    """Corrected exact-ID URMA4-prefix sibling of the 4K bridge baseline."""

    base = (
        september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1()
    )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=(
            _urma_config.ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_token_v2(
                bundle_sha256=(
                    "1b4915f876db20b899403430e16382b71021fb82addfe26a86a0705aca032ec8"
                )
            )
        ),
    )
    return dataclasses.replace(
        base,
        name=(
            "september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_"
            "urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v2"
        ),
        model=model,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params",
            missing_regex=(
                r"(.*lora.*|embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)"
            ),
        ),
    )


def september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1() -> TrainConfig:
    """Equal bridge-4K/calibrated-center mixture with an explicit URMA4 anchor."""

    name = (
        "september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_"
        "urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1"
    )
    base = (
        september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v2()
    )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=(
            _urma_config.ur5e_bridge4k_plus_calibrated_anchor_urma4_backbone_token(
                bundle_sha256=(
                    "aefa87298c1438037c71e86b28fdc9b29926c5f66e78be015d372972fe60dec3"
                )
            )
        ),
    )
    return dataclasses.replace(
        base,
        name=name,
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=(
                "20260910_oer_ur5e_bridge4k_plus_calibratedcenter200_"
                "equalmix_v1"
            ),
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    "20260910_bridge4k_plus_calibratedcenter200_equalmix_"
                    "train_stdnorm_v1"
                ),
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    weight=0.5,
                    data_subdir=(
                        "20260909_oer_ur5e_discreteaxes_slidegripper_"
                        "cam2520_4k_v2"
                    ),
                ),
                dataclasses.replace(
                    base.data.datasets[0],
                    version="1.2.0",
                    weight=0.5,
                    data_subdir=(
                        "20260906_ur5e_vx300s_fingers_v2_center20p10e_camd_v1"
                    ),
                ),
            ),
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=42,
                require_one_episode_per_embodiment=False,
            ),
        ),
        freeze_filter=model.get_freeze_filter(),
        num_train_steps=16_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params",
            missing_regex=(
                r"(.*lora.*|embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)"
            ),
        ),
    )


def september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1() -> TrainConfig:
    """Valid-coordinate-loss sibling of the bridge-4K target-anchor mixture."""

    base = september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1()
    return dataclasses.replace(
        base,
        name=(
            "september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_"
            "urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1"
        ),
        data=dataclasses.replace(
            base.data,
            input_config=dataclasses.replace(
                base.data.input_config,
                action_loss_coordinates="physical",
            ),
        ),
    )


def september10_oer_ur5e_bridge2942_bounded_noretry_plus_calibratedcenter200_equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1() -> TrainConfig:
    """Quality-filtered bridge/center equal mixture with physical action loss."""

    name = (
        "september10_oer_ur5e_bridge2942_bounded_noretry_plus_calibratedcenter200_"
        "equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1"
    )
    base = september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1()
    return dataclasses.replace(
        base,
        name=name,
        data=dataclasses.replace(
            base.data,
            repo_id=(
                "20260910_oer_ur5e_bridge2942_bounded_noretry_plus_"
                "calibratedcenter200_equalmix_v1"
            ),
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    "20260910_bridge2942_bounded_noretry_plus_calibratedcenter200_"
                    "equalmix_train_stdnorm_v1"
                ),
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=(
                        "20260910_oer_ur5e_bridge4k_bounded_noretry2942_"
                        "camd2520_v1"
                    ),
                ),
                base.data.datasets[1],
            ),
        ),
    )


def september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_physicalgripperstate_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1() -> TrainConfig:
    """Physical-aperture-state bridge/center mixture with physical action loss."""

    name = (
        "september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_"
        "physicalgripperstate_urma4_backbone_physical7_lora_"
        "gaussclip5_b128_lr2e4_seed42_h32_16k_v1"
    )
    base = september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1()
    return dataclasses.replace(
        base,
        name=name,
        data=dataclasses.replace(
            base.data,
            repo_id=(
                "20260910_oer_ur5e_bridge4k_plus_calibratedcenter200_"
                "equalmix_physicalgripperstate_v1"
            ),
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    "20260910_bridge4k_plus_calibratedcenter200_equalmix_"
                    "physicalgripperstate_train_stdnorm_v1"
                ),
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=(
                        "20260910_oer_ur5e_discreteaxes_slidegripper_cam2520_"
                        "4k_physicalgripperstate_v1"
                    ),
                ),
                dataclasses.replace(
                    base.data.datasets[1],
                    data_subdir=(
                        "20260910_ur5e_vx300s_fingers_v2_center20p10e_camd_"
                        "physicalgripperstate_v1"
                    ),
                ),
            ),
        ),
    )


def _september11_independent_bimodal_oer8k_lora(
    cell: _sep11_contract.Cell,
    *,
    total_updates: int = _sep11_contract.TOTAL_UPDATES,
) -> TrainConfig:
    """Native realization of one matched Sep-11 OER8K conditioning cell."""

    base = (
        september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1()
    )
    input_config = dataclasses.replace(
        base.data.input_config,
        action_loss_coordinates="physical",
    )
    model = base.model
    missing_regex = base.weight_loader.missing_regex
    if cell == "urma4_backbone":
        model = dataclasses.replace(
            model,
            backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
                extractor=_input_config.Urma4EmbodimentExtractor(
                    bundle_dir=(
                        f"{PI_EMBODIMENT_ASSETS_DIR}/"
                        f"{_sep11_contract.URMA4_BUNDLE_NAME}"
                    ),
                    bundle_sha256=_sep11_contract.URMA4_BUNDLE_SHA256,
                ),
                token_width=2048,
            ),
        )
        missing_regex = (
            r"(.*lora.*|embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)"
        )
    elif cell == "text_t1":
        model = dataclasses.replace(
            model,
            max_token_len=_sep11_contract.TEXT_T1_MAX_TOKEN_LEN,
        )
        input_config = dataclasses.replace(
            input_config,
            prompt=_input_config.TextBundlePrompt(
                bundle_dir=(
                    f"{PI_EMBODIMENT_ASSETS_DIR}/"
                    f"{_sep11_contract.TEXT_T1_BUNDLE_NAME}"
                ),
                bundle_sha256=_sep11_contract.TEXT_T1_BUNDLE_SHA256,
                max_token_len=_sep11_contract.TEXT_T1_MAX_TOKEN_LEN,
                format_id=_sep11_contract.TEXT_T1_FORMAT_ID,
            ),
        )
    else:
        raise ValueError(f"unsupported September 11 conditioning cell: {cell!r}")

    dataset_id = _sep11_contract.DATASET_ID
    return dataclasses.replace(
        base,
        name=(
            _sep11_contract.experiment_name(cell)
            if total_updates == _sep11_contract.TOTAL_UPDATES
            else _sep11_contract.continuation_experiment_name(cell)
        ),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=dataset_id,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep11_contract.NORMALIZATION_ASSET_ID,
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=dataset_id,
                    version="1.2.0",
                    weight=1.0,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
            input_config=input_config,
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=_sep11_contract.SAMPLER_SEED,
                require_one_episode_per_embodiment=True,
            ),
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            rlds_sampler_seed=_sep11_contract.SAMPLER_SEED,
        ),
        num_train_steps=total_updates,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=dataclasses.replace(
            base.weight_loader,
            missing_regex=missing_regex,
        ),
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            decay_steps=total_updates,
        ),
    )


def _public_september11_independent_bimodal_oer8k_builder(
    cell: _sep11_contract.Cell,
    *,
    total_updates: int = _sep11_contract.TOTAL_UPDATES,
) -> Callable[[], TrainConfig]:
    """Create a discoverable builder under the immutable experiment name."""

    def builder() -> TrainConfig:
        return _september11_independent_bimodal_oer8k_lora(
            cell, total_updates=total_updates
        )

    builder.__name__ = (
        _sep11_contract.experiment_name(cell)
        if total_updates == _sep11_contract.TOTAL_UPDATES
        else _sep11_contract.continuation_experiment_name(cell)
    )
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER11_INDEPENDENT_BIMODAL_OER8K_BUILDERS = tuple(
    _public_september11_independent_bimodal_oer8k_builder(
        cell, total_updates=total_updates
    )
    for cell in _sep11_contract.CELLS
    for total_updates in (
        _sep11_contract.TOTAL_UPDATES,
        _sep11_contract.CONTINUATION_TOTAL_UPDATES,
    )
)


def _september20_nativecentered_4k_urma4_backbone_lora() -> TrainConfig:
    """Native realization of the Sep-20 first-4K URMA4 milestone."""

    base = _september11_independent_bimodal_oer8k_lora("urma4_backbone")
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
            extractor=_input_config.Urma4EmbodimentExtractor(
                bundle_dir=(
                    f"{PI_EMBODIMENT_ASSETS_DIR}/"
                    f"{_sep20_contract.URMA4_BUNDLE_NAME}"
                ),
                bundle_sha256=_sep20_contract.URMA4_BUNDLE_SHA256,
            ),
            token_width=2048,
        ),
    )
    dataset_id = _sep20_contract.DATASET_ID
    return dataclasses.replace(
        base,
        name=_sep20_contract.experiment_name(),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=dataset_id,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep20_contract.NORMALIZATION_ASSET_ID,
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=dataset_id,
                    version="1.2.0",
                    weight=1.0,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=_sep20_contract.SAMPLER_SEED,
                require_one_episode_per_embodiment=True,
            ),
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            rlds_sampler_seed=_sep20_contract.SAMPLER_SEED,
        ),
        num_train_steps=_sep20_contract.TOTAL_UPDATES,
        freeze_filter=model.get_freeze_filter(),
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            decay_steps=_sep20_contract.TOTAL_UPDATES,
        ),
    )


def _public_september20_nativecentered_4k_builder() -> Callable[[], TrainConfig]:
    """Expose the native builder under the repository experiment identity."""

    def builder() -> TrainConfig:
        return _september20_nativecentered_4k_urma4_backbone_lora()

    builder.__name__ = _sep20_contract.experiment_name()
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER20_NATIVECENTERED_4K_BUILDER = (
    _public_september20_nativecentered_4k_builder()
)


def _september21_native_mix6k(cell: _sep21_mix_contract.Cell) -> TrainConfig:
    """Native OpenPI twin of one registered Sep-21 Mix6K experiment."""

    if not _sep21_mix_contract.assets_published(cell):
        raise RuntimeError(f"Sep21 Mix6K {cell} assets are not hash-published")
    base = _september20_nativecentered_4k_urma4_backbone_lora()
    if cell == "urma51":
        extractor = _input_config.Urma51EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep21_mix_contract.URMA51_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep21_mix_contract.URMA51_BUNDLE_SHA256,
        )
    elif cell == "urma6":
        extractor = _input_config.Urma6EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep21_mix_contract.URMA6_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep21_mix_contract.URMA6_BUNDLE_SHA256,
        )
    elif cell == "urma5":
        extractor = _input_config.Urma5EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep21_mix_contract.URMA5_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep21_mix_contract.URMA5_BUNDLE_SHA256,
        )
    else:
        extractor = _input_config.Urma4NewNormEmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep21_mix_contract.URMA4_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep21_mix_contract.URMA4_BUNDLE_SHA256,
        )
    backbone = _input_config.ExtractedEmbodimentBackboneToken(
        extractor=extractor,
        token_width=2048,
    )
    if cell == "urma4_flat":
        backbone = _input_config.FlatUrma4BackboneToken(
            extractor=extractor,
            token_width=2048,
        )
    model = dataclasses.replace(base.model, backbone_conditioning=backbone)
    component_version = (
        "1.5.0" if cell in {"urma5", "urma51", "urma6"} else "1.2.0"
    )
    return dataclasses.replace(
        base,
        name=_sep21_mix_contract.experiment_name(cell),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id="20260921_oer_ur5e_dense2k_broad4k_mix6k_v1",
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep21_mix_contract.NORMALIZATION_ASSET_ID,
            ),
            datasets=tuple(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=dataset_id,
                    data_subdir=dataset_id,
                    name="molmospaces_emb_mix",
                    version=component_version,
                    weight=weight,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                )
                for _, dataset_id, _, weight in _sep21_mix_contract.COMPONENTS
            ),
            episode_split=_data_processing.NamespacedHashEmbodimentSplit(
                validation_fraction=0.05,
                seed=_sep21_mix_contract.SEED,
                require_one_episode_per_embodiment=True,
            ),
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            rlds_sampler_seed=_sep21_mix_contract.SEED,
        ),
        freeze_filter=model.get_freeze_filter(),
        num_train_steps=_sep21_mix_contract.TOTAL_UPDATES,
        save_interval=_sep21_mix_contract.CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=(
            _sep21_mix_contract.CHECKPOINT_RETENTION_EVERY_UPDATES
        ),
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            decay_steps=_sep21_mix_contract.TOTAL_UPDATES,
        ),
    )


def _public_september21_native_mix6k_builder(
    cell: _sep21_mix_contract.Cell,
) -> Callable[[], TrainConfig]:
    """Expose one native builder under the repository experiment identity."""

    def builder() -> TrainConfig:
        return _september21_native_mix6k(cell)

    builder.__name__ = _sep21_mix_contract.experiment_name(cell)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER21_NATIVE_MIX6K_BUILDERS = tuple(
    _public_september21_native_mix6k_builder(cell)
    for cell in _sep21_mix_contract.CELLS
)


def _september22_nearcenter2k_urma4() -> TrainConfig:
    """Native twin of the 2K-only response-aware URMA4 control."""

    base = _september21_native_mix6k("urma4")
    extractor = _input_config.Urma4NewNormEmbodimentExtractor(
        bundle_dir=(
            f"{PI_EMBODIMENT_ASSETS_DIR}/"
            f"{_sep22_dense_contract.URMA4_BUNDLE_NAME}"
        ),
        bundle_sha256=_sep22_dense_contract.URMA4_BUNDLE_SHA256,
    )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
            extractor=extractor,
            token_width=2048,
        ),
    )
    return dataclasses.replace(
        base,
        name=_sep22_dense_contract.experiment_name(),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=_sep22_dense_contract.DATASET_ID,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep22_dense_contract.NORMALIZATION_ASSET_ID,
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=_sep22_dense_contract.DATASET_ID,
                    data_subdir=_sep22_dense_contract.DATASET_ID,
                    name="molmospaces_emb_mix",
                    version="1.2.0",
                    weight=1.0,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
            episode_split=_data_processing.AnchoredHashEmbodimentSplit(
                required_validation_embodiment_ids=(
                    _sep22_dense_contract.REFERENCE_VALIDATION_IDS
                ),
                required_training_embodiment_ids=(
                    _sep22_dense_contract.REFERENCE_SEEN_IDS
                ),
                anchors_manifest_sha256=_sep22_dense_contract.REFERENCE_SPLIT_SHA256,
                validation_fraction=0.05,
                seed=_sep22_dense_contract.SEED,
                require_one_episode_per_embodiment=True,
            ),
        ),
        freeze_filter=model.get_freeze_filter(),
    )


def _public_september22_nearcenter2k_urma4_builder() -> Callable[[], TrainConfig]:
    def builder() -> TrainConfig:
        return _september22_nearcenter2k_urma4()

    builder.__name__ = _sep22_dense_contract.experiment_name()
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER22_NEARCENTER2K_URMA4_BUILDER = (
    _public_september22_nearcenter2k_urma4_builder()
)


def _september23_clean2k_repeat100_physical_split() -> _data_processing.PhysicalEmbodimentAnchoredHashSplit:
    """Hold out the Sep-22 clean-2K validation robots in both mixture components."""

    return _data_processing.PhysicalEmbodimentAnchoredHashSplit(
        reference_namespace=_sep23_mix_contract.CLEAN_DATASET_ID,
        repeat_namespaces=(_sep23_mix_contract.REPEAT_DATASET_ID,),
        required_validation_embodiment_ids=_sep22_dense_contract.REFERENCE_VALIDATION_IDS,
        required_training_embodiment_ids=_sep22_dense_contract.REFERENCE_SEEN_IDS,
        anchors_manifest_sha256=_sep22_dense_contract.REFERENCE_SPLIT_SHA256,
        validation_fraction=0.05,
        seed=_sep22_dense_contract.SEED,
    )


def _september23_clean2k_repeat100_datasets(version: str) -> tuple:
    return tuple(
        droid_rlds_dataset.RLDSDataset(
            dataset_id=dataset_id,
            data_subdir=dataset_id,
            name="molmospaces_emb_mix",
            version=version,
            weight=weight,
            joint_layout="ur5e_contiguous_arm6_gripper6_v1",
            target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
        )
        for _, dataset_id, _, weight in _sep23_mix_contract.COMPONENTS
    )


def _september23_clean2k_repeat100_mix6k(cell: _sep23_mix_contract.Cell) -> TrainConfig:
    """Native OpenPI twin of one registered Sep-23 clean-2K + repeat-100 cell."""

    if not _sep23_mix_contract.assets_published(cell):
        raise RuntimeError(f"Sep23 clean2K+repeat100 {cell} assets are not hash-published")
    base = _september21_native_mix6k(cell)
    if cell == "urma6":
        extractor = _input_config.Urma6EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep23_mix_contract.URMA6_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep23_mix_contract.URMA6_BUNDLE_SHA256,
        )
    else:
        extractor = _input_config.Urma4NewNormEmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep23_mix_contract.URMA4_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep23_mix_contract.URMA4_BUNDLE_SHA256,
        )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=(
            _input_config.FlatUrma4BackboneToken(extractor=extractor, token_width=2048)
            if cell == "urma4_flat"
            else _input_config.ExtractedEmbodimentBackboneToken(
                extractor=extractor, token_width=2048
            )
        ),
    )
    return dataclasses.replace(
        base,
        name=_sep23_mix_contract.experiment_name(cell),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=_sep23_mix_contract.DATASET_REPO_ID,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep23_mix_contract.NORMALIZATION_ASSET_ID,
            ),
            datasets=_september23_clean2k_repeat100_datasets(
                _sep23_mix_contract.component_version(cell)
            ),
            episode_split=_september23_clean2k_repeat100_physical_split(),
        ),
        freeze_filter=model.get_freeze_filter(),
    )


def _public_september23_clean2k_repeat100_mix6k_builder(
    cell: _sep23_mix_contract.Cell,
) -> Callable[[], TrainConfig]:
    """Expose one native builder under the repository experiment identity."""

    def builder() -> TrainConfig:
        return _september23_clean2k_repeat100_mix6k(cell)

    builder.__name__ = _sep23_mix_contract.experiment_name(cell)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER23_CLEAN2K_REPEAT100_MIX6K_BUILDERS = tuple(
    _public_september23_clean2k_repeat100_mix6k_builder(cell)
    for cell in _sep23_mix_contract.CELLS
)


def _september24_repeat100_single_robot_view(robot_id: int) -> TrainConfig:
    """Shared single-robot data/schedule view over the Sep-23 URMA4 mixture twin."""

    base = _september23_clean2k_repeat100_mix6k("urma4")
    return dataclasses.replace(
        base,
        name=_sep24_single_contract.experiment_name(robot_id),
        data=dataclasses.replace(
            base.data,
            repo_id=_sep24_single_contract.DATASET_ID,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep24_single_contract.normalization_asset_id(robot_id),
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=_sep24_single_contract.DATASET_ID,
                    data_subdir=_sep24_single_contract.DATASET_ID,
                    name="molmospaces_emb_mix",
                    version=_sep24_single_contract.DATASET_VERSION,
                    weight=1.0,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
            episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                embodiment_ids=(robot_id,),
                episodes_per_embodiment=_sep24_single_contract.EPISODES_PER_ROBOT,
            ),
            episode_split=None,
        ),
        num_train_steps=_sep24_single_contract.TOTAL_UPDATES,
        save_interval=_sep24_single_contract.CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=(
            _sep24_single_contract.CHECKPOINT_RETENTION_EVERY_UPDATES
        ),
        val_interval=0,
        val_batches=0,
    )


def _september24_repeat100_single_robot(robot_id: int) -> TrainConfig:
    """Native OpenPI twin of one registered Sep-24 single-robot overfit cell."""

    if not _sep24_single_contract.assets_published(robot_id):
        raise RuntimeError(
            f"Sep24 repeat100 single-robot c{robot_id:06d} assets are not hash-published"
        )
    return _september24_repeat100_single_robot_view(robot_id)


def _public_september24_repeat100_single_robot_builder(
    robot_id: int,
) -> Callable[[], TrainConfig]:
    """Expose one native builder under the repository experiment identity."""

    def builder() -> TrainConfig:
        return _september24_repeat100_single_robot(robot_id)

    builder.__name__ = _sep24_single_contract.experiment_name(robot_id)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER24_REPEAT100_SINGLE_ROBOT_BUILDERS = tuple(
    _public_september24_repeat100_single_robot_builder(robot_id)
    for robot_id in _sep24_single_contract.ROBOT_IDS
)


def _public_september13_independent_bimodal_oer_data_scaling_builder(
    training_episode_count: int,
) -> Callable[[], TrainConfig]:
    """Create one native OpenPI config for the nested 2K/4K data-size study."""

    def builder() -> TrainConfig:
        base = _september11_independent_bimodal_oer8k_lora(
            "urma4_backbone",
            total_updates=_sep11_contract.CONTINUATION_TOTAL_UPDATES,
        )
        return dataclasses.replace(
            base,
            name=_sep11_contract.data_scaling_experiment_name(
                training_episode_count
            ),
            data=dataclasses.replace(
                base.data,
                assets=dataclasses.replace(
                    base.data.assets,
                    asset_id=(
                        _sep11_contract.DATA_SCALING_NORMALIZATION_ASSET_IDS[
                            training_episode_count
                        ]
                    ),
                ),
                episode_selector=_data_processing.NestedHashTrainingEpisodeSelector(
                    training_episode_count=training_episode_count,
                    validation_episode_count=400,
                    selection_seed=(
                        _sep11_contract.DATA_SCALING_SELECTION_SEED
                    ),
                    split_seed=_sep11_contract.SAMPLER_SEED,
                ),
                episode_split=_data_processing.HashEmbodimentCountSplit(
                    validation_embodiment_count=400,
                    seed=_sep11_contract.SAMPLER_SEED,
                    require_one_episode_per_embodiment=True,
                ),
            ),
        )

    builder.__name__ = _sep11_contract.data_scaling_experiment_name(
        training_episode_count
    )
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER13_INDEPENDENT_BIMODAL_OER_DATA_SCALING_BUILDERS = tuple(
    _public_september13_independent_bimodal_oer_data_scaling_builder(count)
    for count in _sep11_contract.DATA_SCALING_TRAIN_EPISODE_COUNTS
)


def _september16_oer8k_plus_calibratedcenter200_equalmix_warmstart() -> TrainConfig:
    """Native realization of the endpoint-density warm start on the broad OER8K policy."""

    base = _september11_independent_bimodal_oer8k_lora("urma4_backbone")
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
            extractor=_input_config.Urma4EmbodimentExtractor(
                bundle_dir=(
                    f"{PI_EMBODIMENT_ASSETS_DIR}/"
                    f"{_sep16_contract.URMA4_BUNDLE_NAME}"
                ),
                bundle_sha256=_sep16_contract.URMA4_BUNDLE_SHA256,
            ),
            token_width=2048,
        ),
    )
    return dataclasses.replace(
        base,
        name=_sep16_contract.experiment_name(),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=_sep16_contract.MIXTURE_DATASET_ID,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_sep16_contract.NORMALIZATION_ASSET_ID,
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    weight=0.5,
                    data_subdir=_sep16_contract.BROAD_DATASET_ID,
                ),
                dataclasses.replace(
                    base.data.datasets[0],
                    weight=0.5,
                    data_subdir=_sep16_contract.CENTER200_DATASET_ID,
                ),
            ),
            episode_split=_data_processing.HashEmbodimentSplit(
                validation_fraction=0.05,
                seed=_sep16_contract.SAMPLER_SEED,
                require_one_episode_per_embodiment=False,
            ),
        ),
        freeze_filter=model.get_freeze_filter(),
        num_train_steps=_sep16_contract.TOTAL_UPDATES,
        save_interval=_sep16_contract.CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=_sep16_contract.CHECKPOINT_EVERY_UPDATES,
        weight_loader=dataclasses.replace(
            base.weight_loader,
            params_path=_sep16_contract.warm_start_params_uri(),
        ),
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            decay_steps=_sep16_contract.TOTAL_UPDATES,
        ),
    )


def _public_september16_oer8k_plus_calibratedcenter200_builder() -> Callable[[], TrainConfig]:
    """Create a discoverable builder under the immutable experiment name."""

    def builder() -> TrainConfig:
        return _september16_oer8k_plus_calibratedcenter200_equalmix_warmstart()

    builder.__name__ = _sep16_contract.experiment_name()
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER16_OER8K_CENTER200_MIXTURE_BUILDER = (
    _public_september16_oer8k_plus_calibratedcenter200_builder()
)


def _public_september19_center200_urma51_noise_builder(
    cell: _sep19_center_contract.Cell,
) -> Callable[[], TrainConfig]:
    """Create one paired center200 URMA5.1 clean/dynamic-noise config."""

    def builder() -> TrainConfig:
        if not _sep19_center_contract.assets_published():
            raise RuntimeError("center200 URMA5.1 assets are not hash-published")
        base = september6_ur5e_vx300s_fingers_v2_center200_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1()
        extractor = _input_config.Urma51EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep19_center_contract.SOURCE_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep19_center_contract.SOURCE_BUNDLE_SHA256,
        )
        model = dataclasses.replace(
            base.model,
            backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
                extractor=extractor,
                token_width=2048,
            ),
            urma51_dynamic_conditioning_noise_std=(
                _sep19_center_contract.DYNAMIC_NOISE_STANDARD_DEVIATION[cell]
            ),
        )
        return dataclasses.replace(
            base,
            name=_sep19_center_contract.experiment_name(cell),
            model=model,
            data=dataclasses.replace(
                base.data,
                repo_id=_sep19_center_contract.DATASET_ID,
                datasets=(
                    dataclasses.replace(
                        base.data.datasets[0],
                        data_subdir=_sep19_center_contract.RLDS_RELEASE_NAME,
                        version=_sep19_center_contract.DATASET_VERSION,
                        joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                        target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    ),
                ),
                rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            ),
            freeze_filter=model.get_freeze_filter(),
            num_train_steps=_sep19_center_contract.TOTAL_UPDATES,
            lr_schedule=dataclasses.replace(
                base.lr_schedule,
                decay_steps=_sep19_center_contract.TOTAL_UPDATES,
            ),
            save_interval=_sep19_center_contract.SAVE_EVERY_UPDATES,
            keep_every_n_completed_updates=(
                _sep19_center_contract.RETAIN_EVERY_UPDATES
            ),
        )

    builder.__name__ = _sep19_center_contract.experiment_name(cell)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


# Registered only once the study's assets are hash-published: list_configs()
# builds every registered config, so an unpublished fail-closed builder would
# otherwise break the whole registry.
_SEPTEMBER19_CENTER200_URMA51_NOISE_BUILDERS = (
    tuple(
        _public_september19_center200_urma51_noise_builder(cell)
        for cell in _sep19_center_contract.CELLS
    )
    if _sep19_center_contract.assets_published()
    else ()
)


def _public_september14_reuse100_dense_builder(
    robot_count: int,
    preemptible: bool = False,
) -> Callable[[], TrainConfig]:
    """Create one native config for a predetermined dense-embodiment milestone.

    A preemptible lane saves far more often so a reclaim loses less progress; it
    retains the same 2K-aligned checkpoints, so it needs its own native name
    because the adapter requires the native and experiment intervals to agree.
    """

    def builder() -> TrainConfig:
        base = _september11_independent_bimodal_oer8k_lora("urma4_backbone")
        model = dataclasses.replace(
            base.model,
            backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
                extractor=_input_config.Urma4EmbodimentExtractor(
                    bundle_dir=(
                        f"{PI_EMBODIMENT_ASSETS_DIR}/"
                        f"{_sep14_dense_contract.URMA4_BUNDLE_NAMES[robot_count]}"
                    ),
                    bundle_sha256=(
                        _sep14_dense_contract.URMA4_BUNDLE_SHA256[robot_count]
                    ),
                ),
                token_width=2048,
            ),
        )
        return dataclasses.replace(
            base,
            name=_sep14_dense_contract.experiment_name(
                robot_count, preemptible=preemptible
            ),
            model=model,
            data=dataclasses.replace(
                base.data,
                repo_id=_sep14_dense_contract.DATASET_ID,
                assets=dataclasses.replace(
                    base.data.assets,
                    asset_id=(
                        _sep14_dense_contract.NORMALIZATION_ASSET_IDS[robot_count]
                    ),
                ),
                datasets=(
                    dataclasses.replace(
                        base.data.datasets[0],
                        data_subdir=(
                            _sep14_dense_contract.RLDS_RELEASE_NAMES[robot_count]
                        ),
                        version=_sep14_dense_contract.DATASET_VERSION,
                    ),
                ),
                episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                    embodiment_ids=(
                        _sep14_dense_contract.milestone_embodiment_ids(robot_count)
                    ),
                    episodes_per_embodiment=(
                        _sep14_dense_contract.EPISODES_PER_ROBOT
                    ),
                ),
                episode_split=None,
            ),
            num_train_steps=_sep14_dense_contract.TOTAL_UPDATES,
            save_interval=_sep14_dense_contract.checkpoint_every_updates(
                preemptible=preemptible
            ),
            keep_every_n_completed_updates=(
                _sep14_dense_contract.CHECKPOINT_RETENTION_EVERY_UPDATES
            ),
            val_interval=0,
            val_batches=0,
        )

    builder.__name__ = _sep14_dense_contract.experiment_name(
        robot_count, preemptible=preemptible
    )
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER14_REUSE100_DENSE_BUILDERS = tuple(
    _public_september14_reuse100_dense_builder(count, preemptible)
    for count in _sep14_dense_contract.MILESTONE_ROBOT_COUNTS
    for preemptible in (False, True)
)


def _public_september17_urma4_threeway_builder(
    cell: _sep17_threeway_contract.Cell,
    *,
    continuation: bool = False,
) -> Callable[[], TrainConfig]:
    """Create one native config for the matched nested-500 URMA4 study."""

    def builder() -> TrainConfig:
        total_updates = (
            _sep17_threeway_contract.CONTINUATION_TOTAL_UPDATES
            if continuation
            else _sep17_threeway_contract.TOTAL_UPDATES
        )
        base = _september11_independent_bimodal_oer8k_lora(
            "urma4_backbone",
            total_updates=total_updates,
        )
        model = dataclasses.replace(
            base.model,
            backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
                extractor=_input_config.Urma4DynamicEmbodimentExtractor(
                    bundle_dir=(
                        f"{PI_EMBODIMENT_ASSETS_DIR}/"
                        f"{_sep17_threeway_contract.URMA4_BUNDLE_NAME}"
                    ),
                    bundle_sha256=_sep17_threeway_contract.URMA4_BUNDLE_SHA256,
                    dynamic_stats_path=(
                        f"{PI_EMBODIMENT_ASSETS_DIR}/"
                        f"{_sep17_threeway_contract.DYNAMIC_STATS_NAME}"
                    ),
                    dynamic_stats_sha256=(
                        _sep17_threeway_contract.DYNAMIC_STATS_SHA256
                    ),
                    dynamic_profile=cell,
                ),
                token_width=2048,
            ),
        )
        return dataclasses.replace(
            base,
            name=(
                _sep17_threeway_contract.continuation_experiment_name(cell)
                if continuation
                else _sep17_threeway_contract.experiment_name(cell)
            ),
            model=model,
            data=dataclasses.replace(
                base.data,
                repo_id=_sep17_threeway_contract.DATASET_ID,
                assets=dataclasses.replace(
                    base.data.assets,
                    asset_id=_sep17_threeway_contract.NORMALIZATION_ASSET_ID,
                ),
                datasets=(
                    dataclasses.replace(
                        base.data.datasets[0],
                        data_subdir=_sep17_threeway_contract.RLDS_RELEASE_NAME,
                        version=_sep17_threeway_contract.DATASET_VERSION,
                        weight=1.0,
                    ),
                ),
                episode_selector=_data_processing.NestedHashTrainingEpisodeSelector(
                    training_episode_count=(
                        _sep17_threeway_contract.TRAINING_EPISODE_COUNT
                    ),
                    validation_episode_count=(
                        _sep17_threeway_contract.VALIDATION_EPISODE_COUNT
                    ),
                    selection_seed=_sep17_threeway_contract.SELECTION_SEED,
                    split_seed=_sep17_threeway_contract.SPLIT_SEED,
                ),
                episode_split=_data_processing.HashEmbodimentCountSplit(
                    validation_embodiment_count=(
                        _sep17_threeway_contract.VALIDATION_EPISODE_COUNT
                    ),
                    seed=_sep17_threeway_contract.SPLIT_SEED,
                    require_one_episode_per_embodiment=True,
                ),
            ),
            freeze_filter=model.get_freeze_filter(),
            save_interval=500,
            keep_every_n_completed_updates=2_000,
        )

    builder.__name__ = (
        _sep17_threeway_contract.continuation_experiment_name(cell)
        if continuation
        else _sep17_threeway_contract.experiment_name(cell)
    )
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER17_URMA4_THREEWAY_BUILDERS = tuple(
    _public_september17_urma4_threeway_builder(cell, continuation=continuation)
    for cell in _sep17_threeway_contract.CELLS
    for continuation in (False, True)
)


def _public_september18_dense94_urma456_builder(
    cell: _sep18_urma_contract.Cell,
    *,
    preemptible: bool = False,
    urma4_rldsv2: bool = False,
) -> Callable[[], TrainConfig]:
    """Create one native config for the matched dense-94 URMA4/5.1/6 study."""

    if urma4_rldsv2 and cell != "urma4_newnorm":
        raise ValueError("the RLDS-v2 compatibility twin is defined only for URMA4")

    def builder() -> TrainConfig:
        base = _september11_independent_bimodal_oer8k_lora(
            "urma4_backbone",
            total_updates=_sep18_urma_contract.TOTAL_UPDATES,
        )
        extractor = {
            "urma4_newnorm": _input_config.Urma4NewNormEmbodimentExtractor,
            "urma51": _input_config.Urma51EmbodimentExtractor,
            "urma6": _input_config.Urma6EmbodimentExtractor,
        }[cell](
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{_sep18_urma_contract.BUNDLE_NAMES[cell]}"
            ),
            bundle_sha256=_sep18_urma_contract.BUNDLE_SHA256[cell],
        )
        model = dataclasses.replace(
            base.model,
            backbone_conditioning=_input_config.ExtractedEmbodimentBackboneToken(
                extractor=extractor,
                token_width=2048,
            ),
        )
        dataset_cell: _sep18_urma_contract.Cell = (
            "urma51" if urma4_rldsv2 else cell
        )
        return dataclasses.replace(
            base,
            name=(
                _sep18_urma_contract.urma4_rldsv2_experiment_name(
                    preemptible=preemptible
                )
                if urma4_rldsv2
                else _sep18_urma_contract.experiment_name(
                    cell, preemptible=preemptible
                )
            ),
            model=model,
            data=dataclasses.replace(
                base.data,
                repo_id=_sep18_urma_contract.DATASET_ID,
                assets=dataclasses.replace(
                    base.data.assets,
                    asset_id=_sep18_urma_contract.NORMALIZATION_ASSET_IDS[cell],
                ),
                datasets=(
                    dataclasses.replace(
                        base.data.datasets[0],
                        data_subdir=_sep18_urma_contract.RLDS_RELEASE_NAMES[
                            dataset_cell
                        ],
                        version=_sep18_urma_contract.DATASET_VERSIONS[dataset_cell],
                        weight=1.0,
                    ),
                ),
                episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                    embodiment_ids=_sep18_urma_contract.EMBODIMENT_IDS,
                    episodes_per_embodiment=(
                        _sep18_urma_contract.EPISODES_PER_ROBOT
                    ),
                ),
                episode_split=None,
            ),
            freeze_filter=model.get_freeze_filter(),
            num_train_steps=_sep18_urma_contract.TOTAL_UPDATES,
            save_interval=_sep18_urma_contract.checkpoint_every_updates(
                preemptible=preemptible
            ),
            keep_every_n_completed_updates=(
                _sep18_urma_contract.CHECKPOINT_RETENTION_EVERY_UPDATES
            ),
            val_interval=0,
            val_batches=0,
        )

    builder.__name__ = (
        _sep18_urma_contract.urma4_rldsv2_experiment_name(
            preemptible=preemptible
        )
        if urma4_rldsv2
        else _sep18_urma_contract.experiment_name(cell, preemptible=preemptible)
    )
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER18_DENSE94_URMA456_BUILDERS = tuple(
    _public_september18_dense94_urma456_builder(
        cell, preemptible=preemptible
    )
    for cell in _sep18_urma_contract.CELLS
    for preemptible in (False, True)
) + tuple(
    _public_september18_dense94_urma456_builder(
        "urma4_newnorm",
        preemptible=preemptible,
        urma4_rldsv2=True,
    )
    for preemptible in (False, True)
)


def _public_september19_dense94_learned_id_builder(
    *, preemptible: bool = False
) -> Callable[[], TrainConfig]:
    """Create the matched dense-94 unconstrained learned-ID reference."""

    def builder() -> TrainConfig:
        base = _september11_independent_bimodal_oer8k_lora(
            "urma4_backbone",
            total_updates=_sep18_urma_contract.TOTAL_UPDATES,
        )
        model = dataclasses.replace(
            base.model,
            backbone_conditioning=_input_config.LearnedEmbodimentBackboneToken(
                raw_id_codebook=_sep18_urma_contract.EMBODIMENT_IDS,
                token_width=2048,
            ),
        )
        dataset_cell: _sep18_urma_contract.Cell = "urma51"
        return dataclasses.replace(
            base,
            name=_sep18_urma_contract.learned_id_experiment_name(
                preemptible=preemptible
            ),
            model=model,
            data=dataclasses.replace(
                base.data,
                repo_id=_sep18_urma_contract.DATASET_ID,
                assets=dataclasses.replace(
                    base.data.assets,
                    asset_id=(
                        _sep18_urma_contract.NORMALIZATION_ASSET_IDS[
                            dataset_cell
                        ]
                    ),
                ),
                datasets=(
                    dataclasses.replace(
                        base.data.datasets[0],
                        data_subdir=(
                            _sep18_urma_contract.RLDS_RELEASE_NAMES[dataset_cell]
                        ),
                        version=_sep18_urma_contract.DATASET_VERSIONS[dataset_cell],
                        weight=1.0,
                    ),
                ),
                episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                    embodiment_ids=_sep18_urma_contract.EMBODIMENT_IDS,
                    episodes_per_embodiment=(
                        _sep18_urma_contract.EPISODES_PER_ROBOT
                    ),
                ),
                episode_split=None,
            ),
            freeze_filter=model.get_freeze_filter(),
            weight_loader=dataclasses.replace(
                base.weight_loader,
                missing_regex=r"(.*lora.*|embodiment_backbone_embed/.*)",
            ),
            num_train_steps=_sep18_urma_contract.TOTAL_UPDATES,
            save_interval=_sep18_urma_contract.checkpoint_every_updates(
                preemptible=preemptible
            ),
            keep_every_n_completed_updates=(
                _sep18_urma_contract.CHECKPOINT_RETENTION_EVERY_UPDATES
            ),
            val_interval=0,
            val_batches=0,
        )

    builder.__name__ = _sep18_urma_contract.learned_id_experiment_name(
        preemptible=preemptible
    )
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER19_DENSE94_LEARNED_ID_BUILDERS = tuple(
    _public_september19_dense94_learned_id_builder(
        preemptible=preemptible
    )
    for preemptible in (False, True)
)


def september19_dense94_urma4_flat1464pad2048() -> TrainConfig:
    """Direct, parameter-free URMA4 concatenation into one PaliGemma token."""

    base = _september11_independent_bimodal_oer8k_lora(
        "urma4_backbone",
        total_updates=_sep18_urma_contract.FLAT_URMA4_TOTAL_UPDATES,
    )
    extractor = _input_config.Urma4NewNormEmbodimentExtractor(
        bundle_dir=(
            f"{PI_EMBODIMENT_ASSETS_DIR}/"
            f"{_sep18_urma_contract.BUNDLE_NAMES['urma4_newnorm']}"
        ),
        bundle_sha256=_sep18_urma_contract.BUNDLE_SHA256["urma4_newnorm"],
    )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=_input_config.FlatUrma4BackboneToken(
            extractor=extractor,
            token_width=_sep18_urma_contract.FLAT_URMA4_TOKEN_WIDTH,
        ),
    )
    return dataclasses.replace(
        base,
        name=_sep18_urma_contract.flat_urma4_experiment_name(),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=_sep18_urma_contract.DATASET_ID,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    _sep18_urma_contract.NORMALIZATION_ASSET_IDS["urma4_newnorm"]
                ),
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=(
                        _sep18_urma_contract.RLDS_RELEASE_NAMES["urma4_newnorm"]
                    ),
                    version=(
                        _sep18_urma_contract.DATASET_VERSIONS["urma4_newnorm"]
                    ),
                    weight=1.0,
                ),
            ),
            episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                embodiment_ids=_sep18_urma_contract.EMBODIMENT_IDS,
                episodes_per_embodiment=_sep18_urma_contract.EPISODES_PER_ROBOT,
            ),
            episode_split=None,
        ),
        freeze_filter=model.get_freeze_filter(),
        weight_loader=dataclasses.replace(
            base.weight_loader,
            missing_regex=r".*lora.*",
        ),
        num_train_steps=_sep18_urma_contract.FLAT_URMA4_TOTAL_UPDATES,
        save_interval=_sep18_urma_contract.FLAT_URMA4_CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=(
            _sep18_urma_contract.FLAT_URMA4_CHECKPOINT_RETENTION_EVERY_UPDATES
        ),
        val_interval=0,
        val_batches=0,
    )


september19_dense94_urma4_flat1464pad2048.__name__ = (
    _sep18_urma_contract.flat_urma4_experiment_name()
)
september19_dense94_urma4_flat1464pad2048.__qualname__ = (
    september19_dense94_urma4_flat1464pad2048.__name__
)
globals()[september19_dense94_urma4_flat1464pad2048.__name__] = (
    september19_dense94_urma4_flat1464pad2048
)


def internal_september11_oer8k_norm_build_only() -> TrainConfig:
    """Conditioning-free native twin used only for train-view normalization."""

    base = _september11_independent_bimodal_oer8k_lora("text_t1")
    return dataclasses.replace(
        base,
        name="internal_september11_oer8k_norm_build_only",
        data=dataclasses.replace(
            base.data,
            input_config=dataclasses.replace(
                base.data.input_config,
                prompt=_input_config.NoPromptConditioning(),
            ),
        ),
    )


def internal_september20_native4k_norm_build_only() -> TrainConfig:
    """Conditioning-free Sep20 state/action view, independent of unpublished URMA assets."""

    base = internal_september11_oer8k_norm_build_only()
    return dataclasses.replace(
        base,
        name="internal_september20_native4k_norm_build_only",
        data=dataclasses.replace(
            base.data,
            repo_id=_sep20_contract.DATASET_ID,
            assets=dataclasses.replace(
                base.data.assets, asset_id=_sep20_contract.NORMALIZATION_ASSET_ID
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=_sep20_contract.DATASET_ID,
                    version="1.2.0",
                ),
            ),
        ),
    )


def internal_september21_mix6k_norm_build_only() -> TrainConfig:
    """Conditioning-free Sep21 mixture view used only for train-view normalization."""

    base = internal_september20_native4k_norm_build_only()
    return dataclasses.replace(
        base,
        name="internal_september21_mix6k_norm_build_only",
        save_interval=_sep21_mix_contract.CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=(
            _sep21_mix_contract.CHECKPOINT_RETENTION_EVERY_UPDATES
        ),
        data=dataclasses.replace(
            base.data,
            repo_id="20260921_oer_ur5e_dense2k_broad4k_mix6k_v1",
            assets=dataclasses.replace(
                base.data.assets, asset_id=_sep21_mix_contract.NORMALIZATION_ASSET_ID
            ),
        ),
    )


def internal_september23_clean2k_repeat100_mix6k_norm_build_only() -> TrainConfig:
    """Conditioning-free Sep23 mixture view used only for train-view normalization."""

    base = internal_september21_mix6k_norm_build_only()
    return dataclasses.replace(
        base,
        name="internal_september23_clean2k_repeat100_mix6k_norm_build_only",
        data=dataclasses.replace(
            base.data,
            repo_id=_sep23_mix_contract.DATASET_REPO_ID,
            assets=dataclasses.replace(
                base.data.assets, asset_id=_sep23_mix_contract.NORMALIZATION_ASSET_ID
            ),
            datasets=_september23_clean2k_repeat100_datasets("1.2.0"),
            episode_split=_september23_clean2k_repeat100_physical_split(),
        ),
    )


def internal_september24_repeat100_single_robot_norm_build_only() -> TrainConfig:
    """Conditioning-free Sep24 single-robot view used only to fit per-robot normalization.

    The repository adapter substitutes the experiment's selector, datasets, and
    normalization asset ID, so one view serves every registered robot.
    """

    base = _september24_repeat100_single_robot_view(_sep24_single_contract.ROBOT_IDS[0])
    model = dataclasses.replace(
        base.model, backbone_conditioning=_input_config.NoBackboneConditioning()
    )
    return dataclasses.replace(
        base,
        name="internal_september24_repeat100_single_robot_norm_build_only",
        model=model,
        freeze_filter=model.get_freeze_filter(),
    )


def august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h32_60k() -> TrainConfig:
    return _august29_id355_camd40p10e_lora_horizon(32)


def august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h50_60k() -> TrainConfig:
    return _august29_id355_camd40p10e_lora_horizon(50)


def september1_id355_camd400_current32d_lora_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    """Current all-32-coordinate objective control on the historical ID355 corpus."""

    base = _august29_id355_camd40p10e_lora_horizon(32)
    return dataclasses.replace(
        base,
        name="september1_id355_camd400_current32d_lora_gaussclip5_b128_lr1e5_seed42_24k",
        num_train_steps=24_000,
        data=dataclasses.replace(
            base.data,
            # TPU launches bind the common data root to the regional bucket. Keep the
            # exact dataset identity as a subdirectory instead of inheriting the
            # workstation-only single-dataset root, which resolves to bucket `/./`.
            datasets=tuple(
                dataclasses.replace(
                    dataset,
                    data_subdir=(
                        "20260827_ur5e_1emb_id0355_camD_slowdwell40p10e_"
                        "contiguous7_oerasync_v2"
                    ),
                )
                for dataset in base.data.datasets
            ),
        ),
    )


def september6_ur5e_vx300s_fingers_v2_center200_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1() -> TrainConfig:
    """Target-calibrated UR5e source-policy transfer gate."""

    name = (
        "september6_ur5e_vx300s_fingers_v2_center200_nocond_lora_"
        "gaussclip5_b128_lr2e4_seed42_h32_40k_v1"
    )
    dataset_id = "20260906_ur5e_vx300s_fingers_v2_center20p10e_camd_v1"
    base = _clean_pose_lora_native_config(
        name,
        repo_id=dataset_id,
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=40_000,
        dataset_version="1.2.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    "ur5e_vx300s_fingers_v2_center20p10e_"
                    "fulltrain_stdnorm_20260906"
                ),
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=dataset_id,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=dataclasses.replace(
            base.weight_loader,
            params_path="gs://boai-iris-central2/pi05_init/pi05_base/params",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=2e-4,
            decay_steps=40_000,
            decay_lr=2e-4,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=8_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=500, sample_count=4
        ),
    )


def _september13_center200_d70(episodes_per_pose: int) -> TrainConfig:
    """Native counterpart for one logical, pose-balanced center-200 view."""

    if episodes_per_pose not in (1, 2, 4, 6):
        raise ValueError(f"unsupported center-200 D70 episodes-per-pose: {episodes_per_pose}")
    episodes = 20 * episodes_per_pose
    base = september6_ur5e_vx300s_fingers_v2_center200_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1()
    return dataclasses.replace(
        base,
        name=(
            f"september13_center200_d70_n{episodes:03d}_nocond_lora_"
            "gaussclip5_b128_lr2e4_seed42_h32_12k_v1"
        ),
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    "ur5e_vx300s_fingers_v2_"
                    f"center20p{episodes_per_pose}e_d70_stdnorm_20260913"
                ),
            ),
            episode_selector=_data_processing.PoseEpisodeSelector(
                tuple(f"p{pose:02d}" for pose in range(1, 21)),
                episodes_per_pose=episodes_per_pose,
                seed=42,
            ),
        ),
        num_train_steps=12_000,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=12_000),
        save_interval=2_000,
        keep_every_n_completed_updates=2_000,
    )


def september13_center200_d70_n020_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1() -> TrainConfig:
    return _september13_center200_d70(1)


def september13_center200_d70_n040_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1() -> TrainConfig:
    return _september13_center200_d70(2)


def september13_center200_d70_n080_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1() -> TrainConfig:
    return _september13_center200_d70(4)


def september13_center200_d70_n120_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1() -> TrainConfig:
    return _september13_center200_d70(6)


def _september14_center200_d70_n040_replica(training_seed: int) -> TrainConfig:
    if training_seed not in (43, 44):
        raise ValueError(f"unsupported center-200 D70 N40 replica seed: {training_seed}")
    base = _september13_center200_d70(2)
    return dataclasses.replace(
        base,
        name=(
            "september14_center200_d70_n040_nocond_lora_"
            f"gaussclip5_b128_lr2e4_seed{training_seed}_h32_12k_v1"
        ),
        seed=training_seed,
        data=dataclasses.replace(base.data, rlds_sampler_seed=training_seed),
    )


def september14_center200_d70_n040_nocond_lora_gaussclip5_b128_lr2e4_seed43_h32_12k_v1() -> TrainConfig:
    return _september14_center200_d70_n040_replica(43)


def september14_center200_d70_n040_nocond_lora_gaussclip5_b128_lr2e4_seed44_h32_12k_v1() -> TrainConfig:
    return _september14_center200_d70_n040_replica(44)


def _september7_center200_appearance10x(part: str) -> TrainConfig:
    base = september6_ur5e_vx300s_fingers_v2_center200_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1()
    dataset_id = f"20260907_center200_{part}_appearance10x_camd_v1"
    return dataclasses.replace(
        base,
        name=(f"september7_center200_{part}_appearance10x_nocond_lora_"
              "gaussclip5_b128_lr2e4_seed42_h32_24k_v1"),
        num_train_steps=24_000,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=24_000),
        data=dataclasses.replace(
            base.data,
            repo_id=dataset_id,
            datasets=(dataclasses.replace(base.data.datasets[0], data_subdir=dataset_id),),
        ),
    )


def september7_center200_arm_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_24k_v1() -> TrainConfig:
    return _september7_center200_appearance10x("arm")


def september7_center200_gripper_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_24k_v1() -> TrainConfig:
    return _september7_center200_appearance10x("gripper")


def _september7_center200_appearance10x_48k(part: str) -> TrainConfig:
    base = _september7_center200_appearance10x(part)
    return dataclasses.replace(
        base,
        name=base.name.replace("_24k_v1", "_48k_continue_v1"),
        num_train_steps=48_000,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=48_000),
    )


def september7_center200_arm_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_48k_continue_v1() -> TrainConfig:
    return _september7_center200_appearance10x_48k("arm")


def september7_center200_gripper_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_48k_continue_v1() -> TrainConfig:
    return _september7_center200_appearance10x_48k("gripper")


def september7_center200_arm_appearance10x_hardware_smoke_100_v1() -> TrainConfig:
    return dataclasses.replace(
        _september7_center200_appearance10x("arm"),
        name="september7_center200_arm_appearance10x_hardware_smoke_100_v1",
        num_train_steps=100,
    )


def august29_id355_slow200_unique230_camd400_equalmix_lora_h32_b384_lr3e5_seed42_60k() -> TrainConfig:
    """Equal-weight three-source cross-viewpoint aggregate."""

    name = (
        "august29_id355_slow200_unique230_camd400_equalmix_"
        "lora_h32_b384_lr3e5_seed42_60k"
    )
    base = _clean_pose_lora_native_config(
        name,
        repo_id="20260829_ur5e_id0355_slow200_unique230_camd400_equalmix_n830_v1",
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=60_000,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id="id355_slow200_unique230_camd400_equalmix_n830_stdnorm_20260829",
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1 / 3,
                    data_subdir="20260826_ur5e_1emb_id0355_slowdwell20p10e_contiguous7_v1",
                    joint_layout="legacy_left_packed_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1 / 3,
                    data_subdir="20260823_1emb_id0355_bal15x10_plus_online80_n230_movingfirst_r1",
                    joint_layout="cross_family_arm7_gripper7_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1 / 3,
                    data_subdir=(
                        "20260827_ur5e_1emb_id0355_camD_slowdwell40p10e_"
                        "contiguous7_oerasync_v2"
                    ),
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
        ),
        batch_size=384,
        fsdp_devices=32,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=3e-5,
            decay_steps=60_000,
            decay_lr=3e-5,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=500, sample_count=4
        ),
    )


def august29_ur5e_trossen_v6_camd8k_baseline_lora_gaussclip5_b256_seed42_lr4e5_60k() -> TrainConfig:
    """No-conditioning B256 baseline on the frozen motion-qualified Camera-D 8K view."""

    name = (
        "august29_ur5e_trossen_v6_camd8k_baseline_lora_"
        "gaussclip5_b256_seed42_lr4e5_60k"
    )
    base = _clean_pose_lora_native_config(
        name,
        repo_id="20260829_ur5e_trossen_v6_motioncompatible8k_camd_v1",
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=_input_config.GlobalStandardNorm(),
            action_clipping=_input_config.NormalizedActionClip(5.0),
        ),
        num_train_steps=60_000,
        dataset_version="1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir="20260829_ur5e_trossen_v6_motioncompatible8k_camd_v1",
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
        ),
        batch_size=256,
        fsdp_devices=32,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=4e-5,
            decay_steps=60_000,
            decay_lr=4e-5,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=500, sample_count=4
        ),
    )


def august29_ur5e_trossen_v6_camd8k_urma3_lora_gaussclip5_b256_seed42_lr4e5_60k() -> TrainConfig:
    """URMA3 sibling of the frozen Camera-D v6 8K baseline."""

    base = (
        august29_ur5e_trossen_v6_camd8k_baseline_lora_gaussclip5_b256_seed42_lr4e5_60k()
    )
    return dataclasses.replace(
        base,
        name=(
            "august29_ur5e_trossen_v6_camd8k_urma3_lora_"
            "gaussclip5_b256_seed42_lr4e5_60k"
        ),
        model=dataclasses.replace(
            base.model,
            action_expert_conditioning=(
                _urma_config.ur5e_trossen_v6_camd8k_urma3_concat_project_adarms(
                    bundle_sha256=(
                        "b371863ca5416f1a1d26ab58b96e0577eeb52f330c9e2a70eebc1414adb5e29b"
                    )
                )
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params",
            missing_regex=(
                r"(.*lora.*|embodiment_encoder/.*|"
                r"embodiment_fusion_(in|out)/.*)"
            ),
        ),
    )


def _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
    name: str,
    *,
    cell: Literal[
        "learnedid_backbone",
        "urma4_adarms",
        "urma4_backbone",
        "urma4_text",
        "response_history_backbone",
    ],
    normalization_asset_id: str,
    urma4_bundle_sha256: str | None = None,
    response_history_stats_sha256: str | None = None,
) -> TrainConfig:
    """Native realization of one matched September 1 conditioning cell."""

    action_expert_conditioning: _input_config.ActionExpertConditioning = (
        _input_config.TimestepOnly()
    )
    backbone_conditioning: _input_config.BackboneConditioning = (
        _input_config.NoBackboneConditioning()
    )
    input_config = _input_config.Pi05InputConfig(
        normalization=_input_config.GlobalStandardNorm(),
        action_clipping=_input_config.NormalizedActionClip(5.0),
    )
    if cell == "learnedid_backbone":
        backbone_conditioning = _input_config.LearnedEmbodimentBackboneToken(
            raw_id_codebook=_sep1_contract.learned_id_codebook(), token_width=2048
        )
    elif cell in {"urma4_adarms", "urma4_backbone", "urma4_text"}:
        if urma4_bundle_sha256 is None:
            raise ValueError(f"{cell} requires the immutable URMA4 bundle SHA-256")
        if cell == "urma4_adarms":
            action_expert_conditioning = (
                _urma_config.ur5e_trossen_v6_camd8k_urma4_concat_project_adarms(
                    bundle_sha256=urma4_bundle_sha256
                )
            )
        elif cell == "urma4_backbone":
            backbone_conditioning = (
                _urma_config.ur5e_trossen_v6_camd8k_urma4_backbone_token(
                    bundle_sha256=urma4_bundle_sha256
                )
            )
        else:
            input_config = dataclasses.replace(
                input_config,
                prompt=_input_config.Urma4BundleTextPrompt(
                    bundle_dir=(
                        f"{PI_EMBODIMENT_ASSETS_DIR}/"
                        f"{_sep1_contract.URMA4_BUNDLE_NAME}"
                    ),
                    bundle_sha256=urma4_bundle_sha256,
                    max_token_len=768,
                ),
            )
    elif cell == "response_history_backbone":
        if response_history_stats_sha256 is None:
            raise ValueError("response-history cell requires derived Gaussian stats SHA-256")
        input_config = dataclasses.replace(
            input_config,
            response_history=_input_config.CausalResponseHistoryBackboneToken(
                stats_sha256=response_history_stats_sha256
            ),
        )
    else:
        raise ValueError(f"unsupported September 1 conditioning cell: {cell!r}")

    response_history = cell == "response_history_backbone"
    dataset_id = (
        _sep1_contract.RESPONSE_HISTORY_DATASET_ID
        if response_history
        else "20260829_ur5e_trossen_v6_motioncompatible8k_camd_v1"
    )
    base = _clean_pose_lora_native_config(
        name,
        repo_id=dataset_id,
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=input_config,
        action_expert_conditioning=action_expert_conditioning,
        backbone_conditioning=backbone_conditioning,
        num_train_steps=24_000,
        dataset_version="1.2.0" if response_history else "1.1.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(base.data.assets, asset_id=normalization_asset_id),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=dataset_id,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=dataclasses.replace(
            base.weight_loader,
            params_path="gs://boai-iris-central2/pi05_init/pi05_base/params",
        ),
        freeze_filter=base.model.get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=1e-5,
            decay_steps=24_000,
            decay_lr=1e-5,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=8_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000, sample_count=4
        ),
    )


def _public_september1_urma4_cell(
    cell: Literal[
        "learnedid_backbone", "urma4_adarms", "urma4_backbone", "urma4_text"
    ],
) -> TrainConfig:
    name = (
        f"september1_ur5e_trossen_v6_camd8k_{cell}_lora_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )
    return _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
        name,
        cell=cell,
        normalization_asset_id=(
            "ur5e_trossen_v6_motioncompatible8k_camd_fulltrain_stdnorm_20260829"
        ),
        urma4_bundle_sha256=(
            None
            if cell == "learnedid_backbone"
            else _sep1_contract.URMA4_SOURCE_BUNDLE_SHA256
        ),
    )


def september1_ur5e_trossen_v6_camd8k_learnedid_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _public_september1_urma4_cell("learnedid_backbone")


def september1_ur5e_trossen_v6_camd8k_urma4_adarms_lora_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _public_september1_urma4_cell("urma4_adarms")


def september1_ur5e_trossen_v6_camd8k_urma4_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _public_september1_urma4_cell("urma4_backbone")


def september1_ur5e_trossen_v6_camd8k_urma4_text_lora_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _public_september1_urma4_cell("urma4_text")


def september1_ur5e_trossen_v6_camd8k_response_history_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    """Public Cell E bound to the measured full-8K history statistics."""

    return _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
        (
            "september1_ur5e_trossen_v6_camd8k_response_history_backbone_lora_"
            "gaussclip5_b128_seed42_lr1e5_24k"
        ),
        cell="response_history_backbone",
        normalization_asset_id=_sep1_contract.RESPONSE_HISTORY_NORM_ASSET_ID,
        response_history_stats_sha256=(
            _sep1_contract.RESPONSE_HISTORY_NORMALIZATION_STATS_SHA256
        ),
    )


def _september2_ur5e_trossen_v6_camd8k_robotokens_backbone(
    name: str, *, full_ft: bool
) -> TrainConfig:
    """Matched learned-ID protocol with a full static RoboToken prefix sequence."""
    base = september1_ur5e_trossen_v6_camd8k_learnedid_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k()
    conditioning = _input_config.RobotTokenBackboneSequence(
        bundle_dir=(
            f"{PI_EMBODIMENT_ASSETS_DIR}/{_sep1_contract.ROBOTOKEN_BUNDLE_NAME}"
        ),
        bundle_sha256=_sep1_contract.ROBOTOKEN_SOURCE_BUNDLE_SHA256,
    )
    model = dataclasses.replace(
        base.model,
        backbone_conditioning=conditioning,
        **(
            {"paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"}
            if full_ft
            else {}
        ),
    )
    return dataclasses.replace(
        base,
        name=name,
        model=model,
        weight_loader=dataclasses.replace(
            base.weight_loader,
            missing_regex=(
                r"(robotoken_encoder/.*)"
                if full_ft
                else r"(.*lora.*|robotoken_encoder/.*)"
            ),
        ),
        freeze_filter=nnx.Nothing() if full_ft else model.get_freeze_filter(),
    )


def september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _september2_ur5e_trossen_v6_camd8k_robotokens_backbone(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k",
        full_ft=False,
    )


def september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _september2_ur5e_trossen_v6_camd8k_robotokens_backbone(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k",
        full_ft=True,
    )


def _september2_ur5e_trossen_v6_camd8k_robotokens_backbone_40k(
    name: str, *, full_ft: bool
) -> TrainConfig:
    """Exact optimizer-lineage continuation of one RoboToken cell through 40K."""

    base = _september2_ur5e_trossen_v6_camd8k_robotokens_backbone(
        name, full_ft=full_ft
    )
    return dataclasses.replace(
        base,
        num_train_steps=40_000,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=40_000),
    )


def september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _september2_ur5e_trossen_v6_camd8k_robotokens_backbone_40k(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k",
        full_ft=False,
    )


def september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _september2_ur5e_trossen_v6_camd8k_robotokens_backbone_40k(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k",
        full_ft=True,
    )


def _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
    episode_count: Literal[2_000, 4_000],
    *,
    total_updates: Literal[24_000, 40_000],
    process_sharded_file_shuffle_v1: bool = False,
) -> TrainConfig:
    name = (
        _sep1_contract.robotoken_data_scaling_processshard_v1_experiment_name(
            episode_count, total_updates=total_updates
        )
        if process_sharded_file_shuffle_v1
        else _sep1_contract.robotoken_data_scaling_experiment_name(
            episode_count, total_updates=total_updates
        )
    )
    base = _september2_ur5e_trossen_v6_camd8k_robotokens_backbone(
        name,
        full_ft=True,
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            episode_selector=_data_processing.HashEpisodeSelector(
                episode_count=episode_count,
                seed=_sep1_contract.ROBOTOKEN_DATA_SCALING_SELECTOR_SEED,
            ),
            rlds_input_sampling=(
                _data_processing.ProcessShardedFileShuffleV1()
                if process_sharded_file_shuffle_v1
                else _data_processing.LegacyRldsInputSampling()
            ),
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=(
                    _sep1_contract.ROBOTOKEN_DATA_SCALING_NORM_ASSET_IDS[
                        episode_count
                    ]
                ),
            ),
        ),
        num_train_steps=total_updates,
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            decay_steps=total_updates,
        ),
    )


def september4_ur5e_trossen_v6_camd8k_hash2k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
        2_000, total_updates=24_000
    )


def september4_ur5e_trossen_v6_camd8k_hash2k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
        2_000, total_updates=40_000
    )


def september4_ur5e_trossen_v6_camd8k_hash4k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k() -> TrainConfig:
    return _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
        4_000, total_updates=24_000
    )


def september4_ur5e_trossen_v6_camd8k_hash4k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
        4_000, total_updates=40_000
    )


def september4_ur5e_trossen_v6_camd8k_hash2k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k_processshardv1() -> TrainConfig:
    return _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
        2_000, total_updates=24_000, process_sharded_file_shuffle_v1=True
    )


def september4_ur5e_trossen_v6_camd8k_hash4k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k_processshardv1() -> TrainConfig:
    return _september4_ur5e_trossen_v6_camd8k_robotokens_fullft_data_scaling(
        4_000, total_updates=24_000, process_sharded_file_shuffle_v1=True
    )


def _september6_ur5e_trossen_v6_camd8k_fullft(
    name: str, *, cell: _sep6_contract.Cell
) -> TrainConfig:
    """Native twin for one corrected-pipeline September 6 FullFT cell."""

    input_config = _input_config.Pi05InputConfig(
        normalization=_input_config.GlobalStandardNorm(),
        action_clipping=_input_config.NormalizedActionClip(5.0),
    )
    backbone_conditioning: _input_config.BackboneConditioning = (
        _input_config.NoBackboneConditioning()
    )
    missing_regex = r"(?!)"
    if cell == "urma4_backbone":
        backbone_conditioning = (
            _urma_config.ur5e_trossen_v6_camd8k_urma4_backbone_token(
                bundle_sha256=_sep1_contract.URMA4_SOURCE_BUNDLE_SHA256
            )
        )
        missing_regex = r"(embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)"
    elif cell == "robotokens_backbone":
        backbone_conditioning = _input_config.RobotTokenBackboneSequence(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/{_sep1_contract.ROBOTOKEN_BUNDLE_NAME}"
            ),
            bundle_sha256=_sep1_contract.ROBOTOKEN_SOURCE_BUNDLE_SHA256,
        )
        missing_regex = r"(robotoken_encoder/.*)"
    elif cell in {"text_t1", "text_t2"}:
        is_t1 = cell == "text_t1"
        input_config = dataclasses.replace(
            input_config,
            prompt=_input_config.TextBundlePrompt(
                bundle_dir=(
                    f"{PI_EMBODIMENT_ASSETS_DIR}/"
                    f"{_sep6_contract.TEXT_T1_BUNDLE_NAME if is_t1 else _sep6_contract.TEXT_T2_BUNDLE_NAME}"
                ),
                bundle_sha256=(
                    _sep6_contract.TEXT_T1_BUNDLE_SHA256
                    if is_t1
                    else _sep6_contract.TEXT_T2_BUNDLE_SHA256
                ),
                max_token_len=(
                    _sep6_contract.TEXT_T1_MAX_TOKEN_LEN
                    if is_t1
                    else _sep6_contract.TEXT_T2_MAX_TOKEN_LEN
                ),
                format_id=(
                    _sep6_contract.TEXT_T1_FORMAT_ID
                    if is_t1
                    else _sep6_contract.TEXT_T2_FORMAT_ID
                ),
            ),
        )
    elif cell != "nocond":
        raise ValueError(f"unsupported September 6 conditioning cell: {cell!r}")

    dataset_id = "20260829_ur5e_trossen_v6_motioncompatible8k_camd_v1"
    base = _clean_pose_lora_native_config(
        name,
        repo_id=dataset_id,
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=input_config,
        action_expert_conditioning=_input_config.TimestepOnly(),
        backbone_conditioning=backbone_conditioning,
        num_train_steps=_sep6_contract.TOTAL_UPDATES,
        dataset_version="1.1.0",
    )
    model = dataclasses.replace(
        base.model,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    return dataclasses.replace(
        base,
        model=model,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets, asset_id=_sep6_contract.NORM_ASSET_ID
            ),
            datasets=(
                dataclasses.replace(
                    base.data.datasets[0],
                    data_subdir=dataset_id,
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
            episode_split=_sep6_contract.camd8k_validation_holdout_split(),
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            rlds_sampler_seed=_sep6_contract.SAMPLER_SEED,
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=dataclasses.replace(
            base.weight_loader,
            params_path="gs://boai-iris-central2/pi05_init/pi05_base/params",
            missing_regex=missing_regex,
        ),
        freeze_filter=nnx.Nothing(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=2e-5,
            decay_steps=_sep6_contract.TOTAL_UPDATES,
            decay_lr=2e-5,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=8_000,
        val_interval=1_000,
        val_batches=4,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000, sample_count=4
        ),
    )


def _public_september6_camd8k_fullft_builder(
    cell: _sep6_contract.Cell,
) -> Callable[[], TrainConfig]:
    """Create a discoverable builder whose symbol/name is the immutable config ID."""

    def builder() -> TrainConfig:
        return _september6_ur5e_trossen_v6_camd8k_fullft(
            _sep6_contract.experiment_name(cell), cell=cell
        )

    builder.__name__ = _sep6_contract.experiment_name(cell)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER6_CAMD8K_FULLFT_BUILDERS = tuple(
    _public_september6_camd8k_fullft_builder(cell) for cell in _sep6_contract.CELLS
)


def september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_overfit10_noaug_b32_seed42_lr1e4_1k() -> TrainConfig:
    """Real-gradient memorization smoke over a deterministic ten-episode view."""

    base = _september2_ur5e_trossen_v6_camd8k_robotokens_backbone(
        (
            "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_"
            "lora_overfit10_noaug_b32_seed42_lr1e4_1k"
        ),
        full_ft=False,
    )
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, augmentation_mode="none"),
        data=dataclasses.replace(
            base.data,
            episode_selector=_data_processing.HashEpisodeSelector(
                episode_count=10,
                seed=42,
            ),
        ),
        batch_size=32,
        rlds_num_parallel_reads=1,
        rlds_num_parallel_calls=1,
        num_train_steps=1_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=50,
            peak_lr=1e-4,
            decay_steps=1_000,
            decay_lr=1e-4,
            initial_lr=1e-8,
        ),
        save_interval=250,
        keep_period=None,
        keep_every_n_completed_updates=250,
        val_interval=0,
        val_batches=0,
        log_interval=10,
        train_sample_interval=100,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=100,
            sample_count=4,
        ),
    )


def september3_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_overfit_first10_noaug_b32_seed42_lr1e4_1k() -> TrainConfig:
    """Fast memorization smoke that reads exactly ten TFDS examples."""

    base = september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_overfit10_noaug_b32_seed42_lr1e4_1k()
    return dataclasses.replace(
        base,
        name=(
            "september3_ur5e_trossen_v6_camd8k_robotokens_backbone_"
            "lora_overfit_first10_noaug_b32_seed42_lr1e4_1k"
        ),
        data=dataclasses.replace(
            base.data,
            rlds_split="train[:10]",
            episode_selector=None,
        ),
    )


def _public_september6_camd8k_fullft_24k_continuation_builder(
    cell: _sep6_contract.Cell,
) -> Callable[[], TrainConfig]:
    """Create the native twin for one exact 16K-to-24K continuation."""

    public_name = _sep6_contract.continuation_experiment_name(cell)

    def builder() -> TrainConfig:
        base = _september6_ur5e_trossen_v6_camd8k_fullft(public_name, cell=cell)
        return dataclasses.replace(
            base,
            num_train_steps=_sep6_contract.CONTINUATION_TOTAL_UPDATES,
            lr_schedule=dataclasses.replace(
                base.lr_schedule,
                decay_steps=_sep6_contract.CONTINUATION_TOTAL_UPDATES,
            ),
        )

    builder.__name__ = public_name
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_SEPTEMBER6_CAMD8K_FULLFT_24K_CONTINUATION_BUILDERS = tuple(
    _public_september6_camd8k_fullft_24k_continuation_builder(cell)
    for cell in _sep6_contract.CELLS
)


def _public_september6_camd8k_fullft_32k_continuation_builder(
    cell: _sep6_contract.Cell,
) -> Callable[[], TrainConfig]:
    public_name = _sep6_contract.continuation32k_experiment_name(cell)

    def builder() -> TrainConfig:
        base = _september6_ur5e_trossen_v6_camd8k_fullft(public_name, cell=cell)
        return dataclasses.replace(
            base,
            num_train_steps=_sep6_contract.LONG_CONTINUATION_TOTAL_UPDATES,
        )

    builder.__name__ = public_name
    builder.__qualname__ = public_name
    globals()[public_name] = builder
    return builder


_SEPTEMBER6_CAMD8K_FULLFT_32K_CONTINUATION_BUILDERS = tuple(
    _public_september6_camd8k_fullft_32k_continuation_builder(cell)
    for cell in _sep6_contract.LONG_CONTINUATION_CELLS
)


def _oer9_joint_fullft(data: _oer9_contract.TrainingData, cell: _oer9_contract.Cell) -> TrainConfig:
    """Native twin of one OER-9 cell of one training generation: the Sep-6 full-FT shell with
    XEMB variable-DoF data (``oer9_training_contract.V2``/``V3`` differ only in dataset pins).

    Construction never raises for unpublished assets (so registry/alias resolution keeps working);
    training fails closed instead: the repository refuses a missing normalization hash, the
    discovery selector IDs are absent from any OER-9 build, and the zero URMA4 bundle hash
    cannot match a bundle.
    """

    extractor = _input_config.Urma4PhysicalStateEmbodimentExtractor(
        bundle_dir=f"{PI_EMBODIMENT_ASSETS_DIR}/{data.urma4_bundle_name}",
        bundle_sha256=data.urma4_bundle_sha256,
    )
    backbone_conditioning, missing_regex = {
        "urma4": (
            _input_config.ExtractedEmbodimentBackboneToken(extractor=extractor, token_width=2048),
            r"(embodiment_encoder/.*|embodiment_token_(norm|proj)/.*)",
        ),
        # The flat token has no learned parameters (width = PaliGemma width).
        "urma4flat": (_input_config.FlatUrma4BackboneToken(extractor=extractor, token_width=2048), r"(?!)"),
        "none": (_input_config.NoBackboneConditioning(), r"(?!)"),
    }[cell]
    base = _september6_ur5e_trossen_v6_camd8k_fullft(data.experiment_name(cell), cell="nocond")
    n_arm = data.robot_n_arm()
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, backbone_conditioning=backbone_conditioning),
        data=dataclasses.replace(
            base.data,
            repo_id=data.dataset_id,
            assets=dataclasses.replace(base.data.assets, asset_id=data.normalization_asset_id),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=data.dataset_id,
                    data_subdir=data.dataset_id,
                    name="molmospaces_emb_mix",
                    version=_oer9_contract.DATASET_VERSION,
                    weight=1.0,
                    joint_layout=_oer9_contract.JOINT_LAYOUT,
                    target_joint_layout=_oer9_contract.JOINT_LAYOUT,
                ),
            ),
            input_config=_input_config.Pi05InputConfig(
                state_serialization=_input_config.CompactLeftPackedState(),
                normalization=data.normalization(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
                action_loss_coordinates="physical",
            ),
            episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                embodiment_ids=tuple(sorted(n_arm)),
                episodes_per_embodiment=data.episodes_per_robot,
            ),
            episode_split=None,
            rlds_split="train",
        ),
        weight_loader=dataclasses.replace(base.weight_loader, missing_regex=missing_regex),
        num_train_steps=data.total_updates,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=data.total_updates),
        save_interval=_oer9_contract.CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=_oer9_contract.RETAIN_AND_EVALUATE_EVERY_UPDATES,
        val_interval=0,
        val_batches=0,
    )


def _public_oer9_joint_fullft_builder(
    data: _oer9_contract.TrainingData, cell: _oer9_contract.Cell
) -> Callable[[], TrainConfig]:
    def builder() -> TrainConfig:
        return _oer9_joint_fullft(data, cell)

    builder.__name__ = data.experiment_name(cell)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_OCTOBER12_OER9_JOINT_FULLFT_BUILDERS = tuple(
    _public_oer9_joint_fullft_builder(_oer9_contract.V2, cell) for cell in _oer9_contract.CELLS
)
_OCTOBER13_OER9_JOINT_FULLFT_BUILDERS = tuple(
    _public_oer9_joint_fullft_builder(_oer9_contract.V3, cell) for cell in _oer9_contract.CELLS
)


def _public_october14_oer9_joint_fullft88k_builder(cell: _oer9_contract.Cell) -> Callable[[], TrainConfig]:
    """Native twin of the 88K continuation of one october13 cell (checkpoint lineage set by the repository)."""

    public_name = _oer9_contract.extended_experiment_name(cell)

    def builder() -> TrainConfig:
        base = _oer9_joint_fullft(_oer9_contract.V3, cell)
        return dataclasses.replace(
            base,
            name=public_name,
            num_train_steps=_oer9_contract.EXTENDED_TOTAL_UPDATES,
            lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=_oer9_contract.EXTENDED_TOTAL_UPDATES),
        )

    builder.__name__ = public_name
    builder.__qualname__ = public_name
    globals()[public_name] = builder
    return builder


_OCTOBER14_OER9_JOINT_FULLFT88K_BUILDERS = tuple(
    _public_october14_oer9_joint_fullft88k_builder(cell) for cell in _oer9_contract.CELLS
)


def october26_oer9_joint_urma4_queryanchored_fullft() -> TrainConfig:
    """Native twin of the V3 urma4 cell with query-anchored JOINT actions (``oer9_training_contract``)."""
    base = _oer9_joint_fullft(_oer9_contract.V3, _oer9_contract.QUERY_ANCHORED_CELL)
    return dataclasses.replace(
        base,
        name=_oer9_contract.QUERY_ANCHORED_EXPERIMENT_NAME,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets, asset_id=_oer9_contract.QUERY_ANCHORED_NORMALIZATION_ASSET_ID
            ),
            input_config=dataclasses.replace(
                base.data.input_config, arm_action=_input_config.QueryAnchoredJoint()
            ),
        ),
    )


def _october25_twin_gen3_pnp(episodes: int) -> TrainConfig:
    """Native twin of one Oct-25 twin-Gen3 PnP cell: the Sep-24 single-robot LoRA view (the XEMB
    specialists' native base) at H16 with EMA 0.999, no robot token and the OER-9 unified-arm data.

    Construction never raises for an unpublished normalization asset; the repository refuses to
    train without its pinned hash, and the selector refuses a build without exactly the twin's
    ``episodes`` episodes.
    """

    base = _september24_repeat100_single_robot_view(1)
    model = dataclasses.replace(
        base.model,
        action_horizon=_oct25_twin_contract.PREDICTION_HORIZON,
        backbone_conditioning=_input_config.NoBackboneConditioning(),
    )
    dataset_id = _oct25_twin_contract.dataset_id(episodes)
    total_updates = _oct25_twin_contract.TOTAL_UPDATES[episodes]
    every = _oct25_twin_contract.CHECKPOINT_EVERY_UPDATES
    return dataclasses.replace(
        base,
        name=_oct25_twin_contract.experiment_name(episodes),
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=dataset_id,
            assets=dataclasses.replace(
                base.data.assets, asset_id=_oct25_twin_contract.normalization_asset_id(episodes)
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=dataset_id,
                    data_subdir=dataset_id,
                    name="molmospaces_emb_mix",
                    version=_oct25_twin_contract.DATASET_VERSION,
                    weight=1.0,
                    joint_layout=_oct25_twin_contract.JOINT_LAYOUT,
                    target_joint_layout=_oct25_twin_contract.JOINT_LAYOUT,
                ),
            ),
            input_config=_input_config.Pi05InputConfig(
                state_serialization=_input_config.CompactLeftPackedState(),
                normalization=_oct25_twin_contract.normalization(),
                action_clipping=_input_config.NormalizedActionClip(5.0),
                action_loss_coordinates="physical",
            ),
            episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                embodiment_ids=(_oct25_twin_contract.TWIN_EMBODIMENT_ID,),
                episodes_per_embodiment=episodes,
            ),
            episode_split=None,
            rlds_split="train",
        ),
        freeze_filter=model.get_freeze_filter(),
        # Only the LoRA adapters are new; there is no embodiment encoder or token projection.
        weight_loader=dataclasses.replace(base.weight_loader, missing_regex=r".*lora.*"),
        ema_decay=_oct25_twin_contract.EMA_DECAY,
        num_train_steps=total_updates,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=total_updates),
        save_interval=every,
        keep_every_n_completed_updates=every,
        train_sample_interval=every,
        data_visualization=dataclasses.replace(base.data_visualization, interval=every),
        val_interval=0,
        val_batches=0,
    )


def _public_october25_twin_gen3_pnp_builder(episodes: int) -> Callable[[], TrainConfig]:
    def builder() -> TrainConfig:
        return _october25_twin_gen3_pnp(episodes)

    builder.__name__ = _oct25_twin_contract.experiment_name(episodes)
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_OCTOBER25_TWIN_GEN3_PNP_BUILDERS = tuple(
    _public_october25_twin_gen3_pnp_builder(episodes) for episodes in _oct25_twin_contract.EPISODE_COUNTS
)


def _october25_twin_variant(variant: _oct25_twin_contract.Variant) -> TrainConfig:
    """Native twin of one twin follow-up cell: the base twin cell with quantile state normalization (actions
    unchanged), at the variant's batch, constant LR, budget and checkpoint cadence."""

    base = _october25_twin_gen3_pnp(variant.episodes)
    every = variant.checkpoint_every
    return dataclasses.replace(
        base,
        name=variant.name,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=_oct25_twin_contract.normalization_asset_id(variant.episodes, state_quantiles=True),
            ),
            input_config=dataclasses.replace(
                base.data.input_config, normalization=_oct25_twin_contract.normalization(state_quantiles=True)
            ),
        ),
        batch_size=variant.global_batch_size,
        num_train_steps=variant.updates,
        lr_schedule=dataclasses.replace(
            base.lr_schedule, peak_lr=variant.peak_lr, decay_lr=variant.peak_lr, decay_steps=variant.updates
        ),
        save_interval=every,
        keep_every_n_completed_updates=every,
        train_sample_interval=every,
        data_visualization=dataclasses.replace(base.data_visualization, interval=every),
    )


def _public_october25_twin_variant_builder(variant: _oct25_twin_contract.Variant) -> Callable[[], TrainConfig]:
    def builder() -> TrainConfig:
        return _october25_twin_variant(variant)

    builder.__name__ = variant.name
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_OCTOBER25_TWIN_VARIANT_BUILDERS = tuple(
    _public_october25_twin_variant_builder(variant) for variant in _oct25_twin_contract.VARIANTS
)


def _october28_action_freq(run: _oct28_freq_contract.Run) -> TrainConfig:
    """Native twin of one Oct-28 action-frequency cell: the n40 twin cell with query-anchored joint labels on the
    run's rate builds, its normalization and asset, save every 1K and retain every 2K. The time-channel cell adds
    ``action_time_proj`` to the model (zero-initialized, absent from pi05_base, trainable under the LoRA filter)."""

    base = _october25_twin_gen3_pnp(_oct28_freq_contract.EPISODES)
    action_time = _oct28_freq_contract.action_time(run)
    model = dataclasses.replace(
        base.model,
        action_time_conditioning=action_time,
        action_horizon=run.horizon,
        # The URMA4-flat token (R5) has no parameters: weight loader and freeze filter stay the twin's.
        backbone_conditioning=_oct28_freq_contract.backbone_conditioning(run),
    )
    every = _oct28_freq_contract.RETAIN_EVERY_UPDATES
    return dataclasses.replace(
        base,
        name=run.name,
        model=model,
        data=dataclasses.replace(
            base.data,
            repo_id=run.dataset_repo_id,
            assets=dataclasses.replace(
                base.data.assets, asset_id=_oct28_freq_contract.normalization_asset_id(run)
            ),
            datasets=tuple(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=dataset_id,
                    data_subdir=dataset_id,
                    name="molmospaces_emb_mix",
                    version=_oct25_twin_contract.DATASET_VERSION,
                    weight=run.weight,
                    joint_layout=_oct25_twin_contract.JOINT_LAYOUT,
                    target_joint_layout=_oct25_twin_contract.JOINT_LAYOUT,
                )
                for dataset_id in run.dataset_ids
            ),
            input_config=dataclasses.replace(
                base.data.input_config,
                arm_action=_input_config.QueryAnchoredJoint(),
                normalization=_oct28_freq_contract.normalization(run),
                action_time=action_time,
                prompt=_oct28_freq_contract.prompt(run, base.data.input_config.prompt),
            ),
            episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                embodiment_ids=(_oct25_twin_contract.TWIN_EMBODIMENT_ID,),
                episodes_per_embodiment=_oct28_freq_contract.EPISODES * len(run.rates),
            ),
        ),
        freeze_filter=model.get_freeze_filter(),
        weight_loader=(
            dataclasses.replace(base.weight_loader, missing_regex=r"(?:.*lora.*|action_time_proj/.*)")
            if action_time.enabled
            else base.weight_loader
        ),
        save_interval=_oct28_freq_contract.CHECKPOINT_EVERY_UPDATES,
        keep_every_n_completed_updates=every,
        train_sample_interval=every,
        data_visualization=dataclasses.replace(base.data_visualization, interval=every),
    )


def _public_october28_action_freq_builder(run: _oct28_freq_contract.Run) -> Callable[[], TrainConfig]:
    def builder() -> TrainConfig:
        return _october28_action_freq(run)

    builder.__name__ = run.name
    builder.__qualname__ = builder.__name__
    globals()[builder.__name__] = builder
    return builder


_OCTOBER28_ACTION_FREQ_BUILDERS = tuple(
    _public_october28_action_freq_builder(run)
    for run in _oct28_freq_contract.ALL_RUNS
)


def november05_xsrc40_exclv2_qanchor_tact_lbox_gclose_b1024() -> TrainConfig:
    """Native twin of the xsrc40 exclusion-v2 cell (``nov05_xsrc40_exclv2_training_contract``): the Sep-6 full-FT
    shell with the pop34fk query-anchored R2 recipe, one RLDS component per collection family with its embodiment ID
    offset, mobile-base slots at each robot's own control period, letterbox_stretched_v1 frames, closedness gripper
    state at load time, the action time channel (``action_time_proj`` zero-initialized, absent from pi05_base, the
    only weight the loader may initialize), a namespaced 5% robot holdout, no robot token, save every 1K, retain
    every 4K, W&B on."""

    contract = _nov05_xsrc40_contract
    base = _september6_ur5e_trossen_v6_camd8k_fullft(contract.CELL_NAME, cell="nocond")
    input_config = contract.input_config()
    return dataclasses.replace(
        base,
        name=contract.CELL_NAME,
        model=dataclasses.replace(
            base.model,
            action_horizon=contract.PREDICTION_HORIZON,
            augmentation_mode="shared_mild",
            backbone_conditioning=_input_config.NoBackboneConditioning(),
            action_time_conditioning=input_config.action_time,
        ),
        data=dataclasses.replace(
            base.data,
            repo_id=contract.DATASET_ID,
            rlds_data_dir=contract.GPU_DATA_DIR,
            assets=dataclasses.replace(base.data.assets, asset_id=contract.ASSET_ID),
            datasets=tuple(
                droid_rlds_dataset.RLDSDataset(
                    dataset_id=component.dataset_id,
                    data_subdir=component.data_subdir,
                    name=component.name,
                    version=component.version,
                    weight=component.weight,
                    joint_layout=component.joint_layout,
                    target_joint_layout=component.target_joint_layout,
                    embodiment_id_offset=component.embodiment_id_offset,
                )
                for component in contract.components()
            ),
            input_config=input_config,
            episode_selector=None,
            episode_split=contract.episode_split(),
            rlds_split="train",
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            rlds_sampler_seed=contract.SAMPLER_SEED,
        ),
        weight_loader=dataclasses.replace(
            base.weight_loader, params_path=contract.PI05_BASE_PARAMS, missing_regex=r"action_time_proj/.*"
        ),
        batch_size=contract.GLOBAL_BATCH_SIZE,
        seed=contract.SEED,
        num_train_steps=contract.TOTAL_UPDATES,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=contract.WARMUP_UPDATES,
            initial_lr=contract.INITIAL_LR,
            peak_lr=contract.PEAK_LR,
            decay_steps=contract.TOTAL_UPDATES,
            decay_lr=contract.DECAY_LR,
        ),
        optimizer=_optimizer.AdamW(
            weight_decay=contract.WEIGHT_DECAY, clip_gradient_norm=contract.GRADIENT_CLIP_NORM
        ),
        ema_decay=None,
        save_interval=contract.CHECKPOINT_EVERY_UPDATES,
        keep_period=None,
        keep_every_n_completed_updates=contract.RETAIN_EVERY_UPDATES,
        val_interval=contract.RETAIN_EVERY_UPDATES,
        val_batches=contract.VALIDATION_BATCHES,
        train_sample_interval=contract.RETAIN_EVERY_UPDATES,
        data_visualization=dataclasses.replace(base.data_visualization, interval=contract.RETAIN_EVERY_UPDATES),
        wandb_enabled=True,
        project_name=contract.WANDB_PROJECT,
    )


def _november06_xsrc40_exclv2_posquant(io: str) -> TrainConfig:
    """Native twin of oer12-prod (``io2``) or its bridge (``io1``) (``nov06_xsrc40_exclv2_posquant_contract``): the
    exclv2 native cell with R4's per-position quantile statistics, the flat URMA4-absolute token (no learned
    parameters, so ``action_time_proj`` stays the only weight the loader may initialize) and the cell's model I/O."""

    contract = _nov06_posquant_contract
    base = november05_xsrc40_exclv2_qanchor_tact_lbox_gclose_b1024()
    return dataclasses.replace(
        base,
        name=contract.cell_name(io),
        model=dataclasses.replace(base.model, backbone_conditioning=contract.backbone_conditioning(io)),
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(base.data.assets, asset_id=contract.ASSET_IDS[io]),
            input_config=contract.input_config(io),
        ),
    )


def _public_november06_xsrc40_exclv2_posquant_builder(io: str) -> Callable[[], TrainConfig]:
    def builder() -> TrainConfig:
        return _november06_xsrc40_exclv2_posquant(io)

    builder.__name__ = builder.__qualname__ = _nov06_posquant_contract.cell_name(io)
    globals()[builder.__name__] = builder
    return builder


_NOVEMBER06_XSRC40_EXCLV2_POSQUANT_BUILDERS = tuple(
    _public_november06_xsrc40_exclv2_posquant_builder(io) for io in _nov06_posquant_contract.IOS
)


def october27_pop34fk_joint_urma4flat_fullft80k() -> TrainConfig:
    """Native twin of ``october27_pop34fk_joint_urma4flat_fullft80k`` (``pop34fk_training_contract``).

    The OER-9 ``urma4flat`` native cell on generation ``POP34FK`` with the pop34fk recipe: B2048, linear
    warmup 100 updates from 1e-8 to 4e-5, cosine decay to 2e-5 at update 80K (``decay_steps`` counts
    from step 0), save every 1K, retain every 8K, W&B on. A fresh run: no checkpoint lineage.
    """

    contract = _pop34fk_contract
    data = contract.DATA
    base = _oer9_joint_fullft(data, contract.CELL)
    extractor = contract.EXTRACTOR(
        bundle_dir=f"{PI_EMBODIMENT_ASSETS_DIR}/{data.urma4_bundle_name}",
        bundle_sha256=data.urma4_bundle_sha256,
    )
    return dataclasses.replace(
        base,
        name=contract.EXPERIMENT_NAME,
        model=dataclasses.replace(
            base.model,
            action_horizon=contract.PREDICTION_HORIZON,
            augmentation_mode="shared_mild",
            backbone_conditioning=_input_config.FlatUrma4BackboneToken(extractor=extractor, token_width=2048),
        ),
        data=dataclasses.replace(
            base.data,
            rlds_input_sampling=_data_processing.DeterministicReplayV3(),
            rlds_sampler_seed=contract.SAMPLER_SEED,
        ),
        weight_loader=dataclasses.replace(base.weight_loader, params_path=contract.PI05_BASE_PARAMS),
        batch_size=contract.GLOBAL_BATCH_SIZE,
        seed=contract.SEED,
        num_train_steps=contract.TOTAL_UPDATES,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=contract.WARMUP_UPDATES,
            initial_lr=contract.INITIAL_LR,
            peak_lr=contract.PEAK_LR,
            decay_steps=contract.DECAY_UPDATES,
            decay_lr=contract.DECAY_LR,
        ),
        optimizer=_optimizer.AdamW(
            weight_decay=contract.WEIGHT_DECAY, clip_gradient_norm=contract.GRADIENT_CLIP_NORM
        ),
        ema_decay=None,
        save_interval=contract.CHECKPOINT_EVERY_UPDATES,
        keep_period=None,
        keep_every_n_completed_updates=contract.RETAIN_AND_EVALUATE_EVERY_UPDATES,
        wandb_enabled=True,
        project_name=contract.WANDB_PROJECT,
    )


def october27_pop34fk_qanchor_b1024() -> TrainConfig:
    """Native twin of ``october27_pop34fk_qanchor_b1024``: the pop34fk URMA4-flat native cell with query-anchored
    JOINT actions, their own normalization asset, and global batch 1024 (``pop34fk_training_contract.QANCHOR_*``)."""

    contract = _pop34fk_contract
    base = october27_pop34fk_joint_urma4flat_fullft80k()
    return dataclasses.replace(
        base,
        name=contract.QANCHOR_B1024_EXPERIMENT_NAME,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(base.data.assets, asset_id=contract.QANCHOR_NORMALIZATION_ASSET_ID),
            input_config=dataclasses.replace(base.data.input_config, arm_action=_input_config.QueryAnchoredJoint()),
        ),
        batch_size=contract.QANCHOR_B1024_GLOBAL_BATCH_SIZE,
    )


def _october28_pop34fk_qanchor_variant(
    base: TrainConfig,
    name: str,
    *,
    normalization: _input_config.Normalization | None = None,
    asset_id: str | None = None,
    **fields: object,
) -> TrainConfig:
    """One step of the pop34fk query-anchored ablation chain (``pop34fk_training_contract``): ``base`` renamed,
    retained every 4K, with an optional normalization scheme and asset."""

    input_config = base.data.input_config
    return dataclasses.replace(
        base,
        name=name,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(base.data.assets, asset_id=asset_id or base.data.assets.asset_id),
            input_config=dataclasses.replace(input_config, normalization=normalization or input_config.normalization),
        ),
        keep_every_n_completed_updates=_pop34fk_contract.ABLATION_RETAIN_AND_EVALUATE_EVERY_UPDATES,
        **fields,
    )


def october28_pop34fk_qanchor_b2048() -> TrainConfig:
    """Native twin of ablation R1: ``october27_pop34fk_qanchor_b1024`` at global batch 2048."""

    contract = _pop34fk_contract
    return _october28_pop34fk_qanchor_variant(
        october27_pop34fk_qanchor_b1024(), contract.QANCHOR_B2048_EXPERIMENT_NAME, batch_size=contract.GLOBAL_BATCH_SIZE
    )


def october28_pop34fk_qanchor_posgauss_b1024() -> TrainConfig:
    """Native twin of ablation R3: ``october27_pop34fk_qanchor_b1024`` with Gaussian action statistics per chunk
    position and q01/q99 state."""

    contract = _pop34fk_contract
    base = october27_pop34fk_qanchor_b1024()
    return _october28_pop34fk_qanchor_variant(
        base,
        contract.POSGAUSS_EXPERIMENT_NAME,
        normalization=_input_config.GroupedChunkPositionStandardNormQuantileState(
            base.data.input_config.normalization.groups
        ),
        asset_id=contract.PERPOSITION_H32_NORMALIZATION_ASSET_ID,
    )


def october28_pop34fk_qanchor_posquant_b1024() -> TrainConfig:
    """Native twin of ablation R4: R3 with q01/q99 -> -1/+1 action statistics per chunk position."""

    base = october28_pop34fk_qanchor_posgauss_b1024()
    return _october28_pop34fk_qanchor_variant(
        base,
        _pop34fk_contract.POSQUANT_EXPERIMENT_NAME,
        normalization=_input_config.GroupedChunkPositionQuantileNorm(base.data.input_config.normalization.groups),
    )


def october28_pop34fk_qanchor_posquant_h50_b1024() -> TrainConfig:
    """Native twin of ablation R5: R4 at prediction horizon 50 with its own per-position asset."""

    contract = _pop34fk_contract
    base = october28_pop34fk_qanchor_posquant_b1024()
    return _october28_pop34fk_qanchor_variant(
        base,
        contract.POSQUANT_H50_EXPERIMENT_NAME,
        asset_id=contract.PERPOSITION_H50_NORMALIZATION_ASSET_ID,
        model=dataclasses.replace(base.model, action_horizon=contract.H50_PREDICTION_HORIZON),
    )


def _october29_pop34fk_qanchor_embcount(robots: int) -> TrainConfig:
    """Native twin of ``october29_pop34fk_qanchor_emb{robots}_b32``: the qanchor native cell on the first ``robots``
    of ``pop34fk_training_contract.EMBCOUNT_ROBOTS``, read from their shard subset, at B32."""

    contract = _pop34fk_contract
    base = october27_pop34fk_qanchor_b1024()
    (dataset,) = base.data.datasets
    return dataclasses.replace(
        base,
        name=contract.embcount_experiment_name(robots),
        data=dataclasses.replace(
            base.data,
            datasets=(dataclasses.replace(dataset, data_subdir=contract.embcount_data_subdir(robots)),),
            episode_selector=_data_processing.ExplicitEmbodimentEpisodeSelector(
                embodiment_ids=contract.EMBCOUNT_ROBOTS[:robots]
            ),
        ),
        batch_size=contract.EMBCOUNT_GLOBAL_BATCH_SIZE,
        num_train_steps=contract.EMBCOUNT_TOTAL_UPDATES,
        keep_every_n_completed_updates=contract.EMBCOUNT_RETAIN_AND_EVALUATE_EVERY_UPDATES,
    )


def october29_pop34fk_qanchor_emb1_b32() -> TrainConfig:
    return _october29_pop34fk_qanchor_embcount(1)


def october29_pop34fk_qanchor_emb10_b32() -> TrainConfig:
    return _october29_pop34fk_qanchor_embcount(10)


def _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_40k(
    name: str,
    *,
    cell: Literal[
        "learnedid_backbone",
        "urma4_adarms",
        "urma4_backbone",
        "urma4_text",
        "response_history_backbone",
    ],
    normalization_asset_id: str,
    urma4_bundle_sha256: str | None = None,
    response_history_stats_sha256: str | None = None,
) -> TrainConfig:
    """Immutable 40K continuation of one matched September 1 native cell."""

    base = _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
        name,
        cell=cell,
        normalization_asset_id=normalization_asset_id,
        urma4_bundle_sha256=urma4_bundle_sha256,
        response_history_stats_sha256=response_history_stats_sha256,
    )
    return dataclasses.replace(
        base,
        num_train_steps=40_000,
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=40_000),
    )


def _public_september1_urma4_cell_40k(
    cell: Literal[
        "learnedid_backbone", "urma4_adarms", "urma4_backbone", "urma4_text"
    ],
) -> TrainConfig:
    return _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_40k(
        _sep1_contract.conditioning_experiment_name(
            {
                "learnedid_backbone": "a",
                "urma4_adarms": "b",
                "urma4_backbone": "c",
                "urma4_text": "d",
            }[cell],
            total_updates=40_000,
        ),
        cell=cell,
        normalization_asset_id=(
            "ur5e_trossen_v6_motioncompatible8k_camd_fulltrain_stdnorm_20260829"
        ),
        urma4_bundle_sha256=(
            None
            if cell == "learnedid_backbone"
            else _sep1_contract.URMA4_SOURCE_BUNDLE_SHA256
        ),
    )


def september1_ur5e_trossen_v6_camd8k_learnedid_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _public_september1_urma4_cell_40k("learnedid_backbone")


def september1_ur5e_trossen_v6_camd8k_urma4_adarms_lora_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _public_september1_urma4_cell_40k("urma4_adarms")


def september1_ur5e_trossen_v6_camd8k_urma4_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _public_september1_urma4_cell_40k("urma4_backbone")


def september1_ur5e_trossen_v6_camd8k_urma4_text_lora_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _public_september1_urma4_cell_40k("urma4_text")


def september1_ur5e_trossen_v6_camd8k_response_history_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k() -> TrainConfig:
    return _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_40k(
        _sep1_contract.conditioning_experiment_name("e", total_updates=40_000),
        cell="response_history_backbone",
        normalization_asset_id=_sep1_contract.RESPONSE_HISTORY_NORM_ASSET_ID,
        response_history_stats_sha256=(
            _sep1_contract.RESPONSE_HISTORY_NORMALIZATION_STATS_SHA256
        ),
    )


def internal_september1_ur5e_trossen_v6_camd8k_response_history_norm_build_only() -> TrainConfig:
    """Tooling-only input/dataset shape for deriving Cell E normalization stats.

    The impossible digest is intentional: no public training experiment refers
    to this native name, and the final Cell E registration must replace it with
    the measured, manifest-bound statistics digest.
    """

    return _september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
        _sep1_contract.RESPONSE_HISTORY_NORM_BUILD_NATIVE_CONFIG,
        cell="response_history_backbone",
        normalization_asset_id=_sep1_contract.RESPONSE_HISTORY_NORM_ASSET_ID,
        response_history_stats_sha256="0" * 64,
    )


def _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
    name: str,
    *,
    normalization: _input_config.Normalization,
    action_expert_conditioning: _input_config.ActionExpertConditioning,
    normalization_asset_id: str,
    backbone_conditioning: _input_config.BackboneConditioning | None = None,
    full_ft: bool = False,
    action_loss_coordinates: _input_config.ActionLossCoordinates = "model",
    num_train_steps: int = 24_000,
    dataset_mode: Literal["mixed", "i013877"] = "mixed",
) -> TrainConfig:
    """Native realization of one dense two-embodiment Camera-D study cell."""

    base = _clean_pose_lora_native_config(
        name,
        repo_id=(
            "20260831_ur5e_twoemb_id0355_i013877_camD_400each_equalmix_contiguous7_v1"
            if dataset_mode == "mixed"
            else "20260831_ur5e_random8k_i013877_camD_slowdwell40p10e_contiguous7_v1"
        ),
        rlds_data_dir="/iris/u/boai/artifact/rlds_data",
        input_config=_input_config.Pi05InputConfig(
            normalization=normalization,
            action_clipping=_input_config.NormalizedActionClip(5.0),
            action_loss_coordinates=action_loss_coordinates,
        ),
        action_expert_conditioning=action_expert_conditioning,
        backbone_conditioning=(
            _input_config.NoBackboneConditioning()
            if backbone_conditioning is None
            else backbone_conditioning
        ),
        num_train_steps=num_train_steps,
        dataset_version="1.1.0",
    )
    conditioned = not isinstance(action_expert_conditioning, _input_config.TimestepOnly)
    learned_backbone = isinstance(
        backbone_conditioning, _input_config.LearnedEmbodimentBackboneToken
    )
    model = (
        dataclasses.replace(
            base.model,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        )
        if full_ft
        else base.model
    )
    return dataclasses.replace(
        base,
        model=model,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(
                base.data.assets,
                asset_id=normalization_asset_id,
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=0.5,
                    data_subdir=(
                        "20260827_ur5e_1emb_id0355_camD_slowdwell40p10e_"
                        "contiguous7_oerasync_v2"
                    ),
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=0.5,
                    data_subdir=(
                        "20260831_ur5e_random8k_i013877_camD_slowdwell40p10e_"
                        "contiguous7_v1"
                    ),
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            )
            if dataset_mode == "mixed"
            else (
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.1.0",
                    weight=1.0,
                    data_subdir=(
                        "20260831_ur5e_random8k_i013877_camD_slowdwell40p10e_"
                        "contiguous7_v1"
                    ),
                    joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                    target_joint_layout="ur5e_contiguous_arm6_gripper6_v1",
                ),
            ),
        ),
        batch_size=128,
        fsdp_devices=32,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://boai-iris-central2/pi05_init/pi05_base/params",
            missing_regex=(
                r"(embodiment_backbone_embed/.*)"
                if full_ft
                else r"(.*lora.*|embodiment_backbone_embed/.*)"
                if learned_backbone
                else r"(.*lora.*|embodiment_encoder/.*|embodiment_fusion_(in|out)/.*)"
                if conditioned
                else r"(.*lora.*)"
            ),
        ),
        freeze_filter=nnx.Nothing() if full_ft else model.get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=1e-5,
            decay_steps=num_train_steps,
            decay_lr=1e-5,
            initial_lr=1e-8,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=500 if full_ft else 1_000,
        keep_period=None,
        keep_every_n_completed_updates=500 if full_ft else 4_000,
        val_interval=0,
        val_batches=0,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=1_000, sample_count=4
        ),
    )


def august31_twoemb800_pooled_nocond_lora_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "august31_twoemb800_pooled_nocond_lora_gaussclip5_b128_lr1e5_seed42_24k",
        normalization=_input_config.GlobalStandardNorm(),
        action_expert_conditioning=_input_config.TimestepOnly(),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_equalmix_fulltrain_stdnorm_20260831"
        ),
    )


def august31_twoemb800_perrobot_nocond_lora_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "august31_twoemb800_perrobot_nocond_lora_gaussclip5_b128_lr1e5_seed42_24k",
        normalization=_input_config.PerEmbodimentStandardNorm((355, 13_877)),
        action_expert_conditioning=_input_config.TimestepOnly(),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_perrobot_fulltrain_stdnorm_20260831"
        ),
    )


def august31_twoemb800_pooled_urma3_lora_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "august31_twoemb800_pooled_urma3_lora_gaussclip5_b128_lr1e5_seed42_24k",
        normalization=_input_config.GlobalStandardNorm(),
        action_expert_conditioning=(
            _urma_config.ur5e_twoemb800_urma3_concat_project_adarms(
                bundle_sha256=(
                    "bfc29b100962cfdc6083dd864cd763a068974e50115b93bd308c6f355dde32eb"
                )
            )
        ),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_equalmix_fulltrain_stdnorm_20260831"
        ),
    )


def august31_twoemb800_pooled_learnedid_backbone_lora_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "august31_twoemb800_pooled_learnedid_backbone_lora_gaussclip5_b128_lr1e5_seed42_24k",
        normalization=_input_config.GlobalStandardNorm(),
        action_expert_conditioning=_input_config.TimestepOnly(),
        backbone_conditioning=_input_config.LearnedEmbodimentBackboneToken(
            vocab_size=13_878
        ),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_equalmix_fulltrain_stdnorm_20260831"
        ),
    )


def august31_twoemb800_perrobot_learnedid_backbone_lora_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "august31_twoemb800_perrobot_learnedid_backbone_lora_gaussclip5_b128_lr1e5_seed42_24k",
        normalization=_input_config.PerEmbodimentStandardNorm((355, 13_877)),
        action_expert_conditioning=_input_config.TimestepOnly(),
        backbone_conditioning=_input_config.LearnedEmbodimentBackboneToken(
            vocab_size=13_878
        ),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_perrobot_fulltrain_stdnorm_20260831"
        ),
    )


def august31_twoemb800_perrobot_learnedid_backbone_full_gaussclip5_b128_lr1e5_seed42_24k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "august31_twoemb800_perrobot_learnedid_backbone_full_gaussclip5_b128_lr1e5_seed42_24k",
        normalization=_input_config.PerEmbodimentStandardNorm((355, 13_877)),
        action_expert_conditioning=_input_config.TimestepOnly(),
        backbone_conditioning=_input_config.LearnedEmbodimentBackboneToken(
            vocab_size=13_878
        ),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_perrobot_fulltrain_stdnorm_20260831"
        ),
        full_ft=True,
    )


def september1_twoemb800_perrobot_nocond_physical7_lora_gaussclip5_b128_lr1e5_seed42_8k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "september1_twoemb800_perrobot_nocond_physical7_lora_gaussclip5_b128_lr1e5_seed42_8k",
        normalization=_input_config.PerEmbodimentStandardNorm((355, 13_877)),
        action_expert_conditioning=_input_config.TimestepOnly(),
        normalization_asset_id=(
            "twoemb800_id0355_i013877_perrobot_fulltrain_stdnorm_20260831"
        ),
        action_loss_coordinates="physical",
        num_train_steps=8_000,
    )


def september1_i013877_camd400_physical7_lora_gaussclip5_b128_lr1e5_seed42_8k() -> TrainConfig:
    return _august31_twoemb800_gaussclip5_b128_lr1e5_24k(
        "september1_i013877_camd400_physical7_lora_gaussclip5_b128_lr1e5_seed42_8k",
        normalization=_input_config.PerEmbodimentStandardNorm((13_877,)),
        action_expert_conditioning=_input_config.TimestepOnly(),
        normalization_asset_id=(
            "i013877_camd400_projected_perrobot_stdnorm_20260901"
        ),
        action_loss_coordinates="physical",
        num_train_steps=8_000,
        dataset_mode="i013877",
    )


def _august25_id355_balanced150_early4k(
    name: str, *, quantile: bool, full_ft: bool
) -> TrainConfig:
    base = _august24_id355_balanced150_franka_lora_20k(
        name,
        input_config=_input_config.Pi05InputConfig(
            normalization=(
                _input_config.GlobalQuantileNorm()
                if quantile
                else _input_config.GlobalStandardNorm()
            ),
            action_clipping=_input_config.NoActionClipping(),
        ),
    )
    model = (
        dataclasses.replace(
            base.model,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        )
        if full_ft
        else base.model
    )
    return dataclasses.replace(
        base,
        model=model,
        weight_loader=(
            weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params",
                missing_regex=r"(?!)",
            )
            if full_ft
            else base.weight_loader
        ),
        freeze_filter=nnx.Nothing() if full_ft else model.get_freeze_filter(),
        num_train_steps=4_000,
        save_interval=500,
        keep_period=None,
        keep_every_n_completed_updates=500,
        val_interval=500,
        train_sample_interval=500,
        data_visualization=_input_config.DataVisualizationConfig(
            interval=500, sample_count=4
        ),
    )


def august25_id355_balanced150_frankagauss_lora_b128_seed42_save500_4k() -> TrainConfig:
    return _august25_id355_balanced150_early4k(
        "august25_id355_balanced150_frankagauss_lora_b128_seed42_save500_4k",
        quantile=False,
        full_ft=False,
    )


def august25_id355_balanced150_frankaquant_lora_b128_seed42_save500_4k() -> TrainConfig:
    return _august25_id355_balanced150_early4k(
        "august25_id355_balanced150_frankaquant_lora_b128_seed42_save500_4k",
        quantile=True,
        full_ft=False,
    )


def august25_id355_balanced150_frankagauss_fullft_b128_seed42_save500_4k() -> TrainConfig:
    return _august25_id355_balanced150_early4k(
        "august25_id355_balanced150_frankagauss_fullft_b128_seed42_save500_4k",
        quantile=False,
        full_ft=True,
    )


def august25_id355_balanced150_frankaquant_fullft_b128_seed42_save500_4k() -> TrainConfig:
    return _august25_id355_balanced150_early4k(
        "august25_id355_balanced150_frankaquant_fullft_b128_seed42_save500_4k",
        quantile=True,
        full_ft=True,
    )


def august24_id355_unique230_frankagauss_lora_b128_seed42_resume20k_to40k() -> TrainConfig:
    """Native recipe for exact optimizer-state continuation in the original lineage."""

    return dataclasses.replace(
        august23_id355_unique230_frankagauss_lora_b128_seed42_20k(),
        name=(
            "august24_id355_unique230_frankagauss_lora_b128_seed42_"
            "resume20k_to40k"
        ),
        num_train_steps=40_000,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


def august25_id355_balanced150_frankagauss_lora_b128_seed42_resume20k_to40k() -> TrainConfig:
    """Native recipe for exact balanced150 Gaussian continuation through 40K."""

    return dataclasses.replace(
        august24_id355_balanced150_frankagauss_lora_b128_seed42_save1k_20k(),
        name=(
            "august25_id355_balanced150_frankagauss_lora_b128_seed42_"
            "resume20k_to40k"
        ),
        num_train_steps=40_000,
        save_interval=1_000,
        keep_period=None,
        keep_every_n_completed_updates=4_000,
    )


_CONFIG_BUILDERS += (
    july16_cleanpose_h32_open_frankav013_10p10e,
    july16_cleanpose_h32_open_frankav013_5p10e,
    july22_cleanpose_h32_open_frankav013_5p5e,
    july16_cleanpose_h32_open_frankav013_10p5e,
    july18_fixcap_h32_open_frankav013,
    july18_fixcap_h32_open_frankav066,
    july18_fixcap_h32_open_franka2emb,
    july18_fixcap_warmupema_h32_open_frankav066_16k,
    july18_fixcap_warmupema_h32_open_franka2emb_16k,
    july19_fixcap_warmupema_h32_open_frankav066_20k,
    july19_fixcap_warmupema_h32_open_franka2emb_20k,
    july19_fixcap_warmupema_h32_open_franka2emb_b128_32k,
    july19_fixcap_warmupema_h32_open_franka2emb_b256_20k,
    july18_fixcap_warmupema_h32_open_franka9emb_64k,
    july19_fixcap_warmupema_h32_open_franka_remaining81_b2048_64k,
    july20_fixcap_warmupema_h32_open_franka_remaining81_b1024_64k,
    july21_fixcap_remaining81_varianttext_heldout9_inference,
    july20_fixcap_warmupema_h32_open_franka_remaining81_noembtext_b1024_64k,
    july21_fixcap_warmupema_h32_open_franka_remaining81_noembtext_globalnorm_b1024_64k,
    july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm_b1024_64k,
    july24_fixcap_warmupema_h32_open_franka334emb_5p5e_noembtext_globalnorm_b2048_64k,
    july26_fixcap_warmupema_h32_open_r334_exact8095_noembtext_globalnorm_b1024_64k,
    july29_fixcap_warmupema_h32_open_r334_exact8095_noembtext_banknorm_b1024_64k,
    july29_fixcap_warmupema_h32_open_r334_r81hom_exact8095_noembtext_globalnorm_b1024_64k,
    july27_oer_hash8095_noembtext_globalnorm_b1024_64k,
    july27_oer_hash25000_noembtext_globalnorm_b1024_64k,
    july27_oer_hash50000_noembtext_globalnorm_b1024_64k,
    july28_oer_hash8095_noembtext_globalnorm_clip5_val5_b1024_64k,
    july28_oer_hash25000_noembtext_globalnorm_clip5_val5_b1024_64k,
    july28_oer_hash50000_noembtext_globalnorm_clip5_val5_b1024_64k,
    july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k,
    august30_oer_hash8095_noembtext_compactstate_stdnorm_clip5_val5_b1024_64k,
    august23_id355_unique230_frankagauss_lora_b128_seed42_20k,
    august23_id355_unique230_currentgauss_lora_b1024_seed0_20k,
    august15_oer_phase0_repeat1000x30_id999_ep29753_learnedid_token_stdnorm_clip5_b1024_48k,
    august15_oer_phase0_repeat1000x30_id999_ep29753_learnedid_token_stdnorm_clip5_b512_96k,
    august23_oer_phase0_repeat1000x30_id999_ep29753_nocond_seed0_lora_stdnorm_clip5_b1024_48k,
    august23_oer_phase0_repeat1000x30_id999_ep29753_nocond_seed0_full_stdnorm_clip5_b1024_48k,
    august24_id355_balanced150_frankagauss_lora_b128_seed42_save1k_20k,
    august24_id355_balanced150_frankaquant_lora_b128_seed42_save1k_20k,
    august25_id355_balanced150_contiguous_frankagauss_lora_b128_seed42_save1k_20k,
    august25_id355_balanced150_contiguous_frankaquant_lora_b128_seed42_save1k_20k,
    august27_id355_mixed430_contiguous_gaussclip5_lora_b128_seed42_lr1e5_60k,
    august29_ur5e_trossen_v6_camd8k_baseline_lora_gaussclip5_b256_seed42_lr4e5_60k,
    august29_ur5e_trossen_v6_camd8k_urma3_lora_gaussclip5_b256_seed42_lr4e5_60k,
    september1_ur5e_trossen_v6_camd8k_learnedid_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k,
    september1_ur5e_trossen_v6_camd8k_urma4_adarms_lora_gaussclip5_b128_seed42_lr1e5_24k,
    september1_ur5e_trossen_v6_camd8k_urma4_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k,
    september1_ur5e_trossen_v6_camd8k_urma4_text_lora_gaussclip5_b128_seed42_lr1e5_24k,
    september1_ur5e_trossen_v6_camd8k_response_history_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k,
    september1_ur5e_trossen_v6_camd8k_learnedid_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k,
    september1_ur5e_trossen_v6_camd8k_urma4_adarms_lora_gaussclip5_b128_seed42_lr1e5_40k,
    september1_ur5e_trossen_v6_camd8k_urma4_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k,
    september1_ur5e_trossen_v6_camd8k_urma4_text_lora_gaussclip5_b128_seed42_lr1e5_40k,
    september1_ur5e_trossen_v6_camd8k_response_history_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k,
    september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_gaussclip5_b128_seed42_lr1e5_24k,
    september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k,
    september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_gaussclip5_b128_seed42_lr1e5_40k,
    september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k,
    september4_ur5e_trossen_v6_camd8k_hash2k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k,
    september4_ur5e_trossen_v6_camd8k_hash2k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k,
    september4_ur5e_trossen_v6_camd8k_hash4k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k,
    september4_ur5e_trossen_v6_camd8k_hash4k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_40k,
    september4_ur5e_trossen_v6_camd8k_hash2k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k_processshardv1,
    september4_ur5e_trossen_v6_camd8k_hash4k_s42_robotokens_backbone_fullft_gaussclip5_b128_seed42_lr1e5_24k_processshardv1,
    *_SEPTEMBER6_CAMD8K_FULLFT_BUILDERS,
    september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_overfit10_noaug_b32_seed42_lr1e4_1k,
    september3_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_overfit_first10_noaug_b32_seed42_lr1e4_1k,
    *_SEPTEMBER6_CAMD8K_FULLFT_32K_CONTINUATION_BUILDERS,
    *_OCTOBER12_OER9_JOINT_FULLFT_BUILDERS,
    *_OCTOBER13_OER9_JOINT_FULLFT_BUILDERS,
    *_OCTOBER14_OER9_JOINT_FULLFT88K_BUILDERS,
    october26_oer9_joint_urma4_queryanchored_fullft,
    october27_pop34fk_joint_urma4flat_fullft80k,
    october27_pop34fk_qanchor_b1024,
    october28_pop34fk_qanchor_b2048,
    october28_pop34fk_qanchor_posgauss_b1024,
    october28_pop34fk_qanchor_posquant_b1024,
    october28_pop34fk_qanchor_posquant_h50_b1024,
    october29_pop34fk_qanchor_emb1_b32,
    october29_pop34fk_qanchor_emb10_b32,
    *_OCTOBER25_TWIN_GEN3_PNP_BUILDERS,
    *_OCTOBER25_TWIN_VARIANT_BUILDERS,
    *_OCTOBER28_ACTION_FREQ_BUILDERS,
    november05_xsrc40_exclv2_qanchor_tact_lbox_gclose_b1024,
    *_NOVEMBER06_XSRC40_EXCLV2_POSQUANT_BUILDERS,
    *_SEPTEMBER6_CAMD8K_FULLFT_24K_CONTINUATION_BUILDERS,
    internal_september1_ur5e_trossen_v6_camd8k_response_history_norm_build_only,
    august31_twoemb800_pooled_nocond_lora_gaussclip5_b128_lr1e5_seed42_24k,
    august31_twoemb800_perrobot_nocond_lora_gaussclip5_b128_lr1e5_seed42_24k,
    august31_twoemb800_pooled_urma3_lora_gaussclip5_b128_lr1e5_seed42_24k,
    august31_twoemb800_pooled_learnedid_backbone_lora_gaussclip5_b128_lr1e5_seed42_24k,
    august31_twoemb800_perrobot_learnedid_backbone_lora_gaussclip5_b128_lr1e5_seed42_24k,
    august31_twoemb800_perrobot_learnedid_backbone_full_gaussclip5_b128_lr1e5_seed42_24k,
    september1_twoemb800_perrobot_nocond_physical7_lora_gaussclip5_b128_lr1e5_seed42_8k,
    september1_i013877_camd400_physical7_lora_gaussclip5_b128_lr1e5_seed42_8k,
    august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h32_60k,
    august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h50_60k,
    september1_id355_camd400_current32d_lora_gaussclip5_b128_lr1e5_seed42_24k,
    september6_ur5e_vx300s_fingers_v2_center200_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1,
    september13_center200_d70_n020_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1,
    september13_center200_d70_n040_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1,
    september13_center200_d70_n080_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1,
    september13_center200_d70_n120_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_12k_v1,
    september14_center200_d70_n040_nocond_lora_gaussclip5_b128_lr2e4_seed43_h32_12k_v1,
    september14_center200_d70_n040_nocond_lora_gaussclip5_b128_lr2e4_seed44_h32_12k_v1,
    september7_center200_arm_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_24k_v1,
    september7_center200_gripper_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_24k_v1,
    september7_center200_arm_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_48k_continue_v1,
    september7_center200_gripper_appearance10x_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_48k_continue_v1,
    september7_center200_arm_appearance10x_hardware_smoke_100_v1,
    september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_nocond_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1,
    september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v1,
    september9_oer_ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_40k_v2,
    september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1,
    september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1,
    september10_oer_ur5e_bridge2942_bounded_noretry_plus_calibratedcenter200_equalmix_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1,
    september10_oer_ur5e_bridge4k_plus_calibratedcenter200_equalmix_physicalgripperstate_urma4_backbone_physical7_lora_gaussclip5_b128_lr2e4_seed42_h32_16k_v1,
    *_SEPTEMBER11_INDEPENDENT_BIMODAL_OER8K_BUILDERS,
    _SEPTEMBER20_NATIVECENTERED_4K_BUILDER,
    *_SEPTEMBER21_NATIVE_MIX6K_BUILDERS,
    _SEPTEMBER22_NEARCENTER2K_URMA4_BUILDER,
    *_SEPTEMBER23_CLEAN2K_REPEAT100_MIX6K_BUILDERS,
    *_SEPTEMBER24_REPEAT100_SINGLE_ROBOT_BUILDERS,
    *_SEPTEMBER13_INDEPENDENT_BIMODAL_OER_DATA_SCALING_BUILDERS,
    _SEPTEMBER16_OER8K_CENTER200_MIXTURE_BUILDER,
    *_SEPTEMBER19_CENTER200_URMA51_NOISE_BUILDERS,
    *_SEPTEMBER14_REUSE100_DENSE_BUILDERS,
    *_SEPTEMBER17_URMA4_THREEWAY_BUILDERS,
    *_SEPTEMBER18_DENSE94_URMA456_BUILDERS,
    *_SEPTEMBER19_DENSE94_LEARNED_ID_BUILDERS,
    september19_dense94_urma4_flat1464pad2048,
    internal_september11_oer8k_norm_build_only,
    internal_september20_native4k_norm_build_only,
    internal_september21_mix6k_norm_build_only,
    internal_september23_clean2k_repeat100_mix6k_norm_build_only,
    internal_september24_repeat100_single_robot_norm_build_only,
    august29_id355_slow200_unique230_camd400_equalmix_lora_h32_b384_lr3e5_seed42_60k,
    august25_id355_balanced150_frankagauss_lora_b128_seed42_save500_4k,
    august25_id355_balanced150_frankaquant_lora_b128_seed42_save500_4k,
    august25_id355_balanced150_frankagauss_fullft_b128_seed42_save500_4k,
    august25_id355_balanced150_frankaquant_fullft_b128_seed42_save500_4k,
    august24_id355_unique230_frankagauss_lora_b128_seed42_resume20k_to40k,
    august25_id355_balanced150_frankagauss_lora_b128_seed42_resume20k_to40k,
    august24_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_nocond_seed0_lora_stdnorm_clip5_b512_96k,
    august24_oer_phase0_oer1kdist_bal6p5e_id261_ep7830_nocond_seed0_lora_stdnorm_clip5_b1024_48k,
    july29_oer_hash8095_physvalidv2_noembtext_globalnorm_clip5_val5_b1024_64k,
    july29_oer_hash8095_physvalidv2_tailcleanv1_noembtext_globalnorm_clip5_val5_b1024_64k,
    july28_oer_hash8095_noembtext_urma2_concatproj1024_globalnorm_clip5_val5_b1024_64k,
    july28_oer_hash8095_noembtext_urma3_concatproj1024_globalnorm_clip5_val5_b1024_64k,
    july30_oer_hash8095_quantile_clip5_val5_b1024_16k_continue,
    july30_oer_hash8095_standard_clip5_val5_b1024_16k_continue,
    july30_oer_hash8095_urma3_quantile_clip5_val5_b1024_16k_continue,
    july30_oer_hash8095_exactrlds_unioncleanv2_noembtext_globalnorm_clip5_val5_b1024_16k_warmstart1999,
    july30_oer_hash8095_urma3_standard_clip5_val5_b1024_16k,
    july30_oer_full50000_standard_clip5_val5_b1024_16k,
    july30_oer_full100000_standard_clip5_val5_b1024_16k,
    july21_fixcap_warmupema_h32_open_franka_remaining81_adarms_only_b1024_64k,
    july19_fixcap_r81random1_warmupema_h32_open_frankav517_b128_20k,
    july19_fixcap_r81random9_warmupema_h32_open_franka9emb_b256_64k,
    july18_fixcap_warmupema_h32_open_franka9emb_noembtext_64k,
    july20_fixcap_warmupema_h32_open_franka9emb_adarms_only_64k,
    july21_fixcap_warmupema_h32_open_franka9emb_concatadarms_only_64k,
    july21_fixcap_warmupema_h32_open_franka_remaining81_concatadarms_only_b1024_64k,
    july20_fixcap_warmupema_h32_open_franka9emb_urma2_only_64k,
    july22_fixcap_warmupema_h32_open_franka9emb_urma2_add_64k,
    july22_fixcap_warmupema_h32_open_franka9emb_urma2_concat_64k,
    july23_fixcap_warmupema_h32_open_franka_remaining81_urma2_concatproj1024_globalnorm_b1024_64k,
    july23_fixcap_warmupema_h32_open_franka_remaining81_urma2_backbonetoken_globalnorm_b1024_64k,
)

ConfigBuilder = Callable[[], TrainConfig]

_EXPLICIT_COMPATIBILITY_ALIASES: tuple[tuple[str, ConfigBuilder], ...] = (
    ("pi05_lowvar_v2", july11_lowvarv2_h50_open_frankav013_fullft_2k),
    ("pi05_lowvar_v2_fullft_h32", july11_lowvarv2_h32_open_frankav013_fullft_2k),
    ("pi05_lowvar_v2_lora", july11_lowvarv2_h50_open_frankav013_lora_4k),
    (
        "pi05_lowvar_v2_lora_lr5e4",
        july11_lowvarv2_h50_open_frankav013_lora_lr5e4_20k,
    ),
)


def _primary_config(builder: ConfigBuilder) -> TrainConfig:
    """Use the dated builder name as the public config name."""
    return _legacy_sampling.pin_legacy_input_sampling(
        dataclasses.replace(builder(), name=builder.__name__),
        legacy_names=_legacy_sampling.LEGACY_NATIVE_CONFIGS,
    )


def _compatibility_aliases() -> tuple[tuple[str, ConfigBuilder], ...]:
    """Retain old names for restoring recorded runs without listing them as configs."""
    aliases = list(_EXPLICIT_COMPATIBILITY_ALIASES)
    known_names = {alias for alias, _ in aliases}
    for builder in _CONFIG_BUILDERS:
        old_name = builder().name
        if old_name != builder.__name__ and old_name not in known_names:
            aliases.append((old_name, builder))
            known_names.add(old_name)
    return tuple(aliases)


def _all_configs() -> tuple[TrainConfig, ...]:
    configs = tuple(_primary_config(builder) for builder in _CONFIG_BUILDERS)
    names = tuple(config.name for config in configs)
    if len(set(names)) != len(names):
        raise ValueError("Config names must be unique.")
    return configs


def list_configs() -> tuple[str, ...]:
    return tuple(config.name for config in _all_configs())


def cli() -> TrainConfig:
    # Tyro requires a mapping at its API boundary; keep it local rather than maintaining
    # a second global registry that can drift from the individually named builders.
    configs = _all_configs()
    return tyro.extras.overridable_config_cli({config.name: (config.name, config) for config in configs})


def get_config(config_name: str) -> TrainConfig:
    """Resolve one primary config name or an explicit checkpoint-compatibility alias."""
    for builder in _CONFIG_BUILDERS:
        # ``_primary_config`` always publishes the builder's function name.  Match that
        # stable identity before invoking the builder so resolving one ready config does
        # not instantiate unrelated fail-closed configs whose assets are still unpublished.
        if builder.__name__ == config_name:
            return _primary_config(builder)
    for alias, builder in _compatibility_aliases():
        if alias == config_name:
            # Aliases exist only to restore recorded runs; they inherit the pin of their
            # primary name so a legacy lineage never silently resolves to the new default.
            return dataclasses.replace(_primary_config(builder), name=alias)

    choices = (*list_configs(), *(alias for alias, _ in _compatibility_aliases()))
    closest = difflib.get_close_matches(config_name, choices, n=1, cutoff=0.0)
    closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
    raise ValueError(f"Config '{config_name}' not found.{closest_str}")


# pi0.5 knowledge-insulation / data-mixture configs, merged from branch pi05-ki-action-mask-fix.
# Their checkpoints already exist under these names, so each builder keeps the config's recorded
# name (not a dated function name) and pins LegacyRldsInputSampling explicitly: every run of these
# lineages used the legacy contract, and the DROID co-training path resizes DROID frames only in
# the legacy decode, so the DeterministicReplayV3 default would break the DROID mixtures.
# openpi's non-idle DROID step ranges (droid_rlds_dataset.py recipe). Every DROID source in the
# KI mixture sweep drops idle frames with it. Fetched through openpi.shared.download, so it
# must already sit in $OPENPI_DATA_HOME/openpi-assets/droid/ on offline compute nodes.
_DROID_IDLE_RANGES = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

_PI05_KI_MIXTURE_CONFIGS: tuple[TrainConfig, ...] = (
    TrainConfig(
        # molmobot ALONE, identical model/loader settings to pi05_mbdroid_v1_*. Exists so
        # compute_norm_stats can produce a molmobot-only normalization table for comparison
        # against the droid-only and realrig9k-only tables; not intended for training.
        name="pi05_molmobot_fpp20k_normstats_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            discrete_state_input=True,
            augmentation_mode="shared_mild",
            max_token_len=320,
            enable_langact_training=True,
            stop_action_to_vlm_grad=True,
            language_loss_weight=1.0,
            action_loss_weight=1.0,
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_molmobot_fpp20k",
            rlds_data_dir="/scratch/gpfs/SHAHD/users/sg0803/datasets",
            datasets=tuple(
                droid_rlds_dataset.RLDSDataset(
                    name=f"rlds_molmobot_fpp_20k_s{i}/molmospaces_emb_mix",
                    version="1.0.0",
                    weight=0.25,
                )
                for i in range(4)
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/scratch/gpfs/SHAHD/users/sg0803/pi05_ckpts",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=1,
        batch_size=256,
        gradient_accumulation_steps=4,
        fsdp_devices=4,
        num_workers=0,
    ),
    TrainConfig(
        # Pretraining mixture: DROID (pick-and-place excluded) + MolmoBot FrankaPickAndPlace
        # + realrig9k, at 65/30/5. pp9robot and the fr_ur5 pick_place_p1 builds are NOT in
        # this mixture -- unlike pi05_allmix_v1_*, this is franka-only, so the ur5e 6-DoF
        # gripper-dimension hazard documented on that config does not apply here.
        #
        # LR is flat at 5e-5 after a 1k-step warmup: CosineDecaySchedule with
        # peak_lr == decay_lr is the codebase's idiom for a constant schedule (see the
        # 1e-4/1e-4 configs above). warmup_steps counts OPTIMIZER updates, not microbatches
        # -- create_optimizer wraps the schedule in optax.MultiSteps -- so 1k warmup means
        # 1k applied updates at the full 512 global batch.
        # Per-norm full fine-tune on the upstream openpi pi05_full_droid_finetune training recipe:
        # batch 256, 100k steps, 1k warmup then flat 5e-5, EMA 0.99 (TrainConfig default, which that
        # config leaves on), AdamW defaults (clip 1.0), action horizon 16, pi05_native augmentation,
        # keep every 10k. Kept from our v1 sweep on purpose: the DROID/MolmoBot/realrig9k mixture,
        # per-embodiment quantile norm, the KI dual objective (so max_token_len 320 for the FAST
        # tokens) and pi05_base init. Only deviation for infrastructure: saves every 2k steps instead
        # of 5k, because each 24h Slurm job replays from the last save.
        # Global batch 512 (64/GPU on 8 GPUs, no accumulation), by request, instead of upstream's
        # 256. LR, warmup, EMA, horizon, augmentation and the 100k-step length follow upstream, so
        # this run sees 2x the samples of the upstream b256 recipe.
        name="pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=True,
            augmentation_mode="pi05_native",
            # 320, not 224: FASTTokenizer truncates from the END, so an over-long sample would
            # silently drop exactly the action supervision this run is testing.
            max_token_len=320,
            enable_langact_training=True,
            stop_action_to_vlm_grad=True,
            language_loss_weight=1.0,
            action_loss_weight=1.0,
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_mbdroid_pernorm_v1",
            rlds_data_dir="/scratch/gpfs/SHAHD/users/sg0803/datasets",
            # SEPARATE NORM TABLES: real (DROID, embodiment -2) and sim (MolmoBot + realrig9k,
            # embodiment 0) each get their own quantiles, instead of one global table over the
            # mixture. The global table is narrower than DROID's own (0.294 vs 0.330 rad mean arm
            # range) and wider than sim's, so it compresses sim actions and stretches DROID's --
            # the suspected cause of the co-training penalty. State encoding is untouched
            # (NoStateEncoding), so state_conditioning_dim stays 0.
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.PerEmbodimentQuantileNorm((-2, 0)),
            ),
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="droid", version="1.0.1", weight=0.65, exclude_pick_and_place=True, filter_dict_path=_DROID_IDLE_RANGES
                ),
                # molmobot is converted as 4 independent shards (one Slurm job each); they
                # partition the same 20034 episodes, so the 30% share splits evenly.
                *(
                    droid_rlds_dataset.RLDSDataset(
                        name=f"rlds_molmobot_fpp_20k_s{i}/molmospaces_emb_mix",
                        version="1.0.0",
                        weight=0.30 / 4,
                    )
                    for i in range(4)
                ),
                # realrig9k ships as 6 equal shards; the 5% share is split evenly across them.
                *(
                    droid_rlds_dataset.RLDSDataset(
                        name=f"rlds_v3_p8/molmospaces_realrig9k_p{i}",
                        version="1.1.0",
                        weight=0.05 / 6,
                    )
                    for i in range(1, 7)
                ),
            ),
        ),
        # FULL FINE-TUNE from pi05_base. pi05_base must be staged under
        # $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base (compute nodes have no egress).
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/scratch/gpfs/SHAHD/users/sg0803/pi05_ckpts",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=100_000,
        # One 8xH200 node: 512 global batch, 64/GPU, no accumulation.
        batch_size=512,
        gradient_accumulation_steps=1,
        fsdp_devices=8,
        # Restart economy: a 24h wall clock means each relaunch replays from the last
        # checkpoint, so saves come every 2k steps. Every 10k steps (as upstream keep_period)
        # one is kept permanently; the rest roll under max_to_keep=1.
        keep_period=None,
        keep_every_n_completed_updates=10_000,
        # Upstream default EMA. Checkpoints store the EMA weights in place of params; validation
        # still evaluates the raw params.
        ema_decay=0.99,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=1_000,
        val_batches=8,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(interval=1_000, sample_count=4),
        save_interval=2_000,
        log_interval=100,
        num_workers=0,
    ),
    TrainConfig(
        # Sim-only mixture: MolmoBot FrankaPickAndPlace + realrig9k, the co-train run's 30/5
        # shares renormalized to sum to 1 (no DROID). pp9robot and the fr_ur5 pick_place_p1 builds
        # are NOT in this mixture -- unlike pi05_allmix_v1_*, this is franka-only, so the ur5e
        # 6-DoF gripper-dimension hazard documented on that config does not apply here.
        #
        # LR is flat at 5e-5 after a 1k-step warmup: CosineDecaySchedule with
        # peak_lr == decay_lr is the codebase's idiom for a constant schedule (see the
        # 1e-4/1e-4 configs above). warmup_steps counts OPTIMIZER updates, not microbatches
        # -- create_optimizer wraps the schedule in optax.MultiSteps -- so 1k warmup means
        # 1k applied updates at the full 512 global batch.
        # Per-norm full fine-tune on the upstream openpi pi05_full_droid_finetune training recipe:
        # batch 256, 100k steps, 1k warmup then flat 5e-5, EMA 0.99 (TrainConfig default, which that
        # config leaves on), AdamW defaults (clip 1.0), action horizon 16, pi05_native augmentation,
        # keep every 10k. Kept from our v1 sweep on purpose: the DROID/MolmoBot/realrig9k mixture,
        # per-embodiment quantile norm, the KI dual objective (so max_token_len 320 for the FAST
        # tokens) and pi05_base init. Only deviation for infrastructure: saves every 2k steps instead
        # of 5k, because each 24h Slurm job replays from the last save.
        # Global batch 512 (64/GPU on 8 GPUs, no accumulation), by request, instead of upstream's 256.
        # SIM-ONLY arm of pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft: same
        # init, recipe, objective and NORMALIZATION (the very same per-embodiment norm_stats.json,
        # asset id molmospaces_mbdroid_pernorm_v1, so sim actions are scaled identically), with
        # DROID removed and the sim shares renormalized exactly as pi05_simonly_v1_*. The two runs
        # therefore differ in the presence of DROID only.
        name="pi05_simonly_v1_pernormstats_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=True,
            augmentation_mode="pi05_native",
            # 320, not 224: FASTTokenizer truncates from the END, so an over-long sample would
            # silently drop exactly the action supervision this run is testing.
            max_token_len=320,
            enable_langact_training=True,
            stop_action_to_vlm_grad=True,
            language_loss_weight=1.0,
            action_loss_weight=1.0,
        ),
        data=RLDSMolmoSpacesDataConfig(
            repo_id="molmospaces_mbdroid_pernorm_v1",
            rlds_data_dir="/scratch/gpfs/SHAHD/users/sg0803/datasets",
            # SEPARATE NORM TABLES: real (DROID, embodiment -2) and sim (MolmoBot + realrig9k,
            # embodiment 0) each get their own quantiles, instead of one global table over the
            # mixture. The global table is narrower than DROID's own (0.294 vs 0.330 rad mean arm
            # range) and wider than sim's, so it compresses sim actions and stretches DROID's --
            # the suspected cause of the co-training penalty. State encoding is untouched
            # (NoStateEncoding), so state_conditioning_dim stays 0.
            input_config=_input_config.Pi05InputConfig(
                normalization=_input_config.PerEmbodimentQuantileNorm((-2, 0)),
            ),
            datasets=(
                # molmobot is converted as 4 independent shards (one Slurm job each); they
                # partition the same 20034 episodes, so the 30% share splits evenly.
                *(
                    droid_rlds_dataset.RLDSDataset(
                        name=f"rlds_molmobot_fpp_20k_s{i}/molmospaces_emb_mix",
                        version="1.0.0",
                        weight=(0.30 / 0.35) / 4,   # mbdroid's MolmoBot share, renormalized
                    )
                    for i in range(4)
                ),
                # realrig9k ships as 6 equal shards; the 5% share is split evenly across them.
                *(
                    droid_rlds_dataset.RLDSDataset(
                        name=f"rlds_v3_p8/molmospaces_realrig9k_p{i}",
                        version="1.1.0",
                        weight=(0.05 / 0.35) / 6,   # mbdroid's realrig9k share, renormalized
                    )
                    for i in range(1, 7)
                ),
            ),
        ),
        # FULL FINE-TUNE from pi05_base. pi05_base must be staged under
        # $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base (compute nodes have no egress).
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        checkpoint_base_dir="/scratch/gpfs/SHAHD/users/sg0803/pi05_ckpts",
        project_name="embodiment-scaling-manipulation",
        num_train_steps=100_000,
        # One 8xH200 node: 512 global batch, 64/GPU, no accumulation.
        batch_size=512,
        gradient_accumulation_steps=1,
        fsdp_devices=8,
        # Restart economy: a 24h wall clock means each relaunch replays from the last
        # checkpoint, so saves come every 2k steps. Every 10k steps (as upstream keep_period)
        # one is kept permanently; the rest roll under max_to_keep=1.
        keep_period=None,
        keep_every_n_completed_updates=10_000,
        # Upstream default EMA. Checkpoints store the EMA weights in place of params; validation
        # still evaluates the raw params.
        ema_decay=0.99,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        val_interval=1_000,
        val_batches=8,
        train_sample_interval=1_000,
        data_visualization=_input_config.DataVisualizationConfig(interval=1_000, sample_count=4),
        save_interval=2_000,
        log_interval=100,
        num_workers=0,
    ),
)


def _legacy_named_builder(config: TrainConfig) -> ConfigBuilder:
    pinned = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data, rlds_input_sampling=_data_processing.LegacyRldsInputSampling()
        ),
    )

    def builder() -> TrainConfig:
        return pinned

    builder.__name__ = builder.__qualname__ = config.name
    return builder


_CONFIG_BUILDERS += tuple(_legacy_named_builder(config) for config in _PI05_KI_MIXTURE_CONFIGS)


# Continuations of the two Della droidft lineages on the process-sharded reader. Under the legacy
# reader every 24h resume restarted the same fixed-order file stream from its head (fixed seed, no
# skip), so each segment re-trained on roughly the same leading half of the data and the tail was
# never read. These configs differ from their parents ONLY in rlds_input_sampling. They resume from
# the parents' last checkpoints (co-train 17999, sim-only 5999), copied into the new names' run
# directories, so optimizer state, EMA and the step counter carry over.
# V2, not the repository default V3: V3 decodes images after mixing and skips the DROID letterbox
# to the sim 256x256/128x128 sizes, so a DROID mixture cannot batch under it. V2 keeps the
# per-dataset decode path, and on one process it reads the whole file list under a fresh
# permutation seeded from the completed optimizer updates on every resume.
_PI05_DROIDFT_PROCESSSHARDV2_PARENTS = (
    "pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft",
    "pi05_simonly_v1_pernormstats_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft",
)


def _processshardv2_builder(parent: TrainConfig) -> ConfigBuilder:
    config = dataclasses.replace(
        parent,
        name=f"{parent.name}_processshardv2",
        data=dataclasses.replace(
            parent.data, rlds_input_sampling=_data_processing.ProcessShardedFileShuffleV2()
        ),
    )

    def builder() -> TrainConfig:
        return config

    builder.__name__ = builder.__qualname__ = config.name
    return builder


_CONFIG_BUILDERS += tuple(
    _processshardv2_builder(config)
    for config in _PI05_KI_MIXTURE_CONFIGS
    if config.name in _PI05_DROIDFT_PROCESSSHARDV2_PARENTS
)


# TPU (prism pikes) arms of the droidft ablation, derived from the Della co-train config so the
# training recipe (pi05_base full FT, KI, EMA 0.99, 1k warmup then flat 5e-5, H16, pi05_native,
# 100k steps, save 2k / keep 10k) is identical. Each differs from it only in: name, batch_size,
# fsdp_devices, the dataset mixture, GCS data/asset/checkpoint locations, and the reader.
# - Reader: ProcessShardedFileShuffleV2. The legacy reader would give every TPU host the same
#   ordered stream (global batch = process_count near-copies of one local batch). V3 is not usable
#   here: it skips the DROID letterbox, so a DROID mixture cannot batch.
# - Normalization: every arm loads the Della co-train run's norm_stats.json from GCS (same bytes),
#   so action scaling is identical across arms; DROID uses table -2, sim table 0.
# - Topology: the three b512 arms run on pikes-16 (16 chips, 4 hosts, FSDP 16, 32/chip) so they
#   share batch AND reader topology; under V2 the host count sets the file cohorts. The b1024 arm
#   (pikes-32, FSDP 32) is a batch-size arm and must not be compared to the b512 arms as the
#   co-train baseline.
_TPU_RLDS_ROOT = "gs://prism-us-east1-data/users/sg0803/esm_pi05_rlds"
_TPU_DROID_ROOT = "gs://v6_east1d/OXE"
_TPU_CKPT_ROOT = "gs://prism-us-east1-data/users/sg0803/esm_pi05_ckpts"
_TPU_NORM_ASSETS = AssetsConfig(
    assets_dir=(
        "gs://prism-us-east1-data/users/sg0803/esm_pi05_assets/"
        "pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft"
    ),
    asset_id="molmospaces_mbdroid_pernorm_v1",
)


def _tpu_droidft_datasets(droid: float, molmobot: float, realrig9k: float):
    datasets = []
    if droid:
        datasets.append(
            droid_rlds_dataset.RLDSDataset(
                name="droid",
                version="1.0.1",
                weight=droid,
                exclude_pick_and_place=True,
                filter_dict_path=_DROID_IDLE_RANGES,
                data_subdir=_TPU_DROID_ROOT,
            )
        )
    if molmobot:
        datasets.extend(
            droid_rlds_dataset.RLDSDataset(
                name=f"rlds_molmobot_fpp_20k_s{i}/molmospaces_emb_mix", version="1.0.0", weight=molmobot / 4
            )
            for i in range(4)
        )
    if realrig9k:
        datasets.extend(
            droid_rlds_dataset.RLDSDataset(
                name=f"rlds_v3_p8/molmospaces_realrig9k_p{i}", version="1.1.0", weight=realrig9k / 6
            )
            for i in range(1, 7)
        )
    return tuple(datasets)


def _tpu_droidft_builder(
    name: str, *, batch_size: int, fsdp_devices: int, droid: float, molmobot: float, realrig9k: float
) -> ConfigBuilder:
    (parent,) = (
        c
        for c in _PI05_KI_MIXTURE_CONFIGS
        if c.name == "pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft"
    )
    config = dataclasses.replace(
        parent,
        name=name,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        checkpoint_base_dir=_TPU_CKPT_ROOT,
        data=dataclasses.replace(
            parent.data,
            rlds_data_dir=_TPU_RLDS_ROOT,
            assets=_TPU_NORM_ASSETS,
            datasets=_tpu_droidft_datasets(droid, molmobot, realrig9k),
            rlds_input_sampling=_data_processing.ProcessShardedFileShuffleV2(),
        ),
    )

    def builder() -> TrainConfig:
        return config

    builder.__name__ = builder.__qualname__ = config.name
    return builder


_TPU_DROIDFT_ARMS = (
    # Co-train baseline of the ablation: same mixture as Della, batch- and topology-matched to the
    # DROID-only and 70/30 arms below.
    _tpu_droidft_builder(
        "pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft_tpu",
        batch_size=512, fsdp_devices=16, droid=0.65, molmobot=0.30, realrig9k=0.05,
    ),
    # Batch-size arm (pikes-32): not the co-train baseline for the b512 arms.
    _tpu_droidft_builder(
        "pi05_mbdroid_v1_pernorm_fullft_ki_b1024_lr5e5_h16_wu1k_ema99_droidft_tpu",
        batch_size=1024, fsdp_devices=32, droid=0.65, molmobot=0.30, realrig9k=0.05,
    ),
    _tpu_droidft_builder(
        "pi05_droidonly_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft_tpu",
        batch_size=512, fsdp_devices=16, droid=1.0, molmobot=0.0, realrig9k=0.0,
    ),
    _tpu_droidft_builder(
        "pi05_droidmb_70_30_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft_tpu",
        batch_size=512, fsdp_devices=16, droid=0.70, molmobot=0.30, realrig9k=0.0,
    ),
)
_CONFIG_BUILDERS += _TPU_DROIDFT_ARMS


def october25_mobilebase_gen3_smoke() -> TrainConfig:
    """Pipeline smoke of the opt-in holonomic-base slots (``Pi05InputConfig.mobile_base``).

    ``mobile_base_slots_v2`` (``pi05_integration/mobile_base.py``; body-frame base velocities) on the
    one-episode RLDS ``mobile_base_feature_v2`` build of Gen3 mobile episode 465e1681 (state-only smoke
    unit, frames stubbed with zeros). For checking the loader, normalization statistics, transforms and
    one model batch only: one episode with blank images is not a training dataset. Everything else is
    the clean-pose LoRA shell (H32, B128, quantile normalization, model-coordinate loss).

    Moved from v1 to v2 on 2026-10-25 at BoAI's request, before any run used it (v1 had only a
    pipeline smoke; its build ``rlds_mobile_feature_v1_465e1681`` and statistics under
    ``norm_assets`` stay on disk).
    """

    name = "october25_mobilebase_gen3_smoke"
    base = _clean_pose_lora_native_config(
        name,
        repo_id="20261025_dgi_mobile_feature_v2_465e1681",
        rlds_data_dir=f"{H5_DATA_DIR}/20261025_dgi_mobile-data",
        input_config=_input_config.Pi05InputConfig(mobile_base=_input_config.MobileBaseSlotsV2(dt_s=0.066)),
        num_train_steps=1,
        dataset_version="1.2.0",
    )
    return dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="molmospaces_emb_mix",
                    version="1.2.0",
                    weight=1.0,
                    data_subdir="rlds_mobile_feature_v2_465e1681",
                ),
            ),
            assets=AssetsConfig(assets_dir=f"{H5_DATA_DIR}/20261025_dgi_pi_mobile/norm_assets_v2", asset_id=name),
        ),
    )


_CONFIG_BUILDERS += (october25_mobilebase_gen3_smoke,)
