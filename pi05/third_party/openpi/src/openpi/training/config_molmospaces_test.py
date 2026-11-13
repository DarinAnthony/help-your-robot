import dataclasses

import flax.nnx as nnx
import numpy as np
import pytest

import openpi.transforms as transforms
from openpi.policies import libero_policy
from molmo_spaces.training.experiments import get_experiment
from molmo_spaces.training.pi_adapter import resolve_native_config
from molmo_spaces.training.runtimes import RuntimeBinding
from molmo_spaces.training import sep17_urma4_threeway_contract as sep17_contract
from pi05_integration import data_processing
from pi05_integration import input_config

from openpi.training import config as config_lib
from openpi.training import config_utils
from openpi.training import optimizer


def test_september17_urma4_threeway_native_configs_resolve_exact_contract() -> None:
    configs = {
        cell: config_lib.get_config(sep17_contract.experiment_name(cell))
        for cell in sep17_contract.CELLS
    }

    for cell, config in configs.items():
        assert config.name == sep17_contract.experiment_name(cell)
        assert config.num_train_steps == 4_000
        assert config.batch_size == 128
        assert config.lr_schedule.peak_lr == pytest.approx(2e-4)
        assert config.lr_schedule.decay_lr == pytest.approx(2e-4)
        assert config.lr_schedule.decay_steps == 4_000
        assert config.save_interval == 500
        assert config.keep_every_n_completed_updates == 2_000

        assert config.data.repo_id == sep17_contract.DATASET_ID
        assert config.data.assets.asset_id == sep17_contract.NORMALIZATION_ASSET_ID
        assert len(config.data.datasets) == 1
        dataset = config.data.datasets[0]
        assert dataset.data_subdir == sep17_contract.RLDS_RELEASE_NAME
        assert dataset.version == sep17_contract.DATASET_VERSION
        assert dataset.weight == pytest.approx(1.0)
        assert config.data.input_config.action_loss_coordinates == "physical"

        selector = config.data.episode_selector
        assert isinstance(selector, data_processing.NestedHashTrainingEpisodeSelector)
        assert selector.training_episode_count == 500
        assert selector.validation_episode_count == 400
        assert selector.selection_seed == 20260913
        assert selector.split_seed == 42
        split = config.data.episode_split
        assert isinstance(split, data_processing.HashEmbodimentCountSplit)
        assert split.validation_embodiment_count == 400
        assert split.seed == 42
        assert split.require_one_episode_per_embodiment

        conditioning = config.model.backbone_conditioning
        assert isinstance(conditioning, input_config.ExtractedEmbodimentBackboneToken)
        assert conditioning.token_width == 2048
        extractor = conditioning.extractor
        assert isinstance(extractor, input_config.Urma4DynamicEmbodimentExtractor)
        assert extractor.dynamic_profile == cell
        assert extractor.structured_input_shape == (10, 67, 16)
        assert extractor.bundle_sha256 == sep17_contract.URMA4_BUNDLE_SHA256

    assert dataclasses.replace(
        configs["minimal"],
        name=configs["static"].name,
        model=dataclasses.replace(
            configs["minimal"].model,
            backbone_conditioning=configs["static"].model.backbone_conditioning,
        ),
    ) == configs["static"]
    assert dataclasses.replace(
        configs["full"],
        name=configs["static"].name,
        model=dataclasses.replace(
            configs["full"].model,
            backbone_conditioning=configs["static"].model.backbone_conditioning,
        ),
    ) == configs["static"]


def test_september17_urma4_threeway_native_continuations_extend_exact_contract() -> None:
    for cell in sep17_contract.CELLS:
        base = config_lib.get_config(sep17_contract.experiment_name(cell))
        continuation = config_lib.get_config(
            sep17_contract.continuation_experiment_name(cell)
        )
        assert continuation.name == sep17_contract.continuation_experiment_name(cell)
        assert continuation.num_train_steps == 16_000
        assert continuation.lr_schedule.decay_steps == 16_000
        assert dataclasses.replace(
            continuation,
            name=base.name,
            num_train_steps=base.num_train_steps,
            lr_schedule=dataclasses.replace(
                continuation.lr_schedule,
                decay_steps=base.lr_schedule.decay_steps,
            ),
        ) == base


def test_molmospaces_data_config_uses_typed_normalization_mode(tmp_path):
    model = config_lib.get_config("pi05_lowvar_v2_lora_h32_20k").model
    for normalization, expected in (
        (input_config.GlobalStandardNorm(), False),
        (input_config.GlobalQuantileNorm(), True),
    ):
        factory = config_utils.RLDSMolmoSpacesDataConfig(
            repo_id="fake",
            rlds_data_dir=str(tmp_path),
            input_config=input_config.Pi05InputConfig(normalization=normalization),
        )
        assert factory.create(tmp_path, model).use_quantile_norm is expected


def test_molmospaces_data_config_does_not_require_disabled_response_history(tmp_path):
    model = config_lib.get_config("pi05_lowvar_v2_lora_h32_20k").model
    factory = config_utils.RLDSMolmoSpacesDataConfig(
        repo_id="fake",
        rlds_data_dir=str(tmp_path),
        input_config=input_config.Pi05InputConfig(),
    )

    repack = factory.create(tmp_path, model).repack_transforms.inputs[0]
    assert "response_state_history" not in repack.structure
    assert "response_action_history" not in repack.structure


def test_molmospaces_data_config_preserves_dataset_component_namespace(tmp_path):
    model = config_lib.get_config("pi05_lowvar_v2_lora_h32_20k").model
    factory = config_utils.RLDSMolmoSpacesDataConfig(
        repo_id="fake",
        rlds_data_dir=str(tmp_path),
        input_config=input_config.Pi05InputConfig(),
    )

    repack = factory.create(tmp_path, model).repack_transforms.inputs[0]
    assert repack.structure["dataset_component_id"] == "dataset_component_id"
    packed = repack(
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "state": np.zeros(10, dtype=np.float32),
            "state_mask": np.ones(10, dtype=bool),
            "actions": np.zeros((32, 10), dtype=np.float32),
            "action_loss_mask": np.ones((32, 10), dtype=bool),
            "action_semantic_mask": np.ones(10, dtype=bool),
            "prompt": np.asarray("open the drawer"),
            "embodiment_id": np.asarray(0),
            "dataset_component_id": np.asarray("dense2k"),
            "task_id": np.asarray(0),
        }
    )
    assert packed["dataset_component_id"] == "dense2k"
    model_inputs = libero_policy.LiberoInputs(model_type=model.model_type)(packed)
    assert model_inputs["dataset_component_id"] == "dense2k"


def test_compact_state_recipe_is_versioned_and_legacy_recipe_is_unchanged():
    legacy = config_lib.get_config(
        "july28_oer_hash8095_noembtext_stdnorm_clip5_val5_b1024_64k"
    )
    corrected = config_lib.get_config(
        "august30_oer_hash8095_noembtext_compactstate_stdnorm_clip5_val5_b1024_64k"
    )

    assert isinstance(
        legacy.data.input_config.state_serialization,
        input_config.LegacyFixedLayoutState,
    )
    assert isinstance(
        corrected.data.input_config.state_serialization,
        input_config.CompactLeftPackedState,
    )
    assert corrected.seed == 0
    assert dataclasses.replace(
        corrected.data.input_config,
        state_serialization=legacy.data.input_config.state_serialization,
    ) == legacy.data.input_config


def test_camera_d_h50_changes_only_registered_horizon() -> None:
    h32 = config_lib.get_config(
        "august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h32_60k"
    )
    h50 = config_lib.get_config(
        "august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h50_60k"
    )

    assert (h32.model.action_horizon, h50.model.action_horizon) == (32, 50)
    assert dataclasses.replace(
        h50,
        name=h32.name,
        model=dataclasses.replace(h50.model, action_horizon=32),
    ) == h32

    h32_experiment = get_experiment(h32.name)
    h50_experiment = get_experiment(h50.name)
    assert (h32_experiment.prediction_horizon, h50_experiment.prediction_horizon) == (32, 50)
    assert dataclasses.replace(
        h50_experiment,
        name=h32_experiment.name,
        native_config_name=h32_experiment.native_config_name,
        description=h32_experiment.description,
        prediction_horizon=32,
    ) == h32_experiment


@pytest.mark.parametrize("part", ["arm", "gripper"])
def test_center200_appearance_48k_native_config_changes_only_duration(part: str) -> None:
    prefix = (
        f"september7_center200_{part}_appearance10x_nocond_lora_"
        "gaussclip5_b128_lr2e4_seed42_h32"
    )
    base = config_lib.get_config(f"{prefix}_24k_v1")
    continuation = config_lib.get_config(f"{prefix}_48k_continue_v1")

    assert continuation.num_train_steps == 48_000
    assert continuation.lr_schedule.decay_steps == 48_000
    assert dataclasses.replace(
        continuation,
        name=base.name,
        num_train_steps=base.num_train_steps,
        lr_schedule=base.lr_schedule,
    ) == base


def test_threeway_equalmix_native_contract() -> None:
    name = (
        "august29_id355_slow200_unique230_camd400_equalmix_"
        "lora_h32_b384_lr3e5_seed42_60k"
    )
    config = config_lib.get_config(name)
    experiment = get_experiment(name)

    assert config.model.action_horizon == experiment.prediction_horizon == 32
    assert config.batch_size == experiment.global_batch_size == 384
    assert config.lr_schedule.peak_lr == experiment.peak_lr == pytest.approx(3e-5)
    assert config.lr_schedule.decay_lr == experiment.decay_lr == pytest.approx(3e-5)
    assert config.seed == experiment.seed == 42
    assert config.ema_decay is experiment.ema_decay is None
    assert config.save_interval == experiment.checkpoint_every_updates == 1_000
    assert config.keep_every_n_completed_updates == 4_000
    assert experiment.checkpoint_retention_every_updates == 4_000
    assert len(config.data.datasets) == len(experiment.dataset_components) == 3
    assert [dataset.weight for dataset in config.data.datasets] == pytest.approx([1 / 3] * 3)
    assert [component.weight for component in experiment.dataset_components] == pytest.approx(
        [1 / 3] * 3
    )
    assert all(
        dataset.target_joint_layout == "ur5e_contiguous_arm6_gripper6_v1"
        for dataset in config.data.datasets
    )
    assert config.data.assets.asset_id == (
        "id355_slow200_unique230_camd400_equalmix_n830_stdnorm_20260829"
    )


def test_lowvar_fullft_20k_matches_lora_experiment_contract():
    lora = config_lib.get_config("pi05_lowvar_v2_lora_h32_20k")
    full = config_lib.get_config("pi05_lowvar_v2_fullft_h32_20k")

    assert full.model.action_horizon == lora.model.action_horizon == 32
    assert full.model.augmentation_mode == lora.model.augmentation_mode == "shared_mild"
    assert full.num_train_steps == lora.num_train_steps == 20_000
    assert full.lr_schedule == lora.lr_schedule
    assert full.optimizer == lora.optimizer
    assert full.ema_decay == lora.ema_decay is None
    assert full.save_interval == lora.save_interval == 4_000
    assert full.val_interval == lora.val_interval == 500
    assert full.train_sample_interval == lora.train_sample_interval == 500
    assert full.visualization_interval == lora.visualization_interval == 500

    assert full.batch_size == 32
    assert full.fsdp_devices == 1
    assert full.model.paligemma_variant == "gemma_2b"
    assert full.model.action_expert_variant == "gemma_300m"
    assert lora.model.paligemma_variant == "gemma_2b_lora"
    assert lora.model.action_expert_variant == "gemma_300m_lora"


def test_robotoken_lora_and_fullft_are_matched_except_adaptation():
    lora = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )
    full = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )
    learned = config_lib.get_config(
        "september1_ur5e_trossen_v6_camd8k_learnedid_backbone_lora_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )

    assert isinstance(lora.model.backbone_conditioning, input_config.RobotTokenBackboneSequence)
    assert lora.model.backbone_conditioning == full.model.backbone_conditioning
    for field in (
        "batch_size",
        "fsdp_devices",
        "seed",
        "num_train_steps",
        "lr_schedule",
        "optimizer",
        "ema_decay",
        "save_interval",
        "keep_every_n_completed_updates",
        "data_visualization",
    ):
        assert getattr(lora, field) == getattr(full, field) == getattr(learned, field)
    assert lora.model.paligemma_variant == learned.model.paligemma_variant == "gemma_2b_lora"
    assert lora.model.action_expert_variant == learned.model.action_expert_variant == "gemma_300m_lora"
    assert full.model.paligemma_variant == "gemma_2b"
    assert full.model.action_expert_variant == "gemma_300m"
    assert isinstance(full.freeze_filter, type(nnx.Nothing()))

    parameter = nnx.Param(np.zeros(1, dtype=np.float32))
    lora_trainable = nnx.filterlib.to_predicate(lora.trainable_filter)
    assert lora_trainable(("robotoken_encoder", "link_proj", "kernel"), parameter)
    assert lora_trainable(("_pali_gemma", "llm", "layers", "lora_a"), parameter)
    assert not lora_trainable(("_pali_gemma", "llm", "layers", "kernel"), parameter)

    full_trainable = nnx.filterlib.to_predicate(full.trainable_filter)
    assert full_trainable(("robotoken_encoder", "link_proj", "kernel"), parameter)
    assert full_trainable(("_pali_gemma", "llm", "layers", "kernel"), parameter)


def test_robotoken_40k_configs_are_exact_schedule_extensions():
    for adaptation in ("lora", "fullft"):
        prefix = (
            "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_"
            f"{adaptation}_gaussclip5_b128_seed42_lr1e5_"
        )
        initial = config_lib.get_config(prefix + "24k")
        continuation = config_lib.get_config(prefix + "40k")

        assert continuation.num_train_steps == 40_000
        assert continuation.lr_schedule == dataclasses.replace(
            initial.lr_schedule, decay_steps=40_000
        )
        for field in dataclasses.fields(initial):
            if field.name not in {"name", "num_train_steps", "lr_schedule"}:
                assert getattr(continuation, field.name) == getattr(initial, field.name)


def test_robotoken_fullft_data_scaling_native_configs_are_matched_and_nested():
    baseline = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )
    parameter = nnx.Param(np.zeros(1, dtype=np.float32))

    for episode_count in (2_000, 4_000):
        prefix = (
            "september4_ur5e_trossen_v6_camd8k_"
            f"hash{episode_count // 1_000}k_s42_robotokens_backbone_fullft_"
            "gaussclip5_b128_seed42_lr1e5_"
        )
        initial = config_lib.get_config(prefix + "24k")
        continuation = config_lib.get_config(prefix + "40k")

        assert initial.data.episode_selector.to_record() == {
            "type": "HashEpisodeSelector",
            "episode_count": episode_count,
            "seed": 42,
            "identity_key": "episode_metadata.file_path",
            "ranking": "sha256(seed, episode_key)",
        }
        assert initial.model == baseline.model
        for field in (
            "batch_size",
            "fsdp_devices",
            "seed",
            "num_train_steps",
            "lr_schedule",
            "optimizer",
            "ema_decay",
            "save_interval",
            "keep_every_n_completed_updates",
            "data_visualization",
        ):
            assert getattr(initial, field) == getattr(baseline, field)
        assert isinstance(initial.freeze_filter, type(nnx.Nothing()))
        assert nnx.filterlib.to_predicate(initial.trainable_filter)(
            ("_pali_gemma", "llm", "layers", "kernel"), parameter
        )
        assert continuation.num_train_steps == 40_000
        assert continuation.lr_schedule == dataclasses.replace(
            initial.lr_schedule, decay_steps=40_000
        )


def test_robotoken_overfit10_native_config_is_a_bounded_production_fidelity_smoke():
    smoke = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_"
        "lora_overfit10_noaug_b32_seed42_lr1e4_1k"
    )
    official = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_lora_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )

    assert smoke.model.backbone_conditioning == official.model.backbone_conditioning
    assert smoke.data.input_config == official.data.input_config
    assert smoke.data.assets == official.data.assets
    assert smoke.data.datasets == official.data.datasets
    assert smoke.data.episode_selector.to_record() == {
        "type": "HashEpisodeSelector",
        "episode_count": 10,
        "seed": 42,
        "identity_key": "episode_metadata.file_path",
        "ranking": "sha256(seed, episode_key)",
    }
    assert smoke.model.augmentation_mode == "none"
    assert smoke.batch_size == 32
    assert smoke.rlds_num_parallel_reads == 1
    assert smoke.rlds_num_parallel_calls == 1
    assert official.rlds_num_parallel_reads == -1
    assert official.rlds_num_parallel_calls == -1
    assert smoke.num_train_steps == 1_000
    assert smoke.lr_schedule == optimizer.CosineDecaySchedule(
        warmup_steps=50,
        initial_lr=1e-8,
        peak_lr=1e-4,
        decay_steps=1_000,
        decay_lr=1e-4,
    )
    assert smoke.save_interval == smoke.keep_every_n_completed_updates == 250
    assert smoke.keep_period is None
    assert smoke.val_interval == smoke.val_batches == 0
    assert smoke.log_interval == 10
    assert smoke.train_sample_interval == smoke.visualization_interval == 100


def test_robotoken_overfit10_direct_native_config_reads_only_ten_examples():
    smoke = config_lib.get_config(
        "september3_ur5e_trossen_v6_camd8k_robotokens_backbone_"
        "lora_overfit_first10_noaug_b32_seed42_lr1e4_1k"
    )
    original = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_"
        "lora_overfit10_noaug_b32_seed42_lr1e4_1k"
    )

    assert smoke.data.rlds_split == "train[:10]"
    assert smoke.data.episode_selector is None
    for field in dataclasses.fields(original):
        if field.name not in {"name", "data"}:
            assert getattr(smoke, field.name) == getattr(original, field.name)
    assert smoke.data == dataclasses.replace(
        original.data,
        rlds_split="train[:10]",
        episode_selector=None,
    )


def test_repository_adapter_resolves_gpu_and_tpu_without_scientific_drift():
    experiment = get_experiment("pi05_lowvar_v2_lora_h32_20k")
    gpu = resolve_native_config(experiment, "gpu", "adapter-test")
    tpu = resolve_native_config(experiment, "tpu", "adapter-test")

    assert gpu.batch_size == tpu.batch_size == 128
    assert gpu.model == tpu.model
    assert gpu.lr_schedule == tpu.lr_schedule
    assert gpu.fsdp_devices == 1
    assert tpu.fsdp_devices == 4
    assert gpu.data.rlds_data_dir.endswith("/rlds_sanity_lowvar_v2")
    assert tpu.data.rlds_data_dir.startswith("gs://")
    assert "resolved_training_experiment" in gpu.policy_metadata


def test_n60_conditioned_fullft_native_contract():
    experiment = get_experiment("pi05_9emb_n60_idtext_state_fullft_h32_40k")
    native = resolve_native_config(experiment, "gpu", "n60-native-test")

    assert native.model.action_horizon == 32
    assert native.model.max_token_len == 224
    assert native.model.augmentation_mode == "shared_mild"
    assert native.data.embodiment_conditioning == "id_text"
    assert native.data.embodiment_state_ids == (0, 1, 2, 10, 22, 23, 35, 42, 58)
    assert native.batch_size == 64
    assert native.fsdp_devices == 2
    assert native.num_train_steps == 40_000
    assert native.save_interval == 4_000


def test_runtime_fsdp_override_is_explicitly_recorded():
    experiment = get_experiment(
        "pi05_9emb_n30_currentplanner_phasebalanced_idtext_state_fullft_h32_40k"
    )
    native = resolve_native_config(
        experiment, "gpu", "single-h200", fsdp_devices=1
    )

    assert native.fsdp_devices == 1
    record = native.policy_metadata["resolved_training_experiment"]
    assert record["runtime"]["fsdp_devices"] == 2
    assert record["runtime_override"] == {"fsdp_devices": 1}

    with pytest.raises(ValueError, match="fsdp_devices must be positive"):
        resolve_native_config(experiment, "gpu", "invalid", fsdp_devices=0)


def test_accelerated_continuation_overrides_are_explicit_and_effective():
    experiment = get_experiment(
        "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
        "id_state_fullft_h32_40k"
    )
    native = resolve_native_config(
        experiment,
        "gpu",
        "accelerated-resume",
        fsdp_devices=2,
        global_batch_size=128,
        peak_lr=1.4e-4,
        decay_lr=1.4e-4,
        num_train_steps=26_000,
        resume=True,
    )

    assert native.resume
    assert native.fsdp_devices == 2
    assert native.batch_size == 128
    assert native.lr_schedule.peak_lr == pytest.approx(1.4e-4)
    assert native.lr_schedule.decay_lr == pytest.approx(1.4e-4)
    assert native.num_train_steps == 26_000
    record = native.policy_metadata["resolved_training_experiment"]
    assert record["runtime_override"] == {
        "fsdp_devices": 2,
        "global_batch_size": 128,
        "peak_lr": 1.4e-4,
        "decay_lr": 1.4e-4,
        "num_train_steps": 26_000,
    }
    assert record["effective_training"]["global_batch_size"] == 128


def test_n60_currentplanner_native_contract_matches_model_and_optimizer():
    mixed = resolve_native_config(
        get_experiment("pi05_9emb_n60_idtext_state_fullft_h32_40k"), "gpu", "mixed"
    )
    clean = resolve_native_config(
        get_experiment("pi05_9emb_n60_currentplanner_idtext_state_fullft_h32_40k"),
        "gpu",
        "clean",
    )

    assert clean.model == mixed.model
    assert clean.lr_schedule == mixed.lr_schedule
    assert clean.optimizer == mixed.optimizer
    assert clean.batch_size == mixed.batch_size
    assert clean.data.embodiment_conditioning == mixed.data.embodiment_conditioning
    assert "n60_currentplanner_seed277" in clean.data.rlds_data_dir


def test_n30_currentplanner_native_contract_matches_n60_except_data():
    n30 = resolve_native_config(
        get_experiment("pi05_9emb_n30_currentplanner_idtext_state_fullft_h32_40k"),
        "gpu",
        "n30-clean",
    )
    n60 = resolve_native_config(
        get_experiment("pi05_9emb_n60_currentplanner_idtext_state_fullft_h32_40k"),
        "gpu",
        "n60-clean",
    )

    assert n30.model == n60.model
    assert n30.lr_schedule == n60.lr_schedule
    assert n30.optimizer == n60.optimizer
    assert n30.batch_size == n60.batch_size
    assert n30.data.embodiment_conditioning == n60.data.embodiment_conditioning
    assert n30.data.embodiment_state_ids == n60.data.embodiment_state_ids
    assert "n30_currentplanner_seed277" in n30.data.rlds_data_dir


def test_phasebalanced_n30_native_contract_changes_only_sampler():
    uniform = resolve_native_config(
        get_experiment("pi05_9emb_n30_currentplanner_idtext_state_fullft_h32_40k"),
        "gpu",
        "uniform",
    )
    balanced = resolve_native_config(
        get_experiment(
            "pi05_9emb_n30_currentplanner_phasebalanced_idtext_state_fullft_h32_40k"
        ),
        "gpu",
        "phasebalanced",
    )

    assert uniform.model == balanced.model
    assert uniform.lr_schedule == balanced.lr_schedule
    assert uniform.optimizer == balanced.optimizer
    assert uniform.batch_size == balanced.batch_size
    assert uniform.data.rlds_data_dir == balanced.data.rlds_data_dir
    assert uniform.data.post_close_sampling_weight is None
    assert balanced.data.post_close_sampling_weight == 0.5


def test_per_embodiment_normalization_native_contract():
    global_norm = resolve_native_config(
        get_experiment("pi05_9emb_n60_idtext_state_fullft_h32_40k"), "gpu", "global"
    )
    per_emb = resolve_native_config(
        get_experiment("pi05_9emb_n60_idtext_state_perembnorm_fullft_h32_40k"),
        "gpu",
        "per-emb",
    )

    assert not global_norm.data.embodiment_normalization
    assert per_emb.data.embodiment_normalization
    assert per_emb.model == global_norm.model
    assert per_emb.optimizer == global_norm.optimizer
    assert per_emb.lr_schedule == global_norm.lr_schedule
    assert per_emb.data.rlds_data_dir == global_norm.data.rlds_data_dir


def test_per_embodiment_normalization_ids_do_not_require_state_one_hot():
    config = config_lib.get_config(
        "pi05_cleanpose_v013_v066_varianttext_lora_h32_constlr40k"
    ).data

    assert config.embodiment_state_ids == ()
    assert config.embodiment_normalization_ids == (0, 2)
    assert config.embodiment_normalization


def test_combined_n60_native_contract_preserves_base_recipe():
    base = resolve_native_config(
        get_experiment("pi05_9emb_n60_currentplanner_idtext_state_fullft_h32_40k"),
        "gpu",
        "base",
    )
    combined = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "idtext_state_fullft_h32_40k"
        ),
        "gpu",
        "combined",
    )

    assert combined.model == base.model
    assert combined.optimizer == base.optimizer
    assert combined.lr_schedule == base.lr_schedule
    assert combined.batch_size == base.batch_size
    assert combined.data.rlds_data_dir == base.data.rlds_data_dir
    assert combined.data.embodiment_state_ids == base.data.embodiment_state_ids
    assert combined.data.embodiment_normalization
    assert combined.data.post_close_sampling_weight == 0.5


def test_short_id_combined_native_contract_changes_only_text_conditioning():
    long_prompt = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "idtext_state_fullft_h32_40k"
        ),
        "gpu",
        "long-prompt",
    )
    short_prompt = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "id_state_fullft_h32_40k"
        ),
        "gpu",
        "short-prompt",
    )

    assert short_prompt.model == long_prompt.model
    assert short_prompt.optimizer == long_prompt.optimizer
    assert short_prompt.lr_schedule == long_prompt.lr_schedule
    assert short_prompt.data.rlds_data_dir == long_prompt.data.rlds_data_dir
    assert short_prompt.data.embodiment_normalization
    assert short_prompt.data.post_close_sampling_weight == 0.5
    assert long_prompt.data.embodiment_conditioning == "id_text"
    assert short_prompt.data.embodiment_conditioning == "id"


def test_h16_short_id_native_contract_changes_only_action_horizon():
    h32 = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "id_state_fullft_h32_40k"
        ),
        "gpu",
        "h32",
    )
    h16 = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "id_state_fullft_h16_40k"
        ),
        "gpu",
        "h16",
    )

    assert (h16.model.action_horizon, h32.model.action_horizon) == (16, 32)
    assert h16.optimizer == h32.optimizer
    assert h16.lr_schedule == h32.lr_schedule
    assert h16.data.rlds_data_dir == h32.data.rlds_data_dir
    assert h16.data.embodiment_conditioning == h32.data.embodiment_conditioning == "id"
    assert h16.data.embodiment_normalization == h32.data.embodiment_normalization
    assert h16.data.post_close_sampling_weight == h32.data.post_close_sampling_weight


def test_direct_adaemb_candidate_changes_only_model_conditioning():
    baseline = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "id_state_fullft_h16_40k"
        ),
        "gpu",
        "h16-baseline",
    )
    conditioned = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "adaemb_id_state_fullft_h16_40k"
        ),
        "gpu",
        "h16-adaemb",
    )

    assert baseline.model.embodiment_vocab_size == 0
    assert conditioned.model.embodiment_vocab_size == 64
    assert conditioned.model.action_horizon == baseline.model.action_horizon == 16
    assert conditioned.data.rlds_data_dir == baseline.data.rlds_data_dir
    assert conditioned.data.embodiment_conditioning == baseline.data.embodiment_conditioning == "id"
    assert conditioned.data.embodiment_state_ids == baseline.data.embodiment_state_ids
    assert conditioned.data.embodiment_normalization == baseline.data.embodiment_normalization
    assert conditioned.data.post_close_sampling_weight == baseline.data.post_close_sampling_weight
    assert conditioned.optimizer == baseline.optimizer
    assert conditioned.lr_schedule == baseline.lr_schedule


def test_urma2_recipe_registers_structured_condition_and_initializes_new_leaves():
    resolved = resolve_native_config(
        get_experiment("pi05_n60_r6_exactid_urma2ada_lora_h32_b256_lr4e4_30k"),
        "gpu",
        "urma2-config-test",
    )

    conditioning = resolved.model.action_expert_conditioning
    assert type(conditioning).__name__ == "ExtractedEmbodimentAdaRMS"
    assert type(conditioning.extractor).__name__ == "Urma2EmbodimentExtractor"
    assert conditioning.extractor.embedding_dim == 256
    assert "embodiment_encoder/.*" in resolved.weight_loader.missing_regex
    assert "embodiment_fusion_(in|out)/.*" in resolved.weight_loader.missing_regex


def test_fixcap_adarms_and_urma2_cells_match_no_text_native_recipe():
    names = (
        "july18_fixcap_warmupema_h32_open_franka9emb_noembtext_64k",
        "july20_fixcap_warmupema_h32_open_franka9emb_adarms_only_64k",
        "july20_fixcap_warmupema_h32_open_franka9emb_urma2_only_64k",
    )
    runtime = RuntimeBinding(
        platform="tpu",
        data_dir="gs://boai-iris-central2/20260718_fix_cap_pause_9emb_10p10e",
        checkpoint_base_dir="gs://boai-iris-central2/pi05_ckpts",
        fsdp_devices=4,
        norm_assets_dir="gs://boai-iris-central2/pi05_norm_assets",
        norm_asset_id="fix_cap_pause_9emb_10p10e_20260718",
    )
    resolved = [
        resolve_native_config(
            get_experiment(name),
            "tpu",
            f"{index}-comparison-test",
            runtime_profile=runtime,
        )
        for index, name in enumerate(names)
    ]

    for config in resolved:
        assert config.data.repo_id == "fix_cap_pause_9emb_10p10e_20260718"
        assert config.data.rlds_data_dir.endswith(
            "/20260718_fix_cap_pause_9emb_10p10e"
        )
        assert config.data.embodiment_conditioning == "none"
        assert config.data.embodiment_state_ids == ()
        assert config.data.embodiment_normalization
        assert config.batch_size == 256
        assert config.num_train_steps == 64_000
        assert config.save_interval == 4_000
        assert config.ema_decay == 0.999
        assert config.lr_schedule == resolved[0].lr_schedule
        assert config.optimizer == resolved[0].optimizer

    assert resolved[0].model.embodiment_vocab_size == 0
    assert resolved[1].model.embodiment_vocab_size == 64
    assert resolved[2].model.embodiment_vocab_size == 0
    assert "embodiment_embed/.*" in resolved[1].weight_loader.missing_regex
    assert "embodiment_encoder/.*" in resolved[2].weight_loader.missing_regex
    assert "embodiment_fusion_(in|out)/.*" in resolved[2].weight_loader.missing_regex


def test_remaining81_urma_concat_project_resolves_global_norm_recipe():
    runtime = RuntimeBinding(
        platform="tpu",
        data_dir="gs://boai-iris-central2/20260718_fix_cap_pause_remaining81_10p10e",
        checkpoint_base_dir="gs://boai-iris-central2/pi05_ckpts",
        fsdp_devices=32,
        norm_assets_dir="gs://boai-iris-central2/pi05_norm_assets",
        norm_asset_id="fix_cap_pause_remaining81_globalnorm_10p10e_20260721",
    )
    resolved = resolve_native_config(
        get_experiment(
            "july23_fixcap_warmupema_h32_open_franka_remaining81_"
            "urma2_concatproj1024_globalnorm_b1024_64k"
        ),
        "tpu",
        "remaining81-urma-concat-project-test",
        runtime_profile=runtime,
    )

    conditioning = resolved.model.action_expert_conditioning
    assert type(conditioning).__name__ == "ExtractedEmbodimentConcatProjectAdaRMS"
    assert conditioning.projection_dim == 1024
    assert conditioning.extractor.embedding_dim == 256
    assert not resolved.data.embodiment_normalization
    assert resolved.data.embodiment_conditioning == "none"
    assert resolved.data.embodiment_state_ids == ()
    assert resolved.batch_size == 1_024
    assert resolved.num_train_steps == 64_000
    assert resolved.save_interval == 1_000
    assert resolved.keep_every_n_completed_updates == 4_000
    assert resolved.lr_schedule.peak_lr == pytest.approx(4e-4)
    assert resolved.lr_schedule.decay_lr == pytest.approx(8e-5)
    assert resolved.ema_decay == 0.999
    assert "embodiment_encoder/.*" in resolved.weight_loader.missing_regex
    assert "embodiment_fusion_(in|out)/.*" in resolved.weight_loader.missing_regex
    assert resolved.weight_loader.adarms_condition_width_multiplier == 1


def test_remaining81_urma_backbone_token_resolves_global_norm_recipe():
    runtime = RuntimeBinding(
        platform="tpu",
        data_dir="gs://boai-iris-central2/20260718_fix_cap_pause_remaining81_10p10e",
        checkpoint_base_dir="gs://boai-iris-central2/pi05_ckpts",
        fsdp_devices=32,
        norm_assets_dir="gs://boai-iris-central2/pi05_norm_assets",
        norm_asset_id="fix_cap_pause_remaining81_globalnorm_10p10e_20260721",
    )
    resolved = resolve_native_config(
        get_experiment(
            "july23_fixcap_warmupema_h32_open_franka_remaining81_"
            "urma2_backbonetoken_globalnorm_b1024_64k"
        ),
        "tpu",
        "remaining81-urma-backbone-token-test",
        runtime_profile=runtime,
    )

    assert type(resolved.model.action_expert_conditioning).__name__ == "TimestepOnly"
    conditioning = resolved.model.backbone_conditioning
    assert type(conditioning).__name__ == "ExtractedEmbodimentBackboneToken"
    assert conditioning.token_width == 2048
    assert conditioning.extractor.embedding_dim == 256
    assert not resolved.data.embodiment_normalization
    assert resolved.data.embodiment_conditioning == "none"
    assert resolved.data.embodiment_state_ids == ()
    assert resolved.batch_size == 1_024
    assert resolved.num_train_steps == 64_000
    assert resolved.save_interval == 1_000
    assert resolved.keep_every_n_completed_updates == 4_000
    assert resolved.lr_schedule.peak_lr == pytest.approx(4e-4)
    assert resolved.lr_schedule.decay_lr == pytest.approx(8e-5)
    assert resolved.ema_decay == 0.999
    assert "embodiment_encoder/.*" in resolved.weight_loader.missing_regex
    assert "embodiment_token_(norm|proj)/.*" in (
        resolved.weight_loader.missing_regex
    )
    assert "embodiment_fusion_" not in resolved.weight_loader.missing_regex
    assert resolved.weight_loader.adarms_condition_width_multiplier == 1


def test_oer_urma3_initializes_new_conditioning_leaves_from_reference_params():
    resolved = config_lib.get_config(
        "july28_oer_hash8095_noembtext_urma3_concatproj1024_"
        "globalnorm_clip5_val5_b1024_64k"
    )

    assert "embodiment_encoder/.*" in resolved.weight_loader.missing_regex
    assert "embodiment_fusion_(in|out)/.*" in (
        resolved.weight_loader.missing_regex
    )


def test_h32_adaemb_and_strict_no_conditioning_contracts():
    ada = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_"
            "adaemb_id_state_fullft_h32_40k"
        ),
        "gpu",
        "ada-h32",
    )
    no_conditioning = resolve_native_config(
        get_experiment(
            "pi05_9emb_n60_currentplanner_phasebalanced_nocond_fullft_h32_40k"
        ),
        "gpu",
        "nocond-h32",
    )

    assert ada.model.action_horizon == no_conditioning.model.action_horizon == 32
    assert ada.batch_size == no_conditioning.batch_size == 64
    assert ada.lr_schedule.peak_lr == no_conditioning.lr_schedule.peak_lr == 1e-4
    assert ada.model.embodiment_vocab_size == 64
    assert ada.data.embodiment_conditioning == "id"
    assert ada.data.embodiment_normalization
    assert no_conditioning.model.embodiment_vocab_size == 0
    assert no_conditioning.data.embodiment_conditioning == "none"
    assert no_conditioning.data.embodiment_state_ids == ()
    assert not no_conditioning.data.embodiment_normalization


def test_per_embodiment_normalization_selects_scalar_and_batched_tables():
    def stats(offset):
        return transforms.NormStats(
            mean=np.asarray([offset, offset], np.float32),
            std=np.ones(2, np.float32),
            q01=np.asarray([offset, offset], np.float32),
            q99=np.asarray([offset + 2, offset + 2], np.float32),
        )

    norm_stats = {
        transforms.embodiment_norm_key(field, embodiment_id): stats(offset)
        for embodiment_id, offset in ((0, 0.0), (10, 10.0))
        for field in ("state", "actions")
    }
    transform = transforms.NormalizeByEmbodiment(
        norm_stats, (0, 10), use_quantiles=True
    )
    scalar = transform({
        "state": np.asarray([1.0, 1.0]),
        "actions": np.asarray([[1.0, 1.0]]),
        "embodiment_id": np.asarray(0),
    })
    batched = transform({
        "state": np.asarray([[1.0, 1.0], [11.0, 11.0]]),
        "actions": np.asarray([[[1.0, 1.0]], [[11.0, 11.0]]]),
        "embodiment_id": np.asarray([0, 10]),
    })

    np.testing.assert_allclose(scalar["state"], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(batched["state"], np.zeros((2, 2)), atol=1e-6)
    restored = transforms.UnnormalizeByEmbodiment(
        norm_stats, (0, 10), use_quantiles=True
    )(batched)
    np.testing.assert_allclose(restored["actions"], [[[1.0, 1.0]], [[11.0, 11.0]]])


def test_embodiment_one_hot_is_direct_and_rejects_unknown_ids():
    transform = transforms.AppendEmbodimentOneHot((0, 10, 58))
    output = transform({"state": np.zeros(2, np.float32), "embodiment_id": np.int32(10)})

    np.testing.assert_array_equal(output["state"], [0.0, 0.0, -1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="outside configured codebook"):
        transform({"state": np.zeros(2, np.float32), "embodiment_id": np.int32(3)})


def test_native_configs_carry_intended_rlds_input_sampling() -> None:
    from molmo_spaces.training import legacy_rlds_input_sampling as legacy
    from molmo_spaces.training import sep6_camd8k_fullft_contract as sep6_contract
    from pi05_integration import data_processing as pi_data

    names = set(config_lib.list_configs())
    sep6_names = {
        sep6_contract.experiment_name(cell) for cell in sep6_contract.CELLS
    }
    assert legacy.LEGACY_NATIVE_CONFIGS <= names, sorted(legacy.LEGACY_NATIVE_CONFIGS - names)
    assert names >= sep6_names, sorted(sep6_names - names)
    # The merged pi05 KI / mixture lineages pin the legacy contract in their own builders.
    pi05_ki_mixture_names = {config.name for config in config_lib._PI05_KI_MIXTURE_CONFIGS}
    assert pi05_ki_mixture_names <= names, sorted(pi05_ki_mixture_names - names)
    tpu_droidft_names = {builder.__name__ for builder in config_lib._TPU_DROIDFT_ARMS}
    for name in names:
        contract = config_lib.get_config(name).data.rlds_input_sampling
        if name in legacy.LEGACY_NATIVE_CONFIGS or name in pi05_ki_mixture_names:
            assert isinstance(contract, pi_data.LegacyRldsInputSampling), name
        elif name.endswith("_processshardv2") or name in tpu_droidft_names:
            assert isinstance(contract, pi_data.ProcessShardedFileShuffleV2), name
        elif name.endswith("_processshardv1"):
            assert isinstance(contract, pi_data.ProcessShardedFileShuffleV1), name
            assert not isinstance(contract, pi_data.ProcessShardedFileShuffleV2), name
        else:
            assert isinstance(contract, pi_data.DeterministicReplayV3), name
    assert isinstance(
        config_utils.RLDSMolmoSpacesDataConfig.__dataclass_fields__["rlds_input_sampling"].default_factory(),
        pi_data.DeterministicReplayV3,
    )
    # Compatibility aliases restore recorded legacy runs and must keep the legacy contract.
    for alias, builder in config_lib._compatibility_aliases():
        if builder.__name__ in legacy.LEGACY_NATIVE_CONFIGS:
            assert isinstance(
                config_lib.get_config(alias).data.rlds_input_sampling, pi_data.LegacyRldsInputSampling
            ), alias


def test_primary_config_lookup_does_not_construct_unrelated_builders(monkeypatch) -> None:
    def unrelated_unpublished_builder():
        raise RuntimeError("unrelated assets are unpublished")

    unrelated_unpublished_builder.__name__ = "unrelated_unpublished"
    target = config_lib.internal_september20_native4k_norm_build_only
    monkeypatch.setattr(
        config_lib,
        "_CONFIG_BUILDERS",
        (unrelated_unpublished_builder, target),
    )

    resolved = config_lib.get_config(target.__name__)

    assert resolved.name == target.__name__


def test_tpu_droidft_arms_differ_from_della_parent_only_in_placement_and_mixture() -> None:
    import dataclasses

    from pi05_integration import data_processing as pi_data

    parent = config_lib.get_config("pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft")
    expected = {
        "pi05_mbdroid_v1_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft_tpu": (512, 16, {"droid": 0.65, "molmobot": 0.30, "realrig9k": 0.05}),
        "pi05_mbdroid_v1_pernorm_fullft_ki_b1024_lr5e5_h16_wu1k_ema99_droidft_tpu": (1024, 32, {"droid": 0.65, "molmobot": 0.30, "realrig9k": 0.05}),
        "pi05_droidonly_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft_tpu": (512, 16, {"droid": 1.0}),
        "pi05_droidmb_70_30_pernorm_fullft_ki_b512_lr5e5_h16_wu1k_ema99_droidft_tpu": (512, 16, {"droid": 0.70, "molmobot": 0.30}),
    }
    assert {b.__name__ for b in config_lib._TPU_DROIDFT_ARMS} == set(expected)
    for name, (batch, fsdp, shares) in expected.items():
        arm = config_lib.get_config(name)
        top = {f.name for f in dataclasses.fields(parent) if f.name != "data" and getattr(parent, f.name) != getattr(arm, f.name)}
        assert top <= {"name", "batch_size", "fsdp_devices", "checkpoint_base_dir"}, (name, top)
        data = {f.name for f in dataclasses.fields(parent.data) if getattr(parent.data, f.name) != getattr(arm.data, f.name)}
        assert data <= {"rlds_data_dir", "assets", "datasets", "rlds_input_sampling"}, (name, data)
        assert (arm.batch_size, arm.fsdp_devices) == (batch, fsdp)
        assert isinstance(arm.data.rlds_input_sampling, pi_data.ProcessShardedFileShuffleV2)
        assert arm.data.repo_id == parent.data.repo_id
        assert arm.data.assets.asset_id == "molmospaces_mbdroid_pernorm_v1"
        assert arm.checkpoint_base_dir.startswith("gs://") and arm.data.rlds_data_dir.startswith("gs://")
        got = {}
        for d in arm.data.datasets:
            group = "droid" if d.name == "droid" else ("molmobot" if "molmobot" in d.name else "realrig9k")
            got[group] = got.get(group, 0.0) + d.weight
            if d.name == "droid":
                assert d.data_subdir == "gs://v6_east1d/OXE" and d.version == "1.0.1"
                assert d.exclude_pick_and_place and d.filter_dict_path
        assert set(got) == set(shares), (name, got)
        for group, share in shares.items():
            assert abs(got[group] - share) < 1e-9, (name, group, got[group])
        assert abs(sum(d.weight for d in arm.data.datasets) - 1.0) < 1e-9
        # The sim builds are the same names/versions the Della parent reads.
        parent_sim = {(d.name, d.version) for d in parent.data.datasets if d.name != "droid"}
        assert {(d.name, d.version) for d in arm.data.datasets if d.name != "droid"} <= parent_sim
