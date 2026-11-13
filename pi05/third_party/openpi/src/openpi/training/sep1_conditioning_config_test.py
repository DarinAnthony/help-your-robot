import dataclasses
import pathlib

import numpy as np

from molmo_spaces.paths import PI_EMBODIMENT_ASSETS_DIR
from openpi.training import config
from molmo_spaces.training import experiments
from molmo_spaces.training import sep1_conditioning_contract
from molmo_spaces.training.pi_adapter import resolve_native_config
from molmo_spaces.training.runtimes import RuntimeBinding
from pi05_integration import embodiment_bundle, input_config


def _rss_kib() -> int:
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("Linux VmRSS is unavailable")


def _cell_d_tokenize_transform(tmp_path):
    native = config.get_config(
        sep1_conditioning_contract.conditioning_experiment_name("d")
    )
    created = native.data.create(pathlib.Path(tmp_path), native.model)
    return next(
        transform
        for transform in created.model_transforms.inputs
        if type(transform).__name__ == "TokenizePrompt"
    )


def test_sep1_native_conditioning_cells_are_matched_and_fail_closed():
    common = dict(
        normalization_asset_id="asset",
        urma4_bundle_sha256="4" * 64,
        response_history_stats_sha256="5" * 64,
    )
    cells = {
        cell: config._september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
            f"test_{cell}", cell=cell, **common
        )
        for cell in (
            "learnedid_backbone",
            "urma4_adarms",
            "urma4_backbone",
            "urma4_text",
            "response_history_backbone",
        )
    }
    for native in cells.values():
        assert native.batch_size == 128
        assert native.num_train_steps == 24_000
        assert native.save_interval == 1_000
        assert native.keep_every_n_completed_updates == 8_000
        assert native.lr_schedule.peak_lr == native.lr_schedule.decay_lr == 1e-5
        assert native.freeze_filter == native.model.get_freeze_filter()
        assert native.weight_loader.missing_regex != ".*"
    assert sep1_conditioning_contract.urma4_assets_published()

    assert isinstance(
        cells["learnedid_backbone"].model.backbone_conditioning,
        input_config.LearnedEmbodimentBackboneToken,
    )
    assert cells["learnedid_backbone"].model.backbone_conditioning.oov_index == 8_000
    assert "embodiment_backbone_embed" in cells["learnedid_backbone"].weight_loader.missing_regex
    assert "embodiment_fusion" in cells["urma4_adarms"].weight_loader.missing_regex
    assert "embodiment_token" in cells["urma4_backbone"].weight_loader.missing_regex
    assert cells["urma4_text"].weight_loader.missing_regex == ".*lora.*"
    assert cells["urma4_text"].model.max_token_len == 768
    assert "response_history" in cells["response_history_backbone"].weight_loader.missing_regex
    assert cells["response_history_backbone"].data.datasets[0].version == "1.2.0"
    assert all(
        native.data.datasets[0].version == "1.1.0"
        for cell, native in cells.items()
        if cell != "response_history_backbone"
    )


def test_b_to_d_full_native_data_create_uses_matched_token_contract(tmp_path):
    for cell in ("urma4_adarms", "urma4_backbone", "urma4_text"):
        native = config._september1_ur5e_trossen_v6_camd8k_conditioning_lora_gaussclip5_b128_lr1e5_24k(
            f"test_create_{cell}",
            cell=cell,
            normalization_asset_id="missing_test_asset",
            urma4_bundle_sha256=(
                sep1_conditioning_contract.URMA4_SOURCE_BUNDLE_SHA256
            ),
        )
        created = native.data.create(pathlib.Path(tmp_path), native.model)
        assert created.model_transforms is not None
    assert native.model.max_token_len == native.data.input_config.prompt.max_token_len == 768


def test_cell_d_all_immutable_prompts_fit_without_truncation(tmp_path):
    tokenize = _cell_d_tokenize_transform(tmp_path)
    state = np.zeros(32, dtype=np.float32)
    observed = {}
    for label, bundle_name, bundle_sha256 in (
        (
            "source",
            sep1_conditioning_contract.URMA4_BUNDLE_NAME,
            sep1_conditioning_contract.URMA4_SOURCE_BUNDLE_SHA256,
        ),
        (
            "native",
            sep1_conditioning_contract.URMA4_NATIVE_TROSSEN_BUNDLE_NAME,
            sep1_conditioning_contract.URMA4_NATIVE_TROSSEN_BUNDLE_SHA256,
        ),
    ):
        strategy = input_config.Urma4BundleTextPrompt(
            bundle_dir=f"{PI_EMBODIMENT_ASSETS_DIR}/{bundle_name}",
            bundle_sha256=bundle_sha256,
            max_token_len=768,
        )
        builder = strategy.create_descriptor_builder(embodiment_bundle)
        _, rows = embodiment_bundle.load_bundle(
            strategy.bundle_dir, expected_schema="urma4_mjcf_controls_global_v1"
        )
        lengths = []
        rss_after_builder = _rss_kib()
        peak_rss = rss_after_builder
        for index, embodiment_id in enumerate(rows, 1):
            prompt = strategy.condition(
                "open the drawer",
                embodiment_id=embodiment_id,
                descriptor_builder=builder,
            )
            output = tokenize({"prompt": prompt, "state": state.copy()})
            lengths.append(int(output["tokenized_prompt_mask"].sum()))
            if index % 128 == 0:
                peak_rss = max(peak_rss, _rss_kib())
        assert max(lengths) <= 768
        assert peak_rss - rss_after_builder < 256 * 1024
        observed[label] = (len(lengths), min(lengths), max(lengths))
    assert observed == {"source": (8_000, 694, 694), "native": (1, 676, 676)}


def test_public_a_to_d_native_configs_are_discoverable():
    expected = {
        (
            f"september1_ur5e_trossen_v6_camd8k_{cell}_lora_"
            "gaussclip5_b128_seed42_lr1e5_24k"
        )
        for cell in (
            "learnedid_backbone",
            "urma4_adarms",
            "urma4_backbone",
            "urma4_text",
        )
    }
    assert expected <= set(config.list_configs())
    for name in expected:
        native = config.get_config(name)
        assert native.name == name
        assert native.batch_size == 128
        assert native.num_train_steps == 24_000


def test_response_history_norm_build_and_public_configs_are_hash_separated():
    internal_name = (
        sep1_conditioning_contract.RESPONSE_HISTORY_NORM_BUILD_NATIVE_CONFIG
    )
    public_name = (
        "september1_ur5e_trossen_v6_camd8k_response_history_backbone_lora_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )
    assert internal_name in config.list_configs()
    assert public_name in config.list_configs()
    assert public_name in experiments.list_experiments()

    native = config.get_config(internal_name)
    assert native.name == internal_name
    assert native.data.datasets[0].data_subdir == (
        sep1_conditioning_contract.RESPONSE_HISTORY_DATASET_ID
    )
    assert native.data.datasets[0].version == "1.2.0"
    assert native.data.assets.asset_id == (
        sep1_conditioning_contract.RESPONSE_HISTORY_NORM_ASSET_ID
    )
    assert isinstance(
        native.data.input_config.response_history,
        input_config.CausalResponseHistoryBackboneToken,
    )
    assert native.data.input_config.response_history.stats_sha256 == "0" * 64

    public = config.get_config(public_name)
    assert public.name == public_name
    assert public.data.datasets[0].version == "1.2.0"
    assert public.data.input_config.response_history.stats_sha256 == (
        sep1_conditioning_contract.RESPONSE_HISTORY_NORMALIZATION_STATS_SHA256
    )


def test_40k_native_successors_preserve_24k_recipe_loader_and_freeze_contract():
    for cell in ("a", "b", "c", "d", "e"):
        base = config.get_config(
            sep1_conditioning_contract.conditioning_experiment_name(cell)
        )
        continuation = config.get_config(
            sep1_conditioning_contract.conditioning_experiment_name(
                cell, total_updates=40_000
            )
        )
        assert continuation.num_train_steps == 40_000
        assert continuation.save_interval == 1_000
        assert continuation.keep_every_n_completed_updates == 8_000
        assert continuation.weight_loader == base.weight_loader
        assert continuation.weight_loader.missing_regex == base.weight_loader.missing_regex
        assert continuation.freeze_filter == base.freeze_filter

        base_record = dataclasses.asdict(base)
        continuation_record = dataclasses.asdict(continuation)
        base_record.pop("name")
        continuation_record.pop("name")
        base_record.pop("num_train_steps")
        continuation_record.pop("num_train_steps")
        assert base_record["lr_schedule"].pop("decay_steps") == 24_000
        assert continuation_record["lr_schedule"].pop("decay_steps") == 40_000
        assert continuation_record == base_record


def test_40k_runtime_resume_reuses_24k_checkpoint_namespace_without_overwrite():
    continuation_name = sep1_conditioning_contract.conditioning_experiment_name(
        "a", total_updates=40_000
    )
    experiment = experiments.get_experiment(continuation_name)
    native = resolve_native_config(
        experiment,
        "tpu",
        sep1_conditioning_contract.conditioning_run_name("a"),
        runtime_profile=RuntimeBinding(
            platform="tpu",
            data_dir="gs://boai-iris-central2/20260829_ur5e_trossen_v6_motioncompatible8k_camd_v1",
            checkpoint_base_dir="gs://boai-iris-central2/pi05_ckpts",
            fsdp_devices=32,
            norm_assets_dir="gs://boai-iris-central2/pi05_norm_assets",
            norm_asset_id=experiment.normalization_asset_id,
        ),
        resume=True,
        overwrite=False,
    )
    assert native.name == sep1_conditioning_contract.conditioning_experiment_name("a")
    assert native.exp_name == sep1_conditioning_contract.conditioning_run_name("a")
    assert native.num_train_steps == 40_000
    assert native.resume
    assert not native.overwrite
    assert native.policy_metadata["resolved_training_experiment"]["checkpoint_lineage"] == {
        "registered_config": continuation_name,
        "checkpoint_config_name": (
            sep1_conditioning_contract.conditioning_experiment_name("a")
        ),
        "reason": "explicit named continuation of an existing optimizer/checkpoint lineage",
    }
