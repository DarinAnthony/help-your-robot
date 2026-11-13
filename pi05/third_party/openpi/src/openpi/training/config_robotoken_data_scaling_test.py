from __future__ import annotations

import dataclasses

from flax import nnx

from openpi.training import config as config_lib


def test_robotoken_fullft_data_scaling_native_configs_are_nested_and_matched():
    baseline = config_lib.get_config(
        "september2_ur5e_trossen_v6_camd8k_robotokens_backbone_fullft_"
        "gaussclip5_b128_seed42_lr1e5_24k"
    )
    for count in (2_000, 4_000):
        prefix = (
            "september4_ur5e_trossen_v6_camd8k_"
            f"hash{count // 1_000}k_s42_robotokens_backbone_fullft_"
            "gaussclip5_b128_seed42_lr1e5_"
        )
        initial = config_lib.get_config(prefix + "24k")
        continuation = config_lib.get_config(prefix + "40k")
        assert initial.data.episode_selector.to_record() == {
            "type": "HashEpisodeSelector",
            "episode_count": count,
            "seed": 42,
            "identity_key": "episode_metadata.file_path",
            "ranking": "sha256(seed, episode_key)",
        }
        assert initial.data.assets.asset_id.endswith(
            f"hash{count // 1_000}k_s42_stdnorm_20260904"
        )
        assert isinstance(initial.freeze_filter, type(nnx.Nothing()))
        for field in (
            "batch_size",
            "fsdp_devices",
            "seed",
            "optimizer",
            "ema_decay",
            "save_interval",
            "keep_every_n_completed_updates",
            "data_visualization",
        ):
            assert getattr(initial, field) == getattr(baseline, field)
        assert continuation.num_train_steps == 40_000
        assert continuation.lr_schedule == dataclasses.replace(
            initial.lr_schedule, decay_steps=40_000
        )
        for field in dataclasses.fields(initial):
            if field.name not in {"name", "num_train_steps", "lr_schedule"}:
                assert getattr(continuation, field.name) == getattr(initial, field.name)
