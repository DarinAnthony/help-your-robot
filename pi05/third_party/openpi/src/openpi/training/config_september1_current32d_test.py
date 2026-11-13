import dataclasses

from openpi.training import config as config_lib


def test_september1_current32d_control_preserves_single_robot_native_recipe():
    control = config_lib.get_config(
        "september1_id355_camd400_current32d_lora_gaussclip5_b128_lr1e5_seed42_24k"
    )
    historical = config_lib.get_config(
        "august29_id355_camd40p10e_lora_gaussclip5_b128_lr1e5_seed42_h32_60k"
    )
    assert control.num_train_steps == 24_000
    assert dataclasses.replace(control.data, datasets=historical.data.datasets) == historical.data
    assert [dataset.data_subdir for dataset in control.data.datasets] == [
        "20260827_ur5e_1emb_id0355_camD_slowdwell40p10e_contiguous7_oerasync_v2"
    ]
    assert control.model == historical.model
    assert control.batch_size == historical.batch_size == 128
    assert control.lr_schedule == historical.lr_schedule
    assert control.optimizer == historical.optimizer
    assert control.ema_decay is historical.ema_decay is None
    assert control.save_interval == historical.save_interval == 1_000
    assert control.keep_every_n_completed_updates == 4_000
