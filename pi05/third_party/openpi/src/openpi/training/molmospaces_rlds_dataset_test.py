import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training import data_loader
from openpi.training import dataset_mixture
from openpi.training.molmospaces_rlds_dataset import _tensorflow_episode_is_selected
from openpi.training.molmospaces_rlds_dataset import _episode_prompt
from openpi.training.molmospaces_rlds_dataset import approximate_batch_count
from openpi.training.molmospaces_rlds_dataset import canonicalize_embodiment_filter_ids
from openpi.training.molmospaces_rlds_dataset import chunk_action_loss_mask
from openpi.training.molmospaces_rlds_dataset import embodiment_id_is_selected
from openpi.training.molmospaces_rlds_dataset import RESPONSE_HISTORY_KEYS
from openpi.training.molmospaces_rlds_dataset import response_history_fields
from openpi.training.molmospaces_rlds_dataset import response_history_specs


def _response_features(dtype_wrapper=lambda dtype: dtype):
    shapes = {
        "response_state_history": ((8, 10), np.float32),
        "response_state_history_mask": ((8,), np.bool_),
        "response_action_history": ((7, 10), np.float32),
        "response_action_history_mask": ((7,), np.bool_),
    }
    return {
        "proprio": SimpleNamespace(shape=(10,), dtype=np.float32),
        **{
            key: SimpleNamespace(shape=shape, dtype=dtype_wrapper(dtype))
            for key, (shape, dtype) in shapes.items()
        },
    }


def test_embodiment_subset_keeps_original_global_ids():
    filter_ids = canonicalize_embodiment_filter_ids((0, 2))
    global_ids = (0, 1, 2, 10, 22)

    selected = tuple(
        embodiment_id
        for embodiment_id in global_ids
        if embodiment_id_is_selected(embodiment_id, filter_ids)
    )

    assert filter_ids == (0, 2)
    assert selected == (0, 2)
    assert all(embodiment_id_is_selected(value, ()) for value in global_ids)


def test_progress_length_is_in_yielded_batches():
    assert approximate_batch_count(1_024) == 2_637
    assert approximate_batch_count(2_048) == 1_319
    with pytest.raises(ValueError, match="batch_size"):
        approximate_batch_count(0)


def test_chunk_action_loss_mask_separates_model_and_physical_coordinates():
    time_valid = np.asarray([[True, True, False], [True, False, False]])
    semantic = np.broadcast_to(
        np.asarray([True, True, False, False]), (2, 3, 4)
    )

    model = chunk_action_loss_mask(
        time_valid, semantic, coordinates="model"
    )
    physical = chunk_action_loss_mask(
        time_valid, semantic, coordinates="physical"
    )

    assert model.shape == (2, 3, 1)
    np.testing.assert_array_equal(model[..., 0], time_valid)
    assert physical.shape == semantic.shape
    np.testing.assert_array_equal(
        physical,
        np.logical_and(time_valid[..., None], semantic),
    )
    with pytest.raises(ValueError, match="model.*physical"):
        chunk_action_loss_mask(time_valid, semantic, coordinates="unknown")


@pytest.mark.parametrize(
    ("filter_ids", "message"),
    [((-1,), "non-negative"), ((0, 0), "unique")],
)
def test_embodiment_subset_rejects_invalid_global_ids(filter_ids, message):
    with pytest.raises(ValueError, match=message):
        canonicalize_embodiment_filter_ids(filter_ids)


@pytest.mark.parametrize("filter_ids", [(True,), (1.5,), ("2",)])
def test_embodiment_subset_rejects_non_integer_ids(filter_ids):
    with pytest.raises(TypeError):
        canonicalize_embodiment_filter_ids(filter_ids)


def test_tensorflow_episode_predicate_selects_without_reindexing():
    tf = SimpleNamespace(
        int32=np.int32,
        cast=lambda value, dtype: np.asarray(value, dtype=dtype),
        equal=np.equal,
        reduce_any=np.any,
    )
    allowed_ids = np.asarray((0, 2), dtype=np.int32)

    assert bool(_tensorflow_episode_is_selected(np.asarray([2, 2]), allowed_ids, tf))
    assert not bool(_tensorflow_episode_is_selected(np.asarray([10, 10]), allowed_ids, tf))


def test_episode_prompt_prefers_exact_recorded_instruction():
    class Strings:
        @staticmethod
        def join(parts):
            return np.char.add(np.char.add(parts[0], parts[1]), parts[2])

        @staticmethod
        def length(values):
            return np.char.str_len(values)

    tf = SimpleNamespace(
        strings=Strings,
        fill=lambda shape, value: np.full(shape, value),
        where=np.where,
    )
    metadata = {
        "task": np.asarray(["open", "open"]),
        "target": np.asarray(["drawer", "drawer"]),
        "task_instruction": np.asarray(["open the drawer", "open the drawer"]),
    }

    assert _episode_prompt(metadata, 2, tf).tolist() == [
        "open the drawer",
        "open the drawer",
    ]


def test_episode_prompt_preserves_legacy_lowercase_fallback():
    class Strings:
        @staticmethod
        def join(parts):
            return np.char.add(np.char.add(parts[0], parts[1]), parts[2])

    tf = SimpleNamespace(
        strings=Strings,
        fill=lambda shape, value: np.full(shape, value),
    )
    metadata = {
        "task": np.asarray(["open", "open"]),
        "target": np.asarray(["drawer", "drawer"]),
    }

    assert _episode_prompt(metadata, 2, tf).tolist() == [
        "open the drawer",
        "open the drawer",
    ]


def test_rlds_loader_receives_configured_embodiment_subset(monkeypatch):
    captured = {}

    class FakeMolmoSpacesRldsDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openpi.training.molmospaces_rlds_dataset as dataset_module

    monkeypatch.setattr(
        dataset_module, "MolmoSpacesRldsDataset", FakeMolmoSpacesRldsDataset
    )
    data_config = SimpleNamespace(
        datasets=(SimpleNamespace(name="molmospaces_emb_mix", weight=1.0),),
        rlds_data_dir="/unused",
        rlds_split="train[:95%]",
        episode_selector=None,
        episode_split=None,
        episode_split_role="train",
        input_config=SimpleNamespace(
            prompt="none", action_loss_coordinates="physical"
        ),
        post_close_sampling_weight=0.5,
        embodiment_filter_ids=(0, 2),
        behavior_prompt_overlays=("overlay",),
    )

    data_loader.create_rlds_dataset(data_config, action_horizon=32, batch_size=64)

    assert captured["embodiment_filter_ids"] == (0, 2)
    assert captured["action_chunk_size"] == 32
    assert captured["batch_size"] == 64
    assert captured["behavior_prompt_overlays"] == ("overlay",)
    assert captured["action_loss_coordinates"] == "physical"


def test_rlds_loader_can_disable_prompt_overlay_for_normalization(monkeypatch):
    captured = {}

    class FakeMolmoSpacesRldsDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openpi.training.molmospaces_rlds_dataset as dataset_module

    monkeypatch.setattr(
        dataset_module, "MolmoSpacesRldsDataset", FakeMolmoSpacesRldsDataset
    )
    data_config = SimpleNamespace(
        datasets=(SimpleNamespace(name="molmospaces_emb_mix", weight=1.0),),
        rlds_data_dir="/unused",
        rlds_split="train",
        episode_selector=None,
        episode_split=None,
        episode_split_role="train",
        input_config=SimpleNamespace(prompt="none", action_loss_coordinates="model"),
        post_close_sampling_weight=None,
        embodiment_filter_ids=(),
        behavior_prompt_overlays=("overlay",),
    )

    data_loader.create_rlds_dataset(
        data_config,
        action_horizon=None,
        batch_size=64,
        apply_behavior_prompt_overlay=False,
    )

    assert captured["behavior_prompt_overlays"] == ()


def test_startup_audit_counts_only_selected_global_ids(monkeypatch):
    episodes = [
        {
            "episode_metadata": {
                "task": b"open",
                "target": b"drawer",
                "file_path": b"/data/house_1/run",
                "variant": variant.encode(),
                "embodiment_id": embodiment_id,
            },
            "steps": range(num_frames),
        }
        for variant, embodiment_id, num_frames in (
            ("v013", 0, 2),
            ("v018", 1, 7),
            ("v066", 2, 3),
        )
    ]
    fake_tf = SimpleNamespace(
        config=SimpleNamespace(set_visible_devices=lambda *_args, **_kwargs: None)
    )
    fake_tfds = SimpleNamespace(
        builder_from_directory=lambda _path: SimpleNamespace(
            as_dataset=lambda **_kwargs: episodes
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    data_config = SimpleNamespace(
        datasets=(SimpleNamespace(name="molmospaces_emb_mix", version="1.0.0", data_subdir=None),),
        rlds_data_dir="/unused",
        rlds_split="train[:95%]",
        embodiment_filter_ids=(0, 2),
    )

    rows = dataset_mixture.scan_rlds_mixture(data_config, max_episodes=None)
    embodiment_rows = {
        row["value"]: (row["episodes"], row["frames"])
        for row in rows
        if row["group"] == "embodiment"
    }
    dataset_row = next(row for row in rows if row["group"] == "dataset")

    assert embodiment_rows == {"v013": (1, 2), "v066": (1, 3)}
    assert (dataset_row["episodes"], dataset_row["frames"]) == (2, 5)


def test_startup_audit_caps_each_dataset_separately(monkeypatch):
    def episodes(name, count):
        return [
            {
                "episode_metadata": {
                    "task": b"open",
                    "target": b"drawer",
                    "file_path": f"/data/{name}/house_{i}/run".encode(),
                    "variant": b"v013",
                    "embodiment_id": 0,
                },
                "steps": range(3),
            }
            for i in range(count)
        ]

    by_name = {"first": episodes("first", 5), "second": episodes("second", 4)}
    fake_tf = SimpleNamespace(config=SimpleNamespace(set_visible_devices=lambda *_args, **_kwargs: None))
    fake_tfds = SimpleNamespace(
        builder_from_directory=lambda path: SimpleNamespace(
            as_dataset=lambda **_kwargs: by_name["first" if "/first/" in path else "second"]
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    data_config = SimpleNamespace(
        datasets=(
            SimpleNamespace(name="first", version="1.0.0", data_subdir=None),
            SimpleNamespace(name="second", version="1.0.0", data_subdir=None),
        ),
        rlds_data_dir="/unused",
        rlds_split="train",
        embodiment_filter_ids=(),
    )

    rows = dataset_mixture.scan_rlds_mixture(data_config, max_episodes=2)
    per_dataset = {row["value"]: row["episodes"] for row in rows if row["group"] == "dataset"}

    # A shared cap would have spent all of it on "first" and never reached "second".
    assert per_dataset == {"./first:1.0.0": 2, "./second:1.0.0": 2}
    assert all(row["sampled"] for row in rows if row["group"] == "dataset")


def test_letterbox_keeps_droid_aspect_ratio():
    tf = pytest.importorskip("tensorflow")
    from openpi.training.molmospaces_rlds_dataset import letterbox_image

    image = tf.fill([180, 320, 3], tf.constant(200, tf.uint8))
    boxed = letterbox_image(image, 256, 256, tf).numpy()

    assert boxed.shape == (256, 256, 3) and boxed.dtype == np.uint8
    content_rows = np.flatnonzero(boxed[:, 128, 0] > 0)
    # 320 -> 256 wide scales 180 -> 144 tall; the remaining 112 rows are padding.
    assert len(content_rows) == 144
    assert boxed[0].max() == 0 and boxed[-1].max() == 0


def test_instruction_pick_skips_empty_variants():
    tf = pytest.importorskip("tensorflow")
    from openpi.training.molmospaces_rlds_dataset import has_nonempty_instruction
    from openpi.training.molmospaces_rlds_dataset import pick_nonempty_instruction

    instructions = tf.constant(["", "  ", "open the drawer"])
    picks = {pick_nonempty_instruction(instructions, tf).numpy() for _ in range(20)}

    assert picks == {b"open the drawer"}
    assert bool(has_nonempty_instruction(instructions, tf))
    assert not bool(has_nonempty_instruction(tf.constant(["", " ", ""]), tf))


def test_chunk_motion_ignores_absolute_gripper_value():
    tf = pytest.importorskip("tensorflow")
    from openpi.training.molmospaces_rlds_dataset import chunk_motion_norm

    semantic = tf.constant([True] * 8 + [False] * 2)
    model_mask = tf.ones([4, 1], tf.bool)  # model-coordinate loss mask is (H, 1)

    def chunk(arm=0.0, gripper=(0.0, 0.0, 0.0, 0.0)):
        actions = np.zeros((4, 10), np.float32)
        actions[:, :7] = arm
        actions[:, 7] = gripper
        return tf.constant(actions)

    norm = lambda actions, mask=model_mask: float(chunk_motion_norm(actions, mask, semantic, tf))
    # Parked arm: static whether the gripper is held open or held closed.
    assert norm(chunk(gripper=(0, 0, 0, 0))) == 0.0
    assert norm(chunk(gripper=(1, 1, 1, 1))) == 0.0
    # Parked arm while the gripper closes is real supervision, not idle.
    assert norm(chunk(gripper=(0, 0.5, 1, 1))) > 1e-4
    assert norm(chunk(arm=0.01, gripper=(1, 1, 1, 1))) > 1e-4
    # Past-the-end chunk steps (masked out) do not count as a gripper change.
    tail_masked = tf.constant([[True]] * 2 + [[False]] * 2)
    assert norm(chunk(gripper=(1, 1, 0, 0)), tail_masked) == 0.0


def test_dataset_root_accepts_absolute_data_subdir():
    from openpi.training.molmospaces_rlds_dataset import RLDSDataset, dataset_root

    root = "gs://bucket-a/rlds"
    assert dataset_root(root, RLDSDataset("x", "1.0.0")) == "gs://bucket-a/rlds"
    assert dataset_root(root, RLDSDataset("x", "1.0.0", data_subdir="sub/")) == "gs://bucket-a/rlds/sub"
    assert dataset_root(root, RLDSDataset("x", "1.0.0", data_subdir=".")) == "gs://bucket-a/rlds"
    assert dataset_root(root, RLDSDataset("x", "1.0.0", data_subdir="gs://bucket-b/OXE/")) == "gs://bucket-b/OXE"
    assert dataset_root(root, RLDSDataset("x", "1.0.0", data_subdir="/scratch/rlds")) == "/scratch/rlds"


def test_response_history_specs_cover_mixed_schema_components():
    legacy_1_0 = {"proprio": SimpleNamespace(shape=(10,), dtype=np.float32)}
    current = _response_features()
    tf_style = _response_features(lambda dtype: SimpleNamespace(as_numpy_dtype=dtype))

    assert response_history_specs([legacy_1_0, legacy_1_0]) == {}
    specs = response_history_specs([legacy_1_0, current, tf_style])
    assert tuple(specs) == tuple(sorted(RESPONSE_HISTORY_KEYS))
    assert specs["response_state_history"] == ((8, 10), np.dtype(np.float32))
    assert specs["response_action_history_mask"] == ((7,), np.dtype(np.bool_))

    partial = {key: value for key, value in current.items() if key != "response_action_history"}
    with pytest.raises(ValueError, match="partial response-history"):
        response_history_specs([current, partial])
    conflicting = {**current, "response_state_history": SimpleNamespace(shape=(4, 10), dtype=np.float32)}
    with pytest.raises(ValueError, match="disagree on response_state_history"):
        response_history_specs([current, conflicting])


def test_response_history_fields_fill_absent_history_like_schema_1_2():
    tf = pytest.importorskip("tensorflow")
    specs = response_history_specs([_response_features()])
    stored = {
        key: tf.ones([3, *shape], dtype=dtype) for key, (shape, dtype) in specs.items()
    }

    assert response_history_fields({"proprio": tf.zeros([3, 10])}, 3, {}, tf) == {}
    kept = response_history_fields({**stored, "proprio": tf.zeros([3, 10])}, 3, specs, tf)
    assert all(kept[key] is stored[key] for key in specs)
    filled = response_history_fields({"proprio": tf.zeros([3, 10])}, 3, specs, tf)
    assert set(filled) == set(RESPONSE_HISTORY_KEYS)
    for key, (shape, dtype) in specs.items():
        assert tuple(filled[key].shape) == (3, *shape)
        assert filled[key].dtype.as_numpy_dtype == dtype
        assert not np.any(filled[key].numpy())
