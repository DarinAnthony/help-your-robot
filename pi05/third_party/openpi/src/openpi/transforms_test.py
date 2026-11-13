import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_clip_actions_only_clips_action_field():
    item = {
        "state": np.array([-20.0, 20.0]),
        "actions": np.array([[-12.0, 2.0], [7.0, -3.0]]),
    }

    transformed = _transforms.ClipActions(5.0)(item)

    assert np.array_equal(transformed["state"], item["state"])
    assert np.array_equal(
        transformed["actions"],
        np.array([[-5.0, 2.0], [5.0, -3.0]]),
    )


def test_clip_actions_is_noop_when_actions_are_absent():
    item = {"state": np.array([-20.0, 20.0])}
    assert _transforms.ClipActions(5.0)(item) is item


def test_clip_actions_rejects_invalid_limit():
    with pytest.raises(ValueError, match="finite and positive"):
        _transforms.ClipActions(np.inf)


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_tokenize_prompt_forwards_fail_closed_overflow_contract():
    class StrictTokenizer:
        def tokenize(self, prompt, state, *, fail_on_overflow):
            assert prompt == "too long"
            assert state is None
            assert fail_on_overflow is True
            raise ValueError("refusing to truncate")

    transform = _transforms.TokenizePrompt(
        StrictTokenizer(), fail_on_overflow=True
    )
    with pytest.raises(ValueError, match="refusing to truncate"):
        transform({"prompt": "too long"})


def test_compact_left_packed_state_normalizes_before_tokenization():
    data = {
        "state": np.array([10.0, 21.0, 32.0, 43.0], np.float32),
        "state_mask": np.array([True, True, False, False]),
    }
    normalized = _transforms.Normalize(
        {
            "state": _transforms.NormStats(
                mean=np.array([0.0, 1.0, 2.0, 3.0], np.float32),
                std=np.full(4, 10.0, np.float32),
            )
        }
    )(data)

    compact = _transforms.CompactLeftPackedState()(normalized)

    np.testing.assert_allclose(compact["state"], [1.0, 2.0], atol=1e-6)
    with pytest.raises(ValueError, match="left-packed"):
        _transforms.CompactLeftPackedState()({
            "state": np.ones(3),
            "state_mask": np.array([True, False, True]),
        })


def test_invalid_clean_actions_are_zeroed_after_normalization():
    normalized = _transforms.Normalize(
        {
            "actions": _transforms.NormStats(
                mean=np.array([1.0, 5.0, 3.0], np.float32),
                std=np.full(3, 10.0, np.float32),
            )
        }
    )({
        "actions": np.array([[11.0, 25.0, 33.0]], np.float32),
        "action_semantic_mask": np.array([True, True, False]),
    })

    zeroed = _transforms.ZeroInvalidActions()(normalized)

    np.testing.assert_allclose(zeroed["actions"], [[1.0, 2.0, 0.0]], atol=1e-6)


def test_temporal_action_loss_mask_broadcasts_over_full_model_dimension():
    transformed = _transforms.PadStatesAndActions(6)({
        "state": np.ones(3),
        "actions": np.ones((2, 3)),
        "action_loss_mask": np.array([[True], [False]]),
    })

    np.testing.assert_array_equal(
        transformed["action_loss_mask"],
        np.array([[True] * 6, [False] * 6]),
    )


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
