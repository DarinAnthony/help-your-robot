import numpy as np
import pytest

from openpi.training import weight_loaders


def test_expand_adarms_conditioning_kernels_copies_the_pretrained_input_branch():
    loaded = {
        "PaliGemma": {
            "llm": {
                "final_norm_1": {
                    "Dense_0": {
                        "kernel": np.arange(12, dtype=np.float32).reshape(2, 6),
                        "bias": np.arange(6, dtype=np.float32),
                    }
                },
                "layers": {
                    "pre_attention_norm_1": {
                        "Dense_0": {
                            "kernel": np.arange(36, dtype=np.float32).reshape(3, 2, 6),
                        }
                    },
                    "pre_ffw_norm_1": {
                        "Dense_0": {
                            "kernel": np.arange(36, dtype=np.float32).reshape(3, 2, 6),
                        }
                    },
                },
            }
        }
    }
    reference = {
        "PaliGemma": {
            "llm": {
                "final_norm_1": {
                    "Dense_0": {
                        "kernel": np.zeros((4, 6), dtype=np.float32),
                        "bias": np.zeros(6, dtype=np.float32),
                    }
                },
                "layers": {
                    "pre_attention_norm_1": {
                        "Dense_0": {
                            "kernel": np.zeros((3, 4, 6), dtype=np.float32),
                        }
                    },
                    "pre_ffw_norm_1": {
                        "Dense_0": {
                            "kernel": np.zeros((3, 4, 6), dtype=np.float32),
                        }
                    },
                },
            }
        }
    }

    expanded = weight_loaders._expand_adarms_conditioning_kernels(  # noqa: SLF001
        loaded,
        reference,
        multiplier=2,
    )

    for path in (
        ("PaliGemma", "llm", "final_norm_1", "Dense_0", "kernel"),
        ("PaliGemma", "llm", "layers", "pre_attention_norm_1", "Dense_0", "kernel"),
        ("PaliGemma", "llm", "layers", "pre_ffw_norm_1", "Dense_0", "kernel"),
    ):
        source = loaded
        result = expanded
        for key in path:
            source = source[key]
            result = result[key]
        np.testing.assert_array_equal(result, np.concatenate([source, source], axis=-2))

    np.testing.assert_array_equal(
        expanded["PaliGemma"]["llm"]["final_norm_1"]["Dense_0"]["bias"],
        loaded["PaliGemma"]["llm"]["final_norm_1"]["Dense_0"]["bias"],
    )


def _params(*paths: str) -> dict:
    out: dict = {}
    for path in paths:
        node = out
        *parents, leaf = path.split("/")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = np.zeros(2, np.float32)
    return out


_MODEL = (
    "PaliGemma/img/embedding/kernel",
    "PaliGemma/llm/layers/attn/q_einsum/w",
    "PaliGemma/llm/layers/attn/q_einsum_1/w",  # action expert: stays random
    "PaliGemma/llm/final_norm_1/Dense_0/kernel",  # action expert: stays random
    "action_in_proj/kernel",  # outside PaliGemma: stays random
)


def test_paligemma_coverage_accepts_full_vlm_load():
    loaded = _params("PaliGemma/img/embedding/kernel", "PaliGemma/llm/layers/attn/q_einsum/w")
    weight_loaders._check_paligemma_coverage(loaded, _params(*_MODEL))  # noqa: SLF001


def test_paligemma_coverage_rejects_unmatched_checkpoint_tensor():
    loaded = _params("PaliGemma/img/embedding/kernel", "PaliGemma/llm/layers/attn/q_proj/w")
    with pytest.raises(ValueError, match="match no model parameter"):
        weight_loaders._check_paligemma_coverage(loaded, _params(*_MODEL))  # noqa: SLF001


def test_paligemma_coverage_rejects_vlm_parameter_left_random():
    loaded = _params("PaliGemma/img/embedding/kernel")
    with pytest.raises(ValueError, match="would stay randomly initialized"):
        weight_loaders._check_paligemma_coverage(loaded, _params(*_MODEL))  # noqa: SLF001


def test_paligemma_coverage_rejects_siglip_underscore_one_module_left_random():
    # SigLIP's own *_1 modules (LayerNorm_1, MlpBlock_0/Dense_1: ~134M of its ~415M params) are
    # not the action expert and must not slip through the action-expert exemption.
    model = (*_MODEL, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")
    loaded = _params("PaliGemma/img/embedding/kernel", "PaliGemma/llm/layers/attn/q_einsum/w")
    with pytest.raises(ValueError, match="randomly initialized"):
        weight_loaders._check_paligemma_coverage(loaded, _params(*model))  # noqa: SLF001


def test_unspecified_weight_loader_refuses_to_train():
    with pytest.raises(ValueError, match="names no weight_loader"):
        weight_loaders.UnspecifiedWeightLoader().load(_params(*_MODEL))


def test_time_channel_missing_regex_fills_the_projection_absent_from_pi05_base():
    loaded = {"action_in_proj": {"kernel": np.ones((2, 3), np.float32)}}
    reference = {
        "action_in_proj": {"kernel": np.zeros((2, 3), np.float32)},
        "action_time_proj": {"kernel": np.zeros((1, 3), np.float32)},
    }
    merged = weight_loaders._merge_params(  # noqa: SLF001
        loaded, reference, missing_regex=r"(?:.*lora.*|action_time_proj/.*)"
    )
    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"], 1.0)
    np.testing.assert_array_equal(merged["action_time_proj"]["kernel"], np.zeros((1, 3), np.float32))
    # The twin's LoRA-only regex leaves it out, so the trainer's pytree check refuses the load.
    assert "action_time_proj" not in weight_loaders._merge_params(loaded, reference, missing_regex=r".*lora.*")  # noqa: SLF001
