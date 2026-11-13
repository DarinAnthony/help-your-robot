import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0_config import Pi0Config
from pi05_integration import input_config


def test_robotokens_are_masked_fully_visible_prefix_tokens():
    common = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_horizon": 4,
        "action_dim": 4,
        "max_token_len": 8,
        "augmentation_mode": "none",
    }
    vanilla_config = Pi0Config(**common)
    token_config = Pi0Config(
        **common,
        backbone_conditioning=input_config.RobotTokenBackboneSequence(
            bundle_dir="/not/read/by/model",
            bundle_sha256="1" * 64,
            token_width=64,
        ),
    )
    vanilla = vanilla_config.create(jax.random.key(0))
    conditioned = token_config.create(jax.random.key(0))
    observation = token_config.fake_obs(1).replace(
        robotoken_mask=jnp.zeros((1, 192), jnp.bool_).at[:, :3].set(True)
    )

    vanilla_tokens, vanilla_mask, vanilla_ar_mask = vanilla.embed_prefix(
        vanilla_config.fake_obs(1)
    )
    tokens, mask, ar_mask = conditioned.embed_prefix(observation)
    assert tokens.shape == (1, vanilla_tokens.shape[1] + 192, 64)
    np.testing.assert_array_equal(mask[:, :-192], vanilla_mask)
    np.testing.assert_array_equal(mask[0, -192:], observation.robotoken_mask[0])
    np.testing.assert_array_equal(ar_mask[:-192], vanilla_ar_mask)
    assert not np.asarray(ar_mask[-192:]).any()
    np.testing.assert_allclose(np.asarray(tokens[:, -189:]), 0.0, atol=1e-6)
