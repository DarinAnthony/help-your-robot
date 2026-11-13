import jax.numpy as jnp

from flax import nnx
from openpi.models import embodiment
from pi05_integration import input_config


def _config():
    return input_config.RobotTokenBackboneSequence(
        bundle_dir="/not/read/by/model",
        bundle_sha256="1" * 64,
    )


def test_sequence_encoder_preserves_token_axis_and_masks_padding():
    encoder = embodiment.RobotTokenSequenceEncoder(_config(), rngs=nnx.Rngs(0))
    values = jnp.ones((2, 192, 32), jnp.float32)
    references = jnp.full((2, 192, 5), -1, jnp.int32)
    mask = jnp.zeros((2, 192), jnp.bool_).at[:, :3].set(True)
    output = encoder(values, references, mask)
    assert output.shape == (2, 192, 2048)
    assert jnp.all(output[:, 3:] == 0)


def test_token_types_use_distinct_input_projections():
    encoder = embodiment.RobotTokenSequenceEncoder(_config(), rngs=nnx.Rngs(2))
    values = jnp.zeros((1, 192, 32), jnp.float32)
    values = values.at[:, 0].set(1).at[:, 32].set(1)
    references = jnp.full((1, 192, 5), -1, jnp.int32)
    mask = jnp.zeros((1, 192), jnp.bool_).at[:, 0].set(True).at[:, 32].set(True)

    output = encoder(values, references, mask)

    # Identical feature vectors in the link and dynamic-joint blocks must pass
    # through their separate T2-style modality projections.
    assert not jnp.allclose(output[:, 0], output[:, 32])


def test_reference_ids_change_tokens_through_shared_namespace():
    encoder = embodiment.RobotTokenSequenceEncoder(_config(), rngs=nnx.Rngs(1))
    values = jnp.zeros((1, 192, 32), jnp.float32)
    references = jnp.full((1, 192, 5), -1, jnp.int32)
    references = references.at[0, 0, 0].set(4).at[0, 32, 0].set(4)
    mask = jnp.zeros((1, 192), jnp.bool_).at[:, 0].set(True).at[:, 32].set(True)
    output = encoder(values, references, mask)
    # The same link-ID embedding is added to two different modality projections.
    without_reference = encoder(
        values, jnp.full_like(references, -1), mask
    )
    delta = output - without_reference
    assert jnp.allclose(delta[:, 0], delta[:, 32])
