import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from pi05_integration.input_config import Urma6EmbodimentExtractor
from openpi.models.embodiment import create_encoder
from openpi.models.embodiment import UrmaEmbodimentEncoder
from openpi.models.embodiment import flatten_urma4_backbone_token
from openpi.models.embodiment import UrmaGeometryAugmentedEncoder
from openpi.models.embodiment import UrmaPointNetGeometryEncoder


def test_flat_urma4_token_preserves_every_input_in_declared_order():
    description = jnp.arange(2 * 10 * 67, dtype=jnp.float32).reshape(2, 10, 67)
    availability = (description.astype(jnp.int32) % 3) == 0
    control_state = (10000 + jnp.arange(20, dtype=jnp.float32)).reshape(2, 10, 1)
    control_mask = jnp.asarray([[True] * 6 + [False] * 4] * 2)
    global_description = (
        20000 + jnp.arange(2 * 52, dtype=jnp.float32)
    ).reshape(2, 52)
    global_availability = (global_description.astype(jnp.int32) % 2) == 0

    token = flatten_urma4_backbone_token(
        description,
        availability,
        control_state,
        control_mask,
        global_description,
        global_availability,
        token_width=2048,
    )
    expected = np.concatenate(
        [
            np.asarray(description).reshape(2, -1),
            np.asarray(availability, np.float32).reshape(2, -1),
            np.asarray(control_state).reshape(2, -1),
            np.asarray(control_mask, np.float32).reshape(2, -1),
            np.asarray(global_description).reshape(2, -1),
            np.asarray(global_availability, np.float32).reshape(2, -1),
        ],
        axis=-1,
    )

    assert token.shape == (2, 2048)
    assert expected.shape == (2, 1464)
    np.testing.assert_array_equal(token[:, :1464], expected)
    np.testing.assert_array_equal(token[:, 1464:], np.zeros((2, 584)))


def _geometry_inputs(*, batch: int = 2, controls: int = 4):
    points = jax.random.normal(
        jax.random.key(10), (batch, controls, 256, 9), dtype=jnp.float32
    )
    # Part roles are categorical rather than arbitrary random values.
    roles = jax.nn.one_hot(
        jnp.arange(256, dtype=jnp.int32) % 3, 3, dtype=jnp.float32
    )
    points = points.at[..., 6:].set(roles)
    point_mask = jnp.ones((batch, controls, 256), dtype=jnp.bool_)
    control_mask = jnp.asarray(
        [[True] * (controls - 1) + [False]] * batch, dtype=jnp.bool_
    )
    return points, point_mask, control_mask


def test_urma6_pointnet_has_fixed_shape_and_is_permutation_invariant():
    encoder = UrmaPointNetGeometryEncoder(rngs=nnx.Rngs(jax.random.key(0)))
    points, point_mask, control_mask = _geometry_inputs()
    permutation = jax.random.permutation(jax.random.key(11), 256)

    baseline = encoder(points, point_mask, control_mask)
    permuted = encoder(
        points[..., permutation, :], point_mask[..., permutation], control_mask
    )

    assert baseline.shape == (2, 4, 128)
    np.testing.assert_allclose(baseline, permuted, atol=1e-6)


def test_urma6_pointnet_ignores_masked_points_and_zeros_padded_controls():
    encoder = UrmaPointNetGeometryEncoder(rngs=nnx.Rngs(jax.random.key(0)))
    points, point_mask, control_mask = _geometry_inputs()
    point_mask = point_mask.at[:, 0, 128:].set(False)

    baseline = encoder(points, point_mask, control_mask)
    changed = encoder(
        points.at[:, 0, 128:, :].set(1.0e6), point_mask, control_mask
    )

    np.testing.assert_allclose(baseline[:, 0], changed[:, 0], atol=1e-6)
    np.testing.assert_array_equal(baseline[:, -1], jnp.zeros((2, 128)))
    assert np.isfinite(np.asarray(baseline)).all()


def test_urma6_pointnet_preserves_metric_scale():
    encoder = UrmaPointNetGeometryEncoder(rngs=nnx.Rngs(jax.random.key(0)))
    points, point_mask, control_mask = _geometry_inputs(batch=1, controls=2)
    scaled = points.at[..., :3].multiply(2.0)

    baseline = encoder(points, point_mask, control_mask)
    doubled = encoder(scaled, point_mask, control_mask)

    assert not np.allclose(np.asarray(baseline[:, 0]), np.asarray(doubled[:, 0]))


def test_urma6_appends_geometry_without_removing_or_reordering_scalar_fields():
    encoder = UrmaGeometryAugmentedEncoder(
        89,
        128,
        8,
        256,
        3,
        output_normalization="layer_norm",
        global_description_dim=17,
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    points, point_mask, control_mask = _geometry_inputs()
    description = jax.random.normal(jax.random.key(12), (2, 4, 89))
    availability = jnp.ones_like(description, dtype=jnp.bool_)

    augmented, augmented_availability = encoder.augment_description(
        description,
        availability,
        control_mask,
        points,
        point_mask,
    )

    assert augmented.shape == (2, 4, 89 + 128)
    np.testing.assert_array_equal(augmented[..., :89], description)
    np.testing.assert_array_equal(augmented_availability[..., :89], availability)
    assert encoder.scalar_encoder.head0.routing_in.in_features == 2 * (89 + 128)
    assert encoder.scalar_encoder.head0.state_proj.in_features == 7


def test_urma6_extractor_selects_geometry_encoder_with_fixed_contract():
    extractor = Urma6EmbodimentExtractor(
        bundle_dir="unused", bundle_sha256="0" * 64
    )
    encoder = create_encoder(extractor, rngs=nnx.Rngs(jax.random.key(0)))

    assert isinstance(encoder, UrmaGeometryAugmentedEncoder)
    assert extractor.structured_input_shape == (10, 89, 7)
    assert extractor.geometry_input_shape == (10, 256, 9)
    assert encoder.geometry_dim == 128
    assert encoder.scalar_encoder.head0.state_proj.in_features == 7


def test_urma2_returns_one_global_vector_and_ignores_masked_controls():
    encoder = UrmaEmbodimentEncoder(
        35, 128, 8, 256, 3, rngs=nnx.Rngs(jax.random.key(0))
    )
    description = jnp.zeros((2, 4, 35), jnp.float32)
    availability = jnp.ones_like(description, dtype=jnp.bool_)
    state = jnp.zeros((2, 4, 1), jnp.float32)
    mask = jnp.asarray([[True, True, False, False]] * 2)

    baseline = encoder(description, availability, state, mask)
    changed_masked_state = state.at[:, 2:, :].set(100.0)
    changed = encoder(description, availability, changed_masked_state, mask)

    assert baseline.shape == (2, 256)
    assert np.isfinite(np.asarray(baseline)).all()
    np.testing.assert_allclose(baseline, changed, atol=1e-6)


def test_urma51_encoder_accepts_full_25_field_dynamic_state():
    encoder = UrmaEmbodimentEncoder(
        67,
        128,
        8,
        256,
        3,
        control_state_dim=25,
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    output = encoder(
        jnp.zeros((2, 10, 67), jnp.float32),
        jnp.ones((2, 10, 67), jnp.bool_),
        jnp.zeros((2, 10, 25), jnp.float32),
        jnp.asarray([[True] * 7 + [False] * 3] * 2),
    )
    assert output.shape == (2, 256)
    assert encoder.head0.state_proj.in_features == 25


def test_urma2_default_has_no_new_output_normalization_parameters():
    encoder = UrmaEmbodimentEncoder(
        35, 128, 8, 256, 3, rngs=nnx.Rngs(jax.random.key(0))
    )

    assert encoder.output_norm is None
    assert not any(
        "output_norm" in "/".join(str(part) for part in path)
        for path in nnx.state(encoder).flat_state()
    )


def test_urma3_output_layer_norm_sets_a_stable_fusion_scale():
    encoder = UrmaEmbodimentEncoder(
        67,
        128,
        8,
        256,
        3,
        output_normalization="layer_norm",
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    description = jax.random.normal(jax.random.key(1), (3, 10, 67))
    availability = jnp.ones_like(description, dtype=jnp.bool_)
    state = jax.random.normal(jax.random.key(2), (3, 10, 1))
    mask = jnp.asarray([[True] * 8 + [False] * 2] * 3)

    embedding = np.asarray(
        encoder(description, availability, state, mask), np.float32
    )

    np.testing.assert_allclose(np.mean(embedding, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(embedding), axis=-1)),
        1.0,
        atol=1e-4,
    )


def test_urma4_encodes_global_context_separately_before_global_mlp():
    encoder = UrmaEmbodimentEncoder(
        67,
        128,
        8,
        256,
        3,
        output_normalization="layer_norm",
        global_description_dim=52,
        global_embedding_dim=128,
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    description = jnp.zeros((2, 4, 67), jnp.float32)
    availability = jnp.ones_like(description, dtype=jnp.bool_)
    state = jnp.zeros((2, 4, 1), jnp.float32)
    mask = jnp.asarray([[True, True, False, False]] * 2)
    global_description = jnp.zeros((2, 52), jnp.float32)
    global_availability = jnp.ones_like(
        global_description, dtype=jnp.bool_
    )

    baseline = encoder(
        description,
        availability,
        state,
        mask,
        global_description,
        global_availability,
    )
    changed = encoder(
        description,
        availability,
        state,
        mask,
        global_description.at[:, 0].set(1.0),
        global_availability,
    )

    assert baseline.shape == (2, 256)
    assert not np.allclose(np.asarray(baseline), np.asarray(changed))
    assert encoder.global0.in_features == 3 * 128 * 8 + 128


def test_urma4_dynamic_has_a_distinct_masked_multifield_projection():
    encoder = UrmaEmbodimentEncoder(
        67,
        128,
        8,
        256,
        3,
        output_normalization="layer_norm",
        global_description_dim=52,
        global_embedding_dim=128,
        control_state_dim=16,
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    description = jnp.zeros((2, 10, 67), jnp.float32)
    availability = jnp.ones_like(description, dtype=jnp.bool_)
    state = jnp.zeros((2, 10, 16), jnp.float32)
    mask = jnp.asarray([[True] * 7 + [False] * 3] * 2)
    global_description = jnp.zeros((2, 52), jnp.float32)
    global_availability = jnp.ones_like(global_description, dtype=jnp.bool_)
    output = encoder(
        description,
        availability,
        state,
        mask,
        global_description,
        global_availability,
    )
    assert output.shape == (2, 256)
    assert encoder.head0.state_proj.in_features == 16
