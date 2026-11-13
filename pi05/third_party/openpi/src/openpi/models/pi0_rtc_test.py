from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from pi05_integration.input_config import ExtractedEmbodimentAdaRMS
from pi05_integration.input_config import ExtractedEmbodimentBackboneToken
from pi05_integration.input_config import ExtractedEmbodimentConcatProjectAdaRMS
from pi05_integration.input_config import LearnedEmbodimentAdaRMS
from pi05_integration.input_config import LearnedEmbodimentConcatAdaRMS
from pi05_integration.input_config import Urma2EmbodimentExtractor
from pi05_integration.input_config import Urma3EmbodimentExtractor

from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils
from openpi.training import weight_loaders


def test_zero_weight_rtc_is_identical_to_vanilla_sampling():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=4,
        max_token_len=8,
        augmentation_mode="none",
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(1)
    noise = jax.random.normal(jax.random.key(1), (1, 4, 4))

    vanilla = model.sample_actions(jax.random.key(2), observation, noise=noise, num_steps=2)
    zero_weight = model.sample_actions(
        jax.random.key(2),
        observation,
        noise=noise,
        num_steps=2,
        action_prefix=jnp.zeros_like(noise),
        action_prefix_weights=jnp.zeros_like(noise),
    )
    np.testing.assert_allclose(vanilla, zero_weight, atol=1e-5)


def test_guided_sampling_is_finite():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=4,
        max_token_len=8,
        augmentation_mode="none",
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(1)
    noise = jax.random.normal(jax.random.key(1), (1, 4, 4))
    guided = model.sample_actions(
        jax.random.key(2),
        observation,
        noise=noise,
        num_steps=2,
        action_prefix=jnp.zeros_like(noise),
        action_prefix_weights=jnp.ones_like(noise),
        rtc_velocity_rescale=False,
    )
    assert np.isfinite(np.asarray(guided)).all()


def test_zero_initialized_embodiment_conditioning_preserves_pi05_function():
    common = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_horizon": 4,
        "action_dim": 4,
        "max_token_len": 8,
        "augmentation_mode": "none",
    }
    vanilla = Pi0Config(**common).create(jax.random.key(0))
    legacy_config = Pi0Config(**common, embodiment_vocab_size=4)
    typed_config = Pi0Config(
        **common,
        action_expert_conditioning=LearnedEmbodimentAdaRMS(vocab_size=4),
    )
    assert typed_config.embodiment_vocab_size == legacy_config.embodiment_vocab_size == 4
    legacy_conditioned = legacy_config.create(jax.random.key(0))
    conditioned = typed_config.create(jax.random.key(0))
    observation = Pi0Config(**common).fake_obs(1)
    conditioned_observation = observation.replace(embodiment_id=jnp.asarray([2], jnp.int32))
    noise = jax.random.normal(jax.random.key(1), (1, 4, 4))

    vanilla_actions = vanilla.sample_actions(jax.random.key(2), observation, noise=noise, num_steps=2)
    conditioned_actions = conditioned.sample_actions(
        jax.random.key(2), conditioned_observation, noise=noise, num_steps=2
    )
    legacy_conditioned_actions = legacy_conditioned.sample_actions(
        jax.random.key(2), conditioned_observation, noise=noise, num_steps=2
    )
    np.testing.assert_allclose(vanilla_actions, conditioned_actions, atol=1e-5)
    np.testing.assert_allclose(legacy_conditioned_actions, conditioned_actions, atol=1e-5)


def test_zero_initialized_urma_fusion_preserves_pi05_function():
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
    urma_config = Pi0Config(
        **common,
        action_expert_conditioning=ExtractedEmbodimentAdaRMS(
            extractor=Urma2EmbodimentExtractor(
                bundle_dir="unused-by-model-test",
                bundle_sha256="a" * 64,
                max_controls=4,
            )
        ),
    )
    vanilla = vanilla_config.create(jax.random.key(0))
    conditioned = urma_config.create(jax.random.key(0))
    noise = jax.random.normal(jax.random.key(1), (1, 4, 4))

    vanilla_actions = vanilla.sample_actions(
        jax.random.key(2), vanilla_config.fake_obs(1), noise=noise, num_steps=2
    )
    conditioned_actions = conditioned.sample_actions(
        jax.random.key(2), urma_config.fake_obs(1), noise=noise, num_steps=2
    )
    np.testing.assert_allclose(vanilla_actions, conditioned_actions, atol=1e-5)


def test_add_and_concat_urma_conditions_preserve_time_at_initialization():
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
    vanilla = vanilla_config.create(jax.random.key(0))
    noisy_actions = jax.random.normal(jax.random.key(1), (1, 4, 4))
    timestep = jnp.asarray([0.5], jnp.float32)
    *_, time_condition = vanilla.embed_suffix(
        vanilla_config.fake_obs(1), noisy_actions, timestep
    )

    for fusion in ("add", "concat"):
        config = Pi0Config(
            **common,
            action_expert_conditioning=ExtractedEmbodimentAdaRMS(
                extractor=Urma2EmbodimentExtractor(
                    bundle_dir="unused-by-model-test",
                    bundle_sha256="a" * 64,
                    max_controls=4,
                ),
                fusion=fusion,
            ),
        )
        model = config.create(jax.random.key(0))
        *_, condition = model.embed_suffix(
            config.fake_obs(1), noisy_actions, timestep
        )
        np.testing.assert_allclose(
            condition[:, : time_condition.shape[-1]], time_condition, atol=1e-6
        )
        if fusion == "add":
            assert condition.shape == time_condition.shape
        else:
            assert condition.shape[-1] == time_condition.shape[-1] + 256
            np.testing.assert_array_equal(condition[:, time_condition.shape[-1] :], 0)


def test_time_identity_urma_concat_projection_preserves_pi05_function_and_width():
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
    urma_config = Pi0Config(
        **common,
        action_expert_conditioning=ExtractedEmbodimentConcatProjectAdaRMS(
            extractor=Urma2EmbodimentExtractor(
                bundle_dir="unused-by-model-test",
                bundle_sha256="a" * 64,
                max_controls=4,
            ),
            projection_dim=64,
        ),
    )
    vanilla = vanilla_config.create(jax.random.key(0))
    conditioned = urma_config.create(jax.random.key(0))
    observation = urma_config.fake_obs(1)
    noisy_actions = jax.random.normal(jax.random.key(1), (1, 4, 4))
    timestep = jnp.asarray([0.5], jnp.float32)

    *_, condition = conditioned.embed_suffix(observation, noisy_actions, timestep)
    assert condition.shape == (1, 64)

    noise = jax.random.normal(jax.random.key(2), (1, 4, 4))
    vanilla_actions = vanilla.sample_actions(
        jax.random.key(3), vanilla_config.fake_obs(1), noise=noise, num_steps=2
    )
    conditioned_actions = conditioned.sample_actions(
        jax.random.key(3), observation, noise=noise, num_steps=2
    )
    np.testing.assert_allclose(vanilla_actions, conditioned_actions, atol=1e-5)


def test_urma3_layer_norm_still_preserves_time_at_concat_projection_init():
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
    urma3_config = Pi0Config(
        **common,
        action_expert_conditioning=ExtractedEmbodimentConcatProjectAdaRMS(
            extractor=Urma3EmbodimentExtractor(
                bundle_dir="unused-by-model-test",
                bundle_sha256="a" * 64,
                max_controls=4,
            ),
            projection_dim=64,
        ),
    )
    vanilla = vanilla_config.create(jax.random.key(0))
    conditioned = urma3_config.create(jax.random.key(0))
    noisy_actions = jax.random.normal(jax.random.key(1), (1, 4, 4))
    timestep = jnp.asarray([0.5], jnp.float32)

    *_, vanilla_condition = vanilla.embed_suffix(
        vanilla_config.fake_obs(1), noisy_actions, timestep
    )
    *_, conditioned_condition = conditioned.embed_suffix(
        urma3_config.fake_obs(1), noisy_actions, timestep
    )

    np.testing.assert_allclose(
        conditioned_condition, vanilla_condition, atol=1e-6
    )


def test_urma_backbone_token_is_one_fully_visible_prefix_token_not_adarms():
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
        backbone_conditioning=ExtractedEmbodimentBackboneToken(
            extractor=Urma2EmbodimentExtractor(
                bundle_dir="unused-by-model-test",
                bundle_sha256="a" * 64,
                max_controls=4,
            ),
            token_width=64,
        ),
    )
    vanilla = vanilla_config.create(jax.random.key(0))
    conditioned = token_config.create(jax.random.key(0))
    observation = token_config.fake_obs(1)

    vanilla_tokens, vanilla_mask, vanilla_ar_mask = vanilla.embed_prefix(
        vanilla_config.fake_obs(1)
    )
    tokens, mask, ar_mask = conditioned.embed_prefix(observation)
    assert tokens.shape == (1, vanilla_tokens.shape[1] + 1, 64)
    np.testing.assert_allclose(
        np.asarray(tokens[:, :-1], np.float32),
        np.asarray(vanilla_tokens, np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(tokens[:, -1], np.float32), 0.0, atol=1e-6)
    np.testing.assert_array_equal(mask[:, :-1], vanilla_mask)
    np.testing.assert_array_equal(ar_mask[:-1], vanilla_ar_mask)
    assert bool(mask[0, -1])
    assert not bool(ar_mask[-1])

    noisy_actions = jax.random.normal(jax.random.key(1), (1, 4, 4))
    timestep = jnp.asarray([0.5], jnp.float32)
    *_, vanilla_condition = vanilla.embed_suffix(
        vanilla_config.fake_obs(1), noisy_actions, timestep
    )
    *_, token_condition = conditioned.embed_suffix(
        observation, noisy_actions, timestep
    )
    assert token_condition.shape == (1, 64)
    np.testing.assert_allclose(token_condition, vanilla_condition, atol=1e-6)

    conditioned.embodiment_token_proj.kernel.value = 0.01 * jax.random.normal(
        jax.random.key(4),
        conditioned.embodiment_token_proj.kernel.value.shape,
    )
    changed = observation.replace(
        embodiment_description=observation.embodiment_description.at[0, 0, 0].set(2.0)
    )
    token0 = conditioned.embed_embodiment_backbone_token(observation)
    token1 = conditioned.embed_embodiment_backbone_token(changed)
    assert token0.shape == (1, 1, 64)
    assert not np.allclose(token0, token1)

    actions = conditioned.sample_actions(
        jax.random.key(2),
        changed,
        noise=jax.random.normal(jax.random.key(3), (1, 4, 4)),
        num_steps=2,
    )
    assert np.isfinite(np.asarray(actions)).all()


def test_urma_backbone_token_rejects_action_expert_double_conditioning():
    with np.testing.assert_raises_regex(
        ValueError,
        "timestep-only action-expert",
    ):
        Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=4,
            action_dim=4,
            max_token_len=8,
            augmentation_mode="none",
            action_expert_conditioning=LearnedEmbodimentAdaRMS(vocab_size=4),
            backbone_conditioning=ExtractedEmbodimentBackboneToken(
                extractor=Urma2EmbodimentExtractor(
                    bundle_dir="unused-by-model-test",
                    bundle_sha256="a" * 64,
                    max_controls=4,
                ),
                token_width=64,
            ),
        )


def test_urma_backbone_token_projector_bootstraps_encoder_gradients():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=4,
        max_token_len=8,
        augmentation_mode="none",
        backbone_conditioning=ExtractedEmbodimentBackboneToken(
            extractor=Urma2EmbodimentExtractor(
                bundle_dir="unused-by-model-test",
                bundle_sha256="a" * 64,
                max_controls=4,
            ),
            token_width=64,
        ),
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(2)
    probe = jax.random.normal(jax.random.key(1), (2, 1, 64))

    def token_probe_loss(candidate):
        return jnp.mean(
            candidate.embed_embodiment_backbone_token(observation) * probe
        )

    diff_state = nnx.DiffState(
        0,
        nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*embodiment_(encoder|token).*"),
        ),
    )
    _, grads = nnx.value_and_grad(token_probe_loss, argnums=diff_state)(model)

    def state_norm(state):
        return np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf)))
                for leaf in jax.tree.leaves(state)
            )
        )

    # A downstream signal reaches the zero-initialized projector immediately,
    # while the projector intentionally gates gradients to URMA itself.
    assert state_norm(
        grads.filter(nnx_utils.PathRegex(".*embodiment_token_proj.*"))
    ) > 0
    np.testing.assert_allclose(
        state_norm(grads.filter(nnx_utils.PathRegex(".*embodiment_encoder.*"))),
        0.0,
        atol=1e-8,
    )

    # After the zero-initialized token projector moves, gradients reach URMA itself.
    model.embodiment_token_proj.kernel.value = 1e-3 * jax.random.normal(
        jax.random.key(2),
        model.embodiment_token_proj.kernel.value.shape,
    )
    _, grads = nnx.value_and_grad(token_probe_loss, argnums=diff_state)(model)
    assert state_norm(
        grads.filter(nnx_utils.PathRegex(".*embodiment_encoder.*"))
    ) > 0


def test_learned_embodiment_rows_change_action_expert_condition():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=4,
        max_token_len=8,
        augmentation_mode="none",
        embodiment_vocab_size=4,
    )
    model = config.create(jax.random.key(0))
    model.embodiment_embed.embedding.value = model.embodiment_embed.embedding.value.at[1].set(0.25)
    observation = config.fake_obs(1)
    noisy_actions = jax.random.normal(jax.random.key(1), (1, 4, 4))
    timestep = jnp.asarray([0.5], jnp.float32)
    *_, condition0 = model.embed_suffix(
        observation.replace(embodiment_id=jnp.asarray([0], jnp.int32)),
        noisy_actions,
        timestep,
    )
    *_, condition1 = model.embed_suffix(
        observation.replace(embodiment_id=jnp.asarray([1], jnp.int32)),
        noisy_actions,
        timestep,
    )
    np.testing.assert_allclose(np.asarray(condition1 - condition0), 0.25, atol=1e-6)


def test_concatenated_embodiment_condition_preserves_time_half_and_separates_id_half():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=4,
        max_token_len=8,
        augmentation_mode="none",
        action_expert_conditioning=LearnedEmbodimentConcatAdaRMS(vocab_size=4),
    )
    model = config.create(jax.random.key(0))
    model.embodiment_embed.embedding.value = model.embodiment_embed.embedding.value.at[1].set(0.25)
    observation = config.fake_obs(1)
    noisy_actions = jax.random.normal(jax.random.key(1), (1, 4, 4))
    timestep = jnp.asarray([0.5], jnp.float32)
    *_, condition0 = model.embed_suffix(
        observation.replace(embodiment_id=jnp.asarray([0], jnp.int32)),
        noisy_actions,
        timestep,
    )
    *_, condition1 = model.embed_suffix(
        observation.replace(embodiment_id=jnp.asarray([1], jnp.int32)),
        noisy_actions,
        timestep,
    )
    width = condition0.shape[-1] // 2
    assert condition0.shape[-1] == 2 * model.action_in_proj.out_features
    np.testing.assert_allclose(condition1[..., :width], condition0[..., :width], atol=1e-6)
    np.testing.assert_allclose(condition1[..., width:] - condition0[..., width:], 0.25, atol=1e-6)


def test_concatenated_checkpoint_expansion_preserves_pretrained_pi05_function():
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
    concatenated_config = Pi0Config(
        **common,
        action_expert_conditioning=LearnedEmbodimentConcatAdaRMS(vocab_size=4),
    )
    vanilla = vanilla_config.create(jax.random.key(0))
    concatenated = concatenated_config.create(jax.random.key(0))
    _, vanilla_state = nnx.split(vanilla)
    _, concatenated_state = nnx.split(concatenated)
    loaded = weight_loaders._expand_adarms_conditioning_kernels(  # noqa: SLF001
        vanilla_state.to_pure_dict(),
        concatenated_state.to_pure_dict(),
        multiplier=2,
    )
    migrated = weight_loaders._merge_params(  # noqa: SLF001
        loaded,
        concatenated_state.to_pure_dict(),
        missing_regex=".*embodiment_embed.*",
    )
    migrated_model = concatenated_config.load(migrated)

    observation = vanilla_config.fake_obs(1)
    conditioned_observation = observation.replace(embodiment_id=jnp.asarray([2], jnp.int32))
    noise = jax.random.normal(jax.random.key(1), (1, 4, 4))
    vanilla_actions = vanilla.sample_actions(jax.random.key(2), observation, noise=noise, num_steps=2)
    migrated_actions = migrated_model.sample_actions(
        jax.random.key(2), conditioned_observation, noise=noise, num_steps=2
    )
    np.testing.assert_allclose(vanilla_actions, migrated_actions, atol=1e-5)
