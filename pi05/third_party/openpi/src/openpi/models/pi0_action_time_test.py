"""The opt-in action time channel (``ActionTokenTimeSeconds``): tau_k = (k + 1) * action_dt_s through a
zero-initialized ``action_time_proj`` added to the Pi0.5 action tokens."""

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from pi05_integration.input_config import ActionTokenTimeSeconds
import pytest

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config

COMMON = {
    "pi05": True,
    "paligemma_variant": "dummy",
    "action_expert_variant": "dummy",
    "action_horizon": 4,
    "action_dim": 4,
    "max_token_len": 8,
    "augmentation_mode": "none",
}


def _paths(state) -> set[str]:
    return set(flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/"))


def _models():
    vanilla_config, timed_config = Pi0Config(**COMMON), Pi0Config(**COMMON, action_time_conditioning=ActionTokenTimeSeconds())
    return vanilla_config, vanilla_config.create(jax.random.key(0)), timed_config, timed_config.create(jax.random.key(0))


def _timed_obs(config, dt):
    return config.fake_obs(len(dt)).replace(action_dt_s=jnp.asarray(dt, jnp.float32))


def test_zero_initialized_time_channel_preserves_the_function_and_every_other_parameter():
    vanilla_config, vanilla, timed_config, timed = _models()
    vanilla_params, timed_params = nnx.state(vanilla, nnx.Param), nnx.state(timed, nnx.Param)
    # The only new parameter (bias-free: nnx keeps a None placeholder), created last: every other initialization is
    # unchanged.
    assert _paths(timed_params) - _paths(vanilla_params) == {"action_time_proj/kernel", "action_time_proj/bias"}
    flat_vanilla = flax.traverse_util.flatten_dict(vanilla_params.to_pure_dict(), sep="/")
    flat_timed = flax.traverse_util.flatten_dict(timed_params.to_pure_dict(), sep="/")
    for path, value in flat_vanilla.items():
        np.testing.assert_array_equal(flat_timed[path], value, err_msg=path)
    assert flat_timed["action_time_proj/kernel"].shape == (1, timed.action_in_proj.out_features)
    assert not np.any(flat_timed["action_time_proj/kernel"]) and flat_timed["action_time_proj/bias"] is None

    noise = jax.random.normal(jax.random.key(1), (2, 4, 4))
    observation = vanilla_config.fake_obs(2)
    expected = vanilla.sample_actions(jax.random.key(2), observation, noise=noise, num_steps=2)
    for dt in ([0.02, 0.2], [0.066, 0.04]):
        actual = timed.sample_actions(jax.random.key(2), _timed_obs(timed_config, dt), noise=noise, num_steps=2)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    actions = jax.random.normal(jax.random.key(3), (2, 4, 4))
    np.testing.assert_allclose(
        timed.compute_loss(jax.random.key(4), _timed_obs(timed_config, [0.02, 0.2]), actions, train=False),
        vanilla.compute_loss(jax.random.key(4), observation, actions, train=False),
        atol=1e-6,
    )


def test_tau_is_the_chunk_entry_offset_in_seconds():
    vanilla_config, vanilla, timed_config, timed = _models()
    kernel = jax.random.normal(jax.random.key(5), timed.action_time_proj.kernel.value.shape)
    timed.action_time_proj.kernel.value = kernel
    noisy = jax.random.normal(jax.random.key(6), (2, 4, 4))
    timestep = jnp.asarray([0.5, 0.5], jnp.float32)
    dt = np.asarray([0.02, 0.2], np.float32)
    base, *_ = vanilla.embed_suffix(vanilla_config.fake_obs(2), noisy, timestep)
    tokens, *_ = timed.embed_suffix(_timed_obs(timed_config, dt), noisy, timestep)
    # Entry k's label is the target reached at t + (k + 1) dt.
    tau = (np.arange(1, 5, dtype=np.float32)[None, :] * dt[:, None])[..., None]
    np.testing.assert_allclose(tokens - base, tau * np.asarray(kernel)[None], rtol=1e-5, atol=1e-6)
    assert not np.allclose(tokens[0], tokens[1])


def test_time_channel_refuses_a_missing_period_and_pi0():
    _, _, timed_config, timed = _models()
    with pytest.raises(ValueError, match="action_dt_s"):
        timed.embed_suffix(timed_config.fake_obs(1).replace(action_dt_s=None), jnp.zeros((1, 4, 4)), jnp.ones(1))
    with pytest.raises(ValueError, match="only supported for Pi0.5"):
        Pi0Config(**{**COMMON, "pi05": False}, action_time_conditioning=ActionTokenTimeSeconds())
    # The spec (and so fake_obs and the training batch layout) carries the period only when enabled.
    assert Pi0Config(**COMMON).inputs_spec()[0].action_dt_s is None
    assert timed_config.inputs_spec(batch_size=3)[0].action_dt_s.shape == (3,)


def test_time_projection_stays_trainable_under_the_lora_freeze_filter():
    _, vanilla, _, timed = _models()
    lora = Pi0Config(**{**COMMON, "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora"})
    freeze = lora.get_freeze_filter()
    trainable = _paths(nnx.state(timed, nnx.All(nnx.Param, nnx.Not(freeze))))
    assert "action_time_proj/kernel" in trainable
    assert not any(path.startswith("action_time_proj") for path in _paths(nnx.state(timed, freeze)))
    # The filter freezes the backbone as before; disabled models have no such path.
    assert any("llm" in path for path in _paths(nnx.state(timed, freeze)))
    assert vanilla.action_time_proj is None
    assert not any(path.startswith("action_time_proj") for path in _paths(nnx.state(vanilla)))


def test_a_saved_time_conditioned_checkpoint_loads_for_serving(tmp_path):
    vanilla_config, vanilla, timed_config, timed = _models()
    kernel = jax.random.normal(jax.random.key(7), timed.action_time_proj.kernel.value.shape)
    timed.action_time_proj.kernel.value = kernel

    def save_and_restore(model, name):
        # As save_state writes the inference params and create_trained_policy reads them back.
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(tmp_path / name, {"params": nnx.state(model)})
        return _model.restore_params(tmp_path / name, dtype=jnp.bfloat16)

    restored = save_and_restore(timed, "timed")
    assert restored["action_time_proj"]["bias"] is None
    loaded = timed_config.load(restored)
    np.testing.assert_array_equal(
        np.asarray(loaded.action_time_proj.kernel.value), np.asarray(kernel.astype(jnp.bfloat16))
    )
    assert loaded.action_time_proj.bias.value is None
    actions = loaded.sample_actions(jax.random.key(8), _timed_obs(timed_config, [0.02, 0.2]), num_steps=2)
    assert actions.shape == (2, 4, 4) and np.all(np.isfinite(np.asarray(actions, np.float32)))
    # A checkpoint without the channel still fails closed for a time-conditioned config; vanilla loads as before.
    vanilla_restored = save_and_restore(vanilla, "vanilla")
    with pytest.raises(ValueError, match="action_time_proj"):
        timed_config.load(vanilla_restored)
    assert vanilla_config.load(vanilla_restored).action_time_proj is None


def _gp_model(length_scale_s=0.1):
    from pi05_integration.input_config import ActionTokenTimeSecondsGPNoise

    config = Pi0Config(**{**COMMON, "action_horizon": 8},
                       action_time_conditioning=ActionTokenTimeSecondsGPNoise(length_scale_s=length_scale_s))
    return config, config.create(jax.random.key(0))


def test_gp_flow_source_has_the_kernel_covariance_at_every_rate():
    config, model = _gp_model()
    white = jax.random.normal(jax.random.key(1), (4096, 8, 4))
    for dt in (0.02, 0.2):
        noise = model._flow_source_noise(white, _timed_obs(config, [dt] * 4096))
        samples = np.asarray(noise).transpose(0, 2, 1).reshape(-1, 8)  # one function per (sample, action dim)
        tau = np.arange(1, 9) * dt
        kernel = (np.exp(-0.5 * ((tau[:, None] - tau[None, :]) / 0.1) ** 2) + 1e-3 * np.eye(8)) / (1 + 1e-3)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), kernel, atol=0.05)
    # 50 Hz neighbours are nearly identical; 5 Hz neighbours are nearly independent.
    fine, coarse = (np.asarray(model._flow_source_noise(white, _timed_obs(config, [dt] * 4096))) for dt in (0.02, 0.2))
    assert np.corrcoef(fine[:, 0, 0], fine[:, 1, 0])[0, 1] > 0.95
    assert abs(np.corrcoef(coarse[:, 0, 0], coarse[:, 1, 0])[0, 1]) < 0.2


def test_white_source_is_unchanged_and_gp_source_needs_the_period():
    vanilla_config, vanilla, timed_config, timed = _models()
    white = jax.random.normal(jax.random.key(2), (2, 4, 4))
    np.testing.assert_array_equal(vanilla._flow_source_noise(white, vanilla_config.fake_obs(2)), white)
    np.testing.assert_array_equal(timed._flow_source_noise(white, _timed_obs(timed_config, [0.02, 0.2])), white)
    config, model = _gp_model()
    with pytest.raises(ValueError, match="action_dt_s"):
        model._flow_source_noise(jax.random.normal(jax.random.key(2), (2, 8, 4)),
                                 config.fake_obs(2).replace(action_dt_s=None))


def test_gp_sampling_colors_supplied_noise_and_trains():
    config, model = _gp_model()
    obs = _timed_obs(config, [0.02, 0.2])
    white = jax.random.normal(jax.random.key(3), (2, 8, 4))
    a = model.sample_actions(jax.random.key(4), obs, num_steps=2, noise=white)
    b = model.sample_actions(jax.random.key(5), obs, num_steps=2, noise=white)
    np.testing.assert_array_equal(a, b)  # seeded noise stays deterministic after coloring
    loss = model.compute_loss(jax.random.key(6), obs, jnp.zeros((2, 8, 4)), train=True)
    assert loss.shape == (2, 8) and bool(jnp.all(jnp.isfinite(loss)))


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_flow_source_has_the_kernel_covariance_at_every_rate(nu):
    from pi05_integration.input_config import ActionTokenTimeSecondsMaternNoise

    config = Pi0Config(**{**COMMON, "action_horizon": 8},
                       action_time_conditioning=ActionTokenTimeSecondsMaternNoise(length_scale_s=0.1, nu=nu))
    model = config.create(jax.random.key(0))
    white = jax.random.normal(jax.random.key(1), (4096, 8, 4))
    for dt in (0.02, 0.2):
        samples = np.asarray(model._flow_source_noise(white, _timed_obs(config, [dt] * 4096)))
        samples = samples.transpose(0, 2, 1).reshape(-1, 8)
        tau = np.arange(1, 9) * dt
        s = np.sqrt(2 * nu) * np.abs(tau[:, None] - tau[None, :]) / 0.1
        kernel = ({0.5: 1, 1.5: 1 + s, 2.5: 1 + s + s**2 / 3}[nu] * np.exp(-s) + 1e-4 * np.eye(8)) / (1 + 1e-4)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), kernel, atol=0.05)
    loss = model.compute_loss(jax.random.key(6), _timed_obs(config, [0.02, 0.2]), jnp.zeros((2, 8, 4)), train=True)
    assert bool(jnp.all(jnp.isfinite(loss)))
