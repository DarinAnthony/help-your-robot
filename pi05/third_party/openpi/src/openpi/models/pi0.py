import logging
import math

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.embodiment as _embodiment
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.training.sharding as sharding
from pi05_integration import input_config as _input_config

logger = logging.getLogger("openpi")

_URMA51_DYNAMIC_NOISE_RNG_TAG = 0x55524D35


def add_urma51_standardized_dynamic_noise(
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    standard_deviation: float,
) -> _model.Observation:
    """Perturb only deployable, normalized URMA5.1 dynamic values.

    The seven base fields are ``q, qvel, previous target, previous action, anchor xyz``.
    Current q, masks, padded controls, static descriptors, and nine applicability bits
    remain exact.  The nine analytical fields are perturbed only where their paired
    applicability bit is true.  Independent noise deliberately breaks correlations
    among the measured and analytical fields.  Inputs have already undergone their
    clean standardization and [-5, 5] clipping; noise is not clipped again.
    """
    if standard_deviation == 0:
        return observation
    state = observation.embodiment_control_state
    control_mask = observation.embodiment_control_mask
    if state is None or control_mask is None:
        raise ValueError(
            "URMA5.1 dynamic-conditioning noise requires control state and mask"
        )
    if state.shape[-2:] != (10, 25) or control_mask.shape[-1:] != (10,):
        raise ValueError(
            "URMA5.1 dynamic-conditioning noise requires state [...,10,25] "
            "and mask [...,10]"
        )
    active = jnp.asarray(control_mask, dtype=jnp.bool_)[..., :, None]
    selected = jnp.zeros_like(state, dtype=jnp.bool_)
    selected = selected.at[..., 1:7].set(active)
    selected = selected.at[..., 7:16].set(
        active & (state[..., 16:25] > 0.5)
    )
    noise = jax.random.normal(rng, state.shape, dtype=state.dtype)
    noisy_state = jnp.where(
        selected,
        state + jnp.asarray(standard_deviation, dtype=state.dtype) * noise,
        state,
    )
    return observation.replace(embodiment_control_state=noisy_state)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _time_preserving_concat_projection(key, shape, dtype=jnp.float32):
    """Initialize ``[time; embodiment] -> condition`` as the identity on time."""
    del key
    if len(shape) != 2 or shape[0] != 2 * shape[1]:
        raise ValueError(f"expected a [2D, D] concat-projection kernel, got {shape}")
    width = shape[1]
    return jnp.concatenate(
        [jnp.eye(width, dtype=dtype), jnp.zeros((width, width), dtype=dtype)],
        axis=0,
    )


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.augmentation_mode = config.augmentation_mode
        self.urma51_dynamic_conditioning_noise_std = (
            config.urma51_dynamic_conditioning_noise_std
        )
        self.enable_langact_training = config.enable_langact_training
        self.language_loss_weight = config.language_loss_weight
        self.action_loss_weight = config.action_loss_weight
        conditioning = config.action_expert_conditioning
        self.concatenate_embodiment_embedding = (
            conditioning.concatenate_embodiment_embedding
        )
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                stop_action_to_vlm_grad=config.stop_action_to_vlm_grad,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
            adarms_condition_widths=[
                paligemma_config.width,
                conditioning.condition_width(action_expert_config.width),
            ],
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        # Initialize this after every upstream parameter so enabling the opt-in
        # adapter does not perturb any existing random initialization stream.
        self.embodiment_embed = (
            nnx.Embed(
                config.embodiment_vocab_size,
                action_expert_config.width,
                embedding_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
            if config.embodiment_vocab_size
            else None
        )
        self.embodiment_fusion_mode = (
            conditioning.fusion
            if isinstance(conditioning, _input_config.ExtractedEmbodimentAdaRMS)
            else None
        )
        backbone_conditioning = config.backbone_conditioning
        response_conditioning = config.response_history_conditioning
        self.concat_project_embodiment = isinstance(
            conditioning,
            _input_config.ExtractedEmbodimentConcatProjectAdaRMS,
        )
        self.action_expert_uses_embodiment_description = isinstance(
            conditioning,
            _input_config.ExtractedEmbodimentAdaRMS,
        )
        self.backbone_token_embodiment = isinstance(
            backbone_conditioning,
            (
                _input_config.LearnedEmbodimentBackboneToken,
                _input_config.ExtractedEmbodimentBackboneToken,
                _input_config.FlatUrma4BackboneToken,
                _input_config.RobotTokenBackboneSequence,
            ),
        )
        self.backbone_token_learned_id = isinstance(
            backbone_conditioning,
            _input_config.LearnedEmbodimentBackboneToken,
        )
        self.backbone_token_robotokens = isinstance(
            backbone_conditioning, _input_config.RobotTokenBackboneSequence
        )
        self.backbone_token_flat_urma4 = isinstance(
            backbone_conditioning, _input_config.FlatUrma4BackboneToken
        )
        self.flat_urma4_token_width = (
            backbone_conditioning.token_width
            if self.backbone_token_flat_urma4
            else None
        )
        if (
            (self.backbone_token_robotokens or self.backbone_token_flat_urma4)
            and backbone_conditioning.token_width != paligemma_config.width
        ):
            raise ValueError(
                "embodiment token width must match the PaliGemma backbone width: "
                f"{backbone_conditioning.token_width} != {paligemma_config.width}"
            )
        self.robotoken_encoder = (
            _embodiment.RobotTokenSequenceEncoder(backbone_conditioning, rngs=rngs)
            if self.backbone_token_robotokens
            else None
        )
        if self.backbone_token_learned_id:
            if backbone_conditioning.token_width != paligemma_config.width:
                raise ValueError(
                    "learned embodiment backbone token width must match the PaliGemma "
                    f"width: {backbone_conditioning.token_width} != {paligemma_config.width}"
                )
            self.embodiment_backbone_embed = nnx.Embed(
                backbone_conditioning.vocab_size,
                backbone_conditioning.token_width,
                embedding_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
        else:
            self.embodiment_backbone_embed = None
        description_conditioning = (
            conditioning
            if self.action_expert_uses_embodiment_description
            else (
                backbone_conditioning
                if isinstance(
                    backbone_conditioning,
                    _input_config.ExtractedEmbodimentBackboneToken,
                )
                else None
            )
        )
        if self.backbone_token_flat_urma4:
            extractor = backbone_conditioning.extractor
            self.embodiment_encoder = None
            self.embodiment_encoder_uses_global_context = True
            self.embodiment_encoder_uses_geometry = False
            self.embodiment_fusion_in = None
            self.embodiment_fusion_out = None
            self.embodiment_fusion_gate = None
            self.embodiment_token_norm = None
            self.embodiment_token_proj = None
        elif description_conditioning is not None:
            extractor = description_conditioning.extractor
            self.embodiment_encoder = _embodiment.create_encoder(extractor, rngs=rngs)
            self.embodiment_encoder_uses_global_context = (
                extractor.global_description_shape is not None
            )
            self.embodiment_encoder_uses_geometry = (
                extractor.geometry_input_shape is not None
            )
            if self.concat_project_embodiment:
                if conditioning.projection_dim != action_expert_config.width:
                    raise ValueError(
                        "URMA concat-project width must match the action-expert width: "
                        f"{conditioning.projection_dim} != {action_expert_config.width}"
                    )
                self.embodiment_fusion_in = nnx.Linear(
                    extractor.output_dim,
                    conditioning.projection_dim,
                    rngs=rngs,
                )
                self.embodiment_fusion_out = nnx.Linear(
                    2 * conditioning.projection_dim,
                    action_expert_config.width,
                    kernel_init=_time_preserving_concat_projection,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
                self.embodiment_fusion_gate = None
                self.embodiment_token_norm = None
                self.embodiment_token_proj = None
            elif (
                self.action_expert_uses_embodiment_description
                and conditioning.fusion == "residual_mlp"
            ):
                self.embodiment_fusion_in = nnx.Linear(
                    action_expert_config.width + extractor.output_dim,
                    action_expert_config.width,
                    rngs=rngs,
                )
                self.embodiment_fusion_out = nnx.Linear(
                    action_expert_config.width,
                    action_expert_config.width,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
                self.embodiment_fusion_gate = None
                self.embodiment_token_norm = None
                self.embodiment_token_proj = None
            elif (
                self.action_expert_uses_embodiment_description
                and conditioning.fusion == "add"
            ):
                self.embodiment_fusion_in = None
                self.embodiment_fusion_out = nnx.Linear(
                    extractor.output_dim,
                    action_expert_config.width,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
                self.embodiment_fusion_gate = None
                self.embodiment_token_norm = None
                self.embodiment_token_proj = None
            elif self.action_expert_uses_embodiment_description:
                self.embodiment_fusion_in = None
                self.embodiment_fusion_out = None
                self.embodiment_fusion_gate = nnx.Param(
                    jnp.zeros((extractor.output_dim,), dtype=jnp.float32)
                )
                self.embodiment_token_norm = None
                self.embodiment_token_proj = None
            else:
                self.embodiment_token_norm = None
                self.embodiment_token_proj = None
                if backbone_conditioning.token_width != paligemma_config.width:
                    raise ValueError(
                        "URMA backbone token width must match the PaliGemma width: "
                        f"{backbone_conditioning.token_width} != {paligemma_config.width}"
                    )
                self.embodiment_fusion_in = None
                self.embodiment_fusion_out = None
                self.embodiment_token_norm = nnx.LayerNorm(
                    extractor.output_dim,
                    epsilon=1e-6,
                    rngs=rngs,
                )
                self.embodiment_token_proj = nnx.Linear(
                    extractor.output_dim,
                    backbone_conditioning.token_width,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
                self.embodiment_fusion_gate = None
                self.embodiment_fusion_in = None
                self.embodiment_fusion_out = None
        else:
            self.embodiment_encoder = None
            self.embodiment_encoder_uses_global_context = False
            self.embodiment_encoder_uses_geometry = False
            self.embodiment_fusion_in = None
            self.embodiment_fusion_out = None
            self.embodiment_fusion_gate = None
            self.embodiment_token_norm = None
            self.embodiment_token_proj = None

        self.response_history_enabled = response_conditioning.enabled
        if self.response_history_enabled:
            if response_conditioning.token_width != paligemma_config.width:
                raise ValueError(
                    "response-history token width must match the PaliGemma width: "
                    f"{response_conditioning.token_width} != {paligemma_config.width}"
                )
            self.response_history_norm = nnx.LayerNorm(
                response_conditioning.flattened_width,
                epsilon=1e-6,
                rngs=rngs,
            )
            self.response_history_proj = nnx.Linear(
                response_conditioning.flattened_width,
                response_conditioning.token_width,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
        else:
            self.response_history_norm = None
            self.response_history_proj = None

        # The last parameter created, so enabling it leaves every other initialization unchanged. Zero kernel: the
        # released function is exact at step 0 and the channel still gets gradient (tau is nonzero).
        self.action_time_proj = (
            nnx.Linear(
                1,
                action_expert_config.width,
                use_bias=False,
                kernel_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
            if config.action_time_conditioning.enabled
            else None
        )
        # Function-space flow source (ActionTokenTimeSecondsGPNoise / ...MaternNoise): GP over each entry's tau;
        # None = white noise. ``nu`` None = squared exponential, else the Matern smoothness.
        self.action_noise_length_scale_s = getattr(config.action_time_conditioning, "length_scale_s", None)
        self.action_noise_jitter = getattr(config.action_time_conditioning, "jitter", 0.0)
        self.action_noise_matern_nu = getattr(config.action_time_conditioning, "nu", None)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def embed_embodiment_backbone_token(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b t emb"]:
        if not self.backbone_token_embodiment:
            raise ValueError("backbone-token conditioning is not enabled")
        if self.backbone_token_learned_id:
            if obs.embodiment_id is None:
                raise ValueError(
                    "embodiment_id is required when learned backbone-token conditioning "
                    "is enabled"
                )
            return self.embodiment_backbone_embed(
                obs.embodiment_id.astype(jnp.int32)
            )[:, None, :]
        if self.backbone_token_robotokens:
            required = {
                "robotoken_values": obs.robotoken_values,
                "robotoken_types": obs.robotoken_types,
                "robotoken_reference_ids": obs.robotoken_reference_ids,
                "robotoken_mask": obs.robotoken_mask,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"RoboToken backbone conditioning requires: {missing}")
            return self.robotoken_encoder(
                obs.robotoken_values,
                obs.robotoken_reference_ids.astype(jnp.int32),
                obs.robotoken_mask,
            )
        if self.backbone_token_flat_urma4:
            required = {
                "embodiment_description": obs.embodiment_description,
                "embodiment_description_availability": (
                    obs.embodiment_description_availability
                ),
                "embodiment_control_mask": obs.embodiment_control_mask,
                "embodiment_control_state": obs.embodiment_control_state,
                "embodiment_global_description": obs.embodiment_global_description,
                "embodiment_global_description_availability": (
                    obs.embodiment_global_description_availability
                ),
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"flat URMA4 backbone token requires observation fields: {missing}"
                )
            token = _embodiment.flatten_urma4_backbone_token(
                obs.embodiment_description,
                obs.embodiment_description_availability,
                obs.embodiment_control_state,
                obs.embodiment_control_mask,
                obs.embodiment_global_description,
                obs.embodiment_global_description_availability,
                token_width=int(self.flat_urma4_token_width),
            )
            return token[:, None, :]
        required = {
            "embodiment_description": obs.embodiment_description,
            "embodiment_description_availability": obs.embodiment_description_availability,
            "embodiment_control_mask": obs.embodiment_control_mask,
            "embodiment_control_state": obs.embodiment_control_state,
        }
        if self.embodiment_encoder_uses_global_context:
            required.update(
                embodiment_global_description=obs.embodiment_global_description,
                embodiment_global_description_availability=(
                    obs.embodiment_global_description_availability
                ),
            )
        if self.embodiment_encoder_uses_geometry:
            required.update(
                embodiment_geometry_points=obs.embodiment_geometry_points,
                embodiment_geometry_point_mask=(
                    obs.embodiment_geometry_point_mask
                ),
            )
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"URMA backbone-token conditioning requires observation fields: {missing}"
            )
        embodiment_embedding = self.embodiment_encoder(
            obs.embodiment_description,
            obs.embodiment_description_availability,
            obs.embodiment_control_state,
            obs.embodiment_control_mask,
            obs.embodiment_global_description,
            obs.embodiment_global_description_availability,
            obs.embodiment_geometry_points,
            obs.embodiment_geometry_point_mask,
        )
        return self.embodiment_token_proj(
            self.embodiment_token_norm(embodiment_embedding)
        )[:, None, :]

    def embed_response_history_token(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b 1 emb"]:
        required = {
            "response_state_history": obs.response_state_history,
            "response_state_history_mask": obs.response_state_history_mask,
            "response_action_history": obs.response_action_history,
            "response_action_history_mask": obs.response_action_history_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"causal response-history conditioning requires: {missing}")
        q_mask = obs.response_state_history_mask[..., None]
        u_mask = obs.response_action_history_mask[..., None]
        features = jnp.concatenate(
            [
                (obs.response_state_history * q_mask).reshape(
                    obs.response_state_history.shape[0], -1
                ),
                (obs.response_action_history * u_mask).reshape(
                    obs.response_action_history.shape[0], -1
                ),
                obs.response_state_history_mask.astype(jnp.float32),
                obs.response_action_history_mask.astype(jnp.float32),
            ],
            axis=-1,
        )
        return self.response_history_proj(self.response_history_norm(features))[:, None, :]

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        # [s] normally; [b, s] when FAST action tokens supply a per-sample AR mask.
        at.Bool[at.Array, " s"] | at.Bool[at.Array, "b s"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        if self.backbone_token_embodiment:
            embodiment_token = self.embed_embodiment_backbone_token(obs)
            tokens.append(embodiment_token)
            input_mask.append(
                obs.robotoken_mask
                if self.backbone_token_robotokens
                else jnp.ones(embodiment_token.shape[:2], dtype=jnp.bool_)
            )
            # Embodiment tokens are fully visible members of the PaliGemma prefix.
            ar_mask += [False] * embodiment_token.shape[1]
        if self.response_history_enabled:
            response_token = self.embed_response_history_token(obs)
            tokens.append(response_token)
            input_mask.append(jnp.ones(response_token.shape[:2], dtype=jnp.bool_))
            # One fully visible causal summary token; its fields end at the query time.
            ar_mask += [False]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        if obs.token_ar_mask is not None:
            # FAST action tokens live in the prompt stream and must be predicted
            # autoregressively, so use the per-sample AR mask from TokenizeFASTInputs for the
            # language span (images stay bidirectional). Returns [b, s] instead of [s];
            # make_attn_mask broadcasts either.
            image_len = tokens.shape[1] - obs.token_ar_mask.shape[1]
            ar_mask = jnp.concatenate(
                [
                    jnp.zeros((tokens.shape[0], image_len), dtype=bool),
                    obs.token_ar_mask.astype(bool),
                ],
                axis=1,
            )
        else:
            ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b cond"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if self.action_time_proj is not None:
            if obs.action_dt_s is None:
                raise ValueError("action_dt_s (seconds between chunk entries) is required when action time "
                                 "conditioning is enabled")
            # Entry k's label is the target reached at t + (k + 1) dt: one raw-seconds channel per token.
            tau = (jnp.arange(1, self.action_horizon + 1, dtype=jnp.float32)[None, :, None]
                   * obs.action_dt_s.astype(jnp.float32)[:, None, None])
            action_tokens = action_tokens + self.action_time_proj(tau)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            if self.embodiment_embed is not None:
                if obs.embodiment_id is None:
                    raise ValueError("embodiment_id is required when direct Pi0.5 embodiment conditioning is enabled")
                # Zero initialization exactly preserves the released Pi0.5 function
                # before fine-tuning. The concatenated variant expands the restored
                # AdaRMS kernels separately so the new branch starts functionally
                # identical while retaining an independent learned projection.
                embodiment_emb = self.embodiment_embed(obs.embodiment_id.astype(jnp.int32))
                if self.concatenate_embodiment_embedding:
                    time_emb = jnp.concatenate([time_emb, embodiment_emb], axis=-1)
                else:
                    time_emb = time_emb + embodiment_emb
            if self.action_expert_uses_embodiment_description:
                required = {
                    "embodiment_description": obs.embodiment_description,
                    "embodiment_description_availability": (
                        obs.embodiment_description_availability
                    ),
                    "embodiment_control_state": obs.embodiment_control_state,
                    "embodiment_control_mask": obs.embodiment_control_mask,
                }
                if self.embodiment_encoder_uses_global_context:
                    required.update(
                        embodiment_global_description=(
                            obs.embodiment_global_description
                        ),
                        embodiment_global_description_availability=(
                            obs.embodiment_global_description_availability
                        ),
                    )
                if self.embodiment_encoder_uses_geometry:
                    required.update(
                        embodiment_geometry_points=(
                            obs.embodiment_geometry_points
                        ),
                        embodiment_geometry_point_mask=(
                            obs.embodiment_geometry_point_mask
                        ),
                    )
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    raise ValueError(
                        f"URMA embodiment conditioning requires observation fields: {missing}"
                    )
                embodiment_embedding = self.embodiment_encoder(
                    obs.embodiment_description,
                    obs.embodiment_description_availability,
                    obs.embodiment_control_state,
                    obs.embodiment_control_mask,
                    obs.embodiment_global_description,
                    obs.embodiment_global_description_availability,
                    obs.embodiment_geometry_points,
                    obs.embodiment_geometry_point_mask,
                )
                if self.concat_project_embodiment:
                    projected_embodiment = self.embodiment_fusion_in(
                        embodiment_embedding
                    )
                    time_emb = self.embodiment_fusion_out(
                        jnp.concatenate([time_emb, projected_embodiment], axis=-1)
                    )
                elif self.embodiment_fusion_mode == "residual_mlp":
                    fusion = jnp.concatenate(
                        [time_emb, embodiment_embedding], axis=-1
                    )
                    fusion = nnx.gelu(self.embodiment_fusion_in(fusion))
                    time_emb = time_emb + self.embodiment_fusion_out(fusion)
                elif self.embodiment_fusion_mode == "add":
                    time_emb = time_emb + self.embodiment_fusion_out(
                        embodiment_embedding
                    )
                else:
                    time_emb = jnp.concatenate(
                        [
                            time_emb,
                            embodiment_embedding * self.embodiment_fusion_gate,
                        ],
                        axis=-1,
                    )
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _build_prefix_action_mask(
        self, prefix_mask: at.Bool[at.Array, "b s"], obs: _model.Observation
    ) -> at.Bool[at.Array, "b s"]:
        """Build the prefix mask for action attention, excluding the FAST action tokens.

        Ported from lihzha/lap's ``_build_prefix_action_mask``. Action tokens attend to images and
        the prompt, but NOT to the FAST tokens -- those are a FAST encoding of the ground-truth
        actions (tokenizer.py:82-92), so letting the flow-matching expert read them lets it solve
        its objective by copying. The skill does not transfer: at inference actions is None and
        the postfix is empty.

        ``token_ar_mask`` is the langact indicator (LAP's ``tokenized_langact_mask``): False over
        the text prefix, True over the FAST tokens.
        """
        if obs.token_ar_mask is None:
            return prefix_mask
        # Expand to full prefix length (add zeros for image positions).
        img_seq_len = prefix_mask.shape[1] - obs.token_ar_mask.shape[1]
        langact_mask_full = jnp.concatenate(
            [
                jnp.zeros((obs.token_ar_mask.shape[0], img_seq_len), dtype=bool),
                obs.token_ar_mask.astype(bool),
            ],
            axis=1,
        )
        return jnp.logical_and(prefix_mask, jnp.logical_not(langact_mask_full))

    def _build_combined_attention_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"] | at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"] | None,
        suffix_ar_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Bool[at.Array, "b _t _s"]:
        """Combined attention mask for the prefix (VLM) and suffix (action expert).

        Ported from lihzha/lap's ``_build_combined_attention_mask``. Attention pattern:
        - image tokens: bidirectional
        - prompt tokens: bidirectional
        - FAST action tokens: causal, attending to all images + prompt
        - action tokens: attend to images + prompt and causally to each other, NOT to FAST tokens

        On the non-FAST path ``prefix_ar_mask`` is all-False and ``prefix_mask_action`` equals
        ``prefix_mask``, so this reduces exactly to the original single ``make_attn_mask`` call.
        """
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        if suffix_mask is None or suffix_ar_mask is None:
            return prefix_attn

        batch_size, prefix_len = prefix_mask.shape
        suffix_len = suffix_mask.shape[1]
        combined = jnp.zeros((batch_size, prefix_len + suffix_len, prefix_len + suffix_len), dtype=bool)
        combined = combined.at[:, :prefix_len, :prefix_len].set(prefix_attn)

        # For action rows: ar_mask False over the prefix (full attention to images + prompt) and
        # suffix_ar_mask over the suffix (causal among action tokens). prefix_mask_action excludes
        # the FAST tokens, so action tokens cannot attend to them.
        prefix_ar_mask_action = jnp.zeros_like(prefix_mask_action, dtype=bool)
        input_mask = jnp.concatenate([prefix_mask_action, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask_action, suffix_ar_mask], axis=1)
        action_mask = make_attn_mask(input_mask, ar_mask)
        return combined.at[:, prefix_len:, :].set(action_mask[:, prefix_len:, :])

    def _build_combined_positions(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Int[at.Array, "b _t"]:
        """RoPE positions for prefix + suffix.

        Ported from lihzha/lap's ``_build_combined_positions``. The suffix is offset by the number
        of tokens it can actually attend to, not by the full prefix: counting the FAST tokens would
        place the action tokens ~len(postfix) positions further along during training than at
        inference, where the postfix is empty. That train/test mismatch survives attention masking,
        so it has to be handled here. Prefix positions keep the full cumsum -- the FAST tokens do
        occupy positions in the VLM stream.
        """
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if suffix_mask is None:
            return prefix_positions.astype(jnp.int32)
        suffix_positions = jnp.sum(prefix_mask_action, axis=-1, keepdims=True) + jnp.cumsum(suffix_mask, axis=-1) - 1
        return jnp.concatenate([prefix_positions, suffix_positions], axis=1).astype(jnp.int32)

    def _compute_language_loss(
        self, obs: _model.Observation, prefix_pre_logits: at.Float[at.Array, "b s emb"]
    ) -> at.Float[at.Array, " b"]:
        """Next-token cross-entropy over the FAST action tokens carried in the prompt stream.

        This is the language-head half of the pi0.5 objective: the same PaliGemma head that
        reads the prompt also predicts discretized actions, while the action expert predicts
        them continuously via flow matching. Adapted from lihzha/lap's ``_compute_language_loss``
        (its CoT/VQA-specific masks are dropped -- we only supervise FAST action tokens).
        """
        targets = obs.tokenized_prompt[:, 1:]
        # Drop the final prefix position (no next-token target), then align to the text span.
        pre_logits = prefix_pre_logits[:, :-1]
        pre_logits = pre_logits[:, -targets.shape[1] :]
        # Pin the vocab projection to the batch-major layout every other activation uses. Left
        # unconstrained, XLA reshards the [b, t, 257152] logits with an all-to-all whose combined
        # buffers fail its "same element type and count" check at runtime.
        pre_logits = sharding.activation_sharding_constraint(pre_logits)
        logits = sharding.activation_sharding_constraint(self.PaliGemma.llm(pre_logits, method="decode"))

        loss_mask = jnp.logical_and(obs.tokenized_prompt_mask[:, 1:], obs.token_loss_mask[:, 1:])
        # Integer-label CE keeps memory at one vocab-sized array: materializing the [b, t, 257152]
        # one-hot and log_softmax alongside the logits costs ~59 GiB/device at b=64, t=320 and OOMs.
        # The logits are bf16 like every other activation, but the vocab reduction is accumulated
        # in fp32: summing 257k exponentials in bf16 (8-bit mantissa) loses small terms outright.
        # XLA fuses this convert into the reduction, so no fp32 [b, t, vocab] array is materialized.
        token_ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
        return jnp.sum(token_ce * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)

    @override
    def _flow_source_noise(
        self, white: at.Float[at.Array, "b ah ad"], observation: _model.Observation
    ) -> at.Float[at.Array, "b ah ad"]:
        """The flow-matching source from standard normal draws: unchanged (white), or colored per action dimension
        by the Cholesky factor of the squared-exponential or Matern kernel over tau_k = (k + 1) * action_dt_s (plus
        jitter, renormalized to unit variance), so it is one Gaussian-process draw over time in seconds."""
        if self.action_noise_length_scale_s is None:
            return white
        if observation.action_dt_s is None:
            raise ValueError("Gaussian-process action noise needs action_dt_s (seconds between chunk entries)")
        tau = (jnp.arange(1, self.action_horizon + 1, dtype=jnp.float32)[None, :]
               * observation.action_dt_s.astype(jnp.float32)[:, None])
        gap = jnp.abs(tau[:, :, None] - tau[:, None, :]) / self.action_noise_length_scale_s
        nu = self.action_noise_matern_nu
        if nu is None:
            correlation = jnp.exp(-0.5 * gap**2)
        else:
            s = math.sqrt(2 * nu) * gap
            correlation = {0.5: 1.0, 1.5: 1 + s, 2.5: 1 + s + s**2 / 3}[nu] * jnp.exp(-s)
        eye = jnp.eye(self.action_horizon, dtype=jnp.float32)
        kernel = (correlation + self.action_noise_jitter * eye) / (1.0 + self.action_noise_jitter)
        return jnp.einsum("bij,bjd->bid", jnp.linalg.cholesky(kernel), white.astype(jnp.float32)).astype(white.dtype)

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        return_extras: bool = False,
    ) -> at.Float[at.Array, "*b ah"] | tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, augmentation_mode=self.augmentation_mode
        )
        if train and self.urma51_dynamic_conditioning_noise_std > 0:
            observation = add_urma51_standardized_dynamic_noise(
                observation,
                jax.random.fold_in(rng, _URMA51_DYNAMIC_NOISE_RNG_TAG),
                self.urma51_dynamic_conditioning_noise_std,
            )

        batch_shape = actions.shape[:-2]
        noise = self._flow_source_noise(jax.random.normal(noise_rng, actions.shape), observation)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        # Action expert is insulated from the FAST action tokens in both attention and RoPE
        # positions (LAP's _build_prefix_action_mask / _build_combined_attention_mask /
        # _build_combined_positions). On the non-FAST path all three reduce to the original
        # single make_attn_mask + cumsum.
        prefix_mask_action = self._build_prefix_action_mask(prefix_mask, observation)
        attn_mask = self._build_combined_attention_mask(
            prefix_mask,
            prefix_ar_mask,
            prefix_mask_action,
            suffix_mask,
            jnp.broadcast_to(suffix_ar_mask, (prefix_mask.shape[0], suffix_ar_mask.shape[0])),
        )
        positions = self._build_combined_positions(prefix_mask, prefix_mask_action, suffix_mask)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        squared_error = jnp.square(v_t - u_t)
        if observation.action_loss_mask is None:
            action_loss = jnp.mean(squared_error, axis=-1)
        else:
            action_mask = observation.action_loss_mask.astype(squared_error.dtype)
            valid_dimensions = jnp.maximum(jnp.sum(action_mask, axis=-1), 1.0)
            action_loss = jnp.sum(squared_error * action_mask, axis=-1) / valid_dimensions

        if not (self.enable_langact_training and observation.token_loss_mask is not None):
            if return_extras:
                return action_loss, {"action_loss": action_loss}
            return action_loss

        # Dual objective. The two halves reduce over different axes -- flow matching is per
        # chunk step [b, ah], the FAST-token CE is one value per sample [b] -- so they cannot be
        # summed here without silently choosing a reduction for the language half. Broadcasting
        # it across the horizon (what this did until now) hands it to the trainer's masked mean,
        # which turns the language term into a mean *weighted by each sample's number of valid
        # chunk steps* rather than a plain mean. The bias is the covariance between a sample's
        # language loss and how many of its chunk steps are valid, and the two are not
        # independent: a chunk running past the end of its episode has both fewer valid steps
        # and an easier FAST target (the tail is synthetic gripper_hold), so easy samples get
        # down-weighted. At the ~7% truncated fraction measured on droid + molmobot at
        # action_horizon=32 that is ~0.5% from sampling noise alone and ~3% once the tail
        # chunks differ in difficulty. LAP reduces its language half with a plain mean, so the
        # array returned here carries the action half only and the caller adds
        # ``language_loss_weight * mean(lang_loss)`` after its own masked mean.
        if not return_extras:
            raise ValueError(
                "enable_langact_training requires return_extras=True: the language half of the "
                "objective is returned in extras and has to be added by the caller."
            )
        lang_loss = self._compute_language_loss(observation, prefix_out)
        # action_loss stays unweighted in extras -- it is the per-component number logged to
        # wandb. The returned array is the weighted action term the trainer reduces.
        return self.action_loss_weight * action_loss, {"action_loss": action_loss, "lang_loss": lang_loss}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix_weights: at.Float[at.Array, "b ah ad"] | None = None,
        max_guidance_weight: float = 5.0,
        rtc_velocity_rescale: bool = True,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        # Supplied noise (a server's seeded draws) is standard normal too; color it the same way.
        noise = self._flow_source_noise(noise, observation)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Mirror compute_loss: the action expert never attends to FAST action tokens, and its RoPE
        # positions are offset only by what it can see. At serving actions is None, the postfix is
        # empty, and both reduce to LAP's `sample_actions` (which uses plain prefix_mask, because
        # it never has FAST tokens at inference). This generalization matters for offline eval,
        # which feeds batches that still carry ground truth: without it the expert would read the
        # answer. The postfix stays in the KV cache but is fully masked out, and the non-postfix
        # prefix keys are unaffected by its presence (prefix tokens have AR 0, so they never attend
        # to the postfix).
        prefix_mask_action = self._build_prefix_action_mask(prefix_mask, observation)

        def velocity(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            if action_prefix is None:
                v_t = velocity(x_t, time)
            else:
                if action_prefix_weights is None:
                    raise ValueError("action_prefix_weights is required with action_prefix")

                # Openpi integrates from noise t=1 to data t=0.  The corresponding
                # denoised estimate is x_0 = x_t - t*v_t.  The paper's GDM term is a
                # VJP through this estimate; because our integration convention is
                # reversed, it is subtracted from the velocity below.
                def denoise(noisy_action):
                    base_velocity = velocity(noisy_action, time)
                    return noisy_action - time * base_velocity, base_velocity

                denoised, vjp_fn, v_t = jax.vjp(denoise, x_t, has_aux=True)
                weighted_error = action_prefix_weights * (action_prefix - denoised)
                prefix_correction = vjp_fn(weighted_error)[0]
                guidance_weight = jnp.minimum(
                    jnp.asarray(max_guidance_weight, dtype=x_t.dtype),
                    time / jnp.maximum(1.0 - time, jnp.asarray(1e-6, dtype=x_t.dtype)),
                )
                guided_velocity = v_t - guidance_weight * prefix_correction

                base_norm = jnp.linalg.norm(v_t, axis=(-2, -1), keepdims=True)
                guided_norm = jnp.linalg.norm(guided_velocity, axis=(-2, -1), keepdims=True)
                scale = jnp.where((base_norm > 1e-6) & (guided_norm > 1e-6), base_norm / guided_norm, 1.0)
                guided_velocity = jnp.where(jnp.asarray(rtc_velocity_rescale), guided_velocity * scale, guided_velocity)
                v_t = guided_velocity

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
