import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from pi05_integration import input_config as _input_config
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # "shared_mild" is the comparison recipe used by MolmoSpaces x0 and pi0.5.
    # Keep the upstream behavior available for non-comparison configs.
    augmentation_mode: str = "pi05_native"
    # Typed model-side condition for the Pi0.5 action expert.  ``embodiment_vocab_size`` is
    # retained as an input-compatibility alias for old ad-hoc callers; registered configs use
    # ``action_expert_conditioning`` and both resolve to the identical parameter tree.
    action_expert_conditioning: _input_config.ActionExpertConditioning = dataclasses.field(
        default_factory=_input_config.TimestepOnly
    )
    backbone_conditioning: _input_config.BackboneConditioning = dataclasses.field(
        default_factory=_input_config.NoBackboneConditioning
    )
    response_history_conditioning: _input_config.ResponseHistoryConditioning = dataclasses.field(
        default_factory=_input_config.NoResponseHistoryConditioning
    )
    # Chunk-entry time offsets added to the action tokens; equals the data config's input_config.action_time.
    action_time_conditioning: _input_config.ActionTimeConditioning = dataclasses.field(
        default_factory=_input_config.NoActionTime
    )
    # Training-only Gaussian perturbation of normalized URMA5.1 dynamic fields.
    # This lives in the native model config so checkpoints preserve the intervention;
    # inference never applies it.
    urma51_dynamic_conditioning_noise_std: float = 0.0
    embodiment_vocab_size: int = 0

    # --- pi0.5 dual objective ("knowledge insulation", ported from lihzha/lap) ---------------
    # Upstream openpi trains only the flow-matching action expert. The released pi0.5 recipe
    # additionally predicts FAST action tokens through the PaliGemma language head, and blocks
    # gradients from the action expert back into the VLM. Both are opt-in and add NO parameters,
    # so checkpoints stay interchangeable with flow-matching-only runs.
    # Requires the data pipeline to emit token_loss_mask / token_ar_mask (TokenizeFASTInputs).
    enable_langact_training: bool = False
    language_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    # Detach expert-0 (VLM) K/V where the action expert attends to it -- knowledge insulation.
    stop_action_to_vlm_grad: bool = False

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.augmentation_mode not in {"pi05_native", "shared_mild", "none"}:
            raise ValueError(f"unsupported augmentation_mode: {self.augmentation_mode}")
        typed_vocab_size = self.action_expert_conditioning.embodiment_vocab_size
        if self.embodiment_vocab_size and typed_vocab_size:
            if self.embodiment_vocab_size != typed_vocab_size:
                raise ValueError(
                    "legacy embodiment_vocab_size disagrees with action_expert_conditioning"
                )
        elif typed_vocab_size:
            object.__setattr__(self, "embodiment_vocab_size", typed_vocab_size)
        if self.embodiment_vocab_size < 0:
            raise ValueError("embodiment_vocab_size must be non-negative")
        if self.embodiment_vocab_size and not self.pi05:
            raise ValueError("direct embodiment AdaRMS conditioning is only supported for Pi0.5")
        if self.backbone_conditioning.embodiment_vocab_size and not self.pi05:
            raise ValueError(
                "learned embodiment backbone-token conditioning is only supported for Pi0.5"
            )
        if self.action_expert_conditioning.uses_embodiment_description and not self.pi05:
            raise ValueError("URMA embodiment conditioning is only supported for Pi0.5")
        if self.backbone_conditioning.uses_embodiment_description and not self.pi05:
            raise ValueError("URMA backbone-token conditioning is only supported for Pi0.5")
        if self.backbone_conditioning.uses_robotokens and not self.pi05:
            raise ValueError("RoboToken backbone conditioning is only supported for Pi0.5")
        if self.response_history_conditioning.enabled and not self.pi05:
            raise ValueError("causal response-history conditioning is only supported for Pi0.5")
        if self.action_time_conditioning.enabled and not self.pi05:
            raise ValueError("action time conditioning is only supported for Pi0.5")
        if (
            not isinstance(self.urma51_dynamic_conditioning_noise_std, (int, float))
            or not math.isfinite(self.urma51_dynamic_conditioning_noise_std)
            or self.urma51_dynamic_conditioning_noise_std < 0
        ):
            raise ValueError(
                "URMA5.1 dynamic-conditioning noise standard deviation must be finite "
                "and nonnegative"
            )
        if self.urma51_dynamic_conditioning_noise_std > 0:
            extractors = (
                getattr(self.action_expert_conditioning, "extractor", None),
                getattr(self.backbone_conditioning, "extractor", None),
            )
            if not any(
                isinstance(extractor, _input_config.Urma51EmbodimentExtractor)
                for extractor in extractors
            ):
                raise ValueError(
                    "URMA5.1 dynamic-conditioning noise requires an URMA5.1 extractor"
                )
        if (
            self.action_expert_conditioning.uses_embodiment_description
            and self.embodiment_vocab_size
        ):
            raise ValueError("learned-ID and extracted embodiment AdaRMS routes are mutually exclusive")
        if not isinstance(
            self.backbone_conditioning, _input_config.NoBackboneConditioning
        ) and not isinstance(self.action_expert_conditioning, _input_config.TimestepOnly):
            raise ValueError(
                "backbone-token conditioning requires timestep-only action-expert AdaRMS"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            conditioning = self.action_expert_conditioning
            action_extractor = (
                conditioning.extractor
                if isinstance(conditioning, _input_config.ExtractedEmbodimentAdaRMS)
                else None
            )
            backbone_extractor = (
                self.backbone_conditioning.extractor
                if isinstance(
                    self.backbone_conditioning,
                    (
                        _input_config.ExtractedEmbodimentBackboneToken,
                        _input_config.FlatUrma4BackboneToken,
                    ),
                )
                else None
            )
            extractor = (
                action_extractor
                if action_extractor is not None
                else backbone_extractor
            )
            structured_input_shape = (
                None if extractor is None else extractor.structured_input_shape
            )
            description_shape = (
                [batch_size, structured_input_shape[0], structured_input_shape[1]]
                if extractor is not None
                else None
            )
            robotoken_conditioning = (
                self.backbone_conditioning
                if isinstance(
                    self.backbone_conditioning,
                    _input_config.RobotTokenBackboneSequence,
                )
                else None
            )
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                embodiment_description=(
                    None
                    if description_shape is None
                    else jax.ShapeDtypeStruct(description_shape, jnp.float32)
                ),
                embodiment_description_availability=(
                    None
                    if description_shape is None
                    else jax.ShapeDtypeStruct(description_shape, jnp.bool_)
                ),
                embodiment_control_mask=(
                    None
                    if description_shape is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, structured_input_shape[0]], jnp.bool_
                    )
                ),
                embodiment_control_state=(
                    None
                    if description_shape is None
                    else jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            structured_input_shape[0],
                            structured_input_shape[2],
                        ],
                        jnp.float32,
                    )
                ),
                embodiment_geometry_points=(
                    None
                    if extractor is None or extractor.geometry_input_shape is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, *extractor.geometry_input_shape], jnp.float32
                    )
                ),
                embodiment_geometry_point_mask=(
                    None
                    if extractor is None or extractor.geometry_input_shape is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, *extractor.geometry_input_shape[:2]], jnp.bool_
                    )
                ),
                embodiment_global_description=(
                    None
                    if extractor is None
                    or extractor.global_description_shape is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, *extractor.global_description_shape],
                        jnp.float32,
                    )
                ),
                embodiment_global_description_availability=(
                    None
                    if extractor is None
                    or extractor.global_description_shape is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, *extractor.global_description_shape],
                        jnp.bool_,
                    )
                ),
                robotoken_values=(
                    None
                    if robotoken_conditioning is None
                    else jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            robotoken_conditioning.max_tokens,
                            robotoken_conditioning.feature_dim,
                        ],
                        jnp.float32,
                    )
                ),
                robotoken_types=(
                    None
                    if robotoken_conditioning is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, robotoken_conditioning.max_tokens], jnp.int32
                    )
                ),
                robotoken_reference_ids=(
                    None
                    if robotoken_conditioning is None
                    else jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            robotoken_conditioning.max_tokens,
                            robotoken_conditioning.reference_namespaces,
                        ],
                        jnp.int32,
                    )
                ),
                robotoken_mask=(
                    None
                    if robotoken_conditioning is None
                    else jax.ShapeDtypeStruct(
                        [batch_size, robotoken_conditioning.max_tokens], jnp.bool_
                    )
                ),
                response_state_history=(
                    jax.ShapeDtypeStruct(
                        [batch_size, 9, 7], jnp.float32
                    )
                    if self.response_history_conditioning.enabled
                    else None
                ),
                response_state_history_mask=(
                    jax.ShapeDtypeStruct([batch_size, 9], jnp.bool_)
                    if self.response_history_conditioning.enabled
                    else None
                ),
                response_action_history=(
                    jax.ShapeDtypeStruct(
                        [batch_size, 8, 7], jnp.float32
                    )
                    if self.response_history_conditioning.enabled
                    else None
                ),
                response_action_history_mask=(
                    jax.ShapeDtypeStruct([batch_size, 8], jnp.bool_)
                    if self.response_history_conditioning.enabled
                    else None
                ),
                action_dt_s=(
                    jax.ShapeDtypeStruct([batch_size], jnp.float32)
                    if self.action_time_conditioning.enabled
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
