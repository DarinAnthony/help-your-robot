"""Trainable embodiment encoders used by Pi action-expert conditioning.

The URMA2 module is a native NNX port of the three-head, scale-2 routing trunk. It returns
``action_latent``: the first representation with no control dimension, after information has
been pooled across the complete control set. The locomotion policy's general-state branch and
per-joint action decoder are intentionally outside this reusable embodiment extractor.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from pi05_integration import input_config as _input_config


def flatten_urma4_backbone_token(
    description,
    availability,
    control_state,
    control_mask,
    global_description,
    global_availability,
    *,
    token_width: int,
):
    """Concatenate the complete normalized URMA4 input and right-pad one token."""

    if description.ndim != 3:
        raise ValueError(
            f"flat URMA4 description must have shape [batch, control, feature], got {description.shape}"
        )
    if availability.shape != description.shape:
        raise ValueError(
            "flat URMA4 availability must match description: "
            f"{availability.shape} != {description.shape}"
        )
    if control_state.shape[:2] != description.shape[:2]:
        raise ValueError(
            "flat URMA4 control state must match the batch/control prefix: "
            f"{control_state.shape[:2]} != {description.shape[:2]}"
        )
    if control_mask.shape != description.shape[:2]:
        raise ValueError(
            "flat URMA4 control mask must match the batch/control prefix: "
            f"{control_mask.shape} != {description.shape[:2]}"
        )
    if global_description.ndim != 2:
        raise ValueError(
            "flat URMA4 global description must have shape [batch, feature], "
            f"got {global_description.shape}"
        )
    if global_availability.shape != global_description.shape:
        raise ValueError(
            "flat URMA4 global availability must match global description: "
            f"{global_availability.shape} != {global_description.shape}"
        )
    if global_description.shape[0] != description.shape[0]:
        raise ValueError("flat URMA4 local and global batch dimensions must match")

    batch_size = description.shape[0]
    dtype = description.dtype
    flattened = jnp.concatenate(
        [
            description.reshape(batch_size, -1),
            availability.astype(dtype).reshape(batch_size, -1),
            control_state.astype(dtype).reshape(batch_size, -1),
            control_mask.astype(dtype).reshape(batch_size, -1),
            global_description.astype(dtype).reshape(batch_size, -1),
            global_availability.astype(dtype).reshape(batch_size, -1),
        ],
        axis=-1,
    )
    if flattened.shape[-1] > token_width:
        raise ValueError(
            "flat URMA4 input is wider than the backbone token: "
            f"{flattened.shape[-1]} > {token_width}"
        )
    return jnp.pad(flattened, ((0, 0), (0, token_width - flattened.shape[-1])))


class RobotTokenSequenceEncoder(nnx.Module):
    """T2-style modality projections plus shared learned reference-ID embeddings."""

    def __init__(self, config: _input_config.RobotTokenBackboneSequence, *, rngs: nnx.Rngs):
        self.link_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.dyna_joint_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.fixed_joint_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.actuator_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.collision_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.tendon_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.equality_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        self.control_proj = nnx.Linear(config.feature_dim, config.token_width, rngs=rngs)
        # Index zero is the missing-reference sentinel. IDs are shifted by one.
        self.link_ref = nnx.Embed(config.max_tokens + 1, config.token_width, rngs=rngs)
        self.dyna_joint_ref = nnx.Embed(config.max_tokens + 1, config.token_width, rngs=rngs)
        self.fixed_joint_ref = nnx.Embed(config.max_tokens + 1, config.token_width, rngs=rngs)
        self.actuator_ref = nnx.Embed(config.max_tokens + 1, config.token_width, rngs=rngs)
        self.track_link_ref = nnx.Embed(config.max_tokens + 1, config.token_width, rngs=rngs)
        self.type_capacities = config.type_capacities

    def __call__(self, values, reference_ids, token_mask):
        projections = (
            self.link_proj,
            self.dyna_joint_proj,
            self.fixed_joint_proj,
            self.actuator_proj,
            self.collision_proj,
            self.tendon_proj,
            self.equality_proj,
            self.control_proj,
        )
        offset = 0
        blocks = []
        for projection, capacity in zip(projections, self.type_capacities, strict=True):
            blocks.append(projection(values[:, offset : offset + capacity]))
            offset += capacity
        tokens = jnp.concatenate(blocks, axis=1)
        for namespace, embedding in enumerate(
            (
                self.link_ref,
                self.dyna_joint_ref,
                self.fixed_joint_ref,
                self.actuator_ref,
                self.track_link_ref,
            )
        ):
            ids = reference_ids[..., namespace]
            present = ids >= 0
            tokens = tokens + embedding(jnp.maximum(ids + 1, 0)) * present[..., None]
        return tokens * token_mask[..., None]


class UrmaPointNetGeometryEncoder(nnx.Module):
    """Shared PointNet-style encoder for per-control collision geometry.

    Input coordinates remain in meters in the control-local frame.  In particular,
    this module deliberately does not center or normalize each point set: physical
    scale and aspect ratio are part of the embodiment signal.  The final three input
    channels are gripper part-role indicators (palm, moving finger, contact pad); they
    are zero for ordinary arm-link geometry.
    """

    input_dim: int = 9
    output_dim: int = 128

    def __init__(self, *, rngs: nnx.Rngs):
        self.point0 = nnx.Linear(self.input_dim, 64, rngs=rngs)
        self.point0_norm = nnx.LayerNorm(64, epsilon=1e-6, rngs=rngs)
        self.point1 = nnx.Linear(64, 128, rngs=rngs)
        self.point1_norm = nnx.LayerNorm(128, epsilon=1e-6, rngs=rngs)
        self.point2 = nnx.Linear(128, self.output_dim, rngs=rngs)
        self.output_norm = nnx.LayerNorm(
            self.output_dim, epsilon=1e-6, rngs=rngs
        )

    def __call__(self, points, point_mask, control_mask):
        if points.shape[-1] != self.input_dim:
            raise ValueError(
                "URMA6 collision points must contain position[3], normal[3], "
                f"and part role[3], got width {points.shape[-1]}"
            )
        if points.shape[-2] != 256:
            raise ValueError(
                f"URMA6 requires exactly 256 points per control, got {points.shape[-2]}"
            )
        if point_mask.shape != points.shape[:-1]:
            raise ValueError(
                "URMA6 point mask shape must match the point prefix: "
                f"{point_mask.shape} != {points.shape[:-1]}"
            )
        if control_mask.shape != points.shape[:-2]:
            raise ValueError(
                "URMA6 control mask shape must match the control prefix: "
                f"{control_mask.shape} != {points.shape[:-2]}"
            )

        features = jax.nn.elu(self.point0_norm(self.point0(points)))
        features = jax.nn.elu(self.point1_norm(self.point1(features)))
        features = jax.nn.elu(self.point2(features))

        # Invalid points must not participate in the symmetric reduction.  Replace
        # the all-invalid result after max pooling so an empty/padded control has an
        # exact zero geometry vector rather than a learned or non-finite sentinel.
        valid_points = point_mask & control_mask[..., None]
        pooled = jnp.max(
            jnp.where(valid_points[..., None], features, -jnp.inf), axis=-2
        )
        has_points = jnp.any(valid_points, axis=-1)
        pooled = jnp.where(has_points[..., None], pooled, 0.0)
        encoded = self.output_norm(pooled)
        return jnp.where(has_points[..., None], encoded, 0.0)


class UrmaGeometryAugmentedEncoder(nnx.Module):
    """URMA5 scalar encoder augmented by a 128-D PointNet feature per control.

    The scalar description is concatenated unchanged and in its original order;
    URMA6 therefore contains every URMA5 scalar field rather than replacing scalar
    geometry with an opaque learned feature.
    """

    geometry_dim: int = UrmaPointNetGeometryEncoder.output_dim

    def __init__(
        self,
        scalar_description_dim: int,
        routing_dim: int,
        state_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        output_normalization: str = "none",
        global_description_dim: int | None = None,
        global_embedding_dim: int = 128,
        control_state_dim: int = 7,
        *,
        rngs: nnx.Rngs,
    ):
        if scalar_description_dim <= 0:
            raise ValueError("URMA6 scalar description dimension must be positive")
        self.scalar_description_dim = scalar_description_dim
        self.geometry_encoder = UrmaPointNetGeometryEncoder(rngs=rngs)
        self.scalar_encoder = UrmaEmbodimentEncoder(
            scalar_description_dim + self.geometry_dim,
            routing_dim,
            state_feature_dim,
            embedding_dim,
            num_heads,
            output_normalization,
            global_description_dim,
            global_embedding_dim,
            control_state_dim,
            rngs=rngs,
        )

    def augment_description(
        self,
        description,
        availability,
        control_mask,
        geometry_points,
        geometry_point_mask,
    ):
        if description.shape[-1] != self.scalar_description_dim:
            raise ValueError(
                "URMA6 scalar description width disagrees with its extractor: "
                f"{description.shape[-1]} != {self.scalar_description_dim}"
            )
        geometry = self.geometry_encoder(
            geometry_points, geometry_point_mask, control_mask
        )
        geometry_available = jnp.broadcast_to(
            (jnp.any(geometry_point_mask, axis=-1) & control_mask)[..., None],
            geometry.shape,
        )
        return (
            jnp.concatenate([description, geometry], axis=-1),
            jnp.concatenate([availability, geometry_available], axis=-1),
        )

    def __call__(
        self,
        description,
        availability,
        control_state,
        control_mask,
        global_description,
        global_availability,
        geometry_points=None,
        geometry_point_mask=None,
    ):
        if geometry_points is None or geometry_point_mask is None:
            raise ValueError("URMA6 requires collision geometry and its point mask")
        description, availability = self.augment_description(
            description,
            availability,
            control_mask,
            geometry_points,
            geometry_point_mask,
        )
        return self.scalar_encoder(
            description,
            availability,
            control_state,
            control_mask,
            global_description,
            global_availability,
        )


class UrmaRoutingHead(nnx.Module):
    def __init__(
        self,
        description_dim: int,
        routing_dim: int,
        state_feature_dim: int,
        control_state_dim: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        # Availability is concatenated with the numerical description so unavailable
        # URDF fields remain distinguishable from genuine zero values.
        self.routing_in = nnx.Linear(2 * description_dim, routing_dim, rngs=rngs)
        self.routing_norm = nnx.LayerNorm(routing_dim, epsilon=1e-6, rngs=rngs)
        self.routing_out = nnx.Linear(routing_dim, routing_dim, rngs=rngs)
        self.state_proj = nnx.Linear(
            control_state_dim, state_feature_dim, rngs=rngs
        )
        self.log_temperature = nnx.Param(
            jnp.log(jnp.asarray([1.0 - 0.015], jnp.float32))
        )

    def __call__(self, description, availability, control_state, control_mask):
        routing = jnp.concatenate(
            [description, availability.astype(description.dtype)], axis=-1
        )
        routing = self.routing_in(routing)
        routing = jax.nn.elu(self.routing_norm(routing))
        routing = jnp.clip(
            jnp.tanh(self.routing_out(routing)), -1.0 + 1e-8, 1.0 - 1e-8
        )
        routing = jax.nn.softmax(
            routing / (jnp.exp(self.log_temperature.value) + 0.015), axis=-1
        )
        state_features = jax.nn.elu(self.state_proj(control_state))
        interactions = routing[..., :, None] * state_features[..., None, :]
        interactions = interactions.reshape((*interactions.shape[:-2], -1))
        return jnp.sum(interactions * control_mask[..., None], axis=-2)


class UrmaEmbodimentEncoder(nnx.Module):
    """URMA2 routing followed by its compact joint-fused global latent."""

    def __init__(
        self,
        description_dim: int,
        routing_dim: int,
        state_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        output_normalization: str = "none",
        global_description_dim: int | None = None,
        global_embedding_dim: int = 128,
        control_state_dim: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        if (num_heads, routing_dim, state_feature_dim, embedding_dim) != (3, 128, 8, 256):
            raise ValueError("URMA2 requires its three-head, scale-2 dimensions")
        self.head0 = UrmaRoutingHead(
            description_dim,
            routing_dim,
            state_feature_dim,
            control_state_dim,
            rngs=rngs,
        )
        self.head1 = UrmaRoutingHead(
            description_dim,
            routing_dim,
            state_feature_dim,
            control_state_dim,
            rngs=rngs,
        )
        self.head2 = UrmaRoutingHead(
            description_dim,
            routing_dim,
            state_feature_dim,
            control_state_dim,
            rngs=rngs,
        )
        if global_description_dim is None:
            self.global_context_in = None
            self.global_context_norm = None
            self.global_context_out = None
            pooled_dim = num_heads * routing_dim * state_feature_dim
        else:
            if global_description_dim <= 0 or global_embedding_dim <= 0:
                raise ValueError("URMA global context dimensions must be positive")
            self.global_context_in = nnx.Linear(
                2 * global_description_dim, global_embedding_dim, rngs=rngs
            )
            self.global_context_norm = nnx.LayerNorm(
                global_embedding_dim, epsilon=1e-6, rngs=rngs
            )
            self.global_context_out = nnx.Linear(
                global_embedding_dim, global_embedding_dim, rngs=rngs
            )
            pooled_dim = (
                num_heads * routing_dim * state_feature_dim
                + global_embedding_dim
            )
        self.global0 = nnx.Linear(pooled_dim, 512, rngs=rngs)
        self.global_norm = nnx.LayerNorm(512, epsilon=1e-6, rngs=rngs)
        self.global1 = nnx.Linear(512, 256, rngs=rngs)
        self.global2 = nnx.Linear(256, embedding_dim, rngs=rngs)
        if output_normalization == "none":
            self.output_norm = None
        elif output_normalization == "layer_norm":
            self.output_norm = nnx.LayerNorm(
                embedding_dim, epsilon=1e-6, rngs=rngs
            )
        else:
            raise ValueError(
                "unsupported URMA output normalization: "
                f"{output_normalization!r}"
            )

    def __call__(
        self,
        description,
        availability,
        control_state,
        control_mask,
        global_description=None,
        global_availability=None,
        geometry_points=None,
        geometry_point_mask=None,
    ):
        if geometry_points is not None or geometry_point_mask is not None:
            raise ValueError("this URMA encoder does not consume collision geometry")
        pooled = jnp.concatenate(
            [
                head(description, availability, control_state, control_mask)
                for head in (self.head0, self.head1, self.head2)
            ],
            axis=-1,
        )
        if self.global_context_in is not None:
            if global_description is None or global_availability is None:
                raise ValueError(
                    "URMA4 encoder requires global description and availability"
                )
            global_context = jnp.concatenate(
                [
                    global_description,
                    global_availability.astype(global_description.dtype),
                ],
                axis=-1,
            )
            global_context = jax.nn.elu(
                self.global_context_norm(self.global_context_in(global_context))
            )
            global_context = jax.nn.elu(
                self.global_context_out(global_context)
            )
            pooled = jnp.concatenate([pooled, global_context], axis=-1)
        elif global_description is not None or global_availability is not None:
            raise ValueError("legacy URMA encoder does not consume global context")
        embedding = jax.nn.elu(self.global_norm(self.global0(pooled)))
        embedding = jax.nn.elu(self.global1(embedding))
        embedding = self.global2(embedding)
        return (
            embedding
            if self.output_norm is None
            else self.output_norm(embedding)
        )


def create_encoder(extractor: _input_config.EmbodimentEmbeddingExtractor, *, rngs: nnx.Rngs):
    """Materialize a backend encoder from the portable extractor specification."""
    if isinstance(extractor, _input_config.Urma6EmbodimentExtractor):
        return UrmaGeometryAugmentedEncoder(
            extractor.description_dim,
            extractor.routing_dim,
            extractor.state_feature_dim,
            extractor.embedding_dim,
            extractor.num_heads,
            extractor.output_normalization,
            extractor.global_description_shape[0],
            extractor.global_embedding_dim,
            extractor.structured_input_shape[2],
            rngs=rngs,
        )
    if isinstance(extractor, _input_config.Urma2EmbodimentExtractor):
        return UrmaEmbodimentEncoder(
            extractor.description_dim,
            extractor.routing_dim,
            extractor.state_feature_dim,
            extractor.embedding_dim,
            extractor.num_heads,
            extractor.output_normalization,
            (
                None
                if extractor.global_description_shape is None
                else extractor.global_description_shape[0]
            ),
            getattr(extractor, "global_embedding_dim", 128),
            extractor.structured_input_shape[2],
            rngs=rngs,
        )
    raise TypeError(f"unsupported embodiment embedding extractor: {type(extractor).__name__}")
