import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Per-target validity after action chunking. MolmoSpaces uses this only for
    # synthetic future timesteps and broadcasts it across every model action
    # coordinate, matching the released Pi0.5 flow objective.
    action_loss_mask: at.Bool[ArrayT, "*b ah ad"] | None = None
    # The sample's physical (non-padding) action coordinates as the dataset built them, before
    # padding to the model width. Metrics only: the model never reads it, so under the
    # model-coordinate loss above it is what separates real joints from zero padding.
    action_semantic_mask: at.Bool[ArrayT, "*b sd"] | None = None

    # Host-facing grouping metadata. ``embodiment_id`` also drives the legacy learned-ID
    # AdaRMS route; structured extractors consume the explicit tensors below instead.
    embodiment_id: at.Int[ArrayT, "*b"] | None = None
    task_id: at.Int[ArrayT, "*b"] | None = None

    # Optional structured embodiment input. Unlike embodiment_id, these tensors are model inputs:
    # they carry persistent per-control robot descriptions plus the aligned current state.
    embodiment_description: at.Float[ArrayT, "*b ctrl feat"] | None = None
    embodiment_description_availability: at.Bool[ArrayT, "*b ctrl feat"] | None = None
    embodiment_control_mask: at.Bool[ArrayT, "*b ctrl"] | None = None
    embodiment_control_state: at.Float[ArrayT, "*b ctrl cstate"] | None = None
    # URMA6 collision-surface geometry. Coordinates are metric and control-local;
    # the final feature channels contain normals and optional gripper part roles.
    embodiment_geometry_points: at.Float[ArrayT, "*b ctrl point pfeat"] | None = None
    embodiment_geometry_point_mask: at.Bool[ArrayT, "*b ctrl point"] | None = None

    # Exact causal response history. q is measured arm qpos + physical aperture;
    # u is previously sent absolute arm target + semantic gripper target.
    response_state_history: at.Float[ArrayT, "*b qs rd"] | None = None
    response_state_history_mask: at.Bool[ArrayT, "*b qs"] | None = None
    response_action_history: at.Float[ArrayT, "*b us rd"] | None = None
    response_action_history_mask: at.Bool[ArrayT, "*b us"] | None = None
    embodiment_global_description: at.Float[ArrayT, "*b gfeat"] | None = None
    embodiment_global_description_availability: at.Bool[ArrayT, "*b gfeat"] | None = None

    # Static Transformer-Transformer RoboTokens. Values are padded only in feature and
    # sequence dimensions; token_mask is the sole sequence-validity authority. Reference
    # IDs use five shared namespaces (link, dynamic joint, fixed joint, actuator, tracked
    # link), with -1 meaning that a token has no reference in that namespace.
    robotoken_values: at.Float[ArrayT, "*b rt rf"] | None = None
    robotoken_types: at.Int[ArrayT, "*b rt"] | None = None
    robotoken_reference_ids: at.Int[ArrayT, "*b rt rr"] | None = None
    robotoken_mask: at.Bool[ArrayT, "*b rt"] | None = None

    # Seconds between chunk entries (the control period), read only by the opt-in action time channel
    # (Pi0Config.action_time_conditioning).
    action_dt_s: at.Float[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            action_loss_mask=data.get("action_loss_mask"),
            action_semantic_mask=data.get("action_semantic_mask"),
            embodiment_id=data.get("embodiment_id"),
            task_id=data.get("task_id"),
            embodiment_description=data.get("embodiment_description"),
            embodiment_description_availability=data.get(
                "embodiment_description_availability"
            ),
            embodiment_control_mask=data.get("embodiment_control_mask"),
            embodiment_control_state=data.get("embodiment_control_state"),
            embodiment_geometry_points=data.get("embodiment_geometry_points"),
            embodiment_geometry_point_mask=data.get(
                "embodiment_geometry_point_mask"
            ),
            response_state_history=data.get("response_state_history"),
            response_state_history_mask=data.get("response_state_history_mask"),
            response_action_history=data.get("response_action_history"),
            response_action_history_mask=data.get("response_action_history_mask"),
            embodiment_global_description=data.get(
                "embodiment_global_description"
            ),
            embodiment_global_description_availability=data.get(
                "embodiment_global_description_availability"
            ),
            robotoken_values=data.get("robotoken_values"),
            robotoken_types=data.get("robotoken_types"),
            robotoken_reference_ids=data.get("robotoken_reference_ids"),
            robotoken_mask=data.get("robotoken_mask"),
            action_dt_s=data.get("action_dt_s"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    augmentation_mode: str = "pi05_native",
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train and augmentation_mode != "none":
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                if augmentation_mode == "shared_mild":
                    # Uniform-in-log zoom from 1.0 to 1/sqrt(.90) gives crop areas in
                    # [0.90, 1.0]. Unit aspect lies inside the agreed [0.95, 1.05] band and
                    # keeps static JIT shapes. Rotation and flips stay off.
                    transforms += [
                        augmax.RandomSizedCrop(width, height, zoom_range=(1.0, 1.0 / np.sqrt(0.90))),
                    ]
                else:
                    transforms += [
                        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                        augmax.Resize(width, height),
                        augmax.Rotate((-5, 5)),
                    ]
            transforms += [
                augmax.ColorJitter(
                    brightness=0.1 if augmentation_mode == "shared_mild" else 0.3,
                    contrast=0.1 if augmentation_mode == "shared_mild" else 0.4,
                    saturation=0.1 if augmentation_mode == "shared_mild" else 0.5,
                    hue=0.03 if augmentation_mode == "shared_mild" else 0.1,
                ),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        action_loss_mask=observation.action_loss_mask,
        embodiment_id=observation.embodiment_id,
        task_id=observation.task_id,
        embodiment_description=observation.embodiment_description,
        embodiment_description_availability=observation.embodiment_description_availability,
        embodiment_control_mask=observation.embodiment_control_mask,
        embodiment_control_state=observation.embodiment_control_state,
        embodiment_geometry_points=observation.embodiment_geometry_points,
        embodiment_geometry_point_mask=observation.embodiment_geometry_point_mask,
        response_state_history=observation.response_state_history,
        response_state_history_mask=observation.response_state_history_mask,
        response_action_history=observation.response_action_history,
        response_action_history_mask=observation.response_action_history_mask,
        embodiment_global_description=observation.embodiment_global_description,
        embodiment_global_description_availability=(
            observation.embodiment_global_description_availability
        ),
        robotoken_values=observation.robotoken_values,
        robotoken_types=observation.robotoken_types,
        robotoken_reference_ids=observation.robotoken_reference_ids,
        robotoken_mask=observation.robotoken_mask,
        action_dt_s=observation.action_dt_s,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            expected = state.to_pure_dict()
            params = ocp.transform_utils.intersect_trees(expected, params)
            # intersect_trees drops None leaves (JAX treats None as an empty subtree), such as the bias placeholder of
            # the bias-free action_time_proj; put them back. A no-op for models without None leaves.
            none_paths = [path for path, value in traverse_util.flatten_dict(expected).items() if value is None]
            if none_paths:
                params = traverse_util.unflatten_dict(
                    {**traverse_util.flatten_dict(params), **dict.fromkeys(none_paths)}
                )
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = (
        pathlib.Path(params_path).resolve()
        if not str(params_path).startswith(("gs://", "/bigstore/", "/gcs/"))
        else params_path
    )

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
