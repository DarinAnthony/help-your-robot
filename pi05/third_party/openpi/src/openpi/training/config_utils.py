"""Reusable native OpenPI training configuration types and data factories.

Experiment registration belongs in :mod:`openpi.training.config`; this module owns only
the general contracts used by loaders, policies, and registered configurations.
"""

import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from pi05_integration import data_processing as _data_processing
from pi05_integration import behavior_prompt_overlay as _behavior_prompt_overlay
from pi05_integration import embodiment_bundle as _embodiment_bundle
from pi05_integration import input_config as _input_config
from pi05_integration import mobile_base as _mobile_base
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # Immutable provenance/processor record copied beside norm_stats in each checkpoint.
    normalization_bundle: dict[str, Any] | None = None
    # Immutable URMA description bundle copied beside normalization assets.
    embodiment_description_bundle: dict[str, str] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Typed Pi0.5 input contract. Compatibility properties below retain the old scalar API for
    # inference and loader callers while registered recipes use explicit strategy objects.
    input_config: _input_config.Pi05InputConfig = dataclasses.field(default_factory=_input_config.Pi05InputConfig)
    # Training-only target probability for observations after the current absolute gripper command
    # changes to closed. None preserves frame-uniform sampling. Validation and normalization loaders
    # use shuffle=False and retain the natural distribution.
    post_close_sampling_weight: float | None = None
    # Training-only threshold below which a frame's masked action-chunk norm counts as static and
    # the frame is dropped. Default matches the value every completed run used.
    static_action_threshold: float = 1e-4
    # Optional subset of stable global embodiment IDs. Filtering does not remap the IDs.
    embodiment_filter_ids: tuple[int, ...] = ()
    # TFDS split expression. Keep train/validation disjoint while using the same loader.
    rlds_split: str = "train"
    # Typed episode partition. None preserves the historical rlds_split behavior.
    episode_selector: _data_processing.EpisodeSelector | None = None
    episode_split: _data_processing.EpisodeSplit | None = None
    episode_split_role: Literal["train", "validation"] = "train"
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()
    # Hash-pinned, component-complete task-language overlays applied before tokenization.
    behavior_prompt_overlays: tuple[_behavior_prompt_overlay.BehaviorPromptOverlay, ...] = ()
    rlds_input_sampling: _data_processing.RldsInputSampling = dataclasses.field(
        default_factory=_data_processing.default_rlds_input_sampling
    )
    rlds_sampler_seed: int | None = None

    @property
    def embodiment_conditioning(self) -> str:
        return self.input_config.prompt.legacy_mode

    @property
    def embodiment_state_ids(self) -> tuple[int, ...]:
        return self.input_config.state.embodiment_ids

    @property
    def embodiment_normalization_ids(self) -> tuple[int, ...]:
        return self.input_config.normalization.embodiment_ids

    @property
    def embodiment_normalization(self) -> bool:
        return self.input_config.normalization.per_embodiment


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    fail_on_prompt_truncation: bool = False

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            fail_on_overflow=self.fail_on_prompt_truncation,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                if model_config.enable_langact_training:
                    # pi0.5 dual objective: the language head is additionally supervised on FAST
                    # action tokens, so the prompt stream must carry them along with the
                    # token_loss_mask / token_ar_mask that select and causally order them.
                    return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            _transforms.TokenizeFASTInputs(
                                _tokenizer.FASTTokenizer(model_config.max_token_len),
                            ),
                            _transforms.PadStatesAndActions(model_config.action_dim),
                        ],
                    )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            fail_on_overflow=self.fail_on_prompt_truncation,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    normalization_bundle: dict[str, Any] | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            normalization_bundle=self.normalization_bundle,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        if (
            self.input_config.prompt.fail_on_token_overflow
            and self.input_config.prompt.max_token_len != model_config.max_token_len
        ):
            raise ValueError(
                "URMA4 bundle-text prompt max_token_len must equal the model token budget: "
                f"{self.input_config.prompt.max_token_len} != {model_config.max_token_len}"
            )
        model_transforms = ModelTransformFactory(
            fail_on_prompt_truncation=self.input_config.prompt.fail_on_token_overflow
        )(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if (
            self.input_config.response_history
            != model_config.response_history_conditioning
        ):
            raise ValueError(
                "Pi input and model response-history conditioning must be identical"
            )
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Required experiment name. Used for W&B and the metadata/checkpoint directories, including
    # resumed runs, so W&B never falls back to a random display name.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    # No silent default: UnspecifiedWeightLoader refuses to train, so every config states its
    # init (NoOpWeightLoader for deliberate random init).
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.UnspecifiedWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size per loader step, i.e. what actually lands on the devices. With
    # gradient_accumulation_steps > 1 the *effective* batch is
    # batch_size * gradient_accumulation_steps; this field stays the microbatch, so per-device
    # memory is always batch_size / fsdp_devices regardless of accumulation.
    batch_size: int = 32
    # Number of microbatches to accumulate before each optimizer update (1 disables accumulation).
    # Implemented with optax.MultiSteps; the trainer runs all k microbatches inside one step, so
    # every step count -- num_train_steps, save/val/log intervals, warmup_steps/decay_steps, the
    # checkpoint step numbers -- is in optimizer updates.
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # TensorFlow RLDS input parallelism. ``-1`` preserves AUTOTUNE for production runs;
    # bounded diagnostics can set these explicitly to isolate native-reader failures.
    rlds_num_parallel_reads: int = -1
    rlds_num_parallel_calls: int = -1
    # Host input-pipeline tuning (throughput only; frame content and order are unchanged).
    # Historical defaults: a 1 GB tf.data autotune RAM budget (dlimp takes GB) and no terminal prefetch.
    rlds_ram_budget_gb: int = 1
    rlds_final_prefetch: int = 0
    # Permit an explicitly recorded distributional restart when V3 topology or
    # software versions differ from the checkpoint's exact-replay contract.
    rlds_input_topology_restart: bool = False
    # Permit a resume on dataset files that differ from the ones the run recorded at its last start
    # (finetuning or continuing on changed data). The change is recorded, and V3 restarts its
    # stream at a derived seed instead of replaying the old position over the new files.
    rlds_input_dataset_restart: bool = False
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # Validation uses the held-out RLDS slice from ``rlds_split`` (MolmoSpaces defaults
    # to train[:95%]). Set val_interval=0 to disable.
    val_interval: int = 500
    val_batches: int = 2
    # Full sampled-action MSE on a training batch. This is more expensive than surrogate
    # flow loss, so it runs at a coarser cadence like Octo's sample_loss.
    train_sample_interval: int = 500
    # One consolidated W&B panel containing actual train-mode visual inputs, model tokens,
    # state/conditioning partitions, targets, masks, and sampled actions.
    data_visualization: _input_config.DataVisualizationConfig | None = dataclasses.field(
        default_factory=_input_config.DataVisualizationConfig
    )
    # Bound the startup dataset-mixture audit cost (episodes scanned per dataset).
    mixture_report_max_episodes: int = 1000
    # Checkpoint cadence, both in completed updates. A checkpoint is written every
    # ``save_interval`` updates; one whose count is a multiple of
    # ``keep_every_n_completed_updates`` (and the final step) is kept as a milestone
    # <checkpoint_dir>/<N>. Every other save goes to <checkpoint_dir>/resume, which holds only its
    # newest step and is skipped rather than waited for while an earlier save is still writing, so
    # a short ``save_interval`` bounds the work a crash loses without accumulating checkpoints.
    save_interval: int = 1000
    keep_every_n_completed_updates: int | None = 4000
    # Legacy Orbax retention on the zero-based step index, kept for older configs. When set, every
    # save is a milestone, Orbax keeps those with step % keep_period == 0, and
    # ``keep_every_n_completed_updates`` is ignored.
    keep_period: int | None = None
    # Opt-in checkpoint layout (operational; the training trajectory is unchanged). When true, the
    # off-grid ``save_interval`` saves are full checkpoints (parameters, optimizer state, assets) in
    # <checkpoint_dir>/<N> instead of the resume tier, so every save can be evaluated; an off-grid one
    # is kept only until the next save commits. A resume tier an earlier start wrote is still restored
    # from and then cleared. Set per launch with ``molmo_spaces.training.run train --unified-checkpoints``.
    unified_checkpoints: bool = False

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def wandb_name(self) -> str:
        """Return the explicit experiment name used for the W&B run."""
        if not isinstance(self.exp_name, str) or not self.exp_name.strip():
            raise ValueError("--exp_name must be set to a non-empty explicit W&B run name")
        return self.exp_name

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if "://" in str(self.checkpoint_base_dir) or str(self.checkpoint_base_dir).startswith(("/bigstore/", "/gcs/")):  # gs-aware
            return epath.Path(self.checkpoint_base_dir) / self.name / self.wandb_name
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.wandb_name).resolve()

    @property
    def keep_interval(self) -> int | None:
        """The milestone grid in completed updates; None when legacy ``keep_period`` applies."""
        return None if self.keep_period is not None else self.keep_every_n_completed_updates

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

    @property
    def visualization_interval(self) -> int:
        """Compatibility accessor used by repository/native config validation."""
        return 0 if self.data_visualization is None else self.data_visualization.interval

    @property
    def visualization_samples(self) -> int:
        return 0 if self.data_visualization is None else self.data_visualization.sample_count


@dataclasses.dataclass(frozen=True)
class RLDSMolmoSpacesDataConfig(DataConfigFactory):
    """RLDS streaming config for our molmospaces rlds_v3 (reuses LIBERO transforms)."""

    rlds_data_dir: str | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()
    behavior_prompt_overlays: tuple[_behavior_prompt_overlay.BehaviorPromptOverlay, ...] = ()
    input_config: _input_config.Pi05InputConfig = dataclasses.field(default_factory=_input_config.Pi05InputConfig)
    post_close_sampling_weight: float | None = None
    static_action_threshold: float = 1e-4
    embodiment_filter_ids: tuple[int, ...] = ()
    rlds_split: str = "train[:95%]"
    episode_selector: _data_processing.EpisodeSelector | None = None
    episode_split: _data_processing.EpisodeSplit | None = None
    rlds_input_sampling: _data_processing.RldsInputSampling = dataclasses.field(
        default_factory=_data_processing.default_rlds_input_sampling
    )
    rlds_sampler_seed: int | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.input_config.action_time != getattr(
            model_config, "action_time_conditioning", _input_config.NoActionTime()
        ):
            raise ValueError(
                "data input_config.action_time and model action_time_conditioning disagree: "
                f"{self.input_config.action_time} != {getattr(model_config, 'action_time_conditioning', None)}"
            )
        description_conditioning = (
            model_config.backbone_conditioning
            if (
                model_config.backbone_conditioning.uses_embodiment_description
                or model_config.backbone_conditioning.uses_embodiment_codebook
                or model_config.backbone_conditioning.uses_robotokens
            )
            else model_config.action_expert_conditioning
        )
        extractor_name = type(
            getattr(description_conditioning, "extractor", None)
        ).__name__
        dynamic_fields = {
            "Urma4DynamicEmbodimentExtractor": (
                "dynamic_urma_values",
                "dynamic_urma_mask",
            ),
            "Urma5EmbodimentExtractor": ("dynamic_urma5_values",),
            "Urma6EmbodimentExtractor": ("dynamic_urma5_values",),
            "Urma51EmbodimentExtractor": (
                "dynamic_urma51_values",
                "dynamic_urma51_analytic_mask",
            ),
        }.get(extractor_name, ())
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "state_mask": "state_mask",
                        "actions": "actions",
                        "action_loss_mask": "action_loss_mask",
                        "action_semantic_mask": "action_semantic_mask",
                        "prompt": "prompt",
                        "embodiment_id": "embodiment_id",
                        "dataset_component_id": "dataset_component_id",
                        "task_id": "task_id",
                        **(
                            {
                                "response_state_history": "response_state_history",
                                "response_state_history_mask": "response_state_history_mask",
                                "response_action_history": "response_action_history",
                                "response_action_history_mask": "response_action_history_mask",
                            }
                            if self.input_config.response_history.enabled
                            else {}
                        ),
                        **{key: key for key in dynamic_fields},
                        **(
                            {key: key for key in _mobile_base.BASE_KEYS}
                            if self.input_config.mobile_base.enabled
                            else {}
                        ),
                        **({"action_dt_s": "action_dt_s"} if self.input_config.action_time.enabled else {}),
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        if (
            self.input_config.prompt.fail_on_token_overflow
            and self.input_config.prompt.max_token_len != model_config.max_token_len
        ):
            raise ValueError(
                "URMA4 bundle-text prompt max_token_len must equal the model token budget: "
                f"{self.input_config.prompt.max_token_len} != {model_config.max_token_len}"
            )
        model_transforms = ModelTransformFactory(
            fail_on_prompt_truncation=self.input_config.prompt.fail_on_token_overflow
        )(model_config)
        description_transform = description_conditioning.create_description_transform(
            _embodiment_bundle
        )
        if getattr(getattr(description_conditioning, "extractor", None),
                   "attach_before_numeric_normalization", False):
            # Physical-state URMA4 reads q/aperture before the numeric normalizer, so its
            # control state is independent of the (grouped) state/action statistics.
            if description_transform is None:
                raise ValueError("physical-state descriptor attachment requires a description transform")
            data_transforms = data_transforms.push(inputs=[description_transform])
            description_transform = None
        state_serialization_transform = (
            self.input_config.state_serialization.create_transform(_transforms)
        )
        state_transform = self.input_config.state.create_transform(_transforms)
        # Base slots go after the arm-only action zeroing (it needs the left-packed arm mask) and
        # before state codes, tokenization and padding (pi05_integration/mobile_base.py).
        mobile_base_transform = (
            _transforms.AppendMobileBase(
                arm_width=_mobile_base.ARM_WIDTH,
                clip_limit=getattr(self.input_config.action_clipping, "limit", None),
            )
            if self.input_config.mobile_base.enabled
            else None
        )
        additional_transforms = [
            transform
            for transform in (
                description_transform,
                _transforms.ZeroInvalidActions(),
                state_serialization_transform,
                mobile_base_transform,
                state_transform,
            )
            if transform is not None
        ]
        model_transforms = _transforms.Group(
            inputs=[
                *additional_transforms,
                *model_transforms.inputs,
                # The loader's mixture-component name (a string) is consumed only by URMA
                # description attachment, which pops it; without one it must not reach the
                # JAX batch (make_array_from_process_local_data rejects bytes arrays).
                _transforms.DropKeys(("dataset_component_id",)),
            ],
            outputs=model_transforms.outputs,
        )
        assert self.rlds_data_dir is not None, "Need rlds_data_dir"
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            rlds_data_dir=self.rlds_data_dir,
            input_config=self.input_config,
            # Pi0.5 historically inferred this compatibility flag from model type, which
            # incorrectly reports quantile normalization for typed Gaussian recipes. Keep the
            # legacy field synchronized with the authoritative typed input contract so RTC
            # action-prefix normalization and any future compatibility callers use the same
            # transform as training and ordinary inference.
            use_quantile_norm=self.input_config.normalization.use_quantiles,
            embodiment_description_bundle=(
                self.input_config.prompt.checkpoint_bundle_source(_embodiment_bundle)
                if self.input_config.prompt.uses_description_bundle
                else description_conditioning.checkpoint_bundle_source(_embodiment_bundle)
            ),
            post_close_sampling_weight=self.post_close_sampling_weight,
            static_action_threshold=self.static_action_threshold,
            embodiment_filter_ids=self.embodiment_filter_ids,
            rlds_split=self.rlds_split,
            episode_selector=self.episode_selector,
            episode_split=self.episode_split,
            episode_split_role="train",
            rlds_input_sampling=self.rlds_input_sampling,
            rlds_sampler_seed=self.rlds_sampler_seed,
            datasets=self.datasets,
            behavior_prompt_overlays=self.behavior_prompt_overlays,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

    @property
    def embodiment_conditioning(self) -> str:
        return self.input_config.prompt.legacy_mode

    @property
    def embodiment_state_ids(self) -> tuple[int, ...]:
        return self.input_config.state.embodiment_ids

    @property
    def embodiment_normalization_ids(self) -> tuple[int, ...]:
        return self.input_config.normalization.embodiment_ids

    @property
    def embodiment_normalization(self) -> bool:
        return self.input_config.normalization.per_embodiment
