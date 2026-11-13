from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _gripper_endpoints(input_config) -> dict | None:
    """The sha-pinned per-robot endpoints of a ``GripperClosedness`` with a table; None otherwise."""
    value = getattr(input_config, "gripper_value", None)
    if value is None or not value.enabled or not getattr(value, "endpoints_table", ""):
        return None
    if getattr(getattr(input_config, "response_history", None), "enabled", False):
        # The response histories carry the aperture in metres.
        raise ValueError("gripper_closedness_v1 at load does not support response-history conditioning")
    from pi05_integration.gripper_value import load_endpoints_table

    return load_endpoints_table(value.endpoints_table, value.endpoints_table_sha256)


def _joint_frames(input_config) -> dict | None:
    """The sha-pinned per-robot (sign, zero) of a ``CanonicalJointFrame``; None otherwise."""
    frame = getattr(input_config, "joint_frame", None)
    if frame is None or not frame.enabled:
        return None
    from pi05_integration.joint_frame import load_table

    return load_table(frame.table, frame.table_sha256)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int | None,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    deterministic: bool = False,
    shuffle_buffer_size: int = 50_000,
    num_parallel_reads: int = -1,
    num_parallel_calls: int = -1,
    repeat: bool = True,
    decode_images: bool = True,
    apply_behavior_prompt_overlay: bool = True,
    process_count: int = 1,
    process_index: int = 0,
    completed_updates: int = 0,
    input_sampling_seed: int | None = None,
    final_prefetch: int = 0,
    ram_budget_gb: int = 1,
) -> Dataset:
    # Dispatch: RLDSMolmoSpacesDataConfig never sets action_space (only RLDSDroidDataConfig does),
    # so this discriminates by which DataConfigFactory actually built data_config -- not by
    # dataset names -- letting a droid-only entry built via RLDSMolmoSpacesDataConfig still go
    # through our streaming loader (flat schema, delta-joint conversion, pick-and-place filter),
    # while legacy RLDSDroidDataConfig configs (whose repack_transforms expect the raw DROID
    # schema) keep using the original DROID-only loader below.
    if data_config.datasets and getattr(data_config, "action_space", None) is None:
        from openpi.training.molmospaces_rlds_dataset import MolmoSpacesRldsDataset

        return MolmoSpacesRldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=batch_size,
            datasets=data_config.datasets,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            split=data_config.rlds_split,
            episode_selector=data_config.episode_selector,
            episode_split=data_config.episode_split,
            episode_split_role=data_config.episode_split_role,
            prompt_conditioning=data_config.input_config.prompt,
            post_close_sampling_weight=data_config.post_close_sampling_weight,
            static_action_threshold=getattr(data_config, "static_action_threshold", 1e-4),
            embodiment_filter_ids=data_config.embodiment_filter_ids,
            seed=seed,
            deterministic=deterministic,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            repeat=repeat,
            decode_images=decode_images,
            behavior_prompt_overlays=(
                getattr(data_config, "behavior_prompt_overlays", ())
                if apply_behavior_prompt_overlay
                else ()
            ),
            action_loss_coordinates=getattr(
                data_config.input_config, "action_loss_coordinates", "model"
            ),
            rlds_input_sampling=getattr(data_config, "rlds_input_sampling", None),
            process_count=process_count,
            process_index=process_index,
            completed_updates=completed_updates,
            input_sampling_seed=input_sampling_seed,
            final_prefetch=final_prefetch,
            ram_budget_gb=ram_budget_gb,
            # Opt-in holonomic-base fields; the default emits none (pi05_integration/mobile_base.py).
            mobile_base=data_config.input_config.mobile_base.contract,
            mobile_base_dt_s=getattr(data_config.input_config.mobile_base, "dt_s", None),
            # Chunk labels; the default per-step joint delta changes nothing (pi05_integration/arm_action.py).
            arm_action=data_config.input_config.arm_action.contract,
            # Per-frame control period for the action time channel; off by default.
            action_time=data_config.input_config.action_time.enabled,
            # Sim-frame geometry; the default keeps the stored frames (pi05_integration/image_geometry.py).
            letterbox_source=(
                (data_config.input_config.image_geometry.source_width,
                 data_config.input_config.image_geometry.source_height)
                if getattr(getattr(data_config.input_config, "image_geometry", None), "enabled", False)
                else None
            ),
            # Gripper state converted to closedness at load; the default keeps the stored state
            # (pi05_integration/gripper_value.py).
            gripper_endpoints=_gripper_endpoints(data_config.input_config),
            # Arm joints mapped into the canonical joint frame at load; the default keeps each robot's own joints
            # (pi05_integration/joint_frame.py).
            joint_frames=_joint_frames(data_config.input_config),
        )
    if action_horizon is None:
        raise ValueError("unchunked normalization is only implemented for MolmoSpaces RLDS data")
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    normalizer = data_config.input_config.normalization.create_normalizer(
        _transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    response_history_normalizer = data_config.input_config.response_history.create_normalizer(
        _transforms, norm_stats
    )
    mobile_base_normalizer = data_config.input_config.mobile_base.create_normalizer(
        _transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        state_quantiles=data_config.input_config.normalization.state_quantiles,
    )
    action_clipper = (
        None
        if skip_norm_stats
        else data_config.input_config.action_clipping.create_transform(_transforms)
    )
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            normalizer,
            *([] if response_history_normalizer is None else [response_history_normalizer]),
            *([] if mobile_base_normalizer is None else [mobile_base_normalizer]),
            *([] if action_clipper is None else [action_clipper]),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    normalizer = data_config.input_config.normalization.create_normalizer(
        _transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    response_history_normalizer = data_config.input_config.response_history.create_normalizer(
        _transforms, norm_stats
    )
    mobile_base_normalizer = data_config.input_config.mobile_base.create_normalizer(
        _transforms,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        state_quantiles=data_config.input_config.normalization.state_quantiles,
    )
    action_clipper = (
        None
        if skip_norm_stats
        else data_config.input_config.action_clipping.create_transform(_transforms)
    )
    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            normalizer,
            *([] if response_history_normalizer is None else [response_history_normalizer]),
            *([] if mobile_base_normalizer is None else [mobile_base_normalizer]),
            *([] if action_clipper is None else [action_clipper]),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    completed_updates: int = 0,
    input_sampling_seed: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
            seed=config.seed,
            num_parallel_reads=config.rlds_num_parallel_reads,
            num_parallel_calls=config.rlds_num_parallel_calls,
            completed_updates=completed_updates,
            input_sampling_seed=input_sampling_seed,
            final_prefetch=config.rlds_final_prefetch,
            ram_budget_gb=config.rlds_ram_budget_gb,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    process_count = jax.process_count()
    process_index = jax.process_index()
    if batch_size % process_count != 0:
        raise ValueError(f"Global batch size ({batch_size}) must be divisible by JAX process count ({process_count}).")
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // process_count
        if process_count > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=process_count,
                rank=process_index,
                shuffle=shuffle,
                drop_last=True,
                seed=seed,
            )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
    seed: int = 0,
    deterministic: bool = False,
    shuffle_buffer_size: int = 50_000,
    num_parallel_reads: int = -1,
    num_parallel_calls: int = -1,
    completed_updates: int = 0,
    input_sampling_seed: int | None = None,
    final_prefetch: int = 0,
    ram_budget_gb: int = 1,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    process_count = jax.process_count()
    if batch_size % process_count != 0:
        raise ValueError(f"Global batch size ({batch_size}) must be divisible by JAX process count ({process_count}).")
    local_batch_size = batch_size // process_count
    # Each process contributes one local shard to the global array. Offset the
    # deterministic stream so workers do not contribute duplicate examples.
    process_seed = seed + jax.process_index()
    input_sampling_seed = (
        data_config.rlds_sampler_seed
        if input_sampling_seed is None
        else input_sampling_seed
    )
    if input_sampling_seed is None:
        input_sampling_seed = seed
    from pi05_integration.data_processing import DeterministicReplayV3

    if isinstance(data_config.rlds_input_sampling, DeterministicReplayV3):
        deterministic = True
    dataset = create_rlds_dataset(
        data_config,
        action_horizon,
        local_batch_size,
        shuffle=shuffle,
        seed=process_seed,
        deterministic=deterministic,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_reads=num_parallel_reads,
        num_parallel_calls=num_parallel_calls,
        process_count=process_count,
        process_index=jax.process_index(),
        completed_updates=completed_updates,
        input_sampling_seed=input_sampling_seed,
        final_prefetch=final_prefetch,
        ram_budget_gb=ram_budget_gb,
    )
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
