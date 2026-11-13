"""Typed data-distribution processing configured by Pi training experiments."""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable


class RldsInputSampling(abc.ABC):
    """Versioned contract for ordering and distributing RLDS input."""

    @abc.abstractmethod
    def to_record(self) -> dict:
        """Return the persistent scientific identity of this sampling contract."""


@dataclasses.dataclass(frozen=True)
class LegacyRldsInputSampling(RldsInputSampling):
    """Preserve the historical unsharded traversal plus frame-buffer shuffle."""

    def to_record(self) -> dict:
        return {"type": type(self).__name__}


@dataclasses.dataclass(frozen=True)
class ProcessShardedFileShuffleV1(RldsInputSampling):
    """Shared deterministic TFDS-file permutation followed by process sharding.

    Only file identities are permuted before records are fetched. On recovery the
    permutation is deterministically changed from the completed optimizer step;
    this is a versioned distribution contract, not exact iterator restoration.
    """

    def shuffle_seed(self, *, experiment_seed: int, completed_updates: int) -> int:
        if completed_updates < 0:
            raise ValueError("completed_updates must be non-negative")
        digest = hashlib.sha256(
            f"process_sharded_file_shuffle_v1\0{experiment_seed}\0{completed_updates}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def file_order(
        self,
        num_files: int,
        *,
        experiment_seed: int,
        completed_updates: int,
    ) -> tuple[int, ...]:
        if num_files < 1:
            raise ValueError("num_files must be positive")
        order = list(range(num_files))
        random.Random(
            self.shuffle_seed(
                experiment_seed=experiment_seed,
                completed_updates=completed_updates,
            )
        ).shuffle(order)
        return tuple(order)

    def validate_restored_completed_updates(
        self, *, restored_completed_updates: int, seeded_completed_updates: int
    ) -> None:
        if restored_completed_updates != seeded_completed_updates:
            raise ValueError(
                "restored train_state.step does not match the completed-update "
                "identity used to seed ProcessShardedFileShuffleV1: "
                f"{restored_completed_updates} != {seeded_completed_updates}"
            )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "version": 1,
            "pre_fetch_permutation": "tfds_file_identities",
            "distribution": "shared_python_permutation_then_tfds_input_context_process_shard",
            "restart": "new_shuffle_seed_from_completed_optimizer_updates",
            "exact_uninterrupted_resume": False,
            "topology_change": "repartitions_permutation_and_changes_distribution",
        }


@dataclasses.dataclass(frozen=True)
class ProcessShardedFileShuffleV2(ProcessShardedFileShuffleV1):
    """Process-sharded RLDS input: the repository default for every new experiment.

    Distribution per training segment:

    1. one shared deterministic permutation of the TFDS file identities, seeded from
       ``(experiment_seed, completed_updates)`` exactly as in V1;
    2. TFDS ``InputContext`` modulo sharding, so the ``process_count`` JAX processes read
       disjoint file cohorts whose union is the whole split (TFDS fails closed when a
       process would receive no file);
    3. a seeded per-process reshuffle of that cohort on every pass, so repeated passes
       do not present the same file order to the downstream frame buffer.

    Membership of each cohort is identical to V1 for the same seed and topology; only the
    within-cohort order changes across passes. Restart semantics are inherited from V1: a
    resume reseeds the permutation from the completed optimizer updates and fails closed if
    the restored step disagrees. Works unchanged for one process (GPU) or many (TPU pods);
    changing the process count changes the distribution and must be recorded.
    """

    def process_file_shuffle_seed(
        self, *, experiment_seed: int, completed_updates: int, process_index: int
    ) -> int:
        if process_index < 0:
            raise ValueError("process_index must be non-negative")
        digest = hashlib.sha256(
            "process_sharded_file_shuffle_v2_cohort\0"
            f"{experiment_seed}\0{completed_updates}\0{process_index}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def to_record(self) -> dict:
        return {
            **super().to_record(),
            "type": type(self).__name__,
            "version": 2,
            "cohort_epoch_reshuffle": "seeded_per_process_tfds_file_shuffle_each_iteration",
        }


@dataclasses.dataclass(frozen=True)
class DeterministicReplayV3(RldsInputSampling):
    """Deterministic process-sharded input with replay-and-skip continuation.

    Unlike V1/V2, every ordering seed is a function of the sampler seed and process
    index only.  A same-topology resume therefore reconstructs the original encoded
    frame stream and advances it by the exact number of frames already consumed.
    """

    interleave_cycle_length: int = 16
    interleave_block_length: int = 16
    flatten_cycle_length: int = 16
    frame_shuffle_buffer_size: int = 50_000

    def __post_init__(self) -> None:
        for name in (
            "interleave_cycle_length",
            "interleave_block_length",
            "flatten_cycle_length",
            "frame_shuffle_buffer_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _seed(*, purpose: str, sampler_seed: int, process_index: int) -> int:
        if process_index < 0:
            raise ValueError("process_index must be non-negative")
        digest = hashlib.sha256(
            f"deterministic_replay_v3\0{purpose}\0{sampler_seed}\0{process_index}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def file_order(self, num_files: int, *, sampler_seed: int) -> tuple[int, ...]:
        if num_files < 1:
            raise ValueError("num_files must be positive")
        order = list(range(num_files))
        random.Random(
            self._seed(purpose="file_order", sampler_seed=sampler_seed, process_index=0)
        ).shuffle(order)
        return tuple(order)

    def process_file_shuffle_seed(self, *, sampler_seed: int, process_index: int) -> int:
        return self._seed(
            purpose="process_file_shuffle",
            sampler_seed=sampler_seed,
            process_index=process_index,
        )

    def frame_shuffle_seed(self, *, sampler_seed: int, process_index: int) -> int:
        return self._seed(
            purpose="frame_shuffle",
            sampler_seed=sampler_seed,
            process_index=process_index,
        )

    def mixture_seed(self, *, sampler_seed: int, process_index: int, purpose: str) -> int:
        if not purpose:
            raise ValueError("mixture seed purpose must be nonempty")
        return self._seed(
            purpose=purpose,
            sampler_seed=sampler_seed,
            process_index=process_index,
        )

    def skip_frames(self, *, completed_updates: int, local_batch_size: int) -> int:
        if completed_updates < 0:
            raise ValueError("completed_updates must be non-negative")
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        return completed_updates * local_batch_size

    def validate_restored_completed_updates(
        self, *, restored_completed_updates: int, seeded_completed_updates: int
    ) -> None:
        if restored_completed_updates != seeded_completed_updates:
            raise ValueError(
                "restored train_state.step does not match the completed-update cursor "
                "used for DeterministicReplayV3 replay: "
                f"{restored_completed_updates} != {seeded_completed_updates}"
            )

    def validate_resume_topology(
        self,
        *,
        recorded: dict,
        process_count: int,
        local_batch_size: int,
        sampler_seed: int,
        allow_topology_restart: bool,
    ) -> bool:
        """Validate exact-replay inputs; return whether a distributional restart is needed."""
        expected = {
            "input_process_count": process_count,
            "input_local_batch_size": local_batch_size,
            "input_sampler_seed": sampler_seed,
        }
        mismatches = {
            key: (recorded.get(key), value)
            for key, value in expected.items()
            if recorded.get(key) != value
        }
        if not mismatches:
            return False
        if not allow_topology_restart:
            details = ", ".join(
                f"{key}={old!r} -> {new!r}" for key, (old, new) in mismatches.items()
            )
            raise ValueError(
                "DeterministicReplayV3 exact resume requires unchanged input topology "
                f"and sampler seed ({details}); set rlds_input_topology_restart=True "
                "to record an explicit distributional restart"
            )
        return True

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "version": 3,
            "pre_fetch_permutation": "tfds_file_identities_seeded_by_sampler_seed",
            "distribution": "shared_python_permutation_then_tfds_input_context_process_shard",
            "cohort_epoch_reshuffle": "seeded_per_process_tfds_file_shuffle_each_iteration",
            "restart": "exact_replay_skip_completed_updates_times_local_batch",
            "exact_uninterrupted_resume": True,
            "topology_change": "fail_closed_unless_recorded_distributional_restart",
            "deterministic_tf_data": True,
            "decode_after_shuffle": True,
            "interleave_cycle_length": self.interleave_cycle_length,
            "interleave_block_length": self.interleave_block_length,
            "flatten_cycle_length": self.flatten_cycle_length,
            "frame_shuffle_buffer_size": self.frame_shuffle_buffer_size,
        }


def default_rlds_input_sampling() -> RldsInputSampling:
    """The input contract every experiment gets unless it explicitly chooses another.

    ``DeterministicReplayV3`` shards both the training and the validation reader across
    JAX processes and continues the exact frame stream on a same-topology resume. It
    replaced ``ProcessShardedFileShuffleV2`` as the default on 2026-09-06: V2 sharded the
    training reader but restarted the stream from a reseeded cohort head on every resume
    and left the validation reader unsharded. The historical ``LegacyRldsInputSampling``
    traversed the same ordered stream on every process (global batch = ``process_count``
    near-copies of one local batch) and never reached the tail of the file list in a
    40K-update run. Pre-fix experiments keep it only through an explicit, tested pin in
    ``molmo_spaces.training.legacy_rlds_input_sampling``.
    """
    return DeterministicReplayV3()


def recorded_input_sampling(checkpoint_input_contract: dict | None) -> dict:
    """Return the sampling record a checkpoint directory was trained with.

    Checkpoints written before the contract was recorded were all produced by the legacy
    unsharded traversal, so a missing key means ``LegacyRldsInputSampling``.
    """
    if checkpoint_input_contract is None:
        return LegacyRldsInputSampling().to_record()
    return checkpoint_input_contract.get(
        "rlds_input_sampling", LegacyRldsInputSampling().to_record()
    )


def validate_resumed_input_sampling(*, recorded: dict, current: RldsInputSampling) -> None:
    """Fail closed when a resume would silently change the input distribution contract."""
    if recorded != current.to_record():
        raise ValueError(
            "resume would change the RLDS input sampling contract: the checkpoint was "
            f"trained with {recorded.get('type')} but this config resolves to "
            f"{type(current).__name__}. Pin the experiment's rlds_input_sampling to the "
            "recorded contract (LegacyRldsInputSampling for pre-2026-09-05 lineages) to "
            "continue that lineage, or start a new named experiment."
        )


def replay_source_group(episode_key: str) -> str:
    """The saved state trajectory an RLDS episode key renders.

    ``mlspaces_to_rlds.py`` keys a replay record ``<source path>@<replay episode id>`` (plus ``#<space>`` when it
    packs both action spaces), so every visual look (``<source>.<variant>.s<NNN>``) and action space of one
    trajectory shares the part before the last ``@``. A key without ``@`` (non-replay data) is its own group.
    Splits that hold out individual episodes group by this, so looks of one trajectory never straddle train and
    validation.
    """
    head, separator, _ = episode_key.rpartition("@")
    return head if separator else episode_key


@dataclasses.dataclass(frozen=True)
class EpisodeIdentity:
    """Stable metadata needed to assign one RLDS episode to train or validation."""

    embodiment_id: int
    episode_key: str
    pose_id: str = ""
    planner_seed: int = -1
    frame_count: int = -1
    shard_index: int = -1
    # Dataset-component namespace.  Required when independently published RLDS
    # components may reuse local embodiment IDs.
    namespace: str = ""


@dataclasses.dataclass(frozen=True)
class RldsDatasetComponent:
    """One independently registered RLDS build in a typed training mixture."""

    dataset_id: str
    data_subdir: str
    name: str = "molmospaces_emb_mix"
    version: str = "1.0.0"
    weight: float = 1.0
    joint_layout: str | None = None
    target_joint_layout: str | None = None
    # Opt-in: added to the component's raw embodiment IDs (``RLDSDataset.embodiment_id_offset``) so
    # per-family builds that reuse local IDs get disjoint model-facing IDs. 0 keeps the raw IDs.
    embodiment_id_offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.embodiment_id_offset, bool) or not isinstance(self.embodiment_id_offset, int) \
                or self.embodiment_id_offset < 0:
            raise ValueError("dataset component embodiment_id_offset must be a non-negative integer")
        if not self.dataset_id:
            raise ValueError("dataset component ID must be nonempty")
        if not self.data_subdir or self.data_subdir.startswith("/"):
            raise ValueError("dataset component data_subdir must be nonempty and relative")
        if not self.name or not self.version:
            raise ValueError("dataset component name and version must be nonempty")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("dataset component weight must be finite and positive")

    def to_record(self) -> dict:
        record = {
            "dataset_id": self.dataset_id,
            "data_subdir": self.data_subdir,
            "name": self.name,
            "version": self.version,
            "weight": self.weight,
        }
        if self.joint_layout is not None:
            record["joint_layout"] = self.joint_layout
        if self.target_joint_layout is not None:
            record["target_joint_layout"] = self.target_joint_layout
        if self.embodiment_id_offset:
            record["embodiment_id_offset"] = self.embodiment_id_offset
        return record


class EpisodeSelector(abc.ABC):
    """Select an immutable episode-level dataset view before splitting/chunking."""

    @abc.abstractmethod
    def selected_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        """Return stable episode keys included in this dataset view."""

    @abc.abstractmethod
    def to_record(self) -> dict:
        """Return the persistent identity of this selection."""


@dataclasses.dataclass(frozen=True)
class HashEpisodeSelector(EpisodeSelector):
    """Select an exact, order-independent episode budget by stable hash rank.

    Selectors with the same seed form nested views as ``episode_count`` grows. This is
    useful for data-scaling studies that share one physical RLDS build.
    """

    episode_count: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.episode_count < 1:
            raise ValueError("episode_count must be positive")

    def _rank(self, episode: EpisodeIdentity) -> bytes:
        return hashlib.sha256(
            f"{self.seed}\0{episode.episode_key}".encode()
        ).digest()

    def selected_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        episodes = tuple(episodes)
        keys = tuple(episode.episode_key for episode in episodes)
        if any(not key for key in keys):
            raise ValueError("hash episode selection requires stable nonempty episode keys")
        if len(set(keys)) != len(keys):
            raise ValueError("hash episode selection requires unique episode keys")
        if len(episodes) < self.episode_count:
            raise ValueError(
                f"dataset has {len(episodes)} episodes; requested {self.episode_count}"
            )
        return frozenset(
            episode.episode_key
            for episode in sorted(
                episodes,
                key=lambda episode: (self._rank(episode), episode.episode_key),
            )[: self.episode_count]
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "episode_count": self.episode_count,
            "seed": self.seed,
            "identity_key": "episode_metadata.file_path",
            "ranking": "sha256(seed, episode_key)",
        }


@dataclasses.dataclass(frozen=True)
class ExplicitEmbodimentEpisodeSelector(EpisodeSelector):
    """Select complete, explicitly named embodiment groups.

    Unlike ``embodiment_filter_ids``, this scientific selector fails closed when an
    expected embodiment is absent or contributes the wrong episode count.  It is
    intended for dense-per-embodiment scaling studies whose physical RLDS build is
    shared across every milestone.
    """

    embodiment_ids: tuple[int, ...]
    episodes_per_embodiment: int | None = None

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.embodiment_ids)
        if not values or len(set(values)) != len(values):
            raise ValueError("embodiment_ids must be nonempty and unique")
        if any(value < 0 for value in values):
            raise ValueError("embodiment_ids must be nonnegative")
        if self.episodes_per_embodiment is not None and self.episodes_per_embodiment < 1:
            raise ValueError("episodes_per_embodiment must be positive")
        object.__setattr__(self, "embodiment_ids", values)

    def selected_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        episodes = tuple(episodes)
        keys = tuple(str(episode.episode_key) for episode in episodes)
        if any(not key for key in keys):
            raise ValueError("explicit embodiment selection requires stable episode keys")
        if len(set(keys)) != len(keys):
            raise ValueError("explicit embodiment selection requires unique episode keys")
        requested = frozenset(self.embodiment_ids)
        groups: dict[int, list[str]] = defaultdict(list)
        for episode in episodes:
            identity = int(episode.embodiment_id)
            if identity in requested:
                groups[identity].append(str(episode.episode_key))
        missing = requested - groups.keys()
        if missing:
            raise ValueError(
                "dataset is missing requested embodiment IDs: "
                f"{sorted(missing)}"
            )
        if self.episodes_per_embodiment is not None:
            violations = {
                identity: len(groups[identity])
                for identity in self.embodiment_ids
                if len(groups[identity]) != self.episodes_per_embodiment
            }
            if violations:
                raise ValueError(
                    "explicit embodiment selection episode-count mismatch: "
                    f"expected={self.episodes_per_embodiment}, actual={violations}"
                )
        return frozenset(
            key for identity in self.embodiment_ids for key in groups[identity]
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "embodiment_ids": list(self.embodiment_ids),
            "episodes_per_embodiment": self.episodes_per_embodiment,
            "identity_keys": [
                "episode_metadata.embodiment_id",
                "episode_metadata.file_path",
            ],
            "semantics": "complete_explicit_embodiment_groups",
        }


@dataclasses.dataclass(frozen=True)
class NestedHashTrainingEpisodeSelector(EpisodeSelector):
    """Select nested training budgets while preserving one canonical hash holdout.

    The selector sees the complete physical RLDS population. It first identifies the
    canonical embodiment holdout using the same rank as :class:`HashEmbodimentSplit`,
    retains every held-out episode, and then hash-selects an exact nested budget from
    the remaining training episodes. Pair this with ``HashEmbodimentCountSplit`` using
    the same ``validation_episode_count`` and ``split_seed``.
    """

    training_episode_count: int
    validation_episode_count: int
    selection_seed: int = 42
    split_seed: int = 42

    def __post_init__(self) -> None:
        if self.training_episode_count < 1:
            raise ValueError("training_episode_count must be positive")
        if self.validation_episode_count < 1:
            raise ValueError("validation_episode_count must be positive")

    def _split_rank(self, embodiment_id: int) -> tuple[bytes, int]:
        return (
            hashlib.sha256(f"{self.split_seed}\0{embodiment_id}".encode()).digest(),
            embodiment_id,
        )

    def _selection_rank(self, episode: EpisodeIdentity) -> tuple[bytes, str]:
        return (
            hashlib.sha256(
                f"{self.selection_seed}\0{episode.episode_key}".encode()
            ).digest(),
            episode.episode_key,
        )

    def selected_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        episodes = tuple(episodes)
        keys = tuple(str(episode.episode_key) for episode in episodes)
        if any(not key for key in keys):
            raise ValueError("nested hash selection requires stable nonempty episode keys")
        if len(set(keys)) != len(keys):
            raise ValueError("nested hash selection requires unique episode keys")
        counts: dict[int, int] = defaultdict(int)
        for episode in episodes:
            counts[int(episode.embodiment_id)] += 1
        repeated = {
            embodiment_id: count for embodiment_id, count in counts.items() if count != 1
        }
        if repeated:
            raise ValueError(
                "nested hash selection requires exactly one episode per embodiment; "
                f"violations={repeated}"
            )
        if self.validation_episode_count >= len(episodes):
            raise ValueError("validation_episode_count must leave training episodes")
        validation_ids = frozenset(
            sorted(counts, key=self._split_rank)[: self.validation_episode_count]
        )
        training = tuple(
            episode
            for episode in episodes
            if int(episode.embodiment_id) not in validation_ids
        )
        if len(training) < self.training_episode_count:
            raise ValueError(
                f"dataset has {len(training)} canonical training episodes; requested "
                f"{self.training_episode_count}"
            )
        selected_training = sorted(training, key=self._selection_rank)[
            : self.training_episode_count
        ]
        return frozenset(
            episode.episode_key
            for episode in episodes
            if int(episode.embodiment_id) in validation_ids
        ) | frozenset(episode.episode_key for episode in selected_training)

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "training_episode_count": self.training_episode_count,
            "validation_episode_count": self.validation_episode_count,
            "selection_seed": self.selection_seed,
            "split_seed": self.split_seed,
            "identity_keys": [
                "episode_metadata.file_path",
                "episode_metadata.embodiment_id",
            ],
            "selection_ranking": "sha256(selection_seed, episode_key)",
            "holdout_ranking": "sha256(split_seed, embodiment_id)",
            "semantics": "nested_training_budget_plus_canonical_holdout",
        }


class NormalizationEpisodeScope(abc.ABC):
    """Choose which episodes define normalization independently of training sampling."""

    @abc.abstractmethod
    def selector_for(
        self, training_selector: EpisodeSelector | None
    ) -> EpisodeSelector | None: ...

    def split_for(self, training_split: EpisodeSplit | None) -> EpisodeSplit | None:
        return training_split

    def to_record(self) -> dict:
        return {"type": type(self).__name__}


@dataclasses.dataclass(frozen=True)
class TrainingSelectionNormalization(NormalizationEpisodeScope):
    def selector_for(
        self, training_selector: EpisodeSelector | None
    ) -> EpisodeSelector | None:
        return training_selector


@dataclasses.dataclass(frozen=True)
class SelectedViewIncludingValidationNormalization(NormalizationEpisodeScope):
    """Reuse statistics fitted over the selected view before its validation holdout."""

    def selector_for(
        self, training_selector: EpisodeSelector | None
    ) -> EpisodeSelector | None:
        return training_selector

    def split_for(self, training_split: EpisodeSplit | None) -> None:
        del training_split
        return None


@dataclasses.dataclass(frozen=True)
class FullDatasetNormalization(NormalizationEpisodeScope):
    def selector_for(
        self, training_selector: EpisodeSelector | None
    ) -> EpisodeSelector | None:
        del training_selector
        return None


@dataclasses.dataclass(frozen=True)
class ReusedArtifactNormalization(NormalizationEpisodeScope):
    """Statistics deliberately reused from another training set: ``source_selector`` is the episode selection
    that produced them (with the experiment's ``normalization_dataset_repo_id`` naming that set), so the
    artifact's manifest still validates while training samples a different selection."""

    source_selector: EpisodeSelector

    def selector_for(
        self, training_selector: EpisodeSelector | None
    ) -> EpisodeSelector | None:
        del training_selector
        return self.source_selector

    def to_record(self) -> dict:
        return {"type": type(self).__name__, "source_selector": self.source_selector.to_record()}


@dataclasses.dataclass(frozen=True)
class ExplicitSelectionNormalization(NormalizationEpisodeScope):
    """Statistics fitted over a named selection other than the training one.

    A run trained on part of another run's selection (for example a few of its robots) reuses that run's asset:
    the manifest's split identity is checked against ``selector``, not the training selector.
    """

    selector: EpisodeSelector

    def selector_for(
        self, training_selector: EpisodeSelector | None
    ) -> EpisodeSelector | None:
        del training_selector
        return self.selector

    def to_record(self) -> dict:
        return {"type": type(self).__name__, "selector": self.selector.to_record()}


@dataclasses.dataclass(frozen=True)
class PoseEpisodeSelector(EpisodeSelector):
    """Select named poses and optionally cap deterministic repeats per pose."""

    pose_ids: tuple[str, ...]
    episodes_per_pose: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.pose_ids or len(set(self.pose_ids)) != len(self.pose_ids):
            raise ValueError("pose_ids must be nonempty and unique")
        if self.episodes_per_pose is not None and self.episodes_per_pose < 1:
            raise ValueError("episodes_per_pose must be positive")

    def _rank(self, episode: EpisodeIdentity) -> bytes:
        return hashlib.sha256(
            f"{self.seed}\0{episode.pose_id}\0{episode.planner_seed}\0{episode.episode_key}".encode()
        ).digest()

    def selected_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        requested = set(self.pose_ids)
        groups: dict[str, list[EpisodeIdentity]] = defaultdict(list)
        for episode in episodes:
            if episode.pose_id in requested:
                groups[episode.pose_id].append(episode)
        missing = requested - groups.keys()
        if missing:
            raise ValueError(f"dataset is missing requested pose IDs: {sorted(missing)}")
        selected: set[str] = set()
        for pose_id in self.pose_ids:
            ranked = sorted(groups[pose_id], key=self._rank)
            if self.episodes_per_pose is not None:
                if len(ranked) < self.episodes_per_pose:
                    raise ValueError(
                        f"pose {pose_id} has {len(ranked)} episodes; "
                        f"requested {self.episodes_per_pose}"
                    )
                ranked = ranked[: self.episodes_per_pose]
            selected.update(episode.episode_key for episode in ranked)
        return frozenset(selected)

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "pose_ids": list(self.pose_ids),
            "episodes_per_pose": self.episodes_per_pose,
            "seed": self.seed,
            "identity_keys": ["episode_metadata.pose_id", "episode_metadata.planner_seed"],
        }


class EpisodeSplit(abc.ABC):
    """Base class for deterministic episode-level train/validation partitioning."""

    @property
    @abc.abstractmethod
    def source_split(self) -> str:
        """TFDS source split read before applying this strategy."""

    @abc.abstractmethod
    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        """Return stable episode keys assigned to validation."""

    @abc.abstractmethod
    def to_record(self) -> dict:
        """Return the persistent scientific identity of this partition."""


@dataclasses.dataclass(frozen=True)
class ContiguousEpisodeSplit(EpisodeSplit):
    """Compatibility strategy for historical TFDS percentage slices.

    The slices cut by record position, not identity: in a multi-look replay build (K renders of one trajectory,
    ``replay_source_group``) the looks at the cut can land on both sides. Use an embodiment holdout or
    ``StratifiedEmbodimentEpisodeSplit`` (which groups looks) for such builds.
    """

    train: str = "train[:95%]"
    validation: str = "train[95%:]"

    @property
    def source_split(self) -> str:
        return "train"

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        del episodes
        raise NotImplementedError(
            "contiguous splits are executed by TFDS and do not build an episode-key table"
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "train": self.train,
            "validation": self.validation,
        }


@dataclasses.dataclass(frozen=True)
class HashEmbodimentSplit(EpisodeSplit):
    """Hold out a deterministic fraction of complete embodiment identities."""

    validation_fraction: float = 0.05
    seed: int = 42
    minimum_validation_embodiments: int = 1
    require_one_episode_per_embodiment: bool = False
    split: str = "train"

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one")
        if self.minimum_validation_embodiments < 1:
            raise ValueError("minimum_validation_embodiments must be positive")
        if not self.split:
            raise ValueError("source split must be nonempty")

    @property
    def source_split(self) -> str:
        return self.split

    def _rank(self, embodiment_id: int) -> bytes:
        return hashlib.sha256(f"{self.seed}\0{embodiment_id}".encode()).digest()

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        groups: dict[int, list[str]] = defaultdict(list)
        seen_keys: set[str] = set()
        for episode in episodes:
            key = str(episode.episode_key)
            if not key:
                raise ValueError(
                    "hash embodiment splitting requires stable nonempty episode keys"
                )
            if key in seen_keys:
                raise ValueError("hash embodiment splitting requires unique episode keys")
            seen_keys.add(key)
            groups[int(episode.embodiment_id)].append(key)
        if not groups:
            raise ValueError("cannot split an empty episode collection")
        repeated = {
            embodiment_id: len(keys)
            for embodiment_id, keys in groups.items()
            if len(keys) != 1
        }
        if self.require_one_episode_per_embodiment and repeated:
            raise ValueError(
                "expected exactly one episode per embodiment; found "
                f"{dict(sorted(repeated.items()))}"
            )
        requested = max(
            self.minimum_validation_embodiments,
            int(math.floor(len(groups) * self.validation_fraction + 0.5)),
        )
        if requested >= len(groups):
            raise ValueError(
                "hash embodiment splitting requires at least one embodiment on each side"
            )
        validation_ids = set(sorted(groups, key=self._rank)[:requested])
        return frozenset(
            key
            for embodiment_id in validation_ids
            for key in groups[embodiment_id]
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "validation_fraction": self.validation_fraction,
            "minimum_validation_embodiments": self.minimum_validation_embodiments,
            "require_one_episode_per_embodiment": (
                self.require_one_episode_per_embodiment
            ),
            "seed": self.seed,
            "identity_key": "episode_metadata.embodiment_id",
            "ranking": "sha256(seed, embodiment_id)",
            "semantics": "complete_embodiment_holdout",
        }


@dataclasses.dataclass(frozen=True)
class NamespacedHashEmbodimentSplit(EpisodeSplit):
    """Hold out complete ``(component namespace, embodiment_id)`` identities.

    This is the fail-closed split for mixtures whose independently generated
    components can legitimately reuse numeric embodiment IDs.  It never rewrites
    source provenance and therefore keeps raw-ID overlap auditable.
    """

    validation_fraction: float = 0.05
    seed: int = 42
    minimum_validation_embodiments: int = 1
    require_one_episode_per_embodiment: bool = False
    split: str = "train"

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one")
        if self.minimum_validation_embodiments < 1:
            raise ValueError("minimum_validation_embodiments must be positive")
        if not self.split:
            raise ValueError("source split must be nonempty")

    @property
    def source_split(self) -> str:
        return self.split

    def _rank(self, identity: tuple[str, int]) -> bytes:
        namespace, embodiment_id = identity
        return hashlib.sha256(
            f"{self.seed}\0{namespace}\0{embodiment_id}".encode()
        ).digest()

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        groups: dict[tuple[str, int], list[str]] = defaultdict(list)
        seen_keys: set[str] = set()
        for episode in episodes:
            if not episode.namespace:
                raise ValueError("namespaced splitting requires a nonempty namespace")
            key = str(episode.episode_key)
            if not key:
                raise ValueError("namespaced splitting requires stable nonempty episode keys")
            if key in seen_keys:
                raise ValueError("namespaced splitting requires unique episode keys")
            seen_keys.add(key)
            groups[(episode.namespace, int(episode.embodiment_id))].append(key)
        if not groups:
            raise ValueError("cannot split an empty episode collection")
        repeated = {identity: len(keys) for identity, keys in groups.items() if len(keys) != 1}
        if self.require_one_episode_per_embodiment and repeated:
            raise ValueError(
                "expected exactly one episode per namespaced embodiment; found "
                f"{dict(sorted(repeated.items()))}"
            )
        requested = max(
            self.minimum_validation_embodiments,
            int(math.floor(len(groups) * self.validation_fraction + 0.5)),
        )
        if requested >= len(groups):
            raise ValueError("namespaced splitting requires at least one embodiment on each side")
        validation_ids = set(sorted(groups, key=self._rank)[:requested])
        return frozenset(key for identity in validation_ids for key in groups[identity])

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "validation_fraction": self.validation_fraction,
            "minimum_validation_embodiments": self.minimum_validation_embodiments,
            "require_one_episode_per_embodiment": self.require_one_episode_per_embodiment,
            "seed": self.seed,
            "identity_keys": [
                "dataset_component.dataset_id",
                "episode_metadata.embodiment_id",
            ],
            "ranking": "sha256(seed, component_namespace, embodiment_id)",
            "semantics": "global_complete_namespaced_embodiment_holdout",
        }


@dataclasses.dataclass(frozen=True)
class HashEmbodimentCountSplit(EpisodeSplit):
    """Hold out an exact number of complete identities by deterministic hash rank."""

    validation_embodiment_count: int
    seed: int = 42
    require_one_episode_per_embodiment: bool = False
    split: str = "train"

    def __post_init__(self) -> None:
        if self.validation_embodiment_count < 1:
            raise ValueError("validation_embodiment_count must be positive")
        if not self.split:
            raise ValueError("source split must be nonempty")

    @property
    def source_split(self) -> str:
        return self.split

    def _rank(self, embodiment_id: int) -> tuple[bytes, int]:
        return (
            hashlib.sha256(f"{self.seed}\0{embodiment_id}".encode()).digest(),
            embodiment_id,
        )

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        groups: dict[int, list[str]] = defaultdict(list)
        seen_keys: set[str] = set()
        for episode in episodes:
            key = str(episode.episode_key)
            if not key:
                raise ValueError("hash-count splitting requires nonempty episode keys")
            if key in seen_keys:
                raise ValueError("hash-count splitting requires unique episode keys")
            seen_keys.add(key)
            groups[int(episode.embodiment_id)].append(key)
        if self.validation_embodiment_count >= len(groups):
            raise ValueError("validation_embodiment_count must leave training identities")
        repeated = {
            embodiment_id: len(keys)
            for embodiment_id, keys in groups.items()
            if len(keys) != 1
        }
        if self.require_one_episode_per_embodiment and repeated:
            raise ValueError(
                "expected exactly one episode per embodiment; found "
                f"{dict(sorted(repeated.items()))}"
            )
        validation_ids = frozenset(
            sorted(groups, key=self._rank)[: self.validation_embodiment_count]
        )
        return frozenset(
            key
            for embodiment_id in validation_ids
            for key in groups[embodiment_id]
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "validation_embodiment_count": self.validation_embodiment_count,
            "require_one_episode_per_embodiment": (
                self.require_one_episode_per_embodiment
            ),
            "seed": self.seed,
            "identity_key": "episode_metadata.embodiment_id",
            "ranking": "sha256(seed, embodiment_id)",
            "semantics": "fixed_count_complete_embodiment_holdout",
        }


@dataclasses.dataclass(frozen=True)
class FrozenEmbodimentHoldoutSplit(EpisodeSplit):
    """Apply one explicit union-global identity holdout to every component."""

    validation_embodiment_ids: tuple[int, ...]
    seed: int
    represented_embodiment_count: int
    split: str = "train"
    require_one_episode_per_embodiment: bool = False
    artifact_uri: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.validation_embodiment_ids:
            raise ValueError("frozen embodiment holdout must contain at least one ID")
        if len(set(self.validation_embodiment_ids)) != len(self.validation_embodiment_ids):
            raise ValueError("frozen embodiment holdout IDs must be unique")
        if tuple(sorted(self.validation_embodiment_ids)) != self.validation_embodiment_ids:
            raise ValueError("frozen embodiment holdout IDs must be sorted")
        if any(embodiment_id < 0 for embodiment_id in self.validation_embodiment_ids):
            raise ValueError("frozen embodiment holdout IDs must be nonnegative")
        if self.represented_embodiment_count <= len(self.validation_embodiment_ids):
            raise ValueError("frozen embodiment holdout must leave training identities")
        if not self.split:
            raise ValueError("source split must be nonempty")
        if (self.artifact_uri is None) != (self.artifact_sha256 is None):
            raise ValueError("frozen holdout artifact URI and SHA-256 must be provided together")
        if self.artifact_sha256 is not None and (
            len(self.artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.artifact_sha256)
        ):
            raise ValueError("frozen holdout artifact_sha256 must be lowercase SHA-256")

    @property
    def source_split(self) -> str:
        return self.split

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        episodes = tuple(episodes)
        keys = tuple(str(episode.episode_key) for episode in episodes)
        if any(not key for key in keys):
            raise ValueError("frozen embodiment splitting requires stable episode keys")
        if len(set(keys)) != len(keys):
            raise ValueError("frozen embodiment splitting requires unique episode keys")
        validation_ids = frozenset(self.validation_embodiment_ids)
        counts: dict[int, int] = defaultdict(int)
        for episode in episodes:
            counts[int(episode.embodiment_id)] += 1
        missing = validation_ids - counts.keys()
        if missing:
            raise ValueError(
                "RLDS mixture is missing frozen validation identities: "
                f"{sorted(missing)}"
            )
        if self.require_one_episode_per_embodiment:
            if len(counts) != self.represented_embodiment_count:
                raise ValueError(
                    "frozen embodiment holdout represented population mismatch: "
                    f"dataset has {len(counts)} identities but the contract records "
                    f"{self.represented_embodiment_count}"
                )
            non_unique = {identity: count for identity, count in counts.items() if count != 1}
            if non_unique:
                raise ValueError(
                    "frozen embodiment holdout requires exactly one episode per embodiment; "
                    f"violations={non_unique}"
                )
        validation = frozenset(
            episode.episode_key
            for episode in episodes
            if int(episode.embodiment_id) in validation_ids
        )
        if not validation:
            raise ValueError(
                "RLDS mixture contains none of the frozen validation identities"
            )
        if len(validation) == len(episodes):
            raise ValueError("frozen embodiment holdout leaves no training episodes")
        return validation

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "validation_embodiment_ids": list(self.validation_embodiment_ids),
            "validation_embodiment_count": len(self.validation_embodiment_ids),
            "represented_embodiment_count": self.represented_embodiment_count,
            "validation_fraction": (
                len(self.validation_embodiment_ids)
                / self.represented_embodiment_count
            ),
            "seed": self.seed,
            "require_one_episode_per_embodiment": self.require_one_episode_per_embodiment,
            "identity_key": "episode_metadata.embodiment_id",
            "ranking": "frozen_explicit_membership",
            "semantics": "complete_embodiment_holdout",
            **(
                {
                    "artifact_uri": self.artifact_uri,
                    "artifact_sha256": self.artifact_sha256,
                }
                if self.artifact_uri is not None
                else {}
            ),
        }


@dataclasses.dataclass(frozen=True)
class AnchoredHashEmbodimentSplit(EpisodeSplit):
    """Hold out complete embodiments while preserving pinned split membership.

    This is intended for nested dataset-scaling studies.  A smaller reference run can
    pin validation and training identities, while larger populations add validation
    embodiments by the same deterministic hash rank until the requested fraction is
    reached.  The split is computed over the union of all configured dataset components.
    """

    required_validation_embodiment_ids: tuple[int, ...]
    required_training_embodiment_ids: tuple[int, ...] = ()
    anchors_manifest_sha256: str = ""
    validation_fraction: float = 0.05
    seed: int = 42
    minimum_validation_embodiments: int = 1
    require_one_episode_per_embodiment: bool = False
    split: str = "train"

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one")
        if self.minimum_validation_embodiments < 1:
            raise ValueError("minimum_validation_embodiments must be positive")
        if not self.split:
            raise ValueError("source split must be nonempty")
        validation = tuple(int(value) for value in self.required_validation_embodiment_ids)
        training = tuple(int(value) for value in self.required_training_embodiment_ids)
        if not validation or len(validation) != len(set(validation)):
            raise ValueError("required validation embodiment IDs must be nonempty and unique")
        if len(training) != len(set(training)):
            raise ValueError("required training embodiment IDs must be unique")
        if set(validation) & set(training):
            raise ValueError("required validation and training embodiment IDs must be disjoint")
        if any(value < 0 for value in (*validation, *training)):
            raise ValueError("anchored split embodiment IDs must be non-negative")
        if not self.anchors_manifest_sha256 or len(self.anchors_manifest_sha256) != 64:
            raise ValueError("anchored split requires a pinned manifest SHA-256")
        object.__setattr__(self, "required_validation_embodiment_ids", validation)
        object.__setattr__(self, "required_training_embodiment_ids", training)

    @property
    def source_split(self) -> str:
        return self.split

    def _rank(self, embodiment_id: int) -> tuple[bytes, int]:
        return (
            hashlib.sha256(f"{self.seed}\0{embodiment_id}".encode()).digest(),
            embodiment_id,
        )

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        groups: dict[int, list[str]] = defaultdict(list)
        seen_keys: set[str] = set()
        for episode in episodes:
            key = str(episode.episode_key)
            if not key:
                raise ValueError("anchored splitting requires stable nonempty episode keys")
            if key in seen_keys:
                raise ValueError("anchored splitting requires globally unique episode keys")
            seen_keys.add(key)
            groups[int(episode.embodiment_id)].append(key)
        if not groups:
            raise ValueError("cannot split an empty episode collection")
        repeated = {
            embodiment_id: len(keys)
            for embodiment_id, keys in groups.items()
            if len(keys) != 1
        }
        if self.require_one_episode_per_embodiment and repeated:
            raise ValueError(
                "expected exactly one episode per embodiment; found "
                f"{dict(sorted(repeated.items()))}"
            )

        available = set(groups)
        required_validation = set(self.required_validation_embodiment_ids)
        required_training = set(self.required_training_embodiment_ids)
        missing_validation = required_validation - available
        missing_training = required_training - available
        if missing_validation or missing_training:
            raise ValueError(
                "anchored split population is missing required identities: "
                f"validation={sorted(missing_validation)}, "
                f"training={sorted(missing_training)}"
            )
        requested = max(
            self.minimum_validation_embodiments,
            int(math.floor(len(groups) * self.validation_fraction + 0.5)),
        )
        if requested >= len(groups):
            raise ValueError(
                "anchored splitting requires at least one embodiment on each side"
            )
        if len(required_validation) > requested:
            raise ValueError(
                "required validation anchors exceed requested validation population: "
                f"{len(required_validation)} > {requested}"
            )
        candidates = available - required_validation - required_training
        additional_count = requested - len(required_validation)
        if len(candidates) < additional_count:
            raise ValueError("too few unpinned embodiments to complete validation split")
        validation_ids = required_validation | set(
            sorted(candidates, key=self._rank)[:additional_count]
        )
        return frozenset(
            key
            for embodiment_id in validation_ids
            for key in groups[embodiment_id]
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "validation_fraction": self.validation_fraction,
            "minimum_validation_embodiments": self.minimum_validation_embodiments,
            "require_one_episode_per_embodiment": (
                self.require_one_episode_per_embodiment
            ),
            "seed": self.seed,
            "identity_key": "episode_metadata.embodiment_id",
            "ranking": "sha256(seed, embodiment_id)",
            "semantics": "anchored_complete_embodiment_holdout",
            "anchors_manifest_sha256": self.anchors_manifest_sha256,
            "required_validation_embodiment_count": len(
                self.required_validation_embodiment_ids
            ),
            "required_training_embodiment_count": len(
                self.required_training_embodiment_ids
            ),
        }


@dataclasses.dataclass(frozen=True)
class PhysicalEmbodimentAnchoredHashSplit(EpisodeSplit):
    """Hold out complete physical robots across components that re-record them.

    A reference component defines the physical robot population and its holdout by
    the exact :class:`AnchoredHashEmbodimentSplit` rule, so a reference-only run and
    every mixture built on it hold out the same robots.  Each repeat component
    re-records a subset of those robots (same collection candidate indices and
    robot models under new artifact hashes and embodiment names) and may contain
    many episodes per robot.  In every declared component the RLDS
    ``embodiment_id`` is the collection candidate index, i.e. the physical
    identity, so every episode of a held-out robot is validation in every
    component.  Undeclared components fail closed: in an arbitrary mixture, equal
    numeric IDs name unrelated robots.
    """

    reference_namespace: str
    repeat_namespaces: tuple[str, ...]
    required_validation_embodiment_ids: tuple[int, ...]
    required_training_embodiment_ids: tuple[int, ...] = ()
    anchors_manifest_sha256: str = ""
    validation_fraction: float = 0.05
    seed: int = 42
    minimum_validation_embodiments: int = 1
    require_one_reference_episode_per_embodiment: bool = True
    split: str = "train"

    def __post_init__(self) -> None:
        repeat = tuple(str(value) for value in self.repeat_namespaces)
        if not self.reference_namespace:
            raise ValueError("physical-identity split requires a reference namespace")
        if not repeat or any(not value for value in repeat) or len(set(repeat)) != len(repeat):
            raise ValueError("repeat namespaces must be nonempty and unique")
        if self.reference_namespace in repeat:
            raise ValueError("the reference namespace cannot also be a repeat namespace")
        object.__setattr__(self, "repeat_namespaces", repeat)
        # Validates anchors, fraction, seed, and split exactly as the reference rule does.
        self.reference_split()

    @property
    def source_split(self) -> str:
        return self.split

    def reference_split(self) -> AnchoredHashEmbodimentSplit:
        return AnchoredHashEmbodimentSplit(
            required_validation_embodiment_ids=self.required_validation_embodiment_ids,
            required_training_embodiment_ids=self.required_training_embodiment_ids,
            anchors_manifest_sha256=self.anchors_manifest_sha256,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
            minimum_validation_embodiments=self.minimum_validation_embodiments,
            require_one_episode_per_embodiment=(
                self.require_one_reference_episode_per_embodiment
            ),
            split=self.split,
        )

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        episodes = tuple(episodes)
        keys = [str(episode.episode_key) for episode in episodes]
        if any(not key for key in keys):
            raise ValueError("physical-identity splitting requires stable nonempty episode keys")
        if len(set(keys)) != len(keys):
            raise ValueError("physical-identity splitting requires globally unique episode keys")
        present = {episode.namespace for episode in episodes}
        undeclared = present - {self.reference_namespace, *self.repeat_namespaces}
        absent = {self.reference_namespace, *self.repeat_namespaces} - present
        if undeclared or absent:
            raise ValueError(
                "physical-identity split components disagree with the declared namespaces: "
                f"undeclared={sorted(undeclared)}, absent={sorted(absent)}"
            )
        reference = tuple(
            episode for episode in episodes if episode.namespace == self.reference_namespace
        )
        reference_ids = {int(episode.embodiment_id) for episode in reference}
        foreign = {
            int(episode.embodiment_id)
            for episode in episodes
            if episode.namespace != self.reference_namespace
        } - reference_ids
        if foreign:
            raise ValueError(
                "repeat components contain robots outside the reference population: "
                f"{sorted(foreign)}"
            )
        reference_validation = self.reference_split().validation_episode_keys(reference)
        validation_ids = {
            int(episode.embodiment_id)
            for episode in reference
            if str(episode.episode_key) in reference_validation
        }
        return frozenset(
            str(episode.episode_key)
            for episode in episodes
            if int(episode.embodiment_id) in validation_ids
        )

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "reference_namespace": self.reference_namespace,
            "repeat_namespaces": list(self.repeat_namespaces),
            "reference_split": self.reference_split().to_record(),
            "identity_keys": [
                "dataset_component.dataset_id",
                "episode_metadata.embodiment_id",
            ],
            "physical_identity": "episode_metadata.embodiment_id (collection candidate index)",
            "ranking": "anchored sha256(seed, embodiment_id) over the reference component",
            "semantics": "anchored_complete_physical_embodiment_holdout_across_components",
        }


@dataclasses.dataclass(frozen=True)
class StratifiedEmbodimentEpisodeSplit(EpisodeSplit):
    """Hold out a deterministic fraction independently within every embodiment.

    Episodes are ranked by a seeded SHA-256 of their stable RLDS ``file_path``. This is
    invariant to TFRecord shard order and guarantees that every embodiment with at least two
    source trajectories appears in both train and validation.

    The unit held out is a source trajectory (``replay_source_group``): all visual looks of one
    saved state trajectory go to the same side, ranked by their best-ranked look, and the fraction
    counts trajectories. Without multi-look replay records every group is one episode, so the
    membership equals the per-episode ranking used before 2026-10-16.
    """

    validation_fraction: float = 0.05
    seed: int = 42
    minimum_validation_episodes: int = 1
    split: str = "train"

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one")
        if self.minimum_validation_episodes < 1:
            raise ValueError("minimum_validation_episodes must be positive")
        if not self.split:
            raise ValueError("source split must be nonempty")

    @property
    def source_split(self) -> str:
        return self.split

    def _rank(self, episode_key: str) -> bytes:
        return hashlib.sha256(f"{self.seed}\0{episode_key}".encode()).digest()

    def validation_episode_keys(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> frozenset[str]:
        groups: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        seen: set[str] = set()
        for episode in episodes:
            key = str(episode.episode_key)
            if not key:
                raise ValueError("stratified episode splitting requires a stable episode key")
            if key in seen:
                raise ValueError(f"duplicate episode key in RLDS split: {key}")
            seen.add(key)
            groups[int(episode.embodiment_id)][replay_source_group(key)].append(key)
        if not groups:
            raise ValueError("cannot stratify an empty episode collection")

        validation: set[str] = set()
        for embodiment_id, sources in groups.items():
            if len(sources) < 2:
                raise ValueError(
                    "stratified train/validation splitting requires at least two episodes "
                    f"of distinct source trajectories for embodiment {embodiment_id}; "
                    f"found {len(sources)}"
                )
            requested = max(
                self.minimum_validation_episodes,
                int(math.floor(len(sources) * self.validation_fraction + 0.5)),
            )
            count = min(len(sources) - 1, requested)
            ranked = sorted(
                sources.values(),
                key=lambda keys: min((self._rank(key), key) for key in keys),
            )
            validation.update(key for keys in ranked[:count] for key in keys)
        return frozenset(validation)

    def counts_by_embodiment(
        self, episodes: Iterable[EpisodeIdentity]
    ) -> dict[int, dict[str, int]]:
        episodes = tuple(episodes)
        validation = self.validation_episode_keys(episodes)
        counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"train": 0, "validation": 0}
        )
        for episode in episodes:
            role = "validation" if episode.episode_key in validation else "train"
            counts[int(episode.embodiment_id)][role] += 1
        return dict(counts)

    def to_record(self) -> dict:
        return {
            "type": type(self).__name__,
            "source_split": self.source_split,
            "validation_fraction": self.validation_fraction,
            "minimum_validation_episodes": self.minimum_validation_episodes,
            "seed": self.seed,
            "identity_key": "episode_metadata.file_path",
            "stratification_key": "episode_metadata.embodiment_id",
            "holdout_unit": "replay_source_group(file_path)",
        }


@dataclasses.dataclass(frozen=True)
class FrameSampling:
    """Base class for selecting training observations from complete trajectories."""

    @property
    def post_close_weight(self) -> float | None:
        raise NotImplementedError

    def to_record(self) -> dict:
        return {"type": type(self).__name__}


@dataclasses.dataclass(frozen=True)
class NaturalFrameSampling(FrameSampling):
    """Preserve the empirical frame distribution; this is the default."""

    @property
    def post_close_weight(self) -> float | None:
        return None


POST_CLOSE_RESAMPLING_REFUSAL = (
    "post-close frame resampling is not supported: rebalancing training frames around a "
    "task-specific event (the gripper closing) bakes a per-task recipe into the input pipeline "
    "and doubles host input work. The natural frame distribution is the training contract; "
    "change the data, not the sampler. Five July 2026 lineages recorded this knob and stay "
    "loadable for provenance and evaluation, but cannot be trained or resumed."
)


def validation_source_split(*, episode_split, rlds_split: str) -> str:
    """Return the TFDS split the validation reader must open, or refuse.

    Validation data exists only when the config defines an episode split (the reader then
    opens the same source split and keeps the held-out episodes) or when training itself
    uses a leading percent slice such as ``train[:95%]`` whose complement is free. Training
    on the whole ``train`` split with no episode split leaves nothing to validate on.
    """
    if episode_split is not None:
        return rlds_split
    match = re.fullmatch(r"(?P<split>[A-Za-z_][A-Za-z0-9_]*)\[:(?P<percent>\d{1,2})%\]", rlds_split)
    if match is None:
        raise ValueError(
            "validation is enabled but the config defines no validation data: rlds_split="
            f"{rlds_split!r} with no episode split. Define an episode split, train on a leading "
            "percent slice such as 'train[:95%]', or set val_interval=0."
        )
    return f"{match.group('split')}[{match.group('percent')}%:]"


@dataclasses.dataclass(frozen=True)
class PostCloseBalancedFrameSampling(FrameSampling):
    """Historical record of the pre/post-close resampling ablation (refused at load time).

    Kept so the July 2026 configs that used it remain loadable and their provenance intact.
    Building a training dataset with it raises ``POST_CLOSE_RESAMPLING_REFUSAL``.
    """

    weight: float = 0.5

    def __post_init__(self):
        if not 0.0 < self.weight < 1.0:
            raise ValueError("post-close sampling weight must be strictly between zero and one")

    @property
    def post_close_weight(self) -> float | None:
        return self.weight

    def to_record(self) -> dict:
        return {"type": type(self).__name__, "weight": self.weight}
