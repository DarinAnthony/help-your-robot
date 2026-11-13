"""RLDS streaming loader for our molmospaces rlds_v3 (fr_ur5 open/close).

Mirrors DroidRldsDataset (tf.data + dlimp) so we can train on the FULL dataset
(39k episodes / 116 GiB) by streaming from GCS — no LeRobot conversion needed.
Our schema (per step): action (10,), observation/{proprio(10,), image_primary,
image_wrist (jpeg strings)}, plus authoritative episode_metadata/{task,target} and,
for new datasets, the exact task-config-owned task_instruction.

Actions are already delta-joint + absolute-gripper (like LIBERO) -> no delta xform.
Emits FLAT keys {image, wrist_image, state, actions, prompt} so the LIBERO repack +
LiberoInputs transforms can be reused unchanged.
"""

from collections.abc import Sequence
import dataclasses
import logging
import numbers
import re

import numpy as np

from pi05_integration.input_config import NoPromptConditioning
from pi05_integration.input_config import PromptConditioning
from pi05_integration.input_config import prompt_conditioning_from_legacy
from pi05_integration import embodiment_bundle as embodiment_bundle_module
from pi05_integration.behavior_prompt_overlay import BehaviorPromptOverlay
from pi05_integration.behavior_prompt_overlay import load_behavior_prompt_overlay
from pi05_integration.behavior_prompt_overlay import validate_episode_key_coverage
from pi05_integration.data_processing import ContiguousEpisodeSplit
from pi05_integration.data_processing import DeterministicReplayV3
from pi05_integration.data_processing import AnchoredHashEmbodimentSplit
from pi05_integration.data_processing import EpisodeIdentity
from pi05_integration.data_processing import EpisodeSelector
from pi05_integration.data_processing import EpisodeSplit
from pi05_integration.data_processing import FrozenEmbodimentHoldoutSplit
from pi05_integration.data_processing import HashEmbodimentCountSplit
from pi05_integration.data_processing import HashEmbodimentSplit
from pi05_integration.data_processing import NamespacedHashEmbodimentSplit
from pi05_integration.data_processing import PhysicalEmbodimentAnchoredHashSplit
from pi05_integration.data_processing import LegacyRldsInputSampling
from pi05_integration.data_processing import ProcessShardedFileShuffleV1
from pi05_integration.data_processing import ProcessShardedFileShuffleV2
from pi05_integration.data_processing import POST_CLOSE_RESAMPLING_REFUSAL, default_rlds_input_sampling
from pi05_integration.rlds_episode_index import load_episode_views, scan_episode_views
from pi05_integration.data_processing import RldsInputSampling
from pi05_integration.data_processing import StratifiedEmbodimentEpisodeSplit

from molmo_spaces.rlds_layout import JointVectorLayout
from molmo_spaces.rlds_layout import joint_vector_remap_matrix
from molmo_spaces.rlds_layout import joint_vector_remap_pairs

from openpi.training.embodiment_conditioning import make_embodiment_descriptor

# Canonical index of the absolute-gripper column in the 10-wide action/state vectors.
# Every 7-DoF franka build (DROID, realrig9k, pp9robot, the franka half of fr_ur5) already
# stores it here; 6-DoF arms are remapped onto it in prepare() below.
_CANONICAL_GRIPPER_DIM = 7


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float = 1.0
    dataset_id: str | None = None
    # Optional path below the runtime-owned RLDS root. This lets one typed config
    # stream several independently registered TFDS builds without copying them.
    data_subdir: str | None = None
    joint_layout: str | None = None
    target_joint_layout: str | None = None


_EPISODE_SPLIT_CACHE: dict[tuple[str, str, str], tuple[EpisodeIdentity, ...]] = {}
_APPROXIMATE_FRAME_COUNT = 2_700_000


def approximate_batch_count(batch_size: int) -> int:
    """Return a progress-bar estimate in yielded batches, not raw frames."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return max(1, (_APPROXIMATE_FRAME_COUNT + batch_size - 1) // batch_size)


def letterbox_image(image, height: int, width: int, tf):
    """Resize a uint8 HxWx3 image to (height, width), keeping its aspect ratio with zero padding.

    DROID's 180x320 frames have to match the sim builds' square frames for
    sample_from_datasets. A plain resize squashes 16:9 into 1:1; pi05_base was pretrained on
    letterboxed DROID frames (image_tools.resize_with_pad), so pad instead of stretch.
    """
    return tf.cast(tf.round(tf.image.resize_with_pad(image, height, width)), tf.uint8)


def pick_nonempty_instruction(instructions, tf):
    """Uniformly pick one non-empty (after stripping) string from a 1-D string tensor.

    DROID stores three instruction variants per episode and many episodes leave some of them
    blank; shuffling all three would train on an empty prompt. Callers must first drop
    episodes whose instructions are all empty (has_nonempty_instruction).
    """
    nonempty = tf.boolean_mask(instructions, tf.strings.length(tf.strings.strip(instructions)) > 0)
    index = tf.random.uniform([], 0, tf.shape(nonempty)[0], dtype=tf.int32)
    return nonempty[index]


def has_nonempty_instruction(instructions, tf):
    return tf.reduce_any(tf.strings.length(tf.strings.strip(instructions)) > 0)


def chunk_motion_norm(actions, loss_mask, semantic_mask, tf, base_actions=None, base_mask=None, rest=None):
    """Norm of the motion an action chunk asks for: arm deltas plus gripper *change*.

    actions is (H, D); loss_mask is (H, D) or, for model-coordinate masks, (H, 1);
    semantic_mask is (D,). The gripper is the last valid
    semantic dimension and is stored as an absolute position, so its raw value says nothing
    about motion: counting it would keep every chunk held with a closed gripper and drop every
    one held open. Use its change from the chunk's first step instead, which still keeps
    gripper-only chunks (the arm parked while the gripper closes).

    ``base_actions`` (H, 3) and ``base_mask`` (3,), given only under the mobile-base option, add the
    holonomic base's commanded motion, so a drive-only chunk is not static. A fixed arm's base is
    zero, which leaves the norm bit-identical.

    ``rest`` (D,) is the action of no motion where it is not zero: the identity rotation columns
    of a BASE TCP pose label (``_action_rest``). None leaves the norm bit-identical.
    """
    width = tf.shape(actions)[-1]
    gripper_index = tf.cast(tf.reduce_max(tf.where(semantic_mask)), tf.int32)
    valid = tf.broadcast_to(tf.cast(loss_mask, actions.dtype), tf.shape(actions))
    arm_columns = tf.cast(tf.not_equal(tf.range(width), gripper_index), actions.dtype)
    arm = (actions if rest is None else actions - rest[None, :]) * valid * arm_columns[None, :]
    gripper = tf.gather(actions, gripper_index, axis=-1)
    gripper_valid = tf.gather(valid, gripper_index, axis=-1)
    gripper_change = (gripper - gripper[0]) * gripper_valid
    squared = tf.reduce_sum(tf.square(arm)) + tf.reduce_sum(tf.square(gripper_change))
    if base_actions is not None:
        squared = squared + tf.reduce_sum(
            tf.square(base_actions * tf.cast(base_mask, base_actions.dtype)[None, :])
        )
    return tf.sqrt(squared)


def _relative_pose9(goal9, anchor9, tf):
    """BASE TCP label of goal poses against anchor poses (broadcast; ``arm_action.relative_pose9``):
    ``[p_goal - p_anchor, columns 0 and 1 of R_goal R_anchor^T]``."""

    def rotation(columns6):
        first = tf.math.l2_normalize(columns6[..., 0:3], axis=-1)
        second = columns6[..., 3:6] - first * tf.reduce_sum(first * columns6[..., 3:6], axis=-1, keepdims=True)
        second = tf.math.l2_normalize(second, axis=-1)
        return tf.stack([first, second, tf.linalg.cross(first, second)], axis=-1)

    goal_rotation, anchor_rotation = rotation(goal9[..., 3:9]), rotation(anchor9[..., 3:9])
    relative = tf.linalg.matmul(goal_rotation, tf.broadcast_to(anchor_rotation, tf.shape(goal_rotation)),
                                transpose_b=True)
    return tf.concat([goal9[..., :3] - anchor9[..., :3], relative[..., :, 0], relative[..., :, 1]], axis=-1)


_DROID_IDLE_TABLES: dict = {}


def droid_idle_filter_table(filter_dict_path: str, tf):
    """Hash table of DROID step IDs to keep, from openpi's non-idle sample ranges.

    The JSON maps "<recording_folderpath>--<file_path>" to [start, end) ranges of non-idle
    steps (openpi's DROID recipe, droid_rlds_dataset.py). Built once per path per process.
    v1.0.1 keeps ~26M steps, so keys are 64-bit fingerprints of "<episode>--<step>" (about
    200 MB) rather than the strings themselves (several GB); look them up with
    droid_step_fingerprint.
    """
    if filter_dict_path in _DROID_IDLE_TABLES:
        return _DROID_IDLE_TABLES[filter_dict_path]
    import json
    import pathlib

    from openpi.shared import download

    with pathlib.Path(download.maybe_download(filter_dict_path)).open("r") as f:
        filter_dict = json.load(f)
    fingerprints, pending, kept = [], [], 0
    for episode, ranges in filter_dict.items():
        for start, end in ranges:
            pending.extend(f"{episode}--{t}" for t in range(start, end))
        if len(pending) >= 1_000_000:
            fingerprints.append(droid_step_fingerprint(tf.constant(pending), tf).numpy())
            kept += len(pending)
            pending = []
    if pending:
        fingerprints.append(droid_step_fingerprint(tf.constant(pending), tf).numpy())
        kept += len(pending)
    keys = np.unique(np.concatenate(fingerprints))
    logging.info("DROID idle filter: %d episodes, %d kept steps from %s", len(filter_dict), kept, filter_dict_path)
    if len(keys) != kept:
        raise ValueError(f"DROID idle filter: {kept - len(keys)} step fingerprint collisions or duplicate steps")
    table = tf.lookup.StaticHashTable(
        # int64 -> bool has no CPU kernel; store 1/0 and compare at lookup time.
        tf.lookup.KeyValueTensorInitializer(tf.constant(keys, tf.int64), tf.ones([len(keys)], tf.int32)),
        default_value=0,
    )
    _DROID_IDLE_TABLES[filter_dict_path] = table
    return table


def droid_step_fingerprint(step_ids, tf):
    """Deterministic 62-bit fingerprint of DROID step-ID strings (Fingerprint64 based)."""
    return tf.strings.to_hash_bucket_fast(step_ids, 2**62)


def _episode_views(builder, source_split: str, identity_record: dict, tfds) -> tuple[EpisodeIdentity, ...]:
    """Return stable identity plus encoded-frame cardinality, once per process.

    The prebuilt sidecar (``pi05_integration.rlds_episode_index``) is used when it exists and
    matches the split's shard fingerprint; otherwise the split is scanned exactly as before.
    """
    cache_key = (
        str(builder.data_dir),
        source_split,
        repr(identity_record),
    )
    if cache_key in _EPISODE_SPLIT_CACHE:
        return _EPISODE_SPLIT_CACHE[cache_key]
    identities = load_episode_views(builder, source_split)
    if identities is None:
        identities = scan_episode_views(builder, source_split, tfds)
    _EPISODE_SPLIT_CACHE[cache_key] = identities
    return identities


# Compatibility for callers/tests written against the identity-only helper.
_episode_identities = _episode_views


def select_episode_views(
    identities: Sequence[EpisodeIdentity],
    *,
    episode_selector: EpisodeSelector | None,
    episode_split: EpisodeSplit | None,
    episode_split_role: str,
    embodiment_filter_ids: Sequence[int] = (),
) -> tuple[EpisodeIdentity, ...]:
    """Apply the exact selector/split/filter contract used by the streaming loader."""
    selected = tuple(identities)
    if episode_selector is not None:
        selected_keys = episode_selector.selected_episode_keys(selected)
        selected = tuple(item for item in selected if item.episode_key in selected_keys)
    if isinstance(
        episode_split,
        (
            AnchoredHashEmbodimentSplit,
            FrozenEmbodimentHoldoutSplit,
            HashEmbodimentCountSplit,
            HashEmbodimentSplit,
            NamespacedHashEmbodimentSplit,
            PhysicalEmbodimentAnchoredHashSplit,
            StratifiedEmbodimentEpisodeSplit,
        ),
    ):
        validation_keys = episode_split.validation_episode_keys(selected)
        if episode_split_role == "validation":
            selected = tuple(item for item in selected if item.episode_key in validation_keys)
        else:
            selected = tuple(item for item in selected if item.episode_key not in validation_keys)
    filter_ids = canonicalize_embodiment_filter_ids(embodiment_filter_ids)
    if filter_ids:
        selected = tuple(
            item for item in selected if embodiment_id_is_selected(item.embodiment_id, filter_ids)
        )
    return selected


def _embodiment_id_offset(cfg) -> int:
    """The component's opt-in embodiment-ID offset (``RLDSDataset.embodiment_id_offset``; 0 = raw IDs)."""
    offset = getattr(cfg, "embodiment_id_offset", 0) or 0
    if isinstance(offset, bool) or not isinstance(offset, numbers.Integral) or offset < 0 or offset >= 2**31:
        raise ValueError(f"embodiment_id_offset must be a non-negative int32, got {offset!r}")
    return int(offset)


def canonicalize_embodiment_filter_ids(ids: Sequence[int]) -> tuple[int, ...]:
    """Validate a global-ID subset without changing the IDs or their order."""
    values = tuple(ids)
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in values):
        raise TypeError("embodiment_filter_ids must contain only integer global embodiment IDs")
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError("embodiment_filter_ids must contain only non-negative global embodiment IDs")
    if len(set(result)) != len(result):
        raise ValueError("embodiment_filter_ids must be unique")
    return result


def embodiment_id_is_selected(embodiment_id: int, filter_ids: Sequence[int]) -> bool:
    """Return whether a raw global ID passes an optional subset filter."""
    ids = canonicalize_embodiment_filter_ids(filter_ids)
    return not ids or int(embodiment_id) in ids


def chunk_action_loss_mask(
    time_valid,
    semantic_chunk,
    *,
    coordinates: str,
    logical_and=np.logical_and,
):
    """Build the future-chunk loss mask under one explicit coordinate contract."""
    if coordinates == "model":
        return time_valid[..., None]
    if coordinates == "physical":
        return logical_and(time_valid[..., None], semantic_chunk)
    raise ValueError(
        "action_loss_coordinates must be 'model' or 'physical', got "
        f"{coordinates!r}"
    )


def _tensorflow_episode_is_selected(episode_ids, allowed_ids, tf):
    """TensorFlow predicate over repeated episode metadata, kept separate for testing."""
    return tf.reduce_any(tf.equal(tf.cast(episode_ids[0], tf.int32), allowed_ids))


RESPONSE_HISTORY_KEYS = (
    "response_state_history",
    "response_state_history_mask",
    "response_action_history",
    "response_action_history_mask",
)


def response_history_specs(observation_features) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Per-step shape/dtype of the response-history fields any mixture component stores.

    RLDS schema 1.2.0 and later always store all four fields (zeros with false masks when
    no history was materialized); 1.0.0 predates them. Components that store them must agree.
    """
    specs = {}
    for features in observation_features:
        present = {key for key in RESPONSE_HISTORY_KEYS if key in features}
        if present and present != set(RESPONSE_HISTORY_KEYS):
            raise ValueError(f"partial response-history schema: {sorted(present)}")
        for key in sorted(present):
            dtype = features[key].dtype
            spec = (tuple(features[key].shape), np.dtype(getattr(dtype, "as_numpy_dtype", dtype)))
            if specs.setdefault(key, spec) != spec:
                raise ValueError(f"mixture components disagree on {key}: {specs[key]} != {spec}")
    return specs


def response_history_fields(observation, trajectory_length, specs, tf) -> dict:
    """One element structure for a mixture: the stored fields, or the 1.2.0 representation
    of absent history (zeros, false masks) for a component that predates them."""
    return {
        key: observation[key] if key in observation
        else tf.zeros([trajectory_length, *shape], dtype=dtype)
        for key, (shape, dtype) in specs.items()
    }


def _episode_prompt(metadata, trajectory_length, tf):
    """Use the recorded task-config instruction, with a legacy-schema fallback."""
    legacy_prompt = tf.strings.join(
        [metadata["task"], tf.fill([trajectory_length], " the "), metadata["target"]]
    )
    if "task_instruction" not in metadata:
        return legacy_prompt
    recorded = metadata["task_instruction"]
    return tf.where(tf.strings.length(recorded) > 0, recorded, legacy_prompt)


def dataset_root(data_dir: str, dataset: RLDSDataset) -> str:
    if dataset.data_subdir is None:
        return data_dir.rstrip("/")
    # An absolute data_subdir (URL or filesystem path) replaces the root, so one mixture can
    # stream builds that live in different buckets (e.g. DROID in a shared OXE bucket).
    if "://" in dataset.data_subdir or dataset.data_subdir.startswith("/"):
        return dataset.data_subdir.rstrip("/")
    subdir = dataset.data_subdir.strip("/")
    if not subdir or subdir == ".":
        return data_dir.rstrip("/")
    return f"{data_dir.rstrip('/')}/{subdir}"


def _from_rlds_with_sampling_contract(
    dl,
    tf,
    tfds,
    builder,
    *,
    split: str,
    shuffle: bool,
    num_parallel_reads: int,
    sampling: RldsInputSampling,
    experiment_seed: int,
    process_count: int,
    process_index: int,
    completed_updates: int,
):
    """Open RLDS with the legacy unsharded traversal or a process-sharded contract."""
    if isinstance(sampling, LegacyRldsInputSampling) or not shuffle:
        return dl.DLataset.from_rlds(
            builder,
            split=split,
            shuffle=False,
            num_parallel_reads=num_parallel_reads,
        )
    if not isinstance(sampling, (ProcessShardedFileShuffleV1, DeterministicReplayV3)):
        raise TypeError(f"unsupported RLDS input sampling: {type(sampling).__name__}")
    if process_count < 1 or not 0 <= process_index < process_count:
        raise ValueError("invalid RLDS input process topology")

    # Pinned dlimp does not expose TFDS ReadConfig. Reproduce its 12-line
    # from_rlds adapter so sharding happens at file identities before records
    # (and their encoded RGB payloads) are fetched.
    from dlimp import dataset as dlimp_dataset

    def shared_file_permutation(file_instructions):
        # TFDS 4.9 applies InputContext.shard() before shuffle_files. Its
        # pre-shard hook is required to permute the common file list first.
        return [
            file_instructions[index]
            for index in (
                sampling.file_order(len(file_instructions), sampler_seed=experiment_seed)
                if isinstance(sampling, DeterministicReplayV3)
                else sampling.file_order(
                    len(file_instructions),
                    experiment_seed=experiment_seed,
                    completed_updates=completed_updates,
                )
            )
        ]

    # V2 additionally reshuffles each process's own file cohort on every pass. TFDS applies
    # this shuffle after InputContext.shard(), so cohort membership is unchanged from V1.
    epoch_reshuffle = isinstance(
        sampling, (ProcessShardedFileShuffleV2, DeterministicReplayV3)
    )
    v3 = isinstance(sampling, DeterministicReplayV3)
    read_config = tfds.ReadConfig(
        skip_prefetch=True,
        num_parallel_calls_for_interleave_files=(
            sampling.interleave_cycle_length if v3 else num_parallel_reads
        ),
        interleave_cycle_length=(
            sampling.interleave_cycle_length if v3 else num_parallel_reads
        ),
        interleave_block_length=(sampling.interleave_block_length if v3 else 1),
        input_context=tf.distribute.InputContext(
            num_input_pipelines=process_count,
            input_pipeline_id=process_index,
            num_replicas_in_sync=process_count,
        ),
        experimental_interleave_sort_fn=shared_file_permutation,
        shuffle_seed=(
            (
                sampling.process_file_shuffle_seed(
                    sampler_seed=experiment_seed,
                    process_index=process_index,
                )
                if v3
                else sampling.process_file_shuffle_seed(
                    experiment_seed=experiment_seed,
                    completed_updates=completed_updates,
                    process_index=process_index,
                )
            )
            if epoch_reshuffle
            else None
        ),
        shuffle_reshuffle_each_iteration=epoch_reshuffle,
    )
    dataset = dlimp_dataset._wrap(builder.as_dataset, False)(
        split=split,
        shuffle_files=epoch_reshuffle,
        decoders={"steps": tfds.decode.SkipDecoding()},
        read_config=read_config,
    )._apply_options()
    return dataset.enumerate().traj_map(dlimp_dataset._broadcast_metadata_rlds)


def _overlay_for_component(
    dataset_cfg,
    overlays: Sequence[BehaviorPromptOverlay],
) -> BehaviorPromptOverlay | None:
    """Resolve exactly one scientific overlay for a native RLDS component."""
    matches = tuple(
        overlay
        for overlay in overlays
        if overlay.dataset_version == dataset_cfg.version
        and overlay.dataset_name == str(dataset_cfg.name).rsplit("/", 1)[-1]
        and str(dataset_cfg.name) in {
            overlay.dataset_name,
            f"{overlay.component_repo_id}/{overlay.dataset_name}",
        }
    )
    if not overlays:
        return None
    if len(matches) != 1:
        raise ValueError(
            "behavior prompt overlay component mismatch for native dataset "
            f"{dataset_cfg.name}:{dataset_cfg.version}: matched {len(matches)} overlays"
        )
    return matches[0]


class MolmoSpacesRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence,
        *,
        shuffle: bool = True,
        action_chunk_size: int | None = 10,
        split: str = "train",
        episode_selector: EpisodeSelector | None = None,
        episode_split: EpisodeSplit | None = None,
        episode_split_role: str = "train",
        shuffle_buffer_size: int = 50_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        prompt_conditioning: PromptConditioning | str = "none",
        post_close_sampling_weight: float | None = None,
        static_action_threshold: float = 1e-4,
        embodiment_filter_ids: Sequence[int] = (),
        seed: int = 0,
        deterministic: bool = False,
        repeat: bool = True,
        decode_images: bool = True,
        behavior_prompt_overlays: Sequence[BehaviorPromptOverlay] = (),
        action_loss_coordinates: str = "model",
        rlds_input_sampling: RldsInputSampling | None = None,
        process_count: int = 1,
        process_index: int = 0,
        completed_updates: int = 0,
        input_sampling_seed: int | None = None,
        final_prefetch: int = 0,
        ram_budget_gb: int = 1,
        mobile_base: str | None = None,
        mobile_base_dt_s: float | None = None,
        arm_action: str | None = None,
        action_time: bool = False,
        letterbox_source: tuple[int, int] | None = None,
        gripper_endpoints: dict[str, tuple[float, float]] | None = None,
        joint_frames: dict | None = None,
    ):
        """``mobile_base`` (``Pi05InputConfig.mobile_base.contract``, a ``mobile_base.CONTRACTS`` name)
        additionally emits ``base_state``, ``base_state_mask``, ``base_actions`` (chunked like ``actions``)
        and ``base_action_mask`` from that contract's RLDS fields (``mobile_base.RLDS_FIELDS``: v1
        ``mobile_base_feature_v1`` poses, v2 ``mobile_base_feature_v2`` velocities); a component without
        those fields gets zeros with false masks, and one with the other contract's fields is refused.
        ``mobile_base_dt_s`` (v2's ``dt_s``) refuses a mobile episode whose
        ``episode_metadata/base_velocity_dt_s`` differs: its velocities mean other displacements.
        Every other key is computed exactly as without it (pi05_integration/mobile_base.py).

        ``arm_action`` (``Pi05InputConfig.arm_action.contract``, an ``arm_action.CONTRACTS`` name; None is
        the default ``per_step_joint_delta``) selects the chunk labels: the query-anchored forms replace
        ``actions`` (and, for the joint + BASE TCP form, ``state`` and both masks) with the targets of
        each chunk relative to the state measured at its query step (pi05_integration/arm_action.py).

        ``action_time`` (``Pi05InputConfig.action_time.enabled``) additionally emits ``action_dt_s``, each frame's
        control period in seconds, from ``episode_metadata/policy_dt_s`` (``mlspaces_to_rlds.py
        --policy-dt-metadata``); every component must carry it. Disabled, that metadata is never read.

        ``letterbox_source`` (``Pi05InputConfig.image_geometry``, ``LetterboxStretched``: the render's (width, height))
        un-stretches every stored sim frame into the letterbox of that render (``pi05_integration/image_geometry.py``,
        review T5); DROID frames are letterboxed from their raw size as before. None keeps the stored frames.

        ``gripper_endpoints`` (``Pi05InputConfig.gripper_value``, ``GripperClosedness`` with an endpoints table:
        {robot artifact hash: (closed_m, open_m)}) converts each sim episode's gripper state from the measured aperture
        in metres to closedness, (open - w) / (open - closed), with its robot's endpoints (by
        ``episode_metadata/robot_artifact_hash``; a robot outside the table fails; ``pi05_integration/gripper_value.py``,
        review T8). Actions are unchanged (already the command's closedness); DROID frames are already closedness.
        None keeps the stored state.

        ``joint_frames`` (``Pi05InputConfig.joint_frame``, ``CanonicalJointFrame``: {robot artifact hash: (sign[10],
        zero[10])}) maps each sim episode's arm slots into the canonical joint frame: state phi = zero + sign * q,
        joint-difference actions sign * a (``pi05_integration/joint_frame.py``, review T2/item 4); a robot outside the
        table fails, and a build that stores its recorded convention is checked against the table. DROID components
        and dynamic URMA fields (in each robot's own joints) are refused. None keeps each robot's own joints."""
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        from pi05_integration import arm_action as arm_action_contract
        from pi05_integration.gripper_value import EndpointLookup, closedness_state_tf
        from pi05_integration.image_geometry import unstretch_letterbox_tf
        from pi05_integration.joint_frame import FrameLookup
        from pi05_integration.joint_frame import apply_tf as joint_frame_tf
        from pi05_integration.joint_frame import check_recorded_tf as joint_frame_checks

        if arm_action == arm_action_contract.PER_STEP_JOINT_DELTA:
            arm_action = None
        if arm_action is not None:
            if arm_action not in arm_action_contract.CONTRACTS:
                raise ValueError(f"unknown arm action {arm_action!r}; known: {arm_action_contract.CONTRACTS}")
            if action_chunk_size is None:
                # Normalization statistics of these forms pool the chunks (compute_norm_stats.py).
                raise ValueError(f"{arm_action} labels are defined only for action chunks")
            if mobile_base and arm_action != arm_action_contract.QUERY_ANCHORED_JOINT:
                # Query-anchored joints keep the 10-wide arm block; the base chunk is built beside it unchanged.
                raise ValueError("mobile-base slots support only the per-step joint-delta and query-anchored joint arm actions")

        if rlds_input_sampling is None:
            rlds_input_sampling = default_rlds_input_sampling()
        if (
            isinstance(
                rlds_input_sampling,
                (ProcessShardedFileShuffleV1, DeterministicReplayV3),
            )
            and input_sampling_seed is None
        ):
            if process_count != 1:
                raise ValueError(
                    "process-sharded RLDS input requires the shared experiment "
                    "input_sampling_seed, separate from the process-local frame seed"
                )
            # One process: the shared file permutation and the local frame seed may coincide.
            input_sampling_seed = seed

        tf.config.set_visible_devices([], "GPU")
        assert abs(sum(d.weight for d in datasets) - 1.0) < 1e-6, "dataset weights must sum to 1.0"
        gripper_lookup = EndpointLookup(gripper_endpoints, tf) if gripper_endpoints is not None else None
        frame_lookup = FrameLookup(joint_frames, tf) if joint_frames is not None else None
        if frame_lookup is not None and any(d.name == "droid" for d in datasets):
            raise ValueError("the canonical joint frame covers sim robots only; this mixture has a DROID component")
        if isinstance(prompt_conditioning, str):
            prompt_conditioning = prompt_conditioning_from_legacy(prompt_conditioning)
        descriptor_builder = (
            prompt_conditioning.create_descriptor_builder(embodiment_bundle_module)
            if prompt_conditioning.uses_description_bundle
            else make_embodiment_descriptor
        )
        if post_close_sampling_weight is not None:
            raise NotImplementedError(POST_CLOSE_RESAMPLING_REFUSAL)
        if episode_split_role not in {"train", "validation"}:
            raise ValueError("episode_split_role must be 'train' or 'validation'")
        if action_loss_coordinates not in {"model", "physical"}:
            raise ValueError(
                "action_loss_coordinates must be 'model' or 'physical', got "
                f"{action_loss_coordinates!r}"
            )
        embodiment_filter_ids = canonicalize_embodiment_filter_ids(embodiment_filter_ids)
        behavior_prompt_overlays = tuple(behavior_prompt_overlays)
        if behavior_prompt_overlays:
            resolved_overlay_keys = tuple(
                _overlay_for_component(dataset, behavior_prompt_overlays).component_key
                for dataset in datasets
            )
            configured_overlay_keys = tuple(
                overlay.component_key for overlay in behavior_prompt_overlays
            )
            if len(set(configured_overlay_keys)) != len(configured_overlay_keys):
                raise ValueError("behavior prompt overlays must have unique component identities")
            if set(resolved_overlay_keys) != set(configured_overlay_keys):
                raise ValueError(
                    "behavior prompt overlays include an extra or missing dataset component"
                )

        builders = tuple(
            tfds.builder_from_directory(
                f"{dataset_root(data_dir, cfg)}/{cfg.name}/{cfg.version}"
            )
            for cfg in datasets
        )
        response_specs = response_history_specs(
            builder.info.features["steps"]["observation"] for builder in builders
        )
        source_split = split
        if isinstance(episode_split, ContiguousEpisodeSplit):
            source_split = (
                episode_split.train
                if episode_split_role == "train"
                else episode_split.validation
            )
        elif isinstance(
            episode_split,
            (
                AnchoredHashEmbodimentSplit,
                FrozenEmbodimentHoldoutSplit,
                HashEmbodimentCountSplit,
                HashEmbodimentSplit,
                NamespacedHashEmbodimentSplit,
                PhysicalEmbodimentAnchoredHashSplit,
                StratifiedEmbodimentEpisodeSplit,
            ),
        ):
            source_split = episode_split.source_split
        elif episode_split is not None:
            raise TypeError(
                f"unsupported episode split strategy: {type(episode_split).__name__}"
            )

        identity_record = {
            "selector": None if episode_selector is None else episode_selector.to_record(),
            "split": None if episode_split is None else episode_split.to_record(),
        }
        identities = None
        included_paths = None
        if episode_selector is not None or embodiment_filter_ids or isinstance(
            episode_split,
            (
                AnchoredHashEmbodimentSplit,
                FrozenEmbodimentHoldoutSplit,
                HashEmbodimentCountSplit,
                HashEmbodimentSplit,
                NamespacedHashEmbodimentSplit,
                PhysicalEmbodimentAnchoredHashSplit,
                StratifiedEmbodimentEpisodeSplit,
            ),
        ):
            identities = tuple(
                dataclasses.replace(
                    identity,
                    namespace=cfg.dataset_id or cfg.data_subdir or cfg.name,
                    embodiment_id=int(identity.embodiment_id) + _embodiment_id_offset(cfg),
                )
                for cfg, builder in zip(datasets, builders, strict=True)
                for identity in _episode_identities(builder, source_split, identity_record, tfds)
            )
        if identities is not None:
            included_paths = frozenset(
                item.episode_key
                for item in select_episode_views(
                    identities,
                    episode_selector=episode_selector,
                    episode_split=episode_split,
                    episode_split_role=episode_split_role,
                    embodiment_filter_ids=embodiment_filter_ids,
                )
            )

        def prepare(cfg, builder):
            overlay_spec = _overlay_for_component(cfg, behavior_prompt_overlays)
            overlay_table = None
            if overlay_spec is not None:
                loaded_overlay = load_behavior_prompt_overlay(overlay_spec)
                component_identities = _episode_identities(
                    builder, source_split, identity_record, tfds
                )
                validate_episode_key_coverage(
                    loaded_overlay,
                    tuple(identity.episode_key for identity in component_identities),
                )
                overlay_keys = tf.constant(
                    tuple(loaded_overlay.episode_suffixes), dtype=tf.string
                )
                overlay_suffixes = tf.constant(
                    tuple(loaded_overlay.episode_suffixes.values()), dtype=tf.string
                )
                overlay_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(overlay_keys, overlay_suffixes),
                    default_value="",
                )
            remap_matrix = None
            if cfg.target_joint_layout is not None:
                if cfg.joint_layout is None:
                    raise ValueError(
                        "target_joint_layout requires the immutable source joint_layout"
                    )
                target_layout = JointVectorLayout.parse(cfg.target_joint_layout)
                if target_layout is JointVectorLayout.UR5E_CONTIGUOUS_V1:
                    # The historical six-joint UR5e model-facing remap (bit-identical).
                    remap_matrix = np.zeros((10, 10), dtype=np.float32)
                    for source_slot, target_slot in joint_vector_remap_pairs(
                        cfg.joint_layout,
                        cfg.target_joint_layout,
                        n_arm=6,
                        width=10,
                    ):
                        remap_matrix[source_slot, target_slot] = 1.0
                else:
                    # Left-packed onto left-packed is the identity for every arm length (a
                    # mixed 5-7 joint XEMB/OER-9 component); other pairs need the component's
                    # one fixed arm length (an ``n_arm`` attribute on its dataset config).
                    remap_matrix = joint_vector_remap_matrix(
                        cfg.joint_layout, target_layout, n_arm=getattr(cfg, "n_arm", None), width=10
                    )

            # "droid" is the real DROID TFDS dataset (co-training source), not a MolmoSpaces
            # RLDS upload. It loads through the same pipeline below but needs its own
            # restructure/filter branch to land on the identical flat schema (so it can be mixed
            # with MolmoSpaces shards via sample_from_datasets, which requires matching
            # element_spec across inputs).
            is_droid = cfg.name == "droid"
            if is_droid and arm_action is not None:
                raise ValueError(f"{arm_action} is defined for MolmoSpaces joint builds, not DROID")
            # Keep TFDS file traversal stable and apply the seeded frame-level shuffle below.
            # File-only shuffling is not sufficient for a small held-out split: its first batches
            # can contain only a subset of embodiments even though every ID is present.
            dataset = _from_rlds_with_sampling_contract(
                dl,
                tf,
                tfds,
                builder,
                split=source_split,
                shuffle=shuffle,
                num_parallel_reads=num_parallel_reads,
                # V3 must shard the held-out reader too. Otherwise every JAX process opens
                # the complete validation file set and the global validation batch contains
                # overlapping process-local samples. This changes only traversal: the
                # episode filter below still selects the frozen empirical validation view,
                # and validation disables training-only frame resampling.
                sampling=(
                    rlds_input_sampling
                    if episode_split_role == "train"
                    or isinstance(rlds_input_sampling, DeterministicReplayV3)
                    else LegacyRldsInputSampling()
                ),
                experiment_seed=(seed if input_sampling_seed is None else input_sampling_seed),
                process_count=process_count,
                process_index=process_index,
                completed_updates=completed_updates,
            )

            if is_droid:
                # DROID marks successful episodes via the recording file path.
                def is_successful(traj):
                    return tf.strings.regex_full_match(
                        traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                    )

                dataset = dataset.filter(is_successful)

                if getattr(cfg, "exclude_pick_and_place", True):
                    # Co-training filter: realrig9k already covers pick-and-place, so drop any
                    # DROID episode whose language annotation mentions it (checked across all 3
                    # variants). Opt out via RLDSDataset(exclude_pick_and_place=False) to train
                    # on all of DROID.
                    def is_not_pick_and_place(traj):
                        texts = tf.strings.lower(
                            tf.stack(
                                [
                                    traj["language_instruction"][0],
                                    traj["language_instruction_2"][0],
                                    traj["language_instruction_3"][0],
                                ]
                            )
                        )
                        return tf.reduce_all(
                            tf.logical_not(tf.strings.regex_full_match(texts, ".*pick and place.*"))
                        )

                    dataset = dataset.filter(is_not_pick_and_place)

                def has_instruction(traj):
                    return has_nonempty_instruction(
                        tf.stack(
                            [
                                traj["language_instruction"][0],
                                traj["language_instruction_2"][0],
                                traj["language_instruction_3"][0],
                            ]
                        ),
                        tf,
                    )

                dataset = dataset.filter(has_instruction)

            # DROID's non-idle step ranges (openpi's recipe). Like the success filter above this
            # defines the data, so it also applies to validation and normalization scans.
            idle_table = None
            if is_droid and getattr(cfg, "filter_dict_path", None) is not None:
                idle_table = droid_idle_filter_table(cfg.filter_dict_path, tf)

            if included_paths is not None:
                keys = tf.constant(sorted(included_paths), dtype=tf.string)
                table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        keys,
                        tf.ones(tf.shape(keys), dtype=tf.int32),
                    ),
                    default_value=0,
                )

                def selected_partition(traj):
                    metadata = traj["traj_metadata"]["episode_metadata"]
                    return table.lookup(metadata["file_path"][0]) > 0

                dataset = dataset.filter(selected_partition)
            if repeat:
                dataset = dataset.repeat()

            # Only train on joint-delta (+ absolute gripper) episodes; some MolmoSpaces RLDS
            # uploads mix in absolute end-effector ("ee") episodes tagged via episode_metadata.
            # DROID episodes have no such field and are already handled by the filters above.
            def is_joint_action_space(traj):
                action_space = traj["traj_metadata"]["episode_metadata"]["action_space"][0]
                return tf.equal(action_space, "joint")

            if not is_droid:
                dataset = dataset.filter(is_joint_action_space)

            # Optional single-arm narrowing. The fr_ur5 builds span 7-DoF franka_droid and 6-DoF
            # ur5e_2f85 episodes, which put the gripper on different action dimensions -- dim 7
            # vs dim 6. action_loss_mask marks that correctly for the training loss, but
            # compute_norm_stats ignores the mask, so dim 6 would blend franka joint deltas
            # (~0.1 rad) with ur5e gripper flags (0/1) into one column. Selecting one arm keeps
            # every episode of a mixture on a single layout.
            wanted_robot = getattr(cfg, "robot", None)
            if wanted_robot is not None:
                wanted_robot_tensor = tf.constant(wanted_robot.encode())

                def is_wanted_robot(traj):
                    return tf.equal(
                        traj["traj_metadata"]["episode_metadata"]["robot"][0], wanted_robot_tensor
                    )

                dataset = dataset.filter(is_wanted_robot)

            # Canonical action/state layout: joint deltas in the leading dimensions, absolute
            # gripper at _CANONICAL_GRIPPER_DIM, padding elsewhere. Builds store the gripper
            # immediately after that embodiment's own joints, so a 6-DoF ur5e_2f85 lands it on
            # dim 6 while every 7-DoF franka build lands it on dim 7. action_mask does NOT
            # disambiguate those: it marks valid-vs-padding, and both arms mark dim 6 valid, so
            # a shared normalization table would blend franka joint deltas (~0.1 rad) with ur5e
            # gripper flags (0/1) in one column. Pinning the gripper to a fixed dimension makes
            # a single global table correct for every embodiment, so a mixture no longer has to
            # be narrowed to one arm. No-op for episodes already in the canonical layout.
            def canonicalize_gripper_dim(actions, state, mask):
                action_dim = tf.shape(actions)[-1]
                gripper_idx = tf.reduce_sum(tf.cast(mask[0], tf.int32)) - 1

                def remap():
                    col = tf.range(action_dim)
                    joint_sel = col < gripper_idx

                    def move(x):
                        gripper = tf.gather(x, gripper_idx, axis=-1)  # (T,)
                        joints = tf.where(joint_sel[None, :], x, tf.zeros_like(x))
                        onehot = tf.one_hot(_CANONICAL_GRIPPER_DIM, action_dim, dtype=x.dtype)
                        return joints + gripper[:, None] * onehot[None, :]

                    new_mask = tf.broadcast_to(
                        tf.logical_or(joint_sel, tf.equal(col, _CANONICAL_GRIPPER_DIM))[None, :],
                        tf.shape(mask),
                    )
                    return move(actions), move(state), new_mask

                out_actions, out_state, out_mask = tf.cond(
                    tf.equal(gripper_idx, _CANONICAL_GRIPPER_DIM),
                    lambda: (actions, state, mask),
                    remap,
                )
                out_actions.set_shape(actions.shape)
                out_state.set_shape(state.shape)
                out_mask.set_shape(mask.shape)
                return out_actions, out_state, out_mask

            def restructure_droid(traj):
                if action_time:
                    raise ValueError("action time conditioning needs episode_metadata/policy_dt_s; DROID builds have none")
                traj_len = tf.shape(traj["action"])[0]
                # DROID actions are absolute joint position; convert to delta-joint + absolute
                # gripper (columns [0:7]=delta joints, [7]=gripper, [8:10]=padding) to match
                # realrig9k's convention so both can share one action head / normalizer.
                delta_joint = tf.cast(traj["action_dict"]["joint_position"], tf.float32) - tf.cast(
                    traj["observation"]["joint_position"], tf.float32
                )
                abs_gripper = tf.cast(traj["action_dict"]["gripper_position"], tf.float32)
                pad2 = tf.zeros([traj_len, 2], tf.float32)
                actions = tf.concat([delta_joint, abs_gripper, pad2], axis=-1)  # (T,10)
                action_loss_mask = tf.concat(
                    [tf.ones([traj_len, 8], tf.bool), tf.zeros([traj_len, 2], tf.bool)], axis=-1
                )  # (T,10)
                state = tf.concat(
                    [
                        tf.cast(traj["observation"]["joint_position"], tf.float32),
                        tf.cast(traj["observation"]["gripper_position"], tf.float32),
                        tf.zeros([traj_len, 2], tf.float32),
                    ],
                    axis=-1,
                )  # (T,10)
                # One camera / one instruction choice per episode (not per frame).
                image = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                instruction = pick_nonempty_instruction(
                    tf.stack(
                        [
                            traj["language_instruction"][0],
                            traj["language_instruction_2"][0],
                            traj["language_instruction_3"][0],
                        ]
                    ),
                    tf,
                )
                prompt = tf.fill([traj_len], instruction)
                idle_fields = {}
                if idle_table is not None:
                    metadata = traj["traj_metadata"]["episode_metadata"]
                    step_id = (
                        metadata["recording_folderpath"]
                        + "--"
                        + metadata["file_path"]
                        + "--"
                        + tf.as_string(tf.range(traj_len))
                    )
                    # Removed again right after flattening, before the mixture sees it.
                    idle_fields["_passes_idle_filter"] = idle_table.lookup(droid_step_fingerprint(step_id, tf)) > 0
                return {
                    **idle_fields,
                    "actions": actions,
                    # Same keys as restructure(): sample_from_datasets needs identical specs.
                    "action_semantic_mask": action_loss_mask,
                    "action_loss_mask": action_loss_mask,
                    "image": image,                                       # (T,) jpeg, 180x320
                    "wrist_image": traj["observation"]["wrist_image_left"],  # (T,) jpeg, 180x320
                    "state": state,
                    "state_mask": tf.concat(
                        [tf.ones([traj_len, 8], tf.bool), tf.zeros([traj_len, 2], tf.bool)], axis=-1
                    ),
                    "prompt": prompt,
                    "embodiment_id": tf.fill([traj_len], tf.constant(-2, tf.int32)),  # metrics-only sentinel
                    "task_id": tf.fill([traj_len], tf.constant(-1, tf.int32)),
                    **base_fields(traj, traj_len),
                }

            def base_fields(traj, traj_len):
                # Holonomic base (RLDS mobile_base_feature_v1/v2), kept beside the arm vector so
                # that every arm key, mask and gripper rule above stays exactly as it was.
                if not mobile_base:
                    return {}
                from pi05_integration.mobile_base import RLDS_FIELDS

                observation = traj["observation"]
                action_key, action_mask_key, state_key, state_mask_key = RLDS_FIELDS[mobile_base]
                for contract, (other_action, _, other_state, _) in RLDS_FIELDS.items():
                    if contract != mobile_base and (other_action in traj or other_state in observation):
                        # Same slots, different units: a build of another version must not be misread.
                        raise ValueError(
                            f"{mobile_base} reads {action_key}/{state_key}; this component has the "
                            f"{contract} fields {other_action}/{other_state}"
                        )
                zeros = tf.zeros([traj_len, 3], tf.float32)
                false = tf.zeros([traj_len, 3], tf.bool)
                fields = {
                    "base_actions": tf.cast(traj[action_key], tf.float32) if action_key in traj else zeros,
                    "base_action_mask": traj[action_mask_key] if action_mask_key in traj else false,
                    "base_state": (
                        tf.cast(observation[state_key], tf.float32) if state_key in observation else zeros
                    ),
                    "base_state_mask": (
                        observation[state_mask_key] if state_mask_key in observation else false
                    ),
                }
                mask = fields["base_action_mask"]
                period_checks = []
                if mobile_base_dt_s is not None and action_key in traj:
                    metadata = traj["traj_metadata"]["episode_metadata"]
                    if "base_velocity_dt_s" not in metadata:
                        raise ValueError(f"{mobile_base} with dt_s needs episode_metadata/base_velocity_dt_s")
                    period_checks.append(tf.debugging.assert_near(
                        tf.cast(metadata["base_velocity_dt_s"], tf.float32),
                        tf.fill(tf.shape(metadata["base_velocity_dt_s"]), tf.constant(mobile_base_dt_s, tf.float32)),
                        rtol=1e-5, atol=0.0,
                        message=f"episode base_velocity_dt_s differs from the config's dt_s={mobile_base_dt_s}",
                    ))
                with tf.control_dependencies([
                    *period_checks,
                    tf.debugging.assert_equal(
                        mask, fields["base_state_mask"], message="base action and state masks disagree"
                    ),
                    tf.debugging.assert_equal(
                        tf.reduce_any(mask, axis=-1), tf.reduce_all(mask, axis=-1),
                        message="a base mask marks all three coordinates or none",
                    ),
                ]):
                    fields["base_actions"] = tf.identity(fields["base_actions"])
                return fields

            def base_tcp_fields(traj, actions, state, action_mask, state_mask):
                """Per-step absolute targets of the BASE TCP forms (``arm_reference_feature_v1``);
                ``chunk_actions`` anchors them at each query step.

                ``query_anchored_base_tcp``: actions [TCP goal pose9, gripper], all ten valid; the
                joint state is unchanged and the measured TCP travels as ``_base_tcp``.
                ``query_anchored_joint_and_base_tcp`` (a left-packed n-joint layout): actions
                [joint targets (n), TCP goal pose9, gripper] and state [q (n), aperture, TCP pose9],
                n + 10 of 20 slots valid."""
                goal_key, tcp_key = arm_action_contract.BASE_TCP_GOAL_KEY, arm_action_contract.BASE_TCP_KEY
                if goal_key not in traj or tcp_key not in traj["observation"]:
                    raise ValueError(
                        f"{arm_action} reads the {arm_action_contract.ARM_REFERENCE_FEATURE_V1} fields "
                        f"steps/{goal_key} and steps/observation/{tcp_key} of {cfg.name}; convert with "
                        "mlspaces_to_rlds.py --arm-reference-features"
                    )
                goal = tf.cast(traj[goal_key], actions.dtype)
                tcp = tf.cast(traj["observation"][tcp_key], state.dtype)
                traj_len = tf.shape(actions)[0]
                gripper_index = tf.cast(tf.reduce_max(tf.where(action_mask[0])), tf.int32)
                gripper = tf.gather(actions, gripper_index, axis=-1)[:, None]
                checks = [
                    tf.debugging.assert_all_finite(goal, "commanded BASE TCP goals must be finite"),
                    tf.debugging.assert_all_finite(tcp, "measured BASE TCP must be finite"),
                ]
                if arm_action == arm_action_contract.QUERY_ANCHORED_BASE_TCP:
                    with tf.control_dependencies(checks):
                        return (tf.concat([goal, gripper], axis=-1), state,
                                tf.ones([traj_len, arm_action_contract.ARM_WIDTH], tf.bool), state_mask,
                                {"_base_tcp": tcp})
                # Joint + BASE TCP: the joint targets are the deltas plus the measured joints (the
                # converter's steps/action is target - measured on the arm slots).
                n_arm = gripper_index
                width = tf.shape(action_mask)[-1]
                left_packed = tf.broadcast_to((tf.range(width) < n_arm + 1)[None], tf.shape(action_mask))
                slots = tf.range(arm_action_contract.JOINT_AND_BASE_TCP_WIDTH)
                valid = slots < n_arm + 10
                checks += [
                    tf.debugging.assert_equal(action_mask, left_packed, "joint + BASE TCP needs a left-packed action"),
                    tf.debugging.assert_equal(state_mask, left_packed, "joint + BASE TCP needs a left-packed state"),
                ]
                with tf.control_dependencies(checks):
                    arm = (tf.range(width) < n_arm)[None]
                    targets = actions + tf.where(arm, tf.cast(state, actions.dtype), tf.zeros_like(actions))
                    # Gather sources: the targets then [goal, gripper]; the proprio then the TCP.
                    action_index = tf.where(valid, tf.where(slots < n_arm, slots, slots - n_arm + width), 0)
                    state_index = tf.where(valid, tf.where(slots <= n_arm, slots, slots - n_arm - 1 + width), 0)
                    new_actions = tf.gather(tf.concat([targets, goal, gripper], -1), action_index, axis=1)
                    new_state = tf.gather(tf.concat([state, tcp], -1), state_index, axis=1)
                    mask = tf.broadcast_to(valid[None], [traj_len, arm_action_contract.JOINT_AND_BASE_TCP_WIDTH])
                    return (tf.where(mask, new_actions, tf.zeros_like(new_actions)),
                            tf.where(mask, new_state, tf.zeros_like(new_state)), mask, mask, {})

            def restructure(traj):
                if is_droid:
                    return restructure_droid(traj)
                traj_len = tf.shape(traj["action"])[0]
                md = traj["traj_metadata"]["episode_metadata"]
                # Exact task-config instruction for new data; old RLDS schemas retain
                # their established lowercase task/target reconstruction. Builds predating the
                # task/target metadata redesign -- the fr_ur5_100var_pick_place_* v2sub100 sets --
                # carry only the per-step language_instruction; fall back to it so they can join
                # a mixture instead of raising KeyError here.
                has_task_md = "task" in md and "target" in md
                if has_task_md:
                    prompt = _episode_prompt(md, traj_len, tf)
                else:
                    prompt = traj["language_instruction"]
                if overlay_table is not None:
                    episode_key = md["file_path"][0]
                    suffix = overlay_table.lookup(episode_key)
                    with tf.control_dependencies([
                        tf.debugging.assert_positive(
                            tf.strings.length(suffix),
                            message="behavior prompt overlay missing episode key",
                        )
                    ]):
                        overlaid_prompt = tf.strings.join(
                            [tf.strings.strip(prompt[0]), " ", tf.strings.strip(suffix)]
                        )
                    prompt = tf.fill([traj_len], overlaid_prompt)
                embodiment_id = tf.cast(
                    md["embodiment_id"] if "embodiment_id" in md else tf.fill([traj_len], -1),
                    tf.int32,
                )
                if _embodiment_id_offset(cfg) and "embodiment_id" in md:
                    # Opt-in namespacing of a component that reuses local IDs (RLDSDataset.embodiment_id_offset).
                    embodiment_id = embodiment_id + tf.constant(_embodiment_id_offset(cfg), tf.int32)
                if has_task_md:
                    normalized_task = tf.strings.lower(md["task"])
                    task_id = tf.where(
                        tf.equal(normalized_task, "open"),
                        tf.zeros([traj_len], tf.int32),
                        tf.where(
                            tf.equal(normalized_task, "close"),
                            tf.ones([traj_len], tf.int32),
                            tf.fill([traj_len], -1),
                        ),
                    )
                else:
                    # open/close taxonomy does not apply to these builds; -1 = unlabelled.
                    task_id = tf.fill([traj_len], -1)
                if not isinstance(prompt_conditioning, NoPromptConditioning):
                    # dlimp repeats episode metadata over the trajectory. Format it
                    # once per episode in Python (JSON parsing), then broadcast.
                    def first(key, default):
                        return md[key][0] if key in md else tf.constant(default)

                    if prompt_conditioning.uses_control_period and "policy_dt_s" not in md:
                        raise ValueError(
                            f"{type(prompt_conditioning).__name__} reads episode_metadata/policy_dt_s, which "
                            f"{cfg.name} lacks; convert with mlspaces_to_rlds.py --policy-dt-metadata"
                        )

                    def format_conditioned(base, robot, action_space, params, emb_id, variant, period):
                        values = [
                            value.numpy() if hasattr(value, "numpy") else value
                            for value in (base, robot, action_space, params, emb_id, variant, period)
                        ]
                        base, robot, action_space, params, emb_id, variant, period = values
                        return prompt_conditioning.condition(
                            base.decode(),
                            robot=robot,
                            action_space=action_space,
                            embodiment_params=params,
                            embodiment_id=emb_id,
                            variant=variant,
                            descriptor_builder=descriptor_builder,
                            # Only period-reading conditionings take the keyword (others have fixed signatures).
                            **({"control_period_s": float(period)} if prompt_conditioning.uses_control_period else {}),
                        ).encode()

                    conditioned = tf.py_function(
                        format_conditioned,
                        # prompt[0] is the per-episode base string: the joined task/target for
                        # builds that carry it, otherwise the build's language_instruction.
                        inp=[prompt[0], first("robot", b""),
                             first("action_space", b"joint"), first("embodiment_params", b""),
                             first("embodiment_id", -1), first("variant", b""),
                             tf.cast(first("policy_dt_s", 0.0), tf.float32)],
                        Tout=tf.string,
                    )
                    conditioned.set_shape(())
                    prompt = tf.fill([traj_len], conditioned)
                actions = traj["action"]
                action_semantic_mask = traj.get(
                    "action_mask", tf.ones_like(traj["action"], dtype=tf.bool)
                )
                state = traj["observation"]["proprio"]
                state_mask = traj["observation"].get(
                    "proprio_mask", tf.ones_like(state, dtype=tf.bool)
                )
                if gripper_lookup is not None:
                    if "robot_artifact_hash" not in md:
                        raise ValueError(
                            f"gripper_closedness_v1 at load reads episode_metadata/robot_artifact_hash, which "
                            f"{cfg.name} lacks"
                        )
                    closed_m, open_m = gripper_lookup.row(md["robot_artifact_hash"][0])
                    state = closedness_state_tf(state, state_mask, closed_m, open_m, tf)
                if frame_lookup is not None:
                    if "robot_artifact_hash" not in md:
                        raise ValueError(
                            f"the canonical joint frame reads episode_metadata/robot_artifact_hash, which "
                            f"{cfg.name} lacks"
                        )
                    sign, zero = frame_lookup.row(md["robot_artifact_hash"][0])
                    state, actions = joint_frame_tf(
                        state, actions, sign, zero, joint_frame_checks(traj["observation"], sign, zero, tf), tf
                    )
                response_fields = response_history_fields(
                    traj["observation"], traj_len, response_specs, tf
                )
                dynamic_fields = {}
                for key in (
                    "dynamic_urma_values",
                    "dynamic_urma_mask",
                    "dynamic_urma5_values",
                    "dynamic_urma51_values",
                    "dynamic_urma51_analytic_mask",
                ):
                    if key in traj["observation"]:
                        if frame_lookup is not None:
                            raise ValueError(f"the canonical joint frame does not map the dynamic URMA field {key}")
                        dynamic_fields[key] = traj["observation"][key]
                if remap_matrix is not None:
                    matrix = tf.constant(remap_matrix, dtype=actions.dtype)
                    actions = tf.linalg.matmul(actions, matrix)
                    state = tf.linalg.matmul(state, tf.cast(matrix, state.dtype))
                    action_semantic_mask = tf.linalg.matmul(
                        tf.cast(action_semantic_mask, tf.int32),
                        tf.cast(matrix, tf.int32),
                    ) > 0
                    state_mask = tf.linalg.matmul(
                        tf.cast(state_mask, tf.int32), tf.cast(matrix, tf.int32)
                    ) > 0
                elif cfg.target_joint_layout is None:
                    # An explicit target_joint_layout owns the model-facing layout (including the
                    # identity left-packed one, where remap_matrix is None); otherwise pin the
                    # gripper to _CANONICAL_GRIPPER_DIM (no-op for 7-DoF franka builds).
                    actions, state, action_semantic_mask = canonicalize_gripper_dim(
                        actions, state, action_semantic_mask
                    )
                arm_fields = {}
                if arm_action in arm_action_contract.READS_BASE_TCP:
                    actions, state, action_semantic_mask, state_mask, arm_fields = base_tcp_fields(
                        traj, actions, state, action_semantic_mask, state_mask
                    )
                time_fields = {}
                if action_time:
                    if "policy_dt_s" not in md:
                        raise ValueError(
                            f"action time conditioning reads episode_metadata/policy_dt_s, which {cfg.name} lacks; "
                            "convert with mlspaces_to_rlds.py --policy-dt-metadata"
                        )
                    dt = tf.cast(md["policy_dt_s"], tf.float32)  # (T,): dlimp repeats episode metadata
                    with tf.control_dependencies([
                        tf.debugging.assert_positive(dt, message="episode_metadata/policy_dt_s must be positive"),
                        tf.debugging.assert_all_finite(dt, "episode_metadata/policy_dt_s must be finite"),
                    ]):
                        time_fields["action_dt_s"] = tf.identity(dt)
                return {
                    "actions": actions,                              # (T,10) semantic joint vector
                    "action_semantic_mask": action_semantic_mask,    # (T,D), physical controls
                    "action_loss_mask": action_semantic_mask,        # replaced after future chunking
                    "image": traj["observation"]["image_primary"],   # (T,) jpeg
                    "wrist_image": traj["observation"]["image_wrist"],  # (T,) jpeg
                    "state": state,                                  # (T,10), same semantic layout
                    "state_mask": state_mask,                        # (T,10), physical state
                    "prompt": prompt,                                # (T,)
                    "embodiment_id": embodiment_id,                  # (T,), metrics only
                    # Preserve the immutable raw ID above while making independently
                    # published component namespaces available to conditioning bundles.
                    "dataset_component_id": tf.fill(
                        [traj_len], tf.constant(cfg.dataset_id or cfg.data_subdir or cfg.name)
                    ),
                    "task_id": task_id,                              # (T,), metrics only
                    **response_fields,
                    **dynamic_fields,
                    **base_fields(traj, traj_len),
                    **arm_fields,
                    **time_fields,                                   # (T,) seconds, action time only
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def query_anchored_chunk(traj, idx, gripper_index):
                """(T, H, D) labels of a query-anchored form: each step's absolute targets, holding the
                last one past the episode end, relative to the state measured at the query step.
                Also returns the (T, D) label of no motion for the BASE TCP forms (else None)."""
                actions = traj["actions"]
                state = tf.cast(traj["state"], actions.dtype)
                columns = tf.range(tf.shape(actions)[-1])
                if arm_action == arm_action_contract.QUERY_ANCHORED_JOINT:
                    arm = tf.logical_and(traj["action_semantic_mask"], (columns < gripper_index)[None])
                    anchor = tf.where(arm, state, tf.zeros_like(state))
                    return tf.gather(actions + anchor, idx) - anchor[:, None, :], None
                chunk = tf.gather(actions, idx)
                if arm_action == arm_action_contract.QUERY_ANCHORED_BASE_TCP:
                    pose_offset = 0
                    tcp = tf.cast(traj.pop("_base_tcp"), actions.dtype)
                    label = tf.concat(
                        [_relative_pose9(chunk[..., :9], tcp[:, None, :], tf), chunk[..., 9:]], axis=-1
                    )
                else:
                    # [targets (n), goal (9), gripper]: anchor the joints at q(t), the goal at TCP(t).
                    n_arm = gripper_index - arm_action_contract.POSE_DIM
                    pose_offset = n_arm
                    joint_anchor = tf.where((columns < n_arm)[None], state, tf.zeros_like(state))
                    tcp = tf.gather(state, n_arm + 1 + tf.range(9), axis=-1)
                    pose = _relative_pose9(
                        tf.gather(chunk, n_arm + tf.range(9), axis=-1), tcp[:, None, :], tf
                    )
                    in_pose = tf.logical_and(columns >= n_arm, columns < n_arm + 9)
                    label = tf.gather(
                        tf.concat([chunk - joint_anchor[:, None, :], pose], axis=-1),
                        tf.where(in_pose, columns - n_arm + tf.shape(actions)[-1], columns),
                        axis=-1,
                    )
                rest = tf.one_hot(pose_offset + 3, tf.shape(actions)[-1], dtype=actions.dtype) + tf.one_hot(
                    pose_offset + 7, tf.shape(actions)[-1], dtype=actions.dtype
                )
                return label, tf.broadcast_to(rest[None], tf.shape(actions))

            def chunk_actions(traj):
                traj_len = tf.shape(traj["actions"])[0]
                raw_idx = tf.broadcast_to(tf.range(action_chunk_size)[None], [traj_len, action_chunk_size]) + \
                    tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, action_chunk_size])
                idx = tf.minimum(raw_idx, traj_len - 1)
                chunk = tf.gather(traj["actions"], idx)   # (T,H,D), edge-padded temporarily
                time_valid = raw_idx < traj_len

                # Synthetic targets are materialized safely for consumers which inspect the
                # tensor, but they are always excluded from the loss below. The gripper is the
                # LAST valid dimension for every embodiment. Take the last set index rather than
                # the count of set bits: after gripper canonicalization a 6-DoF arm's mask is
                # non-contiguous (joints 0-5 valid, dim 6 padding, gripper at 7), so counting
                # would point at dim 6. Identical to count-1 for any left-packed mask.
                first_dimension_mask = traj["action_semantic_mask"][0]
                gripper_index = tf.cast(tf.reduce_max(tf.where(first_dimension_mask)), tf.int32)
                tf.debugging.assert_greater_equal(gripper_index, 0, "an action needs at least one valid dimension")
                if arm_action is None:
                    final_gripper = traj["actions"][-1, gripper_index]
                    gripper_hold = tf.one_hot(
                        gripper_index, tf.shape(chunk)[-1], on_value=final_gripper, off_value=tf.cast(0, chunk.dtype)
                    )
                    traj["actions"] = tf.where(time_valid[..., None], chunk, gripper_hold[None, None, :])
                else:
                    traj["actions"], rest = query_anchored_chunk(traj, idx, gripper_index)
                    if rest is not None:
                        traj["_action_rest"] = rest
                semantic_chunk = tf.gather(traj["action_semantic_mask"], idx)
                # The default reproduces released Pi0.5 by broadcasting timestep
                # validity over the fixed model width. New named configs may instead
                # retain the left-packed physical-coordinate mask.
                traj["action_loss_mask"] = chunk_action_loss_mask(
                    time_valid,
                    semantic_chunk,
                    coordinates=action_loss_coordinates,
                    logical_and=tf.logical_and,
                )
                if mobile_base:
                    # Past the episode end the base holds: a zero body-frame offset (v1) or velocity (v2).
                    base_chunk = tf.gather(traj["base_actions"], idx)
                    traj["base_actions"] = tf.where(time_valid[..., None], base_chunk, tf.zeros_like(base_chunk))
                return traj

            if action_chunk_size is not None:
                if action_chunk_size <= 0:
                    raise ValueError("action_chunk_size must be positive or None")
                dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
            dataset = dataset.flatten(
                num_parallel_calls=(
                    rlds_input_sampling.flatten_cycle_length
                    if isinstance(rlds_input_sampling, DeterministicReplayV3)
                    else num_parallel_calls
                )
            )

            # Drop fully-static frames: the whole predicted action chunk asks for ~zero arm
            # displacement and no gripper change (idle/dwell moments, overrepresented near episode
            # boundaries and pauses). The gripper is an absolute position, so its value is not
            # motion: see chunk_motion_norm. These are usually trivially easy, but the droid co-train
            # full-FT instability investigation found a subset of them becoming disproportionately
            # high-loss once the flow-matching time-conditioning pathway (time_mlp_in/out) started
            # drifting -- a peaked, near-delta-function target is a harder regime for a continuous
            # flow-matching objective than it looks. Only applied when shuffle=True, matching the
            # post_close_sampling_weight convention below: validation and unchunked normalization
            # call this loader with shuffle=False and must keep the true empirical distribution.
            # The threshold is configurable because the cut is not scale-free: joint deltas have
            # sigma ~0.022-0.036 rad, so raising it from the 1e-4 default trades a larger drop in
            # near-idle frames against starting to remove genuine slow motion. Existing configs
            # keep 1e-4 so their data distribution stays comparable to the completed runs.
            if idle_table is not None:
                dataset = dataset.filter(lambda frame: frame["_passes_idle_filter"])
                dataset = dataset.frame_map(
                    lambda frame: {k: v for k, v in frame.items() if k != "_passes_idle_filter"},
                    num_parallel_calls,
                )

            if shuffle:
                def is_nonstatic(frame):
                    base = (
                        {
                            "base_actions": (
                                frame["base_actions"][None]
                                if action_chunk_size is None
                                else frame["base_actions"]
                            ),
                            "base_mask": frame["base_action_mask"],
                        }
                        if mobile_base
                        else {}
                    )
                    if action_chunk_size is None:
                        # Unchunked frames hold one action; give the helper a length-1 chunk.
                        return chunk_motion_norm(
                            frame["actions"][None],
                            frame["action_loss_mask"][None],
                            frame["action_semantic_mask"],
                            tf,
                            **base,
                        ) > static_action_threshold
                    return chunk_motion_norm(
                        frame["actions"], frame["action_loss_mask"], frame["action_semantic_mask"], tf, **base,
                        rest=frame.get("_action_rest"),
                    ) > static_action_threshold

                dataset = dataset.filter(is_nonstatic)
            if arm_action in arm_action_contract.READS_BASE_TCP:
                dataset = dataset.frame_map(
                    lambda frame: {k: v for k, v in frame.items() if k != "_action_rest"}, num_parallel_calls
                )

            # The accepted open-drawer trajectories spend roughly four times as many observation
            # frames approaching as they do grasping/pulling. Optionally rebalance the *training*
            # observation distribution before image decoding. The gripper is the highest valid
            # action slot for every supported embodiment; command 1 is closed in the
            # canonical absolute-gripper representation. Validation and unchunked normalization
            # call this loader with shuffle=False, so they always preserve the empirical data.
            if shuffle and post_close_sampling_weight is not None:
                def is_post_close(frame):
                    valid_dimensions = tf.reduce_sum(
                        tf.cast(frame["action_semantic_mask"], tf.int32)
                    )
                    gripper_index = valid_dimensions - 1
                    tf.debugging.assert_greater_equal(
                        gripper_index, 0, "an action needs at least one valid dimension"
                    )
                    return tf.gather(frame["actions"][0], gripper_index) >= 0.5

                pre_close = dataset.filter(lambda frame: tf.logical_not(is_post_close(frame)))
                post_close = dataset.filter(is_post_close)
                dataset = dl.DLataset.sample_from_datasets(
                    [pre_close, post_close],
                    weights=[1.0 - post_close_sampling_weight, post_close_sampling_weight],
                    seed=(
                        rlds_input_sampling.mixture_seed(
                            sampler_seed=input_sampling_seed,
                            process_index=process_index,
                            purpose="post_close",
                        )
                        if isinstance(rlds_input_sampling, DeterministicReplayV3)
                        else int(seed)
                    ),
                )

            if decode_images and not isinstance(rlds_input_sampling, DeterministicReplayV3):
                def decode(frame):
                    frame["image"] = tf.io.decode_image(
                        frame["image"], expand_animations=False, dtype=tf.uint8
                    )
                    frame["wrist_image"] = tf.io.decode_image(
                        frame["wrist_image"], expand_animations=False, dtype=tf.uint8
                    )
                    if is_droid:
                        # DROID's native 180x320 must match realrig9k's 256x256 / 128x128
                        # exactly: sample_from_datasets requires identical element_spec across
                        # mixed inputs. Letterbox rather than stretch (see letterbox_image).
                        frame["image"] = letterbox_image(frame["image"], 256, 256, tf)
                        frame["wrist_image"] = letterbox_image(frame["wrist_image"], 128, 128, tf)
                    elif letterbox_source is not None:
                        frame["image"] = unstretch_letterbox_tf(frame["image"], *letterbox_source, 256, tf)
                        frame["wrist_image"] = unstretch_letterbox_tf(frame["wrist_image"], *letterbox_source, 128, tf)
                    return frame

                dataset = dataset.frame_map(decode, num_parallel_calls)
            return dataset

        all_ds = [prepare(cfg, builder) for cfg, builder in zip(datasets, builders, strict=True)]
        final = dl.DLataset.sample_from_datasets(
            all_ds,
            weights=[d.weight for d in datasets],
            seed=(
                rlds_input_sampling.mixture_seed(
                    sampler_seed=input_sampling_seed,
                    process_index=process_index,
                    purpose="dataset_components",
                )
                if isinstance(rlds_input_sampling, DeterministicReplayV3)
                else int(seed)
            ),
        )
        if shuffle:
            final = final.shuffle(
                (
                    rlds_input_sampling.frame_shuffle_buffer_size
                    if isinstance(rlds_input_sampling, DeterministicReplayV3)
                    else shuffle_buffer_size
                ),
                seed=(
                    rlds_input_sampling.frame_shuffle_seed(
                        sampler_seed=input_sampling_seed,
                        process_index=process_index,
                    )
                    if isinstance(rlds_input_sampling, DeterministicReplayV3)
                    else int(seed)
                ),
                reshuffle_each_iteration=not isinstance(
                    rlds_input_sampling, DeterministicReplayV3
                ),
            )
        if isinstance(rlds_input_sampling, DeterministicReplayV3):
            skipped = rlds_input_sampling.skip_frames(
                completed_updates=completed_updates,
                local_batch_size=batch_size,
            )
            if skipped:
                final = final.skip(skipped)
            if decode_images:
                if letterbox_source is not None and any(cfg.name == "droid" for cfg in datasets):
                    # Decoded after mixing: a DROID frame could not be told from a sim frame here.
                    raise NotImplementedError("letterbox_source with DROID components under DeterministicReplayV3")

                def decode(frame):
                    frame["image"] = tf.io.decode_image(
                        frame["image"], expand_animations=False, dtype=tf.uint8
                    )
                    frame["wrist_image"] = tf.io.decode_image(
                        frame["wrist_image"], expand_animations=False, dtype=tf.uint8
                    )
                    if letterbox_source is not None:
                        frame["image"] = unstretch_letterbox_tf(frame["image"], *letterbox_source, 256, tf)
                        frame["wrist_image"] = unstretch_letterbox_tf(frame["wrist_image"], *letterbox_source, 128, tf)
                    return frame

                final = final.frame_map(decode, num_parallel_calls)
        final = final.batch(batch_size)
        # Host input-pipeline tuning. The historical values (1 GB autotune budget, no
        # terminal prefetch) are the defaults so recorded lineages keep their host behaviour;
        # these knobs change throughput only, never which frames are produced or their order.
        if ram_budget_gb < 1:
            raise ValueError("ram_budget_gb must be positive")
        if final_prefetch < 0:
            raise ValueError("final_prefetch must be non-negative")
        final = final.with_ram_budget(ram_budget_gb)
        if final_prefetch:
            final = final.prefetch(final_prefetch)
        if deterministic or isinstance(rlds_input_sampling, DeterministicReplayV3):
            options = tf.data.Options()
            options.experimental_deterministic = True
            final = final.with_options(options)
            if final.options().experimental_deterministic is not True:
                raise RuntimeError("failed to enforce deterministic tf.data execution")

        self.dataset = final
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # ~39k episodes * ~70 steps; approximate, only used for progress bars. The
        # iterator yields batches, so returning the raw frame count exaggerates ETA by
        # approximately the batch size.
        return approximate_batch_count(self.batch_size)
