"""Small, explicit RLDS mixture audit used by pi0.5 training and standalone reports."""

from __future__ import annotations

from collections import Counter
import dataclasses
import re

from pi05_integration.behavior_prompt_overlay import load_behavior_prompt_overlay
from pi05_integration.data_processing import AnchoredHashEmbodimentSplit
from pi05_integration.data_processing import ContiguousEpisodeSplit
from pi05_integration.data_processing import DeterministicReplayV3
from pi05_integration.data_processing import FrozenEmbodimentHoldoutSplit
from pi05_integration.data_processing import HashEmbodimentCountSplit
from pi05_integration.data_processing import HashEmbodimentSplit
from pi05_integration.data_processing import NamespacedHashEmbodimentSplit
from pi05_integration.data_processing import PhysicalEmbodimentAnchoredHashSplit
from pi05_integration.data_processing import StratifiedEmbodimentEpisodeSplit

from openpi.training.molmospaces_rlds_dataset import _episode_views
from openpi.training.molmospaces_rlds_dataset import canonicalize_embodiment_filter_ids
from openpi.training.molmospaces_rlds_dataset import dataset_root
from openpi.training.molmospaces_rlds_dataset import embodiment_id_is_selected
from openpi.training.molmospaces_rlds_dataset import select_episode_views


def _text(value) -> str:
    value = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def scan_rlds_mixture(data_config, *, max_episodes: int | None = 1000) -> list[dict]:
    """Aggregate episode/frame counts by the comparison metadata axes.

    ``max_episodes`` bounds startup I/O per dataset in the mixture. Every returned row states
    whether it is sampled; run the standalone reporter with ``max_episodes=None`` for an
    exhaustive table.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    counts: Counter[tuple[str, str]] = Counter()
    frame_counts: Counter[tuple[str, str]] = Counter()
    seen = 0
    stopped_early = False
    embodiment_filter_ids = canonicalize_embodiment_filter_ids(
        data_config.embodiment_filter_ids
    )

    for dataset_cfg in data_config.datasets:
        builder = tfds.builder_from_directory(
            f"{dataset_root(data_config.rlds_data_dir, dataset_cfg)}/"
            f"{dataset_cfg.name}/{dataset_cfg.version}"
        )
        episodes = builder.as_dataset(split=data_config.rlds_split, shuffle_files=False)
        # The cap is per dataset: a shared cap let the first (large) dataset use it all and the
        # rest of the mixture never appeared in the audit.
        seen_in_dataset = 0
        for episode in episodes:
            if max_episodes is not None and seen_in_dataset >= max_episodes:
                stopped_early = True
                break
            metadata = episode["episode_metadata"]
            embodiment_id = int(metadata.get("embodiment_id", -1))
            if not embodiment_id_is_selected(embodiment_id, embodiment_filter_ids):
                continue
            if "task" in metadata and "target" in metadata:
                task = f"{_text(metadata['task'])} {_text(metadata['target'])}".strip()
            else:
                # Builds predating the task/target metadata redesign -- the
                # fr_ur5_100var_pick_place_* v2sub100 sets -- carry only the per-step
                # language_instruction. Mirror the fallback in
                # molmospaces_rlds_dataset.restructure so the audit reports a real task label
                # instead of raising KeyError and losing the whole mixture breakdown.
                task = next(
                    (_text(step["language_instruction"]) for step in episode["steps"].take(1)), ""
                )
            file_path = _text(metadata.get("file_path", ""))
            house_match = re.search(r"(?:^|/)house_(\d+)(?:/|$)", file_path)
            scene = f"house_{house_match.group(1)}" if house_match else "house_unknown"
            variant = _text(metadata.get("variant", ""))
            embodiment = variant or f"{_text(metadata.get('robot', 'unknown'))}:{_text(metadata.get('embodiment_id', -1))}"
            dataset_name = (
                f"{dataset_cfg.data_subdir or '.'}/"
                f"{dataset_cfg.name}:{dataset_cfg.version}"
            )
            num_frames = sum(1 for _ in episode["steps"])

            axes = {
                "dataset": dataset_name,
                "task": task,
                "scene": scene,
                "embodiment": embodiment,
                "dataset x task": f"{dataset_name} | {task}",
                "task x scene": f"{task} | {scene}",
                "task x embodiment": f"{task} | {embodiment}",
                "task x scene x embodiment": f"{task} | {scene} | {embodiment}",
            }
            for group, value in axes.items():
                counts[group, value] += 1
                frame_counts[group, value] += num_frames
            seen += 1
            seen_in_dataset += 1

    for overlay_spec in getattr(data_config, "behavior_prompt_overlays", ()):
        overlay = load_behavior_prompt_overlay(overlay_spec)
        component = "/".join(overlay_spec.component_key)
        manifest_value = (
            f"{component} | manifest_sha256:{overlay_spec.manifest_sha256} | "
            f"taxonomy_sha256:{overlay_spec.taxonomy_manifest_sha256}"
        )
        counts["behavior prompt overlay", manifest_value] = len(
            overlay.episode_suffixes
        )
        frame_counts["behavior prompt overlay", manifest_value] = 0
        suffix_counts = Counter(overlay.episode_suffixes.values())
        for suffix, count in suffix_counts.items():
            counts["behavior prompt suffix", f"{component} | {suffix}"] = count
            frame_counts["behavior prompt suffix", f"{component} | {suffix}"] = 0

    return [
        {
            "group": group,
            "value": value,
            "episodes": counts[group, value],
            "frames": frame_counts[group, value],
            "sampled": (
                False if group.startswith("behavior prompt ") else stopped_early
            ),
            "sample_size": (
                counts[group, value]
                if group.startswith("behavior prompt ")
                else seen
            ),
        }
        for group, value in sorted(counts)
    ]


def format_markdown(rows: list[dict]) -> str:
    lines = ["| group | value | episodes | frames | sampled |", "|---|---|---:|---:|:---:|"]
    lines += [
        f"| {r['group']} | {r['value']} | {r['episodes']} | {r['frames']} | {r['sampled']} |"
        for r in rows
    ]
    return "\n".join(lines)


def training_view_frames(data_config, *, process_count: int) -> dict:
    """Return exact selected training frames and V3 process-cohort totals."""
    if process_count < 1:
        raise ValueError("process_count must be positive")
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    split = data_config.rlds_split
    episode_split = data_config.episode_split
    if isinstance(episode_split, ContiguousEpisodeSplit):
        split = episode_split.train
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
        split = episode_split.source_split

    identity_record = {
        "selector": (
            None
            if data_config.episode_selector is None
            else data_config.episode_selector.to_record()
        ),
        "split": None if episode_split is None else episode_split.to_record(),
    }
    components = []
    all_views = []
    for dataset_cfg in data_config.datasets:
        builder = tfds.builder_from_directory(
            f"{dataset_root(data_config.rlds_data_dir, dataset_cfg)}/"
            f"{dataset_cfg.name}/{dataset_cfg.version}"
        )
        views = tuple(
            dataclasses.replace(
                view,
                namespace=(
                    dataset_cfg.dataset_id
                    or dataset_cfg.data_subdir
                    or dataset_cfg.name
                ),
            )
            for view in _episode_views(builder, split, identity_record, tfds)
        )
        components.append((builder, views))
        all_views.extend(views)

    selected = select_episode_views(
        all_views,
        episode_selector=data_config.episode_selector,
        episode_split=episode_split,
        episode_split_role="train",
        embodiment_filter_ids=data_config.embodiment_filter_ids,
    )
    selected_keys = frozenset(item.episode_key for item in selected)
    selected = tuple(item for item in all_views if item.episode_key in selected_keys)
    if any(item.frame_count < 0 for item in selected):
        raise ValueError("exact training-view frame count is unavailable from TFDS")

    cohort_frames = [0] * process_count
    if isinstance(data_config.rlds_input_sampling, DeterministicReplayV3):
        sampler_seed = (
            data_config.rlds_sampler_seed
            if data_config.rlds_sampler_seed is not None
            else 0
        )
        for _builder, views in components:
            component_selected = tuple(
                item for item in views if item.episode_key in selected_keys
            )
            num_shards = max((item.shard_index for item in views), default=-1) + 1
            if num_shards < 1 or any(item.shard_index < 0 for item in component_selected):
                raise ValueError("exact V3 cohort frames require TFDS shard identities")
            order = data_config.rlds_input_sampling.file_order(
                num_shards, sampler_seed=sampler_seed
            )
            order_position = {shard: position for position, shard in enumerate(order)}
            for item in component_selected:
                cohort_frames[order_position[item.shard_index] % process_count] += item.frame_count
    else:
        cohort_frames[0] = sum(item.frame_count for item in selected)

    return {
        "training_frames": sum(item.frame_count for item in selected),
        "training_episodes": len(selected),
        "process_count": process_count,
        "cohort_frames": tuple(cohort_frames),
        "cohort_frames_min": min(cohort_frames),
        "cohort_frames_max": max(cohort_frames),
    }


def nominal_repeat_budget(
    rows: list[dict],
    *,
    global_batch_size: int,
    training_frames: int | None = None,
) -> dict | None:
    """Summarize the batch size and steps for one nominal dataset repeat.

    MolmoSpaces training samples an infinite, shuffled stream and can rebalance phases, so this is
    deliberately called a *nominal repeat*, not an epoch. An exact selected-view audit takes
    precedence over ``training_frames``. A partial audit needs a trusted external frame count and
    otherwise returns ``None`` instead of logging a misleading estimate.
    """
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")

    dataset_rows = [row for row in rows if row["group"] == "dataset"]
    audit_is_exact = bool(dataset_rows) and not any(
        row["sampled"] for row in dataset_rows
    )
    if audit_is_exact:
        # Prefer the actual selected training view over any external manifest. This is
        # essential for filtered and cross-normalization experiments.
        training_frames = sum(int(row["frames"]) for row in dataset_rows)
    elif training_frames is None:
        return None
    training_frames = int(training_frames)
    if training_frames <= 0:
        raise ValueError("training_frames must be positive")

    return {
        "batch_size": int(global_batch_size),
        "steps_to_repeat": training_frames / global_batch_size,
    }


def add_repeat_columns(rows: list[dict], repeat_budget: dict | None) -> list[dict]:
    """Put the exposure denominator directly in every W&B mixture-table row.

    Repeating the run-level values makes them visible while filtering or grouping the table.
    Spaces in the public column names are intentional: these are user-facing W&B labels.
    """
    return [
        {
            **row,
            "batch size": (
                None if repeat_budget is None else repeat_budget["batch_size"]
            ),
            "steps to repeat": (
                None if repeat_budget is None else repeat_budget["steps_to_repeat"]
            ),
        }
        for row in rows
    ]
