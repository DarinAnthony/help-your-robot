"""Identity of the RLDS files a run trains on, recorded at every start and checked on resume.

A resume continues the run's input stream over whatever files are on disk. If a dataset is
re-converted, gains or loses episodes, or is edited in place under the same name and version, the
resumed stream silently differs from the one the run trained on. The fingerprint is cheap enough for
every restart: TFDS metadata (a few KB per dataset) and one size query per shard file. It is keyed by
the dataset's path below the data root and leaves out modification times, so a byte-identical copy (a
staged cache, another region's bucket) matches. It does not hash file contents, which would read the
whole dataset, so an in-place edit that keeps every shard's size goes unnoticed.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging

from pi05_integration.rlds_episode_index import split_fingerprint

from openpi.training.molmospaces_rlds_dataset import dataset_root

SCHEMA_VERSION = 1


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def _dataset_key(dataset) -> str:
    """The dataset's path below the data root; an absolute ``data_subdir`` keeps its last component."""
    subdir = (dataset.data_subdir or "").rstrip("/")
    if "://" in subdir or subdir.startswith("/"):
        subdir = subdir.rsplit("/", 1)[-1]
    return "/".join(part for part in (subdir.strip("/"), dataset.name, dataset.version) if part not in ("", "."))


def dataset_fingerprint(data_config) -> dict:
    """Return the fingerprint of every dataset in the mixture, every split included."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    datasets = {}
    # Shard sizes are one metadata request each, which on GCS is a round trip, so they run in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for dataset_cfg in data_config.datasets:
            builder = tfds.builder_from_directory(
                f"{dataset_root(data_config.rlds_data_dir, dataset_cfg)}/"
                f"{dataset_cfg.name}/{dataset_cfg.version}"
            )
            data_dir = str(builder.data_dir).rstrip("/")
            with tf.io.gfile.GFile(f"{data_dir}/features.json", "rb") as features:
                features_sha256 = hashlib.sha256(features.read()).hexdigest()
            splits = {}
            for split in sorted(builder.info.splits):
                shards = split_fingerprint(builder, split)
                sizes = list(
                    pool.map(
                        lambda name: int(tf.io.gfile.stat(f"{data_dir}/{name}").length),
                        shards["filenames"],
                    )
                )
                splits[split] = {
                    "episodes": shards["num_examples"],
                    "shards": shards["num_shards"],
                    "bytes": sum(sizes),
                    "shard_lengths_sha256": _sha256(shards["shard_lengths"]),
                    "shard_files_sha256": _sha256(list(zip(shards["filenames"], sizes))),
                }
            datasets[_dataset_key(dataset_cfg)] = {
                "features_sha256": features_sha256,
                "splits": splits,
            }
    return {"schema_version": SCHEMA_VERSION, "datasets": datasets}


def fingerprint_differences(recorded: dict, current: dict) -> list[str]:
    """Describe each way the current files differ from the recorded ones; empty when they match."""
    differences = []
    for key in sorted(recorded["datasets"].keys() | current["datasets"].keys()):
        old = recorded["datasets"].get(key)
        new = current["datasets"].get(key)
        if old is None or new is None:
            differences.append(f"{key}: {'added to' if old is None else 'removed from'} the mixture")
            continue
        if old["features_sha256"] != new["features_sha256"]:
            differences.append(f"{key}: features.json changed")
        for split in sorted(old["splits"].keys() | new["splits"].keys()):
            old_split = old["splits"].get(split)
            new_split = new["splits"].get(split)
            if old_split is None or new_split is None:
                differences.append(f"{key} [{split}]: split {'added' if old_split is None else 'removed'}")
                continue
            changes = [
                f"{field} {old_split[field]} -> {new_split[field]}"
                for field in ("episodes", "shards", "bytes")
                if old_split[field] != new_split[field]
            ]
            if old_split["shard_lengths_sha256"] != new_split["shard_lengths_sha256"]:
                changes.append("episodes per shard changed")
            if old_split["shard_files_sha256"] != new_split["shard_files_sha256"]:
                changes.append("shard file names or sizes changed")
            if changes:
                differences.append(f"{key} [{split}]: {', '.join(changes)}")
    return differences


def check_resumed_datasets(recorded: dict | None, current: dict, *, allow_change: bool) -> list[str]:
    """Return how the files differ from the lineage's last start, raising unless ``allow_change``.

    A lineage that recorded no fingerprint (it predates them) cannot be checked: it gets a warning
    and an empty list, and this start records the current files for the next resume.
    """
    if recorded is None or recorded.get("schema_version") != current["schema_version"]:
        logging.warning(
            "The run recorded no dataset fingerprint at its last start; this resume cannot check "
            "the dataset files, and records the current ones for the next resume"
        )
        return []
    differences = fingerprint_differences(recorded, current)
    if differences and not allow_change:
        raise ValueError(
            "The dataset files differ from the ones this run trained on at its last start:\n  "
            + "\n  ".join(differences)
            + "\nResume on the recorded files, or, to continue on the changed data (finetuning "
            "or a new mix), pass --rlds-input-dataset-restart (rlds_input_dataset_restart=True)."
        )
    if differences:
        logging.warning(
            "Continuing on changed dataset files (rlds_input_dataset_restart):\n  %s",
            "\n  ".join(differences),
        )
    return differences
