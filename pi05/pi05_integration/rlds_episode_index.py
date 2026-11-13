"""Prebuilt per-episode identity index for one RLDS build.

Episode selection, hash/stratified splits, and V3 cohort accounting need one
``EpisodeIdentity`` per episode of a TFDS split. Deriving them by scanning the split reads
every record's serialized bytes (JPEG payloads included) on every JAX process at every
launch and resume. This module stores the same identities once, as a JSON sidecar inside
the builder directory, and validates it against the split's ``dataset_info`` shard
fingerprint so a stale or foreign index is never used.

Backward compatibility: a build without a sidecar falls back to the scan unchanged. Nothing
writes a sidecar implicitly; ``scripts/data/format_conversion/build_rlds_episode_index.py``
does it once per build.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from collections.abc import Mapping, Sequence

from pi05_integration.data_processing import EpisodeIdentity

INDEX_FILENAME = "molmospaces_episode_index.v1.json"
INDEX_KIND = "molmospaces_episode_index"
SCHEMA_VERSION = 1

_LOG = logging.getLogger(__name__)


def index_path(builder, path: str | None = None) -> str:
    """Return the sidecar location: ``<builder.data_dir>/molmospaces_episode_index.v1.json``."""
    if path:
        return path
    return f"{str(builder.data_dir).rstrip('/')}/{INDEX_FILENAME}"


def split_fingerprint(builder, source_split: str) -> dict:
    """Shard-level facts from ``dataset_info`` that must match for an index to be trusted."""
    info = builder.info.splits[source_split]
    return {
        "num_examples": int(info.num_examples),
        "num_shards": int(info.num_shards),
        "shard_lengths": [int(length) for length in info.shard_lengths],
        "filenames": [str(name) for name in info.filenames],
    }


def _open(path: str, mode: str):
    if re.match(r"^[a-z][a-z0-9+.-]*://", path):
        import tensorflow as tf  # only remote paths need gfile

        return tf.io.gfile.GFile(path, mode)
    return open(path, mode, encoding="utf-8")


def _exists(path: str) -> bool:
    if re.match(r"^[a-z][a-z0-9+.-]*://", path):
        import tensorflow as tf

        return bool(tf.io.gfile.exists(path))
    import os

    return os.path.exists(path)


def episode_record(identity: EpisodeIdentity) -> dict:
    return {
        "embodiment_id": identity.embodiment_id,
        "episode_key": identity.episode_key,
        "pose_id": identity.pose_id,
        "planner_seed": identity.planner_seed,
        "frame_count": identity.frame_count,
        "shard_index": identity.shard_index,
    }


def identity_from_record(record: Mapping) -> EpisodeIdentity:
    return EpisodeIdentity(
        embodiment_id=int(record["embodiment_id"]),
        episode_key=str(record["episode_key"]),
        pose_id=str(record.get("pose_id", "")),
        planner_seed=int(record.get("planner_seed", -1)),
        frame_count=int(record.get("frame_count", -1)),
        shard_index=int(record.get("shard_index", -1)),
    )


def scan_episode_views(builder, source_split: str, tfds) -> tuple[EpisodeIdentity, ...]:
    """Read identity plus encoded-frame cardinality by traversing the split once.

    This is the pre-index behaviour, moved verbatim from the training loader. It reads every
    record (SkipDecoding avoids image decode but not the byte read) in file order with a
    single interleave, so the returned order is shard order then record order.
    """
    skip_decoding_supported = True
    try:
        episodes = builder.as_dataset(
            split=source_split,
            shuffle_files=False,
            decoders={"steps": tfds.decode.SkipDecoding()},
            read_config=tfds.ReadConfig(
                add_tfds_id=True,
                interleave_cycle_length=1,
                num_parallel_calls_for_interleave_files=1,
            ),
        )
    except Exception as exc:
        skip_decoding_supported = False
        _LOG.warning(
            "TFDS SkipDecoding episode-view scan is unsupported; frame and shard counts "
            "will be unavailable: %s",
            exc,
        )
        episodes = builder.as_dataset(
            split=source_split, shuffle_files=False
        ).map(lambda episode: episode["episode_metadata"], num_parallel_calls=1)

    def episode_view(episode):
        metadata = episode["episode_metadata"] if skip_decoding_supported else episode
        steps = episode.get("steps") if skip_decoding_supported else None
        if steps is None:
            frame_count = -1
        elif isinstance(steps, dict):
            leading_dimensions = set()

            def collect_leading_dimensions(value):
                if isinstance(value, dict):
                    for nested in value.values():
                        collect_leading_dimensions(nested)
                elif getattr(value, "shape", None) is not None and value.shape.rank:
                    leading_dimensions.add(int(value.shape[0]))

            collect_leading_dimensions(steps)
            if len(leading_dimensions) != 1:
                raise ValueError(
                    "SkipDecoding returned inconsistent RLDS step cardinalities: "
                    f"{sorted(leading_dimensions)}"
                )
            frame_count = leading_dimensions.pop()
        else:
            try:
                frame_count = int(steps.shape[0])
            except (TypeError, AttributeError):
                try:
                    frame_count = sum(1 for _ in steps)
                except TypeError:
                    frame_count = -1
        tfds_id = episode.get("tfds_id", b"") if skip_decoding_supported else b""
        tfds_id = tfds_id.numpy() if hasattr(tfds_id, "numpy") else tfds_id
        tfds_id = tfds_id.decode("utf-8") if isinstance(tfds_id, bytes) else str(tfds_id)
        shard_match = re.search(r"\.tfrecord-(\d+)(?:-of-\d+)?__", tfds_id)
        return metadata, frame_count, int(shard_match.group(1)) if shard_match else -1

    return tuple(
        EpisodeIdentity(
            embodiment_id=int(metadata["embodiment_id"]),
            episode_key=(
                metadata["file_path"].numpy().decode("utf-8")
                if hasattr(metadata["file_path"], "numpy")
                else metadata["file_path"].decode("utf-8")
            ),
            pose_id=(
                metadata.get("pose_id", b"").numpy().decode("utf-8")
                if hasattr(metadata.get("pose_id", b""), "numpy")
                else metadata.get("pose_id", b"").decode("utf-8")
            ),
            planner_seed=int(metadata.get("planner_seed", -1)),
            frame_count=frame_count,
            shard_index=shard_index,
        )
        for metadata, frame_count, shard_index in (
            episode_view(episode) for episode in episodes
        )
    )


def build_index_document(
    builder, splits: Mapping[str, Sequence[EpisodeIdentity]], *, created_utc: str | None = None
) -> dict:
    if not splits:
        raise ValueError("an episode index needs at least one split")
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": INDEX_KIND,
        "dataset_name": str(builder.name),
        "dataset_version": str(builder.version),
        "created_utc": created_utc
        or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "splits": {},
    }
    for split, identities in splits.items():
        fingerprint = split_fingerprint(builder, split)
        if len(identities) != fingerprint["num_examples"]:
            raise ValueError(
                f"split {split!r}: scanned {len(identities)} episodes but dataset_info "
                f"declares {fingerprint['num_examples']}"
            )
        document["splits"][split] = {
            "fingerprint": fingerprint,
            "episodes": [episode_record(identity) for identity in identities],
        }
    return document


def write_episode_index(
    builder,
    splits: Mapping[str, Sequence[EpisodeIdentity]],
    *,
    path: str | None = None,
    overwrite: bool = False,
) -> str:
    """Write the sidecar for ``splits`` and return its path."""
    target = index_path(builder, path)
    if _exists(target) and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    document = build_index_document(builder, splits)
    with _open(target, "w") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return target


def load_episode_views(
    builder, source_split: str, *, path: str | None = None
) -> tuple[EpisodeIdentity, ...] | None:
    """Return the indexed identities for ``source_split``, or None when the scan must run.

    None means: no sidecar, unreadable sidecar, foreign dataset name/version, split absent,
    or a shard fingerprint that disagrees with ``dataset_info``. Every rejection is logged
    at WARNING except the plain absence of a sidecar.
    """
    target = index_path(builder, path)
    if not _exists(target):
        _LOG.info("no episode index at %s; scanning the split instead", target)
        return None
    try:
        with _open(target, "r") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as exc:
        _LOG.warning("episode index %s is unreadable (%s); scanning instead", target, exc)
        return None
    problems = []
    if document.get("kind") != INDEX_KIND or document.get("schema_version") != SCHEMA_VERSION:
        problems.append("kind/schema_version mismatch")
    if document.get("dataset_name") != str(builder.name):
        problems.append(f"dataset_name {document.get('dataset_name')!r} != {builder.name!r}")
    if document.get("dataset_version") != str(builder.version):
        problems.append(
            f"dataset_version {document.get('dataset_version')!r} != {str(builder.version)!r}"
        )
    split_document = (document.get("splits") or {}).get(source_split)
    if split_document is None:
        problems.append(f"split {source_split!r} not indexed")
    else:
        expected = split_fingerprint(builder, source_split)
        if split_document.get("fingerprint") != expected:
            problems.append("shard fingerprint differs from dataset_info")
        elif len(split_document.get("episodes", ())) != expected["num_examples"]:
            problems.append("episode count differs from dataset_info")
    if problems:
        _LOG.warning(
            "ignoring episode index %s (%s); scanning the split instead. Rebuild it with "
            "scripts/data/format_conversion/build_rlds_episode_index.py --overwrite",
            target,
            "; ".join(problems),
        )
        return None
    return tuple(identity_from_record(record) for record in split_document["episodes"])
