"""Hash-pinned, component-scoped behavior-mode prompt overlays.

The overlay is deliberately an input-only view: it maps stable RLDS episode keys to
globally semantic prompt suffixes without materializing a second copy of observations
or targets.  The loader validates the complete mapping before constructing its stream.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "molmospaces_behavior_prompt_overlay_v1"


@dataclasses.dataclass(frozen=True)
class BehaviorPromptOverlay:
    """One immutable behavior-prompt manifest for one RLDS component."""

    component_repo_id: str
    dataset_name: str
    dataset_version: str
    manifest_path: str
    manifest_sha256: str
    taxonomy_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in ("component_repo_id", "dataset_name", "dataset_version", "manifest_path"):
            if not getattr(self, name):
                raise ValueError(f"behavior prompt overlay {name} must be nonempty")
        if "/" in self.component_repo_id:
            raise ValueError("behavior prompt overlay component_repo_id must be one path segment")
        for name in ("manifest_sha256", "taxonomy_manifest_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(
                    f"behavior prompt overlay {name} must be 64 lowercase hex characters"
                )

    @property
    def component_key(self) -> tuple[str, str, str]:
        return (self.component_repo_id, self.dataset_name, self.dataset_version)

    def to_record(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LoadedBehaviorPromptOverlay:
    spec: BehaviorPromptOverlay
    episode_suffixes: Mapping[str, str]
    episode_factors: Mapping[str, Mapping[str, str]]

    def prompt_for_episode(self, episode_key: str, base_prompt: str) -> str:
        try:
            suffix = self.episode_suffixes[episode_key]
        except KeyError as error:
            raise ValueError(
                f"behavior prompt overlay has no entry for episode key {episode_key!r}"
            ) from error
        return append_prompt_suffix(base_prompt, suffix)


def append_prompt_suffix(base_prompt: str, suffix: str) -> str:
    """Append one normalized suffix without modifying the recorded base instruction."""
    if not base_prompt:
        raise ValueError("behavior prompt overlay requires a nonempty base prompt")
    normalized_suffix = suffix.strip()
    if not normalized_suffix:
        raise ValueError("behavior prompt overlay suffix must be nonempty")
    return f"{base_prompt.rstrip()} {normalized_suffix}"


def render_semantic_suffix(factors: Mapping[str, str]) -> str:
    """Render the only accepted text form from structured global behavior factors."""
    if not isinstance(factors, Mapping) or not factors:
        raise ValueError("behavior prompt overlay factors must be a nonempty mapping")
    rendered = []
    for name, value in sorted(factors.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(
                "behavior prompt overlay factor names must be lowercase snake_case"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("behavior prompt overlay factor values must be nonempty strings")
        rendered.append(f"{name.replace('_', ' ')}={value.strip()}")
    return "Behavior mode: " + "; ".join(rendered) + "."


def _read_bytes(path: str) -> bytes:
    """Read local or gfile-compatible manifests without importing TensorFlow eagerly."""
    if "://" not in path:
        return Path(path).read_bytes()
    import tensorflow as tf

    with tf.io.gfile.GFile(path, "rb") as manifest_file:
        return manifest_file.read()


def load_behavior_prompt_overlay(
    spec: BehaviorPromptOverlay,
) -> LoadedBehaviorPromptOverlay:
    raw = _read_bytes(spec.manifest_path)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != spec.manifest_sha256:
        raise ValueError(
            "behavior prompt overlay manifest hash mismatch: "
            f"expected {spec.manifest_sha256}, got {actual_sha256}"
        )
    try:
        payload: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("behavior prompt overlay manifest is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("behavior prompt overlay manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"behavior prompt overlay schema_version must be {SCHEMA_VERSION!r}"
        )
    if payload.get("taxonomy_manifest_sha256") != spec.taxonomy_manifest_sha256:
        raise ValueError(
            "behavior prompt overlay taxonomy manifest hash mismatch: "
            f"config expects {spec.taxonomy_manifest_sha256}, "
            f"manifest records {payload.get('taxonomy_manifest_sha256')!r}"
        )
    expected_component = {
        "repo_id": spec.component_repo_id,
        "dataset_name": spec.dataset_name,
        "dataset_version": spec.dataset_version,
    }
    if payload.get("component") != expected_component:
        raise ValueError(
            "behavior prompt overlay component mismatch: "
            f"config expects {expected_component!r}, manifest records {payload.get('component')!r}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("behavior prompt overlay entries must be a nonempty list")
    episode_suffixes: dict[str, str] = {}
    episode_factors: dict[str, Mapping[str, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "episode_key",
            "semantic_suffix",
            "factors",
        }:
            raise ValueError(
                "behavior prompt overlay entry "
                f"{index} must contain exactly episode_key, semantic_suffix, and factors"
            )
        episode_key = entry["episode_key"]
        semantic_suffix = entry["semantic_suffix"]
        factors = entry["factors"]
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError(f"behavior prompt overlay entry {index} has an invalid episode_key")
        if episode_key in episode_suffixes:
            raise ValueError(
                f"behavior prompt overlay contains duplicate episode key {episode_key!r}"
            )
        if not isinstance(semantic_suffix, str) or not semantic_suffix.strip():
            raise ValueError(
                f"behavior prompt overlay entry {index} has an invalid semantic_suffix"
            )
        rendered_suffix = render_semantic_suffix(factors)
        if semantic_suffix != rendered_suffix:
            raise ValueError(
                "behavior prompt overlay semantic_suffix does not match its structured "
                f"factors for episode {episode_key!r}: expected {rendered_suffix!r}"
            )
        episode_suffixes[episode_key] = semantic_suffix
        episode_factors[episode_key] = dict(factors)
    return LoadedBehaviorPromptOverlay(
        spec=spec,
        episode_suffixes=episode_suffixes,
        episode_factors=episode_factors,
    )


def validate_episode_key_coverage(
    overlay: LoadedBehaviorPromptOverlay,
    episode_keys: tuple[str, ...],
) -> None:
    """Require an exact one-to-one overlay for the component's complete source split."""
    if len(set(episode_keys)) != len(episode_keys):
        raise ValueError("RLDS component contains duplicate episode keys")
    expected = set(episode_keys)
    actual = set(overlay.episode_suffixes)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "behavior prompt overlay episode coverage mismatch: "
            f"missing={missing[:5]!r} ({len(missing)} total), "
            f"extra={extra[:5]!r} ({len(extra)} total)"
        )
