"""Host-side loading and checkpointing for URMA embodiment descriptions."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.embodiment.bundle import (
    BUNDLE_MANIFEST,
    BUNDLE_NPZ,
    build_bundle_from_manifest,
    load_bundle,
    load_bundle_identity_map as load_conditioning_identity_map,
    load_bundle_geometry,
    load_bundle_global_context,
    load_bundle_text_index,
    serialize_bundle_description,
    serialize_bundle_descriptions,
    sha256,
)
from molmo_spaces.paths import EMBODIMENT_DESCRIPTION_CACHE_DIR

DYNAMIC_NORMALIZATION = "dynamic_normalization.json"


def validate_dynamic_normalization(
    path: str | Path, expected_sha256: str
) -> tuple[str, str]:
    from molmo_spaces.dynamic_urma import FIELD_NAMES, DynamicNormalization

    path = Path(path)
    payload_bytes = path.read_bytes()
    actual = hashlib.sha256(payload_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"URMA4 dynamic-statistics hash mismatch: {actual} != {expected_sha256}"
        )
    payload = json.loads(payload_bytes)
    if payload.get("schema") != "urma4_dynamic_normalization_v1":
        raise ValueError("unsupported URMA4 dynamic normalization schema")
    if tuple(payload.get("field_names", ())) != FIELD_NAMES:
        raise ValueError("URMA4 dynamic normalization field order changed")
    DynamicNormalization(payload["mean"], payload["std"], payload["count"])
    return str(path), actual


def load_text_bundle_index(
    bundle_dir: str | Path,
    *,
    expected_schema: str,
    expected_format_id: str | None = None,
):
    """Load the versioned text index without importing it on legacy routes."""

    from molmo_spaces.embodiment.text_description.bundle import (
        load_text_bundle_index as load_index,
    )

    return load_index(
        bundle_dir,
        expected_schema=expected_schema,
        expected_format_id=expected_format_id,
    )


def prepare_robotoken_bundle(
    bundle_dir: str | Path,
    *,
    expected_sha256: str | None,
    build_manifest: str | None,
    expected_build_manifest_sha256: str | None,
) -> tuple[str, str]:
    """Validate a prebuilt, immutable static RoboToken bundle.

    Bundle construction is deliberately an explicit offline audit step because the sidecar
    coverage contract must be reviewed before training; a data-loader must never invent or
    silently omit robot fields.
    """
    from molmo_spaces.embodiment import robotokens

    if build_manifest is not None:
        actual = sha256(build_manifest)
        if actual != expected_build_manifest_sha256:
            raise ValueError(f"RoboToken build manifest hash mismatch: {actual} != {expected_build_manifest_sha256}")
    try:
        manifest, _ = robotokens.load_bundle(bundle_dir)
    except FileNotFoundError:
        if build_manifest is None:
            raise
        Path(bundle_dir).parent.mkdir(parents=True, exist_ok=True)
        with (Path(bundle_dir).parent / f".{Path(bundle_dir).name}.lock").open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                manifest, _ = robotokens.load_bundle(bundle_dir)
            except FileNotFoundError:
                robotokens.build_bundle_from_manifest(build_manifest, bundle_dir)
                manifest, _ = robotokens.load_bundle(bundle_dir)
    actual = manifest["npz_sha256"]
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"configured RoboToken bundle hash mismatch: {actual} != {expected_sha256}")
    if build_manifest is not None:
        recorded = manifest.get("coverage", {}).get("build_manifest_sha256")
        if recorded != expected_build_manifest_sha256:
            raise ValueError("RoboToken bundle does not record the pinned build manifest")
    return str(bundle_dir), actual


@dataclasses.dataclass(frozen=True)
class AttachRobotTokens:
    bundle_dir: str
    expected_sha256: str
    max_tokens: int
    feature_dim: int
    reference_namespaces: int

    def __post_init__(self) -> None:
        from molmo_spaces.embodiment import robotokens

        manifest, entries = robotokens.load_bundle(self.bundle_dir)
        if manifest["npz_sha256"] != self.expected_sha256:
            raise ValueError("configured RoboToken bundle hash changed")
        if (self.max_tokens, self.feature_dim, self.reference_namespaces) != (
            robotokens.MAX_TOKENS,
            robotokens.FEATURE_DIM,
            len(robotokens.REFERENCE_NAMESPACES),
        ):
            raise ValueError("RoboToken bundle shape disagrees with model configuration")
        object.__setattr__(self, "entries", entries)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "embodiment_id" not in data:
            raise ValueError("RoboToken conditioning requires embodiment_id")
        embodiment_id = int(np.asarray(data["embodiment_id"]).item())
        try:
            values, types, references, mask = self.entries[embodiment_id]
        except KeyError as error:
            raise ValueError(
                f"embodiment_id {embodiment_id} is absent from RoboToken bundle {self.bundle_dir}"
            ) from error
        return {
            **data,
            "robotoken_values": values,
            "robotoken_types": types,
            "robotoken_reference_ids": references,
            "robotoken_mask": mask,
        }


def prepare_description_bundle(
    bundle_dir: str | Path,
    *,
    expected_sha256: str | None,
    build_manifest: str | None,
    expected_build_manifest_sha256: str | None,
    expected_schema: str | None = None,
) -> tuple[str, str]:
    """Validate an immutable bundle or build it once from pinned robot folders."""
    bundle_dir = Path(bundle_dir)
    if expected_schema == "embodiment_text_bundle_v1":
        if build_manifest is not None or expected_build_manifest_sha256 is not None:
            raise ValueError("text bundles must be built by the offline text-bundle builder")
        from molmo_spaces.embodiment.text_description.bundle import load_text_bundle

        manifest, _ = load_text_bundle(
            bundle_dir, expected_schema=expected_schema
        )
        actual = manifest["npz_sha256"]
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(
                f"configured embodiment bundle hash does not match {bundle_dir}: "
                f"{actual} != {expected_sha256}"
            )
        return str(bundle_dir), actual
    build_manifest_path = None if build_manifest is None else Path(build_manifest)
    if build_manifest_path is not None:
        actual_build_hash = sha256(build_manifest_path)
        if actual_build_hash != expected_build_manifest_sha256:
            raise ValueError(
                "URMA2 build manifest hash mismatch: "
                f"{actual_build_hash} != {expected_build_manifest_sha256}"
            )

    def ready() -> tuple[str, str] | None:
        try:
            manifest, _ = load_bundle(
                bundle_dir, expected_schema=expected_schema
            )
        except FileNotFoundError:
            return None
        actual = manifest["npz_sha256"]
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(
                f"configured embodiment bundle hash does not match {bundle_dir}: "
                f"{actual} != {expected_sha256}"
            )
        if expected_build_manifest_sha256 is not None:
            recorded = manifest.get("build", {}).get("manifest_sha256")
            if recorded != expected_build_manifest_sha256:
                return None
        return str(bundle_dir), actual

    if build_manifest_path is None:
        existing = ready()
        if existing is None:
            raise FileNotFoundError(f"embodiment description bundle is absent: {bundle_dir}")
        return existing

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    with (bundle_dir.parent / f".{bundle_dir.name}.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = ready()
        if existing is not None:
            return existing
        build_bundle_from_manifest(
            build_manifest_path,
            bundle_dir,
            fallback_cache_root=EMBODIMENT_DESCRIPTION_CACHE_DIR,
        )
    built = ready()
    if built is None:
        raise RuntimeError(f"failed to build embodiment description bundle: {bundle_dir}")
    return built


@dataclasses.dataclass(frozen=True)
class AttachEmbodimentDescription:
    bundle_dir: str
    expected_sha256: str
    max_controls: int
    description_dim: int
    expected_schema: str | None = None
    global_description_dim: int | None = None

    def __post_init__(self) -> None:
        manifest, entries = load_bundle(
            self.bundle_dir, expected_schema=self.expected_schema
        )
        actual = manifest["npz_sha256"]
        if actual != self.expected_sha256:
            raise ValueError(
                f"configured embodiment bundle hash does not match {self.bundle_dir}: "
                f"{actual} != {self.expected_sha256}"
            )
        if not entries:
            raise ValueError("embodiment description bundle is empty")
        first = next(iter(entries.values()))
        if first[0].shape != (self.max_controls, self.description_dim):
            raise ValueError(
                "embodiment bundle tensor shape disagrees with the configured extractor: "
                f"{first[0].shape} != {(self.max_controls, self.description_dim)}"
            )
        object.__setattr__(self, "entries", entries)
        identity_map = load_conditioning_identity_map(
            self.bundle_dir, expected_sha256=self.expected_sha256
        )
        object.__setattr__(
            self,
            "conditioning_identity_map",
            identity_map if identity_map else None,
        )
        globals_by_id = load_bundle_global_context(
            self.bundle_dir, expected_schema=self.expected_schema
        )
        if self.global_description_dim is None:
            if globals_by_id:
                raise ValueError(
                    "bundle supplies global context but extractor has no global shape"
                )
        else:
            if set(globals_by_id) != set(entries):
                raise ValueError(
                    "global context IDs disagree with embodiment description IDs"
                )
            first_global = next(iter(globals_by_id.values()))
            if first_global[0].shape != (self.global_description_dim,):
                raise ValueError(
                    "bundle global tensor shape disagrees with configured extractor: "
                    f"{first_global[0].shape} != {(self.global_description_dim,)}"
                )
        object.__setattr__(self, "global_entries", globals_by_id)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "embodiment_id" not in data:
            raise ValueError("URMA embodiment conditioning requires embodiment_id")
        raw_embodiment_id = int(np.asarray(data["embodiment_id"]).item())
        embodiment_id = raw_embodiment_id
        if self.conditioning_identity_map is not None:
            if "dataset_component_id" not in data:
                raise ValueError(
                    "namespaced URMA bundle requires dataset_component_id; raw-ID fallback is forbidden"
                )
            namespace_value = np.asarray(data["dataset_component_id"]).item()
            if isinstance(namespace_value, bytes):
                namespace_value = namespace_value.decode("utf-8")
            identity = (str(namespace_value), raw_embodiment_id)
            try:
                embodiment_id = self.conditioning_identity_map[identity]
            except KeyError as error:
                raise ValueError(
                    f"composite embodiment identity {identity!r} is absent from description bundle"
                ) from error
        try:
            description, availability, control_mask = self.entries[embodiment_id]
        except KeyError as error:
            raise ValueError(
                f"embodiment_id {embodiment_id} is absent from description bundle "
                f"{self.bundle_dir}"
            ) from error
        state = np.asarray(data["state"], np.float32)
        if state.ndim != 1:
            raise ValueError(
                f"description attachment expects one unbatched state, got {state.shape}"
            )
        control_state = np.zeros((self.max_controls, 1), np.float32)
        count = min(self.max_controls, state.shape[0])
        control_state[:count, 0] = state[:count]
        output = dict(data)
        # This string is transport metadata used only to disambiguate raw IDs
        # across mixture components.  Do not expose it to JAX model batches,
        # which accept only numeric array leaves.
        output.pop("dataset_component_id", None)
        output.update(
            conditioning_embodiment_id=np.asarray(embodiment_id, np.int32),
            embodiment_description=description,
            embodiment_description_availability=availability,
            embodiment_control_mask=control_mask,
            embodiment_control_state=control_state,
        )
        if self.global_description_dim is not None:
            global_description, global_availability = self.global_entries[
                embodiment_id
            ]
            output.update(
                embodiment_global_description=global_description,
                embodiment_global_description_availability=global_availability,
            )
        return output


@dataclasses.dataclass(frozen=True)
class AttachDynamicEmbodimentDescription(AttachEmbodimentDescription):
    """Attach immutable URMA4 context and normalized causal dynamic fields."""

    dynamic_stats_path: str = ""
    expected_dynamic_stats_sha256: str = ""
    dynamic_profile: str = "full"

    def __post_init__(self) -> None:
        super().__post_init__()
        from molmo_spaces.dynamic_urma import DYNAMIC_PROFILES, DynamicNormalization

        if self.dynamic_profile not in DYNAMIC_PROFILES:
            raise ValueError(
                f"unknown URMA4 dynamic profile {self.dynamic_profile!r}; "
                f"choices={DYNAMIC_PROFILES}"
            )

        path, _ = validate_dynamic_normalization(
            self.dynamic_stats_path, self.expected_dynamic_stats_sha256
        )
        payload = json.loads(Path(path).read_text())
        object.__setattr__(
            self,
            "dynamic_normalization",
            DynamicNormalization(payload["mean"], payload["std"], payload["count"]),
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from molmo_spaces.dynamic_urma import MASK_KEY, VALUES_KEY

        missing = [key for key in (VALUES_KEY, MASK_KEY) if key not in data]
        if missing:
            raise ValueError(f"URMA4-dynamic input is missing causal fields: {missing}")
        output = super().__call__(data)
        output["embodiment_control_state"] = self.dynamic_normalization.apply(
            data[VALUES_KEY],
            data[MASK_KEY],
            max_controls=self.max_controls,
            profile=self.dynamic_profile,
        )
        # A dynamic field must never make a statically padded control visible.
        padded = ~np.asarray(output["embodiment_control_mask"], dtype=bool)
        if np.any(np.asarray(data[MASK_KEY], dtype=bool)[padded]):
            raise ValueError("URMA4 dynamic mask marks a padded static control as available")
        return output


@dataclasses.dataclass(frozen=True)
class AttachUrma4NewNormDescription(AttachEmbodimentDescription):
    """Attach URMA4's historical fields under the complete-standard contract."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from molmo_spaces.embodiment.normalization_v4_newnorm import (
            validate_normalization,
        )

        manifest = json.loads(
            (Path(self.bundle_dir) / BUNDLE_MANIFEST).read_text(encoding="utf-8")
        )
        normalization = manifest.get("description_normalization")
        if normalization is None:
            raise ValueError(
                "URMA4-newnorm requires one immutable complete normalization record"
            )
        validate_normalization(normalization)
        object.__setattr__(self, "urma4_newnorm_normalization", normalization)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from molmo_spaces.embodiment.normalization_v4_newnorm import (
            normalize_control_state,
        )

        output = super().__call__(data)
        output["embodiment_control_state"] = normalize_control_state(
            np.asarray(data["state"], np.float32),
            normalized_static_values=np.asarray(
                output["embodiment_description"], np.float32
            ),
            control_mask=np.asarray(output["embodiment_control_mask"], bool),
            normalization=self.urma4_newnorm_normalization,
        )
        return output


@dataclasses.dataclass(frozen=True)
class AttachUrma4AbsoluteDescription(AttachEmbodimentDescription):
    """Attach absolute URMA4 (``urma4_absolute_standard_v1``) rows and the role-standardized physical state."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from molmo_spaces.embodiment.normalization_v4_absolute import validate_normalization

        manifest = json.loads((Path(self.bundle_dir) / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
        normalization = manifest.get("description_normalization")
        if normalization is None:
            raise ValueError("URMA4-absolute requires one immutable training normalization record")
        validate_normalization(normalization)
        object.__setattr__(self, "urma4_absolute_normalization", normalization)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from molmo_spaces.embodiment.normalization_v4_absolute import normalize_control_state

        output = super().__call__(data)
        output["embodiment_control_state"] = normalize_control_state(
            np.asarray(data["state"], np.float32),
            normalized_static_values=np.asarray(output["embodiment_description"], np.float32),
            control_mask=np.asarray(output["embodiment_control_mask"], bool),
            normalization=self.urma4_absolute_normalization,
        )
        return output


@dataclasses.dataclass(frozen=True)
class AttachUrma5EmbodimentDescription(AttachEmbodimentDescription):
    """Attach strict URMA5 context, causal control state, and live anchors."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from molmo_spaces.embodiment.normalization_v5 import validate_normalization

        manifest = json.loads(
            (Path(self.bundle_dir) / BUNDLE_MANIFEST).read_text(encoding="utf-8")
        )
        normalization = manifest.get("description_normalization")
        if normalization is None:
            raise ValueError("URMA5 requires one immutable complete normalization record")
        validate_normalization(normalization)
        object.__setattr__(self, "urma5_normalization", normalization)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from molmo_spaces.dynamic_urma5 import VALUES_KEY, validate_dynamic_fields
        from molmo_spaces.embodiment.description_v5 import VALUE_FEATURES
        from molmo_spaces.embodiment.normalization_v5 import normalize_dynamic_values

        if VALUES_KEY not in data:
            raise ValueError(f"URMA5 input is missing causal field {VALUES_KEY!r}")
        output = super().__call__(data)
        control_mask = np.asarray(output["embodiment_control_mask"], bool)
        gripper_mask = (
            np.asarray(output["embodiment_description"], np.float32)[
                :, VALUE_FEATURES.index("type_gripper_aggregate")
            ]
            > 0.5
        ) & control_mask
        values, _, _ = validate_dynamic_fields(
            data[VALUES_KEY],
            control_mask,
            gripper_mask,
            max_controls=self.max_controls,
        )
        output["embodiment_control_state"] = normalize_dynamic_values(
            values,
            control_mask=control_mask,
            gripper_mask=gripper_mask,
            record=self.urma5_normalization,
        )
        return output


@dataclasses.dataclass(frozen=True)
class AttachUrma6EmbodimentDescription(AttachUrma5EmbodimentDescription):
    """Attach every URMA5 field plus hash-bound collision-surface points."""

    def __post_init__(self) -> None:
        super().__post_init__()
        geometry_entries = load_bundle_geometry(
            self.bundle_dir, expected_schema=self.expected_schema
        )
        if set(geometry_entries) != set(self.entries):
            raise ValueError(
                "URMA6 collision-geometry IDs disagree with scalar description IDs"
            )
        object.__setattr__(self, "geometry_entries", geometry_entries)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = super().__call__(data)
        # ``super()`` resolves the wire identity
        # (dataset_component_id, source embodiment_id) into the immutable
        # bundle row. Geometry is keyed by that same global row, not by the
        # component-local source ID that remains in ``data``.
        embodiment_id = int(
            np.asarray(output["conditioning_embodiment_id"]).item()
        )
        points, point_mask = self.geometry_entries[embodiment_id]
        output.update(
            embodiment_geometry_points=points,
            embodiment_geometry_point_mask=point_mask,
        )
        return output


@dataclasses.dataclass(frozen=True)
class AttachUrma51EmbodimentDescription(AttachEmbodimentDescription):
    """Attach URMA5.1 base state, arm-only analytics, and applicability bits."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from molmo_spaces.embodiment.normalization_v51 import validate_normalization

        manifest = json.loads(
            (Path(self.bundle_dir) / BUNDLE_MANIFEST).read_text(encoding="utf-8")
        )
        normalization = manifest.get("description_normalization")
        if normalization is None:
            raise ValueError("URMA5.1 requires one immutable complete normalization record")
        validate_normalization(normalization)
        object.__setattr__(self, "urma51_normalization", normalization)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from molmo_spaces.dynamic_urma51 import (
            ANALYTIC_MASK_KEY,
            VALUES_KEY,
            validate_dynamic_fields,
        )
        from molmo_spaces.embodiment.description_v5 import VALUE_FEATURES
        from molmo_spaces.embodiment.normalization_v51 import normalize_dynamic_values

        missing = [key for key in (VALUES_KEY, ANALYTIC_MASK_KEY) if key not in data]
        if missing:
            raise ValueError(f"URMA5.1 input is missing causal fields: {missing}")
        output = super().__call__(data)
        control_mask = np.asarray(output["embodiment_control_mask"], bool)
        gripper_mask = (
            np.asarray(output["embodiment_description"], np.float32)[
                :, VALUE_FEATURES.index("type_gripper_aggregate")
            ]
            > 0.5
        ) & control_mask
        values, analytic_mask, _, _ = validate_dynamic_fields(
            data[VALUES_KEY],
            data[ANALYTIC_MASK_KEY],
            control_mask,
            gripper_mask,
            max_controls=self.max_controls,
        )
        output["embodiment_control_state"] = normalize_dynamic_values(
            values,
            analytic_mask,
            control_mask=control_mask,
            gripper_mask=gripper_mask,
            record=self.urma51_normalization,
        )
        return output


@dataclasses.dataclass(frozen=True)
class RemapEmbodimentId:
    """Map sparse raw IDs into a compact learned codebook with one OOV row."""

    raw_ids: tuple[int, ...]
    oov_index: int

    def __post_init__(self) -> None:
        raw_ids = tuple(int(value) for value in self.raw_ids)
        if not raw_ids or len(raw_ids) != len(set(raw_ids)):
            raise ValueError("raw embodiment ID codebook must be non-empty and unique")
        if self.oov_index != len(raw_ids):
            raise ValueError("OOV index must immediately follow the compact codebook")
        object.__setattr__(self, "raw_ids", raw_ids)
        object.__setattr__(self, "mapping", {
            value: index for index, value in enumerate(raw_ids)
        })

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "embodiment_id" not in data:
            raise ValueError("embodiment ID remapping requires embodiment_id")
        raw_id = int(np.asarray(data["embodiment_id"]).item())
        return {
            **data,
            "embodiment_id": np.asarray(
                self.mapping.get(raw_id, self.oov_index), np.int32
            ),
        }


def checkpoint_bundle_source(
    bundle_dir: str | Path,
    expected_sha256: str,
    *,
    expected_schema: str | None = None,
) -> dict[str, str]:
    bundle_dir = Path(bundle_dir)
    if expected_schema == "t2_robotokens_plus_mjcf_sidecars_v1":
        from molmo_spaces.embodiment import robotokens

        manifest, _ = robotokens.load_bundle(bundle_dir)
    elif expected_schema == "embodiment_text_bundle_v1":
        from molmo_spaces.embodiment.text_description.bundle import load_text_bundle

        manifest, _ = load_text_bundle(bundle_dir, expected_schema=expected_schema)
    else:
        manifest, _ = load_bundle(bundle_dir, expected_schema=expected_schema)
    if manifest["npz_sha256"] != expected_sha256:
        raise ValueError(
            f"configured embodiment bundle hash mismatch: "
            f"{manifest['npz_sha256']} != {expected_sha256}"
        )
    return {
        "source_dir": str(bundle_dir),
        "npz_sha256": expected_sha256,
        "manifest_sha256": sha256(bundle_dir / BUNDLE_MANIFEST),
        "schema": manifest["schema"],
    }


# Description schemas whose checkpoints record the schema beside the bundle copy and refuse a mismatch at restore
# and serve time (docs/urma_embodiment_conditioning.md §12.3 item 5: a flat token has no parameters to fail on).
SCHEMA_GUARDED_DESCRIPTIONS = frozenset({"urma4_absolute_standard_v1"})
DESCRIPTION_SCHEMA_RECORD = "embodiment_description_schema.json"


def description_schema_record(manifest: dict[str, Any]) -> dict[str, Any]:
    """The identity a checkpoint was trained on: description schema, feature lists and widths."""
    processor = manifest.get("shared_source", {}).get("processor", {})
    return {
        "kind": "embodiment_description_schema_v1",
        "description_schema": manifest["schema"],
        "value_features": list(processor.get("value_features", ())),
        "global_features": list(processor.get("global_features", ())),
        "normalization_schema": (manifest.get("description_normalization") or {}).get("schema"),
    }


def check_checkpoint_description_schema(bundle_dir: Any, expected_schema: str | None) -> None:
    """Refuse a checkpoint whose recorded description schema differs from the configured one.

    A checkpoint of a guarded schema must carry the record; a configuration that expects a guarded schema refuses a
    checkpoint without one (it was trained on another description)."""
    if "://" in str(bundle_dir) or str(bundle_dir).startswith(("/bigstore/", "/gcs/")):
        from etils import epath

        bundle_dir = epath.Path(str(bundle_dir))
    else:
        bundle_dir = Path(str(bundle_dir))
    record_path = bundle_dir / DESCRIPTION_SCHEMA_RECORD
    recorded = json.loads(record_path.read_text()) if record_path.is_file() else None
    manifest_path = bundle_dir / BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if recorded is None:
        if expected_schema in SCHEMA_GUARDED_DESCRIPTIONS or (
                manifest is not None and manifest.get("schema") in SCHEMA_GUARDED_DESCRIPTIONS):
            raise ValueError(
                f"checkpoint embodiment assets {bundle_dir} have no {DESCRIPTION_SCHEMA_RECORD}; refusing to pair "
                f"them with description schema {expected_schema!r}")
        return
    if manifest is None or recorded != description_schema_record(manifest):
        raise ValueError(f"checkpoint description schema record disagrees with its bundle copy in {bundle_dir}")
    if expected_schema is not None and recorded["description_schema"] != expected_schema:
        raise ValueError(
            "checkpoint was trained on embodiment description schema "
            f"{recorded['description_schema']!r}, not the configured {expected_schema!r}")


def copy_checkpoint_bundle(destination: Any, source: dict[str, str]) -> None:
    source_dir = Path(source["source_dir"])
    if sha256(source_dir / BUNDLE_MANIFEST) != source["manifest_sha256"]:
        raise ValueError("embodiment bundle manifest changed after data configuration")
    if source.get("schema") == "t2_robotokens_plus_mjcf_sidecars_v1":
        from molmo_spaces.embodiment import robotokens

        manifest, _ = robotokens.load_bundle(source_dir)
    elif source.get("schema") == "embodiment_text_bundle_v1":
        from molmo_spaces.embodiment.text_description.bundle import load_text_bundle

        manifest, _ = load_text_bundle(
            source_dir, expected_schema=source.get("schema")
        )
    else:
        manifest, _ = load_bundle(
            source_dir, expected_schema=source.get("schema")
        )
    if manifest["npz_sha256"] != source["npz_sha256"]:
        raise ValueError("embodiment bundle tensor changed after data configuration")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (BUNDLE_NPZ, BUNDLE_MANIFEST):
        target = destination / name
        target.write_bytes((source_dir / name).read_bytes())
    if manifest["schema"] in SCHEMA_GUARDED_DESCRIPTIONS:
        (destination / DESCRIPTION_SCHEMA_RECORD).write_text(
            json.dumps(description_schema_record(manifest), indent=2, sort_keys=True) + "\n"
        )
    if "dynamic_stats_source" in source:
        dynamic_stats_path, dynamic_stats_sha256 = validate_dynamic_normalization(
            source["dynamic_stats_source"], source["dynamic_stats_sha256"]
        )
        target = destination / DYNAMIC_NORMALIZATION
        target.write_bytes(Path(dynamic_stats_path).read_bytes())
        if hashlib.sha256(target.read_bytes()).hexdigest() != dynamic_stats_sha256:
            raise ValueError("checkpoint URMA4 dynamic-statistics copy changed")


def resolve_checkpoint_bundle(
    directory: str | Path,
    expected_sha256: str | None,
    *,
    expected_schema: str | None = None,
) -> tuple[str, str]:
    directory = Path(directory)
    if expected_schema == "t2_robotokens_plus_mjcf_sidecars_v1":
        from molmo_spaces.embodiment import robotokens

        manifest, _ = robotokens.load_bundle(directory)
    elif expected_schema == "embodiment_text_bundle_v1":
        from molmo_spaces.embodiment.text_description.bundle import load_text_bundle

        manifest, _ = load_text_bundle(directory, expected_schema=expected_schema)
    else:
        check_checkpoint_description_schema(directory, expected_schema)
        manifest, _ = load_bundle(directory, expected_schema=expected_schema)
    if expected_sha256 is not None and manifest["npz_sha256"] != expected_sha256:
        raise ValueError(
            f"checkpoint embodiment bundle hash mismatch: "
            f"{manifest['npz_sha256']} != {expected_sha256}"
        )
    return str(directory), manifest["npz_sha256"]


def resolve_transfer_bundle(
    checkpoint_directory: str | Path,
    checkpoint_sha256: str,
    transfer_directory: str | Path,
    transfer_sha256: str,
) -> tuple[str, str]:
    """Validate an inference-only bundle against the checkpoint bundle schema and tensor shapes."""
    try:
        raw_checkpoint_manifest = json.loads(
            (Path(checkpoint_directory) / BUNDLE_MANIFEST).read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError:
        # Unit-test and adapter callers may supply a virtual loader instead of
        # a filesystem bundle. Ordinary schema/shape validation below still
        # applies to those loaders.
        raw_checkpoint_manifest = {}
    if raw_checkpoint_manifest.get("schema") == "embodiment_text_bundle_v1":
        from molmo_spaces.embodiment.text_description.bundle import load_text_bundle

        checkpoint_manifest, checkpoint_entries = load_text_bundle(
            checkpoint_directory,
            expected_schema="embodiment_text_bundle_v1",
        )
        transfer_manifest, transfer_entries = load_text_bundle(
            transfer_directory,
            expected_schema="embodiment_text_bundle_v1",
        )
        if checkpoint_manifest["npz_sha256"] != checkpoint_sha256:
            raise ValueError("checkpoint text bundle changed after validation")
        if transfer_manifest["npz_sha256"] != transfer_sha256:
            raise ValueError("transfer text bundle hash mismatch")
        for field in ("format_id", "format_version", "field_order", "reference_robot"):
            if transfer_manifest.get(field) != checkpoint_manifest.get(field):
                raise ValueError(
                    f"transfer text bundle {field} disagrees with checkpoint"
                )
        if (
            transfer_manifest.get("build", {}).get("transfer_source_bundle_sha256")
            != checkpoint_sha256
        ):
            raise ValueError(
                "transfer text bundle is not bound to the checkpoint text bundle"
            )
        if not checkpoint_entries or not transfer_entries:
            raise ValueError("checkpoint and transfer text bundles must both be non-empty")
        return str(Path(transfer_directory)), transfer_manifest["npz_sha256"]
    if raw_checkpoint_manifest.get("schema") == "t2_robotokens_plus_mjcf_sidecars_v1":
        from molmo_spaces.embodiment import robotokens

        checkpoint_manifest, _ = robotokens.load_bundle(checkpoint_directory)
        transfer_manifest, transfer_entries = robotokens.load_bundle(transfer_directory)
        if checkpoint_manifest["npz_sha256"] != checkpoint_sha256:
            raise ValueError("checkpoint RoboToken bundle changed after validation")
        if transfer_manifest["npz_sha256"] != transfer_sha256:
            raise ValueError("transfer RoboToken bundle hash mismatch")
        if (
            checkpoint_manifest["schema"] != transfer_manifest["schema"]
            or checkpoint_manifest["shape"] != transfer_manifest["shape"]
            or checkpoint_manifest["reference_namespaces"]
            != transfer_manifest["reference_namespaces"]
            or checkpoint_manifest["token_types"] != transfer_manifest["token_types"]
            or checkpoint_manifest["token_capacities"]
            != transfer_manifest["token_capacities"]
        ):
            raise ValueError("transfer RoboToken schema disagrees with checkpoint")
        if transfer_manifest["normalization"].get("source") != checkpoint_sha256:
            raise ValueError("transfer RoboToken bundle did not reuse checkpoint normalization")
        if not transfer_entries:
            raise ValueError("transfer RoboToken bundle is empty")
        return str(Path(transfer_directory)), transfer_manifest["npz_sha256"]
    checkpoint_manifest, checkpoint_entries = load_bundle(checkpoint_directory)
    transfer_manifest, transfer_entries = load_bundle(transfer_directory)
    checkpoint_has_global = "global_tensors" in checkpoint_manifest
    transfer_has_global = "global_tensors" in transfer_manifest
    checkpoint_global_entries = (
        load_bundle_global_context(checkpoint_directory)
        if checkpoint_has_global
        else {}
    )
    transfer_global_entries = (
        load_bundle_global_context(transfer_directory)
        if transfer_has_global
        else {}
    )
    if checkpoint_manifest["npz_sha256"] != checkpoint_sha256:
        raise ValueError("checkpoint embodiment bundle changed after validation")
    if transfer_manifest["npz_sha256"] != transfer_sha256:
        raise ValueError(
            "transfer embodiment bundle hash mismatch: "
            f"{transfer_manifest['npz_sha256']} != {transfer_sha256}"
        )
    if transfer_manifest["schema"] != checkpoint_manifest["schema"]:
        raise ValueError(
            "transfer embodiment bundle schema disagrees with checkpoint: "
            f"{transfer_manifest['schema']} != {checkpoint_manifest['schema']}"
        )
    checkpoint_normalization = checkpoint_manifest.get(
        "description_normalization"
    )
    transfer_normalization = transfer_manifest.get("description_normalization")
    if checkpoint_normalization != transfer_normalization:
        raise ValueError(
            "transfer embodiment bundle normalization disagrees with checkpoint"
        )
    if not checkpoint_entries or not transfer_entries:
        raise ValueError("checkpoint and transfer embodiment bundles must both be non-empty")
    checkpoint_shapes = tuple(
        value.shape for value in next(iter(checkpoint_entries.values()))
    )
    transfer_shapes = tuple(value.shape for value in next(iter(transfer_entries.values())))
    if transfer_shapes != checkpoint_shapes:
        raise ValueError(
            "transfer embodiment bundle tensor shapes disagree with checkpoint: "
            f"{transfer_shapes} != {checkpoint_shapes}"
        )
    if bool(transfer_global_entries) != bool(checkpoint_global_entries):
        raise ValueError(
            "transfer and checkpoint bundles disagree on global context presence"
        )
    if checkpoint_global_entries:
        checkpoint_global_shapes = tuple(
            value.shape
            for value in next(iter(checkpoint_global_entries.values()))
        )
        transfer_global_shapes = tuple(
            value.shape
            for value in next(iter(transfer_global_entries.values()))
        )
        if transfer_global_shapes != checkpoint_global_shapes:
            raise ValueError(
                "transfer global tensor shapes disagree with checkpoint: "
                f"{transfer_global_shapes} != {checkpoint_global_shapes}"
            )
    return str(transfer_directory), transfer_manifest["npz_sha256"]
