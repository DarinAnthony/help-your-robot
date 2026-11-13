"""Model I/O presets: the named input representations a checkpoint is trained in.

Three load-time choices decide what the model sees of a robot, apart from the recipe:

- the frame geometry (``Pi05InputConfig.image_geometry``, review T5);
- the gripper value (``gripper_value``, T8);
- the arm-joint frame (``joint_frame``, T2/item 4).

Each is opt-in and serialized in the input record. A preset names one consistent set of all three, so production
configs move between representations as a unit, and an evaluation client can be set from the checkpoint's preset
instead of three independent flags:

- ``io_v1_stored``: every default. Frames as stored (stretched to squares), the gripper value as stored, each robot's
  own joint convention. This is every checkpoint up to oer11-prod (``october28_pop34fk_qanchor_posquant_b1024``).
- ``io_v2_canonical``: letterboxed frames (``letterbox_stretched_v1``), gripper closedness (``gripper_closedness_v1``,
  0 = open, 1 = closed), and the canonical joint frame (``canonical_mdh_tool_v1``). The gripper endpoints table and the
  joint-frame table are per dataset, so the preset takes both with their sha256.

A robot description (URMA4-absolute bundle) must be compiled in the same joint frame: a ``stored`` bundle goes with
``io_v1_stored`` and a canonical bundle with ``io_v2_canonical`` (``PiExperimentConfig`` refuses a mismatch).

Not yet built: the server does not report its preset to the client, so the evaluation config still sets the fields
returned by ``ModelIO.policy_fields``.
"""

from __future__ import annotations

import dataclasses

from pi05_integration import input_config as pi_input

IO_V1_STORED = "io_v1_stored"
IO_V2_CANONICAL = "io_v2_canonical"


@dataclasses.dataclass(frozen=True)
class ModelIO:
    """One named model I/O representation: frame geometry, gripper value and arm-joint frame."""

    name: str
    image_geometry: pi_input.ImageGeometry
    gripper_value: pi_input.GripperValue
    joint_frame: pi_input.JointFrame

    def apply(self, config: pi_input.Pi05InputConfig) -> pi_input.Pi05InputConfig:
        """``config`` with this preset's three fields; every other field is kept."""
        return dataclasses.replace(
            config, image_geometry=self.image_geometry, gripper_value=self.gripper_value, joint_frame=self.joint_frame)

    @property
    def contracts(self) -> dict[str, str]:
        return {"image_geometry": self.image_geometry.contract, "gripper_value": self.gripper_value.contract,
                "joint_frame": self.joint_frame.contract}

    def policy_fields(self) -> dict[str, str]:
        """The ``Pi05PolicyConfig`` fields an evaluation of this preset sets. A stored gripper value depends on how
        the builds stored it, so ``io_v1_stored`` leaves ``proprio_gripper_source`` to the evaluation config."""
        fields = {"image_geometry": self.image_geometry.contract, "joint_frame": self.joint_frame.contract}
        if self.gripper_value.enabled:
            fields["proprio_gripper_source"] = self.gripper_value.contract
        return fields


def io_v1_stored() -> ModelIO:
    return ModelIO(IO_V1_STORED, pi_input.StoredImageGeometry(), pi_input.StoredGripperValue(),
                   pi_input.StoredJointFrame())


def io_v2_canonical(*, gripper_endpoints_table: str, gripper_endpoints_table_sha256: str, joint_frame_table: str,
                    joint_frame_table_sha256: str) -> ModelIO:
    return ModelIO(
        IO_V2_CANONICAL,
        pi_input.LetterboxStretched(),
        pi_input.GripperClosedness(endpoints_table=gripper_endpoints_table,
                                   endpoints_table_sha256=gripper_endpoints_table_sha256),
        pi_input.CanonicalJointFrame(table=joint_frame_table, table_sha256=joint_frame_table_sha256),
    )


_PRESET_CONTRACTS = {
    IO_V1_STORED: {"image_geometry": "stretch", "gripper_value": "stored", "joint_frame": "stored"},
    IO_V2_CANONICAL: {"image_geometry": "letterbox_stretched_v1", "gripper_value": "gripper_closedness_v1",
                      "joint_frame": "canonical_mdh_tool_v1"},
}


def identify(config: pi_input.Pi05InputConfig) -> str | None:
    """The preset an input config is in, or None for a mix (e.g. the exclv2 cell's letterbox + closedness in the
    stored joint frame)."""
    contracts = {"image_geometry": config.image_geometry.contract, "gripper_value": config.gripper_value.contract,
                 "joint_frame": config.joint_frame.contract}
    return next((name for name, expected in _PRESET_CONTRACTS.items() if expected == contracts), None)
