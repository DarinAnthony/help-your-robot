"""The gripper value a Pi0.5 checkpoint is trained on (``Pi05InputConfig.gripper_value``; review T8, BoAI 2026-11-05).

``gripper_closedness_v1`` (``molmo_spaces/robots/gripper_closedness.py``, ``README_RLDS.md`` "Canonical gripper
value"): state and action c in [0, 1], 0 = open, 1 = closed. The action of every unified-arm build is already the
command's closedness (command / 255). The state is either already c (a build converted with
``--gripper-proprio-source gripper_closedness_v1``) or the measured pad aperture in metres (``physical_aperture_m``,
e.g. the xsrc40 exclv2 builds), which the loader converts at load with each robot's own (closed, open) endpoints from
the sha-pinned endpoints table (``gripper_endpoints_table.py``), keyed by ``episode_metadata/robot_artifact_hash``:
c = (open - w) / (open - closed), clipped. No rebuild is needed.

The evaluation client names the contract (``image_geometry``'s pattern): a client with
``proprio_gripper_source="gripper_closedness_v1"`` sends ``gripper_value``, and the server refuses a request whose
contract differs from the checkpoint's. This module imports no simulator; TensorFlow is passed in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STORED = "stored"
GRIPPER_CLOSEDNESS_V1 = "gripper_closedness_v1"
CONTRACTS = (STORED, GRIPPER_CLOSEDNESS_V1)
#: The endpoints table contract (``scripts/data/format_conversion/gripper_endpoints_table.py``).
ENDPOINTS_TABLE_SCHEMA, ENDPOINTS_V1 = "gripper_endpoints_table_v1", "gripper_endpoints_v1"
#: A state above the robot's open endpoint by more than this is not an aperture in metres (a closedness build reads
#: up to 1): the loader refuses it rather than map it.
APERTURE_SLACK_M = 0.02


def load_endpoints_table(path, sha256: str) -> dict[str, tuple[float, float]]:
    """{robot artifact hash: (closed_m, open_m)} of a sha-pinned endpoints table."""
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha256:
        raise ValueError(f"gripper endpoints table {path}: sha256 {actual} != {sha256}")
    table = json.loads(payload)
    if table.get("schema") != ENDPOINTS_TABLE_SCHEMA or table.get("version") != ENDPOINTS_V1:
        raise ValueError(f"{path} is not a {ENDPOINTS_TABLE_SCHEMA} table of {ENDPOINTS_V1}")
    rows = {}
    for key, row in table["robots"].items():
        closed, open_ = float(row["closed_m"]), float(row["open_m"])
        if not 0.0 <= closed < open_:
            raise ValueError(f"{path}: robot {key} endpoints ({closed}, {open_}) are not (closed, open)")
        rows[str(key)] = (closed, open_)
    return rows


def closedness(aperture_m, closed_m, open_m):
    """c = (open - w) / (open - closed), clipped to [0, 1] (NumPy; the loader's TF form is ``closedness_state_tf``)."""
    return np.clip((np.asarray(open_m) - np.asarray(aperture_m)) / (np.asarray(open_m) - np.asarray(closed_m)), 0.0, 1.0)


class EndpointLookup:
    """The table as TF constants: ``row(hash)`` -> (closed, open) of one episode's robot; a robot outside the table
    fails the episode (never a guessed range)."""

    def __init__(self, rows: dict[str, tuple[float, float]], tf):
        keys = sorted(rows)
        if not keys:
            raise ValueError("empty gripper endpoints table")
        self._tf = tf
        self._index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.range(len(keys), dtype=tf.int64)), -1)
        self._closed = tf.constant([rows[k][0] for k in keys], tf.float32)
        self._open = tf.constant([rows[k][1] for k in keys], tf.float32)

    def row(self, artifact_hash):
        tf = self._tf
        index = self._index.lookup(artifact_hash)
        with tf.control_dependencies([tf.debugging.assert_non_negative(
                index, message="robot artifact hash is not in the gripper endpoints table")]):
            index = tf.identity(index)
        return tf.gather(self._closed, index), tf.gather(self._open, index)


def closedness_state_tf(state, state_mask, closed, open_, tf):
    """A (T, D) left-packed unified-arm state with the gripper (the last valid slot) converted from metres to c."""
    slot = tf.reduce_sum(tf.cast(state_mask[0], tf.int32)) - 1
    aperture = tf.cast(state[:, slot], tf.float32)
    with tf.control_dependencies([tf.debugging.assert_less_equal(
            aperture, open_ + APERTURE_SLACK_M,
            message="gripper state exceeds the robot's open aperture: not a physical_aperture_m build")]):
        value = tf.clip_by_value((open_ - aperture) / (open_ - closed), 0.0, 1.0)
    column = tf.equal(tf.range(tf.shape(state)[1]), slot)[None]
    return tf.where(column, tf.cast(value, state.dtype)[:, None], state)
