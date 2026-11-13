"""The arm-joint frame a Pi0.5 checkpoint is trained in (``Pi05InputConfig.joint_frame``; review T2/item 4, BoAI
2026-11-05).

``canonical_mdh_tool_v1`` (``molmo_spaces/robots/unified/joint_frame.py``, ``DISTRIBUTION.md`` section 7): per arm
joint a direction s (+-1) and a zero theta (rad) of the robot's canonical modified-DH description, with the last joint's
zero anchored to the gripper's closing line. In that frame

- state: phi = theta + s * q;
- a joint action that is a difference of two joint values (``per_step_joint_delta``, ``query_anchored_joint``):
  a_phi = s * a (theta cancels; s is its own inverse, so deployment executes s * a_phi).

Training: the loader maps each sim episode's arm slots with its robot's (s, theta) from the sha-pinned joint-frame
table (``scripts/data/format_conversion/joint_frame_table.py``), keyed by ``episode_metadata/robot_artifact_hash``; a
build that also stores the recorded convention (``joint_convention_feature_v1``) is checked against the table. No
rebuild is needed.

Serving: the evaluation client measures the robot's own (s, theta) with the code that built the table and sends them
in its slot layout (``joint_frame``, ``joint_frame_sign``, ``joint_frame_zero``); the server maps the proprio and the
RTC prefix into the frame and the predicted arm actions back, and refuses a client in another frame. This module
imports no simulator; TensorFlow is passed in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STORED = "stored"
CANONICAL_MDH_TOOL_V1 = "canonical_mdh_tool_v1"
CONTRACTS = (STORED, CANONICAL_MDH_TOOL_V1)
TABLE_SCHEMA = "joint_frame_table_v1"
#: The arm-slot width of the RLDS state and action (``mlspaces_to_rlds.JOINT_CONVENTION_DIM``).
SLOTS = 10
#: The arm action forms whose arm slots are a difference of two joint values (``pi05_integration/arm_action.py``).
JOINT_DIFFERENCE_FORMS = ("per_step_joint_delta", "query_anchored_joint")
#: Agreement of the table's recorded zeros with a build's ``joint_zero`` (float32 radians).
RECORDED_ZERO_ATOL = 1e-4


def slot_vectors(sign, zero) -> tuple[np.ndarray, np.ndarray]:
    """(sign, zero) of one robot's n arm joints in the ``SLOTS`` left-packed arm slots; sign 0 marks a slot that is
    not an arm joint (left unchanged)."""
    sign, zero = np.asarray(sign, np.float32), np.asarray(zero, np.float32)
    n = len(sign)
    if n > SLOTS or zero.shape != (n,) or not np.all(np.abs(sign) == 1) or not np.all(np.isfinite(zero)):
        raise ValueError(f"not an arm joint frame: sign {sign.tolist()}, zero {zero.tolist()}")
    out_sign, out_zero = np.zeros(SLOTS, np.float32), np.zeros(SLOTS, np.float32)
    out_sign[:n], out_zero[:n] = sign, zero
    return out_sign, out_zero


def load_table(path, sha256: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{robot artifact hash: (sign[SLOTS], zero[SLOTS])} of a sha-pinned joint-frame table."""
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha256:
        raise ValueError(f"joint frame table {path}: sha256 {actual} != {sha256}")
    table = json.loads(payload)
    if table.get("schema") != TABLE_SCHEMA or table.get("version") != CANONICAL_MDH_TOOL_V1:
        raise ValueError(f"{path} is not a {TABLE_SCHEMA} table of {CANONICAL_MDH_TOOL_V1}")
    return {str(key): slot_vectors(row["sign"], row["zero_rad"]) for key, row in table["robots"].items()}


def apply_state(state, sign, zero):
    """phi = zero + sign * q on the arm slots (sign != 0) of a (..., W >= SLOTS) state; other slots unchanged."""
    state = np.array(state, np.float32, copy=True)
    sign, zero = np.asarray(sign, np.float32), np.asarray(zero, np.float32)
    head = state[..., :SLOTS]
    state[..., :SLOTS] = np.where(sign != 0, zero + sign * head, head)
    return state


def apply_action(action, sign):
    """sign * a on the arm slots of a joint-difference action (its own inverse); other slots unchanged."""
    action = np.array(action, np.float32, copy=True)
    sign = np.asarray(sign, np.float32)
    head = action[..., :SLOTS]
    action[..., :SLOTS] = np.where(sign != 0, sign * head, head)
    return action


class FrameLookup:
    """The table as TF constants: ``row(hash)`` -> (sign[SLOTS], zero[SLOTS]) of one episode's robot; a robot outside
    the table fails the episode."""

    def __init__(self, rows: dict[str, tuple[np.ndarray, np.ndarray]], tf):
        keys = sorted(rows)
        if not keys:
            raise ValueError("empty joint frame table")
        self._tf = tf
        self._index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.range(len(keys), dtype=tf.int64)), -1)
        self._sign = tf.constant(np.stack([rows[k][0] for k in keys]), tf.float32)
        self._zero = tf.constant(np.stack([rows[k][1] for k in keys]), tf.float32)

    def row(self, artifact_hash):
        tf = self._tf
        index = self._index.lookup(artifact_hash)
        with tf.control_dependencies([tf.debugging.assert_non_negative(
                index, message="robot artifact hash is not in the joint frame table")]):
            index = tf.identity(index)
        return tf.gather(self._sign, index), tf.gather(self._zero, index)


def check_recorded_tf(observation, sign, zero, tf):
    """Control dependencies that a build's recorded convention (``joint_sign``/``joint_zero``/``joint_convention_mask``,
    when stored) agrees with the table row: the same signs, and the same zeros except the last arm joint's (the
    table's is tool-anchored)."""
    if "joint_sign" not in observation:
        return []
    mask = tf.cast(observation["joint_convention_mask"][0], tf.bool)
    arm = sign != 0
    last = tf.reduce_sum(tf.cast(arm, tf.int32)) - 1
    not_last = tf.not_equal(tf.range(SLOTS), last)
    recorded_sign = tf.cast(observation["joint_sign"][0], tf.float32)
    recorded_zero = tf.cast(observation["joint_zero"][0], tf.float32)
    # A robot whose params record no convention has an all-false mask: nothing to check against.
    unrecorded = tf.logical_not(tf.reduce_any(mask))
    return [
        tf.debugging.Assert(tf.logical_or(unrecorded, tf.reduce_all(tf.equal(mask, arm))),
                            ["joint frame table and build disagree on the arm joints"]),
        tf.debugging.Assert(tf.logical_or(unrecorded, tf.reduce_all(tf.equal(tf.where(arm, recorded_sign, 0.0), sign))),
                            ["joint frame table and build disagree on the joint signs"]),
        tf.debugging.Assert(tf.logical_or(unrecorded, tf.reduce_all(
            tf.abs(tf.where(tf.logical_and(arm, not_last), recorded_zero - zero, 0.0)) <= RECORDED_ZERO_ATOL)),
                            ["joint frame table and build disagree on the joint zeros"]),
    ]


def apply_tf(state, actions, sign, zero, checks, tf):
    """(state, actions) of a (T, D >= SLOTS) left-packed episode mapped into the frame (``apply_state``,
    ``apply_action``)."""
    def widen(v, fill, like):
        return tf.concat([v, tf.fill([tf.shape(like)[1] - SLOTS], fill)], 0)[None]

    with tf.control_dependencies(checks):
        arm = sign != 0
        state_f = tf.cast(state, tf.float32)
        state = tf.cast(tf.where(widen(arm, False, state),
                                 widen(zero, 0.0, state) + widen(sign, 0.0, state) * state_f, state_f), state.dtype)
        actions = tf.where(widen(arm, False, actions),
                           tf.cast(widen(sign, 0.0, actions), actions.dtype) * actions, actions)
    return state, actions
