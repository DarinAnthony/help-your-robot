"""Arm action forms of a Pi0.5 recipe (``Pi05InputConfig.arm_action``) and their pose geometry.

Every form is a chunk of H targets predicted at query step t (the robot executes the first K):

- ``per_step_joint_delta`` (default, every recipe before October 26): the RLDS ``steps/action``
  itself, ``q_cmd(t+k) - q_meas(t+k)``: each target relative to the joints measured at its own step.
- ``query_anchored_joint``: ``q_cmd(t+k) - q_meas(t)``: every target of the chunk relative to the
  arm joints measured at the query step (openpi ``DeltaActions``). Built from any joint build: the
  absolute target is ``steps/action + proprio`` on the arm slots.
- ``query_anchored_base_tcp``: the commanded TCP pose relative to the TCP measured at the query
  step, both in the robot base-link frame: ``dp = p_cmd(t+k) - p_meas(t)`` along base axes and
  ``dR = R_cmd(t+k) R_meas(t)^T`` (left-multiplied, base frame) stored as its first two columns
  (rotation6D). The client rebuilds ``p = p_meas(t) + dp``, ``R = dR R_meas(t)`` and solves IK.
  Needs a build with ``arm_reference_feature_v1`` (``mlspaces_to_rlds.py --arm-reference-features``).
- ``query_anchored_joint_and_base_tcp``: both in one 20-slot chunk, with the TCP in the state.
  For an n-joint arm, n + 10 of ``JOINT_AND_BASE_TCP_WIDTH`` slots are valid:
  state ``[q (n), aperture, TCP pose9]``, action ``[q target - q(t) (n), TCP goal vs TCP(t) (9),
  gripper]``. The joint state stays in front, where URMA4 reads its per-control state.

The gripper is always the absolute command. ``pose9`` is ``[x, y, z, R[:, 0], R[:, 1]]`` in metres.
Evidence for the forms: notes/20261005_xemb1k_ctr_unif_joint_vs_ee_vs_eestate.md (finding 6).
This module imports no simulator, TensorFlow or JAX runtime.
"""

from __future__ import annotations

import numpy as np

PER_STEP_JOINT_DELTA = "per_step_joint_delta"
QUERY_ANCHORED_JOINT = "query_anchored_joint"
QUERY_ANCHORED_BASE_TCP = "query_anchored_base_tcp"
QUERY_ANCHORED_JOINT_AND_BASE_TCP = "query_anchored_joint_and_base_tcp"
CONTRACTS = (PER_STEP_JOINT_DELTA, QUERY_ANCHORED_JOINT, QUERY_ANCHORED_BASE_TCP, QUERY_ANCHORED_JOINT_AND_BASE_TCP)
QUERY_ANCHORED = frozenset(CONTRACTS[1:])
READS_BASE_TCP = frozenset((QUERY_ANCHORED_BASE_TCP, QUERY_ANCHORED_JOINT_AND_BASE_TCP))
# The joint + BASE TCP checkpoint predicts both; a client executes one of them.
HEADS = ("joint", "ee")

# RLDS fields of ``arm_reference_feature_v1`` (float32 pose9, robot base-link frame):
# the TCP measured at the step (``obs/extra/tcp_pose``) and the commanded TCP goal paired with the
# step's action (``actions/ee_pose["gripper"]``: FK of the commanded joint target).
ARM_REFERENCE_FEATURE_V1 = "arm_reference_feature_v1"
BASE_TCP_KEY = "base_tcp"  # steps/observation/base_tcp
BASE_TCP_GOAL_KEY = "base_tcp_goal"  # steps/base_tcp_goal
POSE_DIM = 9
ARM_WIDTH = 10
JOINT_AND_BASE_TCP_WIDTH = 20
# The relative pose of "no motion": zero translation, identity rotation columns.
IDENTITY_POSE9 = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
ROTATION6D_DEGENERACY_EPS = 1e-8


def width(contract: str) -> int:
    """Model-facing action/state width of a form."""
    return JOINT_AND_BASE_TCP_WIDTH if contract == QUERY_ANCHORED_JOINT_AND_BASE_TCP else ARM_WIDTH


def pose9_from_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.concatenate([pose[:3, 3], pose[:3, 0], pose[:3, 1]])


def rotation_from_6d(columns6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt rotation from two (approximate) columns; raises on a degenerate pair."""
    columns6 = np.asarray(columns6, dtype=float)
    first = columns6[:3]
    norm = np.linalg.norm(first)
    if not np.isfinite(norm) or norm < ROTATION6D_DEGENERACY_EPS:
        raise ValueError("degenerate rotation6D: first column has no direction")
    first = first / norm
    second = columns6[3:6] - first * float(first @ columns6[3:6])
    norm = np.linalg.norm(second)
    if not np.isfinite(norm) or norm < ROTATION6D_DEGENERACY_EPS:
        raise ValueError("degenerate rotation6D: second column is parallel to the first")
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def matrix_from_pose9(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = rotation_from_6d(pose9[3:9])
    pose[:3, 3] = pose9[:3]
    return pose


def relative_pose9(goal9: np.ndarray, anchor9: np.ndarray) -> np.ndarray:
    """Label of a goal against an anchor (the loader's TensorFlow twin is ``_relative_pose9``)."""
    goal, anchor = matrix_from_pose9(goal9), matrix_from_pose9(anchor9)
    rotation = goal[:3, :3] @ anchor[:3, :3].T
    return np.concatenate([goal[:3, 3] - anchor[:3, 3], rotation[:, 0], rotation[:, 1]])


def compose_pose9(anchor: np.ndarray, relative9: np.ndarray) -> np.ndarray:
    """Inverse of ``relative_pose9``: the 4x4 goal of a label against a 4x4 anchor (same frame)."""
    anchor = np.asarray(anchor, dtype=float)
    relative9 = np.asarray(relative9, dtype=float)
    goal = np.eye(4)
    goal[:3, :3] = rotation_from_6d(relative9[3:9]) @ anchor[:3, :3]
    goal[:3, 3] = anchor[:3, 3] + relative9[:3]
    return goal


def joint_and_base_tcp_state(proprio: np.ndarray, tcp9: np.ndarray, n_arm: int) -> np.ndarray:
    """20-slot state of a left-packed ``[q (n), aperture]`` proprio plus the measured TCP pose9."""
    state = np.zeros(JOINT_AND_BASE_TCP_WIDTH, np.float32)
    state[: n_arm + 1] = np.asarray(proprio, np.float32)[: n_arm + 1]
    state[n_arm + 1 : n_arm + 10] = np.asarray(tcp9, np.float32)
    return state


def joint_and_base_tcp_head(chunk: np.ndarray, n_arm: int, head: str) -> np.ndarray:
    """One head of a (H, 20) joint + BASE TCP chunk as a (H, 10) chunk of that single form.

    ``joint``: ``[q target - q(t) (n), gripper]`` left-packed (a ``query_anchored_joint`` chunk);
    ``ee``: ``[TCP goal vs TCP(t) (9), gripper]`` (a ``query_anchored_base_tcp`` chunk)."""
    chunk = np.asarray(chunk)
    if head == "ee":
        return chunk[:, n_arm : n_arm + 10].copy()
    if head != "joint":
        raise ValueError(f"unknown joint + BASE TCP head {head!r}; expected one of {HEADS}")
    joint = np.zeros((len(chunk), ARM_WIDTH), chunk.dtype)
    joint[:, :n_arm] = chunk[:, :n_arm]
    joint[:, n_arm] = chunk[:, n_arm + 9]
    return joint
