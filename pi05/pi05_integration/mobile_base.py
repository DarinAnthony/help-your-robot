"""Holonomic-base slots of the Pi0.5 model vector and their base math.

Two contracts share the slot layout and differ in what the three base numbers mean:

* ``mobile_base_slots_v2`` (current) reads RLDS ``mobile_base_feature_v2``
  (``scripts/data/format_conversion/README_RLDS.md``): the action is the commanded body-frame base
  velocity ``steps/base_velocity_command`` and the state the measured one
  ``steps/observation/base_velocity``, both ``[v_forward m/s, v_left m/s, yaw_rate rad/s]``.
* ``mobile_base_slots_v1`` (superseded) reads ``mobile_base_feature_v1``: the action is the base
  command minus the measured pose in the current base frame and the state the measured pose in the
  episode's first base frame, ``[forward m, left m, yaw rad wrapped]`` (that state jumps when the base
  turns past 180 degrees).

The 10-wide arm vector (joint deltas, gripper, padding) is unchanged. The base takes the three model
slots right after it, ``BASE_SLOTS = (10, 11, 12)``, after normalization; a fixed arm leaves them zero,
which is what padding already puts there. Inside the loader and transforms both contracts use the same
keys (``BASE_KEYS``) and normalization tables (``BASE_NORM_KEYS``); ``RLDS_FIELDS`` maps each contract
to its own RLDS step keys, so a build of the other version is refused instead of being misread. This
module is numpy only: the Pi server and the MuJoCo eval client (separate environments) share it.
"""

from __future__ import annotations

import numpy as np

CONTRACT_V1 = "mobile_base_slots_v1"
CONTRACT_V2 = "mobile_base_slots_v2"
CONTRACTS = (CONTRACT_V1, CONTRACT_V2)
ARM_WIDTH = 10
BASE_DIM = 3
BASE_SLOTS = tuple(range(ARM_WIDTH, ARM_WIDTH + BASE_DIM))
# Loader/transform keys. ``base_actions`` is chunked like ``actions``; the masks are per frame.
BASE_KEYS = ("base_state", "base_state_mask", "base_actions", "base_action_mask")
# Normalization-asset keys (beside ``state`` and ``actions``, which keep their arm-only meaning).
BASE_NORM_KEYS = ("base_state", "base_actions")
# RLDS step keys of each contract: action, action mask (under ``steps``), state, state mask (under
# ``steps/observation``). Must match the converter's ``BASE_FEATURE_KEYS``.
RLDS_FIELDS = {
    CONTRACT_V1: ("base_action", "base_action_mask", "base_state", "base_state_mask"),
    CONTRACT_V2: ("base_velocity_command", "base_velocity_command_mask", "base_velocity", "base_velocity_mask"),
}


def wrap_angle(angle):
    return (np.asarray(angle, np.float64) + np.pi) % (2 * np.pi) - np.pi


def pose_in_frame(origin, pose) -> np.ndarray:
    """``pose`` ([x, y, theta], world) expressed in the frame of ``origin``: [forward, left, yaw wrapped].

    The same frame change as the converter's ``mobile_base_body_delta`` (v1): base_state is
    ``pose_in_frame(first_pose, pose)``, base_action is ``pose_in_frame(pose, command)``."""
    origin = np.asarray(origin, np.float64)[:3]
    pose = np.asarray(pose, np.float64)[:3]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    d = pose[:2] - origin[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], wrap_angle(pose[2] - origin[2])])


def compose_base_target(pose, delta) -> np.ndarray:
    """World ``[x, y, theta]`` position target of the ``base`` move group for a base-frame ``delta``.

    Inverse of ``pose_in_frame(pose, target)``: ``delta`` is [forward m, left m, yaw rad] relative to the
    CURRENT measured base pose ``pose``. Theta is not wrapped, so the target stays continuous with the
    hinge joint's measured angle."""
    pose = np.asarray(pose, np.float64)[:3]
    delta = np.asarray(delta, np.float64)[:3]
    if not (np.all(np.isfinite(pose)) and np.all(np.isfinite(delta))):
        raise ValueError("base pose and base action must be finite")
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([pose[0] + c * delta[0] - s * delta[1], pose[1] + s * delta[0] + c * delta[1], pose[2] + delta[2]])


def body_velocity(previous_pose, pose, dt_s: float) -> np.ndarray:
    """Measured body-frame velocity (v2 base state): ``(pose - previous_pose) / dt`` in the PREVIOUS base frame.

    The converter's ``mobile_base_body_velocity`` state; [v_forward m/s, v_left m/s, yaw_rate rad/s]."""
    if not dt_s or dt_s <= 0:
        raise ValueError("body velocity needs a positive dt")
    return pose_in_frame(previous_pose, pose) / float(dt_s)


def velocity_base_target(reference, yaw, velocity, dt_s: float) -> np.ndarray:
    """World ``[x, y, theta]`` base target one step after ``reference`` for a body-frame ``velocity`` (v2).

    ``reference + R(yaw) * velocity * dt``, ``yaw`` the CURRENT measured heading. With ``reference`` the
    previous target this inverts the converter's label ``R(yaw)^T (target_t - target_(t-1)) / dt``
    exactly; with the measured pose it is a target one step ahead of where the base is."""
    reference = np.asarray(reference, np.float64)[:3]
    velocity = np.asarray(velocity, np.float64)[:3]
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(velocity)) and np.isfinite(yaw)):
        raise ValueError("base reference, heading and base velocity must be finite")
    if not dt_s or dt_s <= 0:
        raise ValueError("velocity execution needs a positive dt")
    c, s = np.cos(yaw), np.sin(yaw)
    return reference + np.array([c * velocity[0] - s * velocity[1], s * velocity[0] + c * velocity[1], velocity[2]]) * dt_s
