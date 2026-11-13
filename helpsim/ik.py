"""Damped least squares IK for the station DROID's grasp site (position + full orientation)."""

from __future__ import annotations

import mujoco
import numpy as np

from helpsim.envs.kitchen_station import PREFIX

ARM = [f"{PREFIX}fr3_joint{i}" for i in range(1, 8)]


class ArmIK:
    def __init__(self, model: mujoco.MjModel, site: str = f"{PREFIX}gripper/grasp_site"):
        self.model = model
        self.site = model.site(site).id
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM])
        self.dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM])
        self.range = np.array([model.jnt_range[model.joint(n).id] for n in ARM])
        self.scratch = mujoco.MjData(model)

    def solve(self, qpos_full: np.ndarray, pos: np.ndarray, rot: np.ndarray, q0: np.ndarray, iters: int = 200):
        """Arm joints putting the site at (pos, rot) from seed q0, with the rest of the state from qpos_full.
        Returns (q, position error m, rotation error rad)."""
        m, d = self.model, self.scratch
        d.qpos[:] = qpos_full
        d.qpos[self.qadr] = q0
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cur = d.site_xmat[self.site].reshape(3, 3)
            ep = pos - d.site_xpos[self.site]
            er = 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + 0.7 * dq, self.range[:, 0], self.range[:, 1])
        mujoco.mj_kinematics(m, d)
        cur = d.site_xmat[self.site].reshape(3, 3)
        rot_err = float(np.arccos(np.clip((np.trace(cur.T @ rot) - 1) / 2, -1, 1)))
        return d.qpos[self.qadr].copy(), float(np.linalg.norm(pos - d.site_xpos[self.site])), rot_err


def grasp_frame(approach: np.ndarray, roll_rad: float = 0.0) -> np.ndarray:
    """Site rotation whose z is the approach direction and whose y (the 2F-85's closing axis) is horizontal, then
    rolled about the approach by roll_rad."""
    z = approach / np.linalg.norm(approach)
    y = np.cross(z, [0.0, 0.0, 1.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    x, y = c * x + s * y, -s * x + c * y
    return np.column_stack([x, y, z])
