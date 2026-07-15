"""Equilibrium search and finite-difference linearization about the stance.

The equilibrium follows the MuJoCo LQR tutorial recipe (also how mjlab's
stance torques were computed): with the stance joint angles fixed and
qacc = 0, contact-aware inverse dynamics gives the generalized force needed
to hold the pose. At the correct base height the wheel contacts supply the
base share, leaving a near-zero residual on the free-joint DOFs; the joint
components are the equilibrium torques u_eq.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import scipy.linalg

from lqr.model import (
    JOINT_NAMES,
    STANCE_JOINT_POS,
    STANDING_BASE_HEIGHT,
    RobotModel,
    reset_to_stance,
)


def base_residual(rm: RobotModel, data: mujoco.MjData, height: float) -> tuple[float, np.ndarray]:
    """Norm of the free-joint residual force and joint torques at this height."""
    reset_to_stance(rm, data, base_height=height)
    data.qacc[:] = 0.0
    mujoco.mj_inverse(rm.model, data)
    resid = float(np.linalg.norm(data.qfrc_inverse[:6]))
    return resid, data.qfrc_inverse[rm.joint_dof_adr].copy()


def find_equilibrium(
    rm: RobotModel,
    lo: float = STANDING_BASE_HEIGHT - 0.005,
    hi: float = STANDING_BASE_HEIGHT + 0.002,
    tol: float = 1e-10,
) -> tuple[float, np.ndarray, float]:
    """Bisect base height on the signed vertical base residual.

    Below the equilibrium height the wheels over-penetrate and contact pushes
    up harder than gravity (fz < 0 needed); above it contact breaks and fz
    jumps to +weight. The zero crossing is the height where wheel contact
    exactly supports the robot. Returns (height, u_eq, residual_norm).
    """

    def fz(h: float) -> float:
        reset_to_stance(rm, data, base_height=h)
        data.qacc[:] = 0.0
        mujoco.mj_inverse(rm.model, data)
        return float(data.qfrc_inverse[2])

    data = mujoco.MjData(rm.model)
    a, b = lo, hi
    fa, fb = fz(a), fz(b)
    if not (fa < 0.0 < fb):
        raise RuntimeError(
            f"no sign change in base z-residual on [{lo}, {hi}]: {fa:.1f}, {fb:.1f}"
        )
    while b - a > tol:
        m = 0.5 * (a + b)
        if fz(m) < 0.0:
            a = m
        else:
            b = m
    h = 0.5 * (a + b)
    resid, u_eq = base_residual(rm, data, h)
    return h, u_eq, resid


@dataclasses.dataclass(frozen=True)
class Linearization:
    height: float
    u_eq: np.ndarray  # equilibrium joint torques, JOINT_NAMES order
    A: np.ndarray  # (2nv, 2nv) discrete-time, one physics step
    B: np.ndarray  # (2nv, nu), columns in JOINT_NAMES order
    qpos_eq: np.ndarray
    keep: np.ndarray  # indices into the 2nv tangent state kept for LQR


def tangent_state_labels(rm: RobotModel) -> list[str]:
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    jnames = [""] * rm.model.njnt
    for n in JOINT_NAMES:
        jnames[rm.model.joint(n).id] = n
    for dof in range(6, rm.nv):
        jid = rm.model.dof_jntid[dof]
        labels.append(jnames[jid].removesuffix("_joint"))
    return labels + ["d" + s for s in labels]


def reduced_state_indices(rm: RobotModel) -> np.ndarray:
    """Tangent-state indices kept for LQR.

    Dropped positions: x, y (translation invariance), yaw (rotation
    invariance), wheel angles (rotation invariance).

    All velocities are kept. Base linear velocity and height have no direct
    sensor on hardware; at runtime they come from a kinematic/complementary
    estimator (wheel odometry through frozen contact Jacobians at DC, IMU
    forward acceleration at high frequency, kinematic height map) — see
    LqrController. The accelerometer is an exogenous information source,
    which is what breaks the static-substitution gain-reshaping pathology
    documented in PROGRESS.md iterations 3-4.
    """
    labels = tangent_state_labels(rm)
    drop = {"x", "y", "yaw", "wheel_l", "wheel_r"}
    return np.array([i for i, lab in enumerate(labels) if lab not in drop])


def linearize(rm: RobotModel, eps: float = 1e-6) -> Linearization:
    height, u_eq, resid = find_equilibrium(rm)
    if resid > 5.0:
        raise RuntimeError(f"equilibrium residual too large: {resid:.3f} N")
    data = mujoco.MjData(rm.model)
    reset_to_stance(rm, data, base_height=height)
    data.ctrl[rm.actuator_ids] = u_eq
    nv, nu = rm.nv, rm.model.nu
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(rm.model, data, eps, 1, A, B, None, None)
    # Reorder B columns from mujoco actuator order to JOINT_NAMES order.
    B = B[:, rm.actuator_ids]
    return Linearization(
        height=height,
        u_eq=u_eq,
        A=A,
        B=B,
        qpos_eq=data.qpos.copy(),
        keep=reduced_state_indices(rm),
    )


def lqr_gain(
    lin: Linearization,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Discrete LQR gain on the reduced state. Returns K with u = u_eq - K dx."""
    keep = lin.keep
    A = lin.A[np.ix_(keep, keep)]
    B = lin.B[keep, :]
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def lqr_gain_integral(
    lin: Linearization,
    Q: np.ndarray,
    R: np.ndarray,
    int_S: np.ndarray,
    Qi: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete LQR with integral action, designed on the augmented system.

    The design model is augmented with one integrator per row of ``int_S``
    (each row a linear functional of the reduced state):
    z' = z + dt * int_S @ x. Solving the DARE on the augmented system keeps
    LQR guarantees instead of bolting an integrator onto a proportional
    design. Returns (Kx, Ki) with u = u_eq - Kx (x - x_ref) - Ki z.
    """
    keep = lin.keep
    A = lin.A[np.ix_(keep, keep)]
    B = lin.B[keep, :]
    n, m, ni = A.shape[0], B.shape[1], int_S.shape[0]
    A_aug = np.zeros((n + ni, n + ni))
    A_aug[:n, :n] = A
    A_aug[n:, n:] = np.eye(ni)
    A_aug[n:, :n] = dt * int_S
    B_aug = np.vstack([B, np.zeros((ni, m))])
    Q_aug = scipy.linalg.block_diag(Q, Qi)
    P = scipy.linalg.solve_discrete_are(A_aug, B_aug, Q_aug, R)
    K = np.linalg.solve(R + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_aug)
    return K[:, :n], K[:, n:]


if __name__ == "__main__":
    from lqr.model import build_model

    rm = build_model()
    lin = linearize(rm)
    labels = tangent_state_labels(rm)
    print(f"equilibrium height: {lin.height:.6f} m")
    print("u_eq:", np.round(lin.u_eq, 3))
    keep = lin.keep
    print(f"reduced state ({len(keep)}):", [labels[i] for i in keep])
    Ar = lin.A[np.ix_(keep, keep)]
    Br = lin.B[keep, :]
    eig = np.linalg.eigvals(Ar)
    print("max |eig(A)|:", np.abs(eig).max())
    # Controllability (stabilizability check via staircase rank).
    n = Ar.shape[0]
    C = Br
    blocks = [Br]
    for _ in range(n - 1):
        blocks.append(Ar @ blocks[-1])
    C = np.hstack(blocks)
    print("ctrb rank:", np.linalg.matrix_rank(C, tol=1e-7), "/", n)
