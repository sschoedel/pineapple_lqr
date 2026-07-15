"""Steady-state Kalman filter over the reduced linearization.

Measurements are exactly what the real robot has (and what the RL deploy
stack uses): joint positions and velocities from the DAMIAO encoders, and
roll/pitch tilt + body angular rates from the IMU/EKF. Base linear velocity
and base height are never measured — the filter reconstructs them through
the model's contact dynamics (the principled version of wheel odometry).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.linalg

from lqr.linearize import Linearization, tangent_state_labels
from lqr.model import RobotModel

# Measurement noise std, hardware-spec tier (DAMIAO encoders, EKF'd IMU).
MEAS_STD_JOINT_POS = 0.002  # rad
MEAS_STD_JOINT_VEL = 0.1  # rad/s
MEAS_STD_TILT = 0.008  # rad
MEAS_STD_GYRO = 0.02  # rad/s


@dataclasses.dataclass(frozen=True)
class KalmanDesign:
    C: np.ndarray  # (ny, nx) measurement matrix on the reduced state
    L: np.ndarray  # (nx, ny) steady-state Kalman gain
    y_labels: list[str]


def measurement_matrix(rm: RobotModel, lin: Linearization) -> tuple[np.ndarray, list[str]]:
    labels_full = tangent_state_labels(rm)
    labels = [labels_full[i] for i in lin.keep]
    idx = {lab: i for i, lab in enumerate(labels)}
    joints = [
        "hip_l", "thigh_l", "calf_l", "hip_r", "thigh_r", "calf_r",
    ]
    y_labels = (
        joints  # positions (wheel angles are dropped states — not measured here)
        + ["d" + j for j in joints]
        + ["dwheel_l", "dwheel_r"]
        + ["roll", "pitch"]
        + ["droll", "dpitch", "dyaw"]
    )
    C = np.zeros((len(y_labels), len(labels)))
    for r, lab in enumerate(y_labels):
        C[r, idx[lab]] = 1.0
    return C, y_labels


def measurement_noise() -> np.ndarray:
    stds = (
        [MEAS_STD_JOINT_POS] * 6
        + [MEAS_STD_JOINT_VEL] * 6
        + [MEAS_STD_JOINT_VEL] * 2
        + [MEAS_STD_TILT] * 2
        + [MEAS_STD_GYRO] * 3
    )
    return np.diag(np.square(stds))


def process_noise(labels: list[str]) -> np.ndarray:
    """Process noise per state: pushes and model error act on velocities."""
    var = []
    for lab in labels:
        if lab.startswith("d"):
            var.append(1e-2)  # rad^2/s^2-ish per step: disturbances
        else:
            var.append(1e-8)
    return np.diag(var)


def kalman_gain(
    rm: RobotModel,
    lin: Linearization,
    Qw: np.ndarray | None = None,
    Rv: np.ndarray | None = None,
) -> KalmanDesign:
    keep = lin.keep
    A = lin.A[np.ix_(keep, keep)]
    C, y_labels = measurement_matrix(rm, lin)
    labels_full = tangent_state_labels(rm)
    labels = [labels_full[i] for i in keep]
    if Qw is None:
        Qw = process_noise(labels)
    if Rv is None:
        Rv = measurement_noise()
    # Observability check (PBH-ish via staircase rank).
    n = A.shape[0]
    O = [C]
    for _ in range(n - 1):
        O.append(O[-1] @ A)
    rank = np.linalg.matrix_rank(np.vstack(O), tol=1e-9)
    if rank < n:
        raise RuntimeError(f"(A, C) not observable: rank {rank} < {n}")
    # Steady-state predictive covariance via the dual DARE, then the
    # current-estimate (filtered) gain.
    P = scipy.linalg.solve_discrete_are(A.T, C.T, Qw, Rv)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + Rv)
    return KalmanDesign(C=C, L=L, y_labels=y_labels)
