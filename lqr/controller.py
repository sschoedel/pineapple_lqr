"""LQG controller (Kalman observer + full-state LQR) emitting DAMIAO
MIT-mode commands.

Structure per controller tick:
  1. y = measurements the real robot has: joint encoders (pos+vel), IMU
     tilt (roll/pitch) and gyro. Base linear velocity / height are never
     read — the Kalman filter reconstructs them through the model.
  2. x_hat = KF predict (with previously applied torque) + correct with y.
  3. u = u_eq - K (x_hat - x_ref) - Ki z, with slew-limited command
     references and integral action on forward velocity and yaw rate.
  4. u is decomposed into an MIT-mode command (q, dq, kp, kd, tau): the
     board layer carries the deploy stack's proven HW PD gains (within the
     DAMIAO packet ranges kp [0,500], kd [0,5]) and tau_ff carries the LQG
     delta, exact at the sensing instant:

    board:  tau_applied = kp (q_cmd - q) + kd (dq_cmd - dq) + tau_ff

Heading (yaw) invariance: tilt is extracted relative to the yaw-aligned
frame; gyro rates are body-frame; the KF model is heading-invariant.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import mujoco
import numpy as np

from lqr.model import (
    BOARD_KP,
    KD_DESIGN,
    KD_EMIT,
    KD_RANGE,
    KP_RANGE,
    PHYSICS_DT,
    STANCE_JOINT_POS,
    WHEEL_RADIUS,
    RobotModel,
)
from lqr.linearize import Linearization, lqr_gain_integral, tangent_state_labels


@dataclasses.dataclass
class MitCommand:
    """One MIT-mode command per joint, JOINT_NAMES order."""

    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray


def default_weights(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Initial Q, R for the reduced state; tuned in M4."""
    qdiag = {
        "z": 500.0,
        "roll": 300.0,
        "pitch": 1000.0,
        "hip_l": 150.0,
        "hip_r": 150.0,
        "thigh_l": 150.0,
        "thigh_r": 150.0,
        "calf_l": 150.0,
        "calf_r": 150.0,
        "dx": 150.0,
        "dy": 5.0,
        "dz": 20.0,
        "droll": 10.0,
        "dpitch": 30.0,
        # Moderate. High proportional yaw gain (1000) tracked well against
        # wheel-scrub friction but amplified gyro noise into instability
        # during turns; the steady-state error is closed by integral action
        # on dyaw instead (see lqr_gain_integral).
        "dyaw": 50.0,
        "dhip_l": 1.0,
        "dhip_r": 1.0,
        "dthigh_l": 1.0,
        "dthigh_r": 1.0,
        "dcalf_l": 1.0,
        "dcalf_r": 1.0,
        "dwheel_l": 0.5,
        "dwheel_r": 0.5,
    }
    Q = np.diag([qdiag[lab] for lab in labels])
    # Legs are stiffer/stronger than wheels; penalize wheel torque more since
    # its budget is only 11 Nm.
    rdiag = {"hip": 0.5, "thigh": 0.5, "calf": 0.5, "wheel": 2.0}
    R = np.diag([rdiag[n.split("_")[0]] for n in JOINT_ORDER])
    return Q, R


JOINT_ORDER = (
    "hip_l",
    "thigh_l",
    "calf_l",
    "wheel_l",
    "hip_r",
    "thigh_r",
    "calf_r",
    "wheel_r",
)
JOINT_NAMES_SHORT = JOINT_ORDER  # JOINT_NAMES order without the _joint suffix


def quat_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class LqrController:
    # Command slew limits: a raw step command is ramped at these rates so the
    # wheel torque budget (11 Nm tapering to 0 at 25 rad/s) isn't blown on the
    # transient. Real teleop should apply the same ramp.
    V_SLEW = 1.0  # m/s^2
    W_SLEW = 3.0  # rad/s^2

    # Integral action on forward velocity and yaw rate: closes the
    # steady-state error left by wheel friction / scrub / model mismatch
    # without high proportional gain. Anti-windup: hard clamp, plus reset
    # when tilted (falling — the integral is meaningless there).
    QI_V = 200.0
    QI_W = 200.0
    # Roll-error integrator: trims hip splay until the roll moment balances.
    # Centrifugal load in turns is a v*w product term invisible to the
    # linearization, and lateral COM offsets are model mismatch — both show
    # up as a persistent roll disturbance this integrator absorbs.
    QI_R = 300.0
    # Complementary-filter crossover for the forward-velocity estimate:
    # below it, wheel odometry (encoder-clean, drift-free) dominates; above
    # it, integrated IMU forward acceleration (exogenous, slip-immune).
    VEL_XOVER_HZ = 1.5
    INTEG_CLAMP = np.array([0.4, 1.0, 0.3])  # [m, rad, rad*s]
    INTEG_RESET_TILT = 0.3  # rad

    def __init__(self, rm: RobotModel, lin: Linearization, Q=None, R=None):
        self.rm = rm
        self.lin = lin
        self._v_smooth = 0.0
        self._w_smooth = 0.0
        labels_full = tangent_state_labels(rm)
        self.labels = [labels_full[i] for i in lin.keep]
        if Q is None or R is None:
            Q0, R0 = default_weights(self.labels)
            Q = Q if Q is not None else Q0
            R = R if R is not None else R0
        self.Q, self.R = Q, R
        self._idx = {lab: i for i, lab in enumerate(self.labels)}
        # Half track width from wheel body lateral offset at stance.
        data = mujoco.MjData(rm.model)
        data.qpos[:] = lin.qpos_eq
        mujoco.mj_forward(rm.model, data)
        wl = data.body("wheel_l").xpos
        wr = data.body("wheel_r").xpos
        self.half_track = float(abs(wl[1] - wr[1]) / 2.0)
        # Column indices in K for each joint's own position/velocity state.
        self._own_pos_col = np.full(8, -1)
        self._own_vel_col = np.full(8, -1)
        for j, name in enumerate(JOINT_ORDER):
            if name in self._idx:
                self._own_pos_col[j] = self._idx[name]
            if "d" + name in self._idx:
                self._own_vel_col[j] = self._idx["d" + name]
        self._dq_scratch = np.zeros(rm.nv)
        self._odo_Mvx, self._odo_M3 = self._build_odometry_map(data)

        # 23-state LQR with integral action on forward velocity, yaw rate,
        # and roll (the estimator below supplies unmeasured states). The
        # design is on the plain plant; the extra emitted wheel damping
        # (KD_EMIT vs KD_DESIGN, see lqr.model) rides on top, unmodeled.
        keep = lin.keep
        int_S = np.zeros((3, len(self.labels)))
        int_S[0, self._idx["dx"]] = 1.0
        int_S[1, self._idx["dyaw"]] = 1.0
        int_S[2, self._idx["roll"]] = 1.0
        self._int_S = int_S
        self.K, self.Ki = lqr_gain_integral(
            lin, Q, R, int_S,
            np.diag([self.QI_V, self.QI_W, self.QI_R]), PHYSICS_DT,
        )
        self._integ = np.zeros(3)
        Kb = np.zeros((8, len(self.labels)))
        for j in range(8):
            pc, vc = self._own_pos_col[j], self._own_vel_col[j]
            if pc >= 0:
                Kb[j, pc] = BOARD_KP[j]
            if vc >= 0:
                Kb[j, vc] = KD_EMIT[j]
        self._Kb = Kb
        # Complementary-filter forward velocity estimate.
        self._vx_hat = 0.0
        self._alpha_hz = self.VEL_XOVER_HZ
        # Delay compensation buffers (prediction with the board-closed
        # matrix and per-tick bias; proven near-neutral, kept for tests).
        self._A = lin.A[np.ix_(keep, keep)]
        self._B = lin.B[keep, :]
        self._A_board = self._A - self._B @ Kb
        self._bias_hist: deque[np.ndarray] = deque(maxlen=32)
        self.delay_comp_steps = 0  # physics steps of loop delay to predict over

    def tilt(self, qpos: np.ndarray) -> np.ndarray:
        """(roll, pitch) tangent relative to the yaw-aligned frame."""
        yaw = quat_yaw(qpos[3:7])
        q_yaw_inv = np.array([np.cos(-yaw / 2.0), 0.0, 0.0, np.sin(-yaw / 2.0)])
        q_rel = np.empty(4)
        mujoco.mju_mulQuat(q_rel, q_yaw_inv, qpos[3:7])
        tangent = np.empty(3)
        mujoco.mju_quat2Vel(tangent, q_rel, 1.0)
        return tangent[:2]

    def measure(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Measurement vector in estimator.measurement_matrix order:
        leg joint pos (6, rel. stance), leg joint vel (6), wheel vel (2),
        tilt (2), gyro (3). Only encoder/IMU quantities."""
        q_j = qpos[self.rm.joint_qpos_adr]
        v_j = qvel[self.rm.joint_dof_adr]
        legs = [0, 1, 2, 4, 5, 6]  # JOINT_NAMES order, wheels excluded
        wheels = [3, 7]
        return np.concatenate(
            [
                q_j[legs] - STANCE_JOINT_POS[legs],
                v_j[legs],
                v_j[wheels],
                self.tilt(qpos),
                qvel[3:6],  # body-frame angular rates (gyro)
            ]
        )

    def reduced_state(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Ground-truth reduced tangent state (debug/analysis only — uses
        base states the real robot cannot measure)."""
        rm = self.rm
        mujoco.mj_differentiatePos(
            rm.model, self._dq_scratch, 1.0, self.lin.qpos_eq, qpos
        )
        dx_full = np.concatenate([self._dq_scratch, qvel])
        x = dx_full[self.lin.keep].copy()
        yaw = quat_yaw(qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        x[self._idx["dx"]] = c * qvel[0] + s * qvel[1]
        x[self._idx["dy"]] = -s * qvel[0] + c * qvel[1]
        x[self._idx["roll"]], x[self._idx["pitch"]] = self.tilt(qpos)
        return x

    def estimated_state(
        self, qpos: np.ndarray, qvel: np.ndarray, accel_x: float, dt: float
    ) -> np.ndarray:
        """Reduced state from real sensors only: encoders, tilt, gyro, and
        forward accelerometer. Base velocity: complementary filter (odometry
        at DC, integrated accel above VEL_XOVER_HZ). Lateral/vertical
        velocity and height: kinematic no-slip maps (small LQR gains, so
        the static-map approximation is benign there)."""
        x = np.zeros(len(self.labels))
        q_j = qpos[self.rm.joint_qpos_adr]
        v_j = qvel[self.rm.joint_dof_adr]
        roll, pitch = self.tilt(qpos)
        x[self._idx["roll"]] = roll
        x[self._idx["pitch"]] = pitch
        for k, jn in enumerate(JOINT_NAMES_SHORT):
            if jn in self._idx:  # wheel angles are dropped states
                x[self._idx[jn]] = q_j[k] - STANCE_JOINT_POS[k]
            x[self._idx["d" + jn]] = v_j[k]
        x[self._idx["droll"]], x[self._idx["dpitch"]], x[self._idx["dyaw"]] = qvel[3:6]
        odo = self.odometry_vel(qvel)
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * self._alpha_hz))
        self._vx_hat = (1.0 - alpha) * (self._vx_hat + accel_x * dt) + alpha * odo[0]
        x[self._idx["dx"]] = self._vx_hat
        x[self._idx["dy"]] = odo[1]
        x[self._idx["dz"]] = odo[2]
        # Height from the position-level version of the vz odometry row
        # (same frozen contact Jacobians, tangent coordinates).
        pos_meas = np.concatenate([[roll, pitch, 0.0], q_j - STANCE_JOINT_POS])
        x[self._idx["z"]] = float(self._odo_M3[2] @ pos_meas)
        return x

    def reference(self, v_cmd: float, w_cmd: float) -> np.ndarray:
        """Reduced-state reference for a forward-velocity / yaw-rate command.

        No explicit centripetal lean term: hand-computed lean references
        made curves worse in testing — the roll-error integrator finds the
        correct hip-splay trim for the centrifugal load on its own.
        """
        x_ref = np.zeros(len(self.labels))
        x_ref[self._idx["dx"]] = v_cmd
        x_ref[self._idx["dyaw"]] = w_cmd
        w_avg = v_cmd / WHEEL_RADIUS
        w_diff = w_cmd * self.half_track / WHEEL_RADIUS
        # Positive yaw (CCW): right wheel faster. wheel axes are +y (left side
        # spins positive forward); verified in simulate.py smoke test.
        x_ref[self._idx["dwheel_l"]] = w_avg - w_diff
        x_ref[self._idx["dwheel_r"]] = w_avg + w_diff
        return x_ref

    def _build_odometry_map(self, data: mujoco.MjData) -> np.ndarray:
        """Constant map from measured rates to base forward velocity.

        No-slip rolling: each wheel's material point at the ground contact
        has zero world velocity. Writing that with translational Jacobians
        at the equilibrium and splitting qvel into base-linear (unknown),
        base-angular (gyro) and joints (encoders):

            J_lin v_lin = -[J_ang, J_joint] [w; qdot]

        gives v_lin by least squares from measured quantities only. The
        matrices are frozen at the equilibrium, so on hardware this is one
        constant (1 x 11) row vector. Returns M with
        vx = M @ [gyro(3), joint_vel(8 in JOINT_NAMES order)].
        """
        rm = self.rm
        model = rm.model
        rows_lin, rows_meas = [], []
        for wheel in ("wheel_l", "wheel_r"):
            body = model.body(wheel).id
            point = data.xpos[body].copy()
            point[2] -= WHEEL_RADIUS
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, None, point, body)
            # x/y world rows (tangential no-slip).
            rows_lin.append(jacp[:2, 0:3])
            rows_meas.append(
                np.hstack([jacp[:2, 3:6], jacp[:2, rm.joint_dof_adr]])
            )
        A_lin = np.vstack(rows_lin)  # (4, 3): rows alternate x,y per wheel
        A_meas = np.vstack(rows_meas)  # (4, 11)
        # Full least-squares solve (all four tangential rows) for (vx,vy,vz):
        # used for the low-gain dy/dz channels only.
        M3 = -np.linalg.pinv(A_lin) @ A_meas
        # vx from the LONGITUDINAL (rolling-direction, world-x at eq) rows
        # only: the lateral rows are violated by wheel scrub during yaw and
        # contaminate vx exactly when the dx gain can least afford it.
        # Per wheel: vx = -(A_meas_row @ meas) / A_lin_row[vx], averaged.
        long_rows = [0, 2]
        Mvx = np.zeros(11)
        for r in long_rows:
            Mvx += -A_meas[r] / A_lin[r, 0] / len(long_rows)
        return Mvx, M3

    def odometry_vel(self, qvel: np.ndarray) -> np.ndarray:
        """No-slip (vx, vy, vz) estimate from gyro + encoders. vx uses the
        scrub-immune longitudinal rows; vy/vz the full tangential solve."""
        meas = np.concatenate([qvel[3:6], qvel[self.rm.joint_dof_adr]])
        vx = float(self._odo_Mvx @ meas)
        v3 = self._odo_M3 @ meas
        return np.array([vx, v3[1], v3[2]])

    def reset(self) -> None:
        self._v_smooth = 0.0
        self._w_smooth = 0.0
        self._integ[:] = 0.0
        self._vx_hat = 0.0
        self._bias_hist.clear()

    # Command governor: the single-point linearization handles pure driving
    # (|v| to 1.5) and pure in-place turning (|w| to 2.0) robustly, but
    # turning WHILE translating is limited by centrifugal load and wheel
    # scrub outside the linear model. Verified safe envelope: |w| <= 0.6
    # while translating, tapering as 0.3/|v| at speed; full |w| only near
    # v = 0. Gain scheduling on (v, w) is the future-work path to widen it.
    GOV_V_INPLACE = 0.05  # below this |v|, full yaw rate allowed
    GOV_W_TRANSLATE = 0.6
    GOV_VW_MAX = 0.3

    def govern_command(self, v_cmd: float, w_cmd: float) -> tuple[float, float]:
        if abs(v_cmd) <= self.GOV_V_INPLACE:
            return v_cmd, w_cmd
        w_lim = min(self.GOV_W_TRANSLATE, self.GOV_VW_MAX / abs(v_cmd))
        return v_cmd, float(np.clip(w_cmd, -w_lim, w_lim))

    def slew_command(self, v_cmd: float, w_cmd: float, dt: float) -> tuple[float, float]:
        v_cmd, w_cmd = self.govern_command(v_cmd, w_cmd)
        dv = np.clip(v_cmd - self._v_smooth, -self.V_SLEW * dt, self.V_SLEW * dt)
        dw = np.clip(w_cmd - self._w_smooth, -self.W_SLEW * dt, self.W_SLEW * dt)
        self._v_smooth += dv
        self._w_smooth += dw
        return self._v_smooth, self._w_smooth

    def mit_command(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        v_cmd: float = 0.0,
        w_cmd: float = 0.0,
        dt: float = PHYSICS_DT,
        accel_x: float = 0.0,
    ) -> MitCommand:
        x = self.estimated_state(qpos, qvel, accel_x, dt)
        # Smith-predictor-style delay compensation: the measurement is
        # delay_comp_steps old by the time our command acts, so propagate
        # the estimate forward with the linear model and the torque deltas
        # issued during that window. Short-horizon open-loop prediction —
        # no covariance feedback, so the FD-contact stiffness that broke
        # the Kalman filter does not bite. On hardware, calibrate the loop
        # delay and set delay_comp_steps accordingly.
        k = self.delay_comp_steps
        if k > 0:
            d = max(1, int(round(dt / PHYSICS_DT)))  # physics steps per tick
            nticks = -(-k // d)
            hist = list(self._bias_hist)[-nticks:]
            if len(hist) < nticks:
                hist = [np.zeros(len(self.labels))] * (nticks - len(hist)) + hist
            seq = [b for b in hist for _ in range(d)][-k:]
            for b in seq:
                x = self._A_board @ x + b
        x_ref = self.reference(v_cmd, w_cmd)
        err = x - x_ref
        u = self.lin.u_eq - self.K @ err - self.Ki @ self._integ

        tilt = max(abs(x[self._idx["roll"]]), abs(x[self._idx["pitch"]]))
        if tilt > self.INTEG_RESET_TILT:
            self._integ[:] = 0.0
        else:
            self._integ += dt * (self._int_S @ err)
            np.clip(self._integ, -self.INTEG_CLAMP, self.INTEG_CLAMP, out=self._integ)

        q_meas = qpos[self.rm.joint_qpos_adr]
        v_meas = qvel[self.rm.joint_dof_adr]
        cmd = self.compose(u, x_ref, q_meas, v_meas)
        # Command-history bias for the delay predictor (state-independent
        # part of the board torque in deviation coordinates).
        self._bias_hist.append(
            self._B @ (cmd.kd * cmd.dq + cmd.tau - self.lin.u_eq)
        )
        return cmd

    def compose(
        self,
        u: np.ndarray,
        x_ref: np.ndarray,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
    ) -> MitCommand:
        """MIT command reproducing the designed torque u at the sensing
        instant, plus extra live wheel damping on top.

        tau_ff subtracts the DESIGN-accounted PD (BOARD_KP / KD_DESIGN) at
        the measured state, so the total equals u exactly at the tick. The
        EMITTED kd is higher on the wheels (KD_EMIT): the difference is
        never subtracted and acts as zero-delay damping at the 500 Hz board
        rate — this is what buys 20-30 ms loop-delay tolerance in the
        balance/velocity regimes (see lqr.model)."""
        kp = np.clip(BOARD_KP, *KP_RANGE)
        kd = np.clip(KD_EMIT, *KD_RANGE)
        q_cmd = STANCE_JOINT_POS.copy()
        dq_cmd = np.zeros(8)
        for j in range(8):
            vc = self._own_vel_col[j]
            if vc >= 0:
                dq_cmd[j] = x_ref[vc]
        tau = u - BOARD_KP * (q_cmd - q_meas) - KD_DESIGN * (dq_cmd - v_meas)
        return MitCommand(q=q_cmd, dq=dq_cmd, kp=kp, kd=kd, tau=tau)
