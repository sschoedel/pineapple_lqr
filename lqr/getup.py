"""Time-varying LQR for get-up-from-ground and go-to-ground.

    uv run python -m lqr.getup           # synthesize + smoke-verify

Pipeline:

1. **Knot tables** — per interpolation knot s in [0, 1] (sit -> stance):
   quasi-static balance pitch (COM over the wheel contact, bisected),
   equilibrium base height (sign change of the base z inverse-dynamics
   residual, the find_equilibrium recipe) and the exact static joint
   torques from mj_inverse at that height. Below the first balanced knot
   (deep fold — the robot rests statically on its rear supports, no
   wheel-only equilibrium exists) the feedforward blends linearly from
   the settled-sit inverse dynamics.

2. **Nominal rollout** — 500 Hz closed-loop sim: joint servo along the
   interpolation with the knot feedforward, plus a hand segway wheel loop
   (pitch / pitch-rate / vx / clamped vx-integral). This produced states
   captured by the stance balance LQR in exploration; the recorded
   (qpos, qvel, ctrl) trajectory is the TVLQR reference. The reverse
   rollout (stance -> sit) is generated the same way with s reversed and
   must end settled on the rear supports.

3. **TVLQR** — mjd_transitionFD along the recording (refreshed every
   DECIM physics steps), same 23-state reduction as the balance
   controller, backward Riccati with terminal cost = the stance DARE
   solution. The get-up gains therefore converge to the balance gains at
   the top, which is what makes the final handoff a no-op.

State/reference convention: x_ref knots are built with the SAME frozen
stance kinematic maps the runtime estimator uses (tilt, joints - stance,
frozen odometry rows), so estimator quirks cancel in (x_hat - x_ref)
instead of injecting phantom errors.
"""

from __future__ import annotations

import dataclasses
import sys

import mujoco
import numpy as np
import scipy.linalg

from lqr.controller import LqrController, default_weights, quat_yaw
from lqr.linearize import linearize, reduced_state_indices, tangent_state_labels
from lqr.model import (
    PHYSICS_DT,
    STANCE_JOINT_POS,
    RobotModel,
    torque_speed_clip,
)

SIT_ANGLES = np.array([0.093, 1.49, -3.14, 0.0, 0.093, 1.49, -3.14, 0.0])
WHEELS = [3, 7]
KP_SERVO = np.array([40.0, 25.0, 25.0, 0.0] * 2)
# board-rate damping (per physics step); wheels get 1.0 — the 100 Hz-held
# wheel-loop torque needs board-rate damping to bridge the ticks, the
# schedule analog of the balance controller's KD_EMIT trick
KD_BOARD = np.array([1.0, 0.5, 0.5, 1.0] * 2)
# DAMIAO MIT-packet tau field range per joint (hip J4340P 28, thigh J4340
# 28, knee J6248P 120, wheel J6006 20 Nm — motor_control damiao.py
# MODEL_LIMITS). tau_ff is clipped HERE, not at EFFORT_LIMIT: the
# feedforward legitimately exceeds the physical effort limit when it
# compensates the board damping term at speed (the board's own
# torque-speed curve enforces physics); clipping tau_ff at EFFORT_LIMIT
# silently corrupted the replayed schedule at wheel speeds beyond ~10 rad/s.
TAU_PACKET = np.array([28.0, 28.0, 120.0, 20.0] * 2)
# hand segway wheel loop for the NOMINAL rollout only (TVLQR replaces it).
# Tuned at the REAL 100 Hz update rate with board-rate damping bridging
# (swept in sim; the 500 Hz-tuned gains ran away backward at 100 Hz).
WHEEL_LOOP = dict(A=40.0, B=6.0, C=16.0, D=6.0)
T_RISE = 5.0     # s, sit -> stance ramp
T_HOLD = 0.0     # schedule ends at the rise end; balance mode takes over
T_DOWN = 4.0     # s, stance -> sit ramp (descent is easier)
T_SETTLE = 1.5   # s, settle onto the supports at the bottom
N_KNOTS = 41     # s-grid for the quasi-static tables
DECIM = 2        # physics steps per knot: 2 x 5 ms = one real 100 Hz tick


def _pitch_of(qpos: np.ndarray) -> float:
    qw, qx, qy, qz = qpos[3:7]
    return float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))


class _RefPoser:
    """Reference-pose machinery on a scratch MjData."""

    def __init__(self, rm: RobotModel):
        self.rm = rm
        self.data = mujoco.MjData(rm.model)
        m = rm.model
        self.gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wheel_l_collision")
        self.gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wheel_r_collision")
        self.wheel_r = float(m.geom_size[self.gl][0])

    def joint_path(self, s: float) -> np.ndarray:
        return s * STANCE_JOINT_POS + (1.0 - s) * SIT_ANGLES

    def set_pose(self, s: float, theta: float, z: float) -> None:
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.qpos[3:7] = [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0]
        d.qpos[self.rm.joint_qpos_adr] = self.joint_path(s)
        d.qpos[2] = z
        mujoco.mj_forward(self.rm.model, d)

    def wheel_touch_z(self, s: float, theta: float) -> float:
        self.set_pose(s, theta, 1.0)
        lo = min(self.data.geom_xpos[self.gl][2], self.data.geom_xpos[self.gr][2])
        return 1.0 - (lo - self.wheel_r)

    def com_err(self, s: float, theta: float) -> float:
        self.set_pose(s, theta, self.wheel_touch_z(s, theta))
        wx = 0.5 * (self.data.geom_xpos[self.gl][0] + self.data.geom_xpos[self.gr][0])
        return float(self.data.subtree_com[1][0] - wx)

    def balance_pitch(self, s: float) -> float | None:
        lo, hi = -0.2, 1.2
        elo, ehi = self.com_err(s, lo), self.com_err(s, hi)
        if elo * ehi > 0:
            return None
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if self.com_err(s, mid) * elo > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def knot_equilibrium(self, s: float, theta: float):
        """Bisect base z on the base-fz inverse-dynamics residual."""
        z0 = self.wheel_touch_z(s, theta)

        def fz(z: float) -> float:
            self.set_pose(s, theta, z)
            self.data.qacc[:] = 0.0
            mujoco.mj_inverse(self.rm.model, self.data)
            return float(self.data.qfrc_inverse[2])

        a, b = z0 - 0.004, z0 + 0.002
        if not (fz(a) < 0.0 < fz(b)):
            return None, None
        for _ in range(60):
            m = 0.5 * (a + b)
            if fz(m) < 0.0:
                a = m
            else:
                b = m
        z = 0.5 * (a + b)
        self.set_pose(s, theta, z)
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.rm.model, self.data)
        return z, self.data.qfrc_inverse[self.rm.joint_dof_adr].copy()


@dataclasses.dataclass
class KnotTables:
    s_grid: np.ndarray       # (N_KNOTS,)
    theta: np.ndarray        # (N_KNOTS,) balance pitch
    u_ff: np.ndarray         # (N_KNOTS, 8) static joint torques


def settle_sit(rm: RobotModel, data: mujoco.MjData) -> None:
    """Drop + servo into the resting sit pose (3 s, from 0.30 m)."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[rm.joint_qpos_adr] = SIT_ANGLES
    data.qpos[2] = 0.30
    for _ in range(int(3.0 / PHYSICS_DT)):
        q = data.qpos[rm.joint_qpos_adr]
        dq = data.qvel[rm.joint_dof_adr]
        tau = KP_SERVO * (SIT_ANGLES - q) - KD_BOARD * dq
        data.ctrl[:] = torque_speed_clip(tau, dq)
        mujoco.mj_step(rm.model, data)


def build_knot_tables(rm: RobotModel) -> KnotTables:
    poser = _RefPoser(rm)
    s_grid = np.linspace(0.0, 1.0, N_KNOTS)
    theta = np.zeros(N_KNOTS)
    u_ff = np.zeros((N_KNOTS, 8))
    ok = np.zeros(N_KNOTS, bool)
    for k, s in enumerate(s_grid):
        th = poser.balance_pitch(s)
        if th is None:
            continue
        z, ff = poser.knot_equilibrium(s, th)
        if z is None:
            continue
        theta[k], u_ff[k], ok[k] = th, ff, True
    k0 = int(np.argmax(ok))
    if not ok[k0]:
        raise RuntimeError("no balanced knot found on the s-grid")
    # deep-fold knots: rest on supports; ff blends from the settled sit
    data = mujoco.MjData(rm.model)
    settle_sit(rm, data)
    ref = poser.data
    ref.qpos[:] = data.qpos
    ref.qvel[:] = 0.0
    ref.qacc[:] = 0.0
    mujoco.mj_forward(rm.model, ref)
    mujoco.mj_inverse(rm.model, ref)
    ff_sit = ref.qfrc_inverse[rm.joint_dof_adr].copy()
    for k in range(k0):
        a = s_grid[k] / s_grid[k0]
        u_ff[k] = (1.0 - a) * ff_sit + a * u_ff[k0]
        theta[k] = theta[k0]
    assert np.isfinite(u_ff).all()
    return KnotTables(s_grid=s_grid, theta=theta, u_ff=u_ff)


@dataclasses.dataclass
class Trajectory:
    qpos: np.ndarray    # (N, nq) recorded each physics step
    qvel: np.ndarray    # (N, nv)
    ctrl: np.ndarray    # (N, 8) applied joint torques (JOINT_NAMES order)
    s: np.ndarray       # (N,) interpolation phase
    dt: float


def rollout_nominal(rm: RobotModel, kt: KnotTables, direction: str) -> Trajectory:
    """500 Hz nominal: joint servo + knot ff + hand segway wheel loop."""
    assert direction in ("up", "down")
    data = mujoco.MjData(rm.model)
    if direction == "up":
        T, tail = T_RISE, T_HOLD
        settle_sit(rm, data)
    else:
        T, tail = T_DOWN, T_SETTLE
        # start at the stance equilibrium, at rest
        lin = linearize(rm)
        data.qpos[:] = lin.qpos_eq
        data.qvel[:] = 0.0
        mujoco.mj_forward(rm.model, data)
    n = int((T + tail) / PHYSICS_DT)
    out = Trajectory(
        qpos=np.zeros((n, rm.model.nq)),
        qvel=np.zeros((n, rm.model.nv)),
        ctrl=np.zeros((n, 8)),
        s=np.zeros(n),
        dt=PHYSICS_DT,
    )
    ivx = 0.0
    q_ref = SIT_ANGLES if direction == "up" else STANCE_JOINT_POS
    ff = np.zeros(8)
    seg_held = 0.0
    s = 0.0 if direction == "up" else 1.0
    for i in range(n):
        # The FEEDFORWARD parts (q_ref, ff, wheel-loop torque) update at
        # the real 100 Hz loop rate and are held between ticks; the servo
        # PD terms are evaluated EVERY physics step — that is what the
        # motor board does with a held MIT command on hardware.
        if i % DECIM == 0:
            t = i * PHYSICS_DT
            a = 0.5 - 0.5 * np.cos(np.pi * min(t, T) / T)
            s = a if direction == "up" else 1.0 - a
            q_ref = s * STANCE_JOINT_POS + (1.0 - s) * SIT_ANGLES
            ff = np.array(
                [np.interp(s, kt.s_grid, kt.u_ff[:, j]) for j in range(8)]
            )
            pitch = _pitch_of(data.qpos)
            vx = data.qvel[0]
            ivx = float(np.clip(ivx + vx * DECIM * PHYSICS_DT, -0.5, 0.5))
            wl = WHEEL_LOOP
            seg = (wl["A"] * pitch + wl["B"] * data.qvel[4]
                   + wl["C"] * vx + wl["D"] * ivx)
            # near the ground the robot rests statically on its supports:
            # the segway loop has nothing to balance and only scoots the
            # robot — fade it out below s=0.25
            seg_held = float(np.clip(s / 0.25, 0.0, 1.0)) * seg
        q = data.qpos[rm.joint_qpos_adr]
        dq = data.qvel[rm.joint_dof_adr]
        tau = KP_SERVO * (q_ref - q) - KD_BOARD * dq + ff
        tau[WHEELS] = seg_held - KD_BOARD[WHEELS] * dq[WHEELS]
        tau_c = torque_speed_clip(tau, dq)
        out.qpos[i] = data.qpos
        out.qvel[i] = data.qvel
        out.ctrl[i] = tau_c
        out.s[i] = s
        data.ctrl[:] = tau_c
        mujoco.mj_step(rm.model, data)
    return out


def check_rollout(rm: RobotModel, traj: Trajectory, direction: str) -> dict:
    """End-state metrics; raises if the nominal is not usable."""
    qpos, qvel = traj.qpos[-1], traj.qvel[-1]
    pitch = _pitch_of(qpos)
    z = qpos[2]
    speed = float(np.hypot(qvel[0], qvel[1]))
    m = dict(pitch=round(pitch, 3), z=round(float(z), 3), speed=round(speed, 3))
    if direction == "up":
        # loose gate — the REAL acceptance is the balance-capture test in
        # the suite (sim-verified: balance absorbs these end states)
        ok = abs(pitch) < 0.2 and z > 0.34 and speed < 0.6
    else:
        ok = abs(pitch) < 0.25 and z < 0.06 and speed < 0.25
    if not ok:
        raise RuntimeError(f"nominal rollout ({direction}) unusable: {m}")
    return m


def _board_matrix(labels: list[str]) -> np.ndarray:
    """Kb (8 x n): the board PD as state feedback — kp on the joint's own
    position error, kd on its own velocity (dq_cmd = 0). Closed into the
    per-step linearization so the Riccati designs around the real plant."""
    from lqr.controller import JOINT_ORDER
    idx = {lab: i for i, lab in enumerate(labels)}
    Kb = np.zeros((8, len(labels)))
    for j, name in enumerate(JOINT_ORDER):
        if name in idx:
            Kb[j, idx[name]] = KP_SERVO[j]
        if "d" + name in idx:
            Kb[j, idx["d" + name]] = KD_BOARD[j]
    return Kb


def _riccati_gains(rm: RobotModel, traj: Trajectory, P_term: np.ndarray,
                   Q: np.ndarray, R: np.ndarray, keep: np.ndarray,
                   Kb: np.ndarray):
    """Backward Riccati along the trajectory; K exported every DECIM steps."""
    model = rm.model
    data = mujoco.MjData(model)
    nv = rm.nv
    n = traj.qpos.shape[0]
    n_knots = (n + DECIM - 1) // DECIM
    nk = len(keep)
    K_out = np.zeros((n_knots, 8, nk))
    P = P_term.copy()
    A_full = np.zeros((2 * nv, 2 * nv))
    B_full = np.zeros((2 * nv, model.nu))
    # Per-PHYSICS-STEP Riccati recursion — the same design timescale the
    # stance balance controller uses (5 ms design, slower deployment,
    # board PD absorbing the rate gap; validated by the suite's
    # decimation tests). Composing 10 ms windows from FD products
    # amplified contact-FD noise into the stiff joint directions and the
    # resulting gains chattered the joints at playback.
    for i in range(n - 1, -1, -1):
        data.qpos[:] = traj.qpos[i]
        data.qvel[:] = traj.qvel[i]
        data.ctrl[rm.actuator_ids] = traj.ctrl[i]
        # eps large enough to average over contact nonsmoothness
        mujoco.mjd_transitionFD(model, data, 1e-4, 1, A_full, B_full, None, None)
        B_ord = B_full[:, rm.actuator_ids]
        A_j = A_full[np.ix_(keep, keep)] - B_ord[keep, :] @ Kb
        B_j = B_ord[keep, :]
        BtP = B_j.T @ P
        K = np.linalg.solve(R + BtP @ B_j, BtP @ A_j)
        P = Q + A_j.T @ P @ (A_j - B_j @ K)
        P = 0.5 * (P + P.T)
        if i % DECIM == 0:
            K_out[i // DECIM] = K
    return K_out


def _ref_state(ctrl_est: LqrController, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """x_ref with the runtime estimator's own conventions (frozen maps)."""
    x = np.zeros(len(ctrl_est.labels))
    idx = ctrl_est._idx
    rm = ctrl_est.rm
    q_j = qpos[rm.joint_qpos_adr]
    v_j = qvel[rm.joint_dof_adr]
    roll, pitch = ctrl_est.tilt(qpos)
    x[idx["roll"]] = roll
    x[idx["pitch"]] = pitch
    from lqr.controller import JOINT_NAMES_SHORT
    for k, jn in enumerate(JOINT_NAMES_SHORT):
        if jn in idx:
            x[idx[jn]] = q_j[k] - STANCE_JOINT_POS[k]
        x[idx["d" + jn]] = v_j[k]
    x[idx["droll"]], x[idx["dpitch"]], x[idx["dyaw"]] = qvel[3:6]
    yaw = quat_yaw(qpos[3:7])
    x[idx["dx"]] = np.cos(yaw) * qvel[0] + np.sin(yaw) * qvel[1]
    odo = ctrl_est.odometry_vel(qvel)
    x[idx["dy"]] = odo[1]
    x[idx["dz"]] = odo[2]
    pos_meas = np.concatenate([[roll, pitch, 0.0], q_j - STANCE_JOINT_POS])
    x[idx["z"]] = float(ctrl_est._odo_M3[2] @ pos_meas)
    return x


def synthesize(rm: RobotModel | None = None) -> dict:
    """Full pipeline; returns the table dict (getup_* / getdown_* keys)."""
    rm = rm or __import__("lqr.model", fromlist=["build_model"]).build_model()
    kt = build_knot_tables(rm)
    lin = linearize(rm)
    ctrl_est = LqrController(rm, lin)
    keep = lin.keep
    labels = [tangent_state_labels(rm)[i] for i in keep]
    Q_st, R = default_weights(labels)
    # terminal cost = stance DARE on the DECIMATED (100 Hz) system, the
    # same timescale as the knot recursion — the terminal gains are then
    # the 100 Hz equivalent of the balance gains
    Kb = _board_matrix(labels)
    # terminal cost: stance DARE on the same 5 ms board-closed system
    A_st = lin.A[np.ix_(keep, keep)] - lin.B[keep, :] @ Kb
    B_st = lin.B[keep, :]
    P_term = scipy.linalg.solve_discrete_are(A_st, B_st, Q_st, R)
    Q = Q_st

    out: dict = dict(getup_dt=DECIM * PHYSICS_DT)
    for direction, prefix in (("up", "getup"), ("down", "getdown")):
        traj = rollout_nominal(rm, kt, direction)
        metrics = check_rollout(rm, traj, direction)
        print(f"{prefix}: nominal ok {metrics}")
        K = _riccati_gains(rm, traj, P_term, Q, R, keep, Kb)
        n_knots = K.shape[0]
        sel = np.arange(n_knots) * DECIM
        # Fade the feedback out in the rest-on-supports phase (s < 0.25):
        # the robot is statically stable there and needs none, while FD
        # linearization through the sticking support contacts is noise the
        # Riccati amplifies into absurd gains (peaks of 3000-9000 vs the
        # ~150-290 of the wheel-balancing phase). Mirrors the feedforward
        # blend and the nominal's wheel-loop fade.
        s_knots = traj.s[sel]
        K = K * np.clip(s_knots / 0.25, 0.0, 1.0)[:, None, None]
        # clamp residual contact-FD spikes (support liftoff zone) to the
        # healthy wheel-phase magnitude range
        K = np.clip(K, -350.0, 350.0)
        prof = np.abs(K).max(axis=(1, 2))
        print(f"{prefix}: |K| profile (every 20 knots):",
              np.round(prof[::20], 0).astype(int).tolist())
        x_ref = np.array([_ref_state(ctrl_est, traj.qpos[i], traj.qvel[i]) for i in sel])
        q_ref = traj.qpos[sel][:, rm.joint_qpos_adr]
        dq_ref = traj.qvel[sel][:, rm.joint_dof_adr]
        u_ref = traj.ctrl[sel]
        out[f"{prefix}_K"] = K
        out[f"{prefix}_x_ref"] = x_ref
        out[f"{prefix}_q_ref"] = q_ref
        out[f"{prefix}_dq_ref"] = dq_ref
        out[f"{prefix}_u_ref"] = u_ref
        print(f"{prefix}: {n_knots} knots, K range |max| = {np.abs(K).max():.1f}")
    return out


class GetupController:
    """Design-side schedule player (mirrors the future runtime exactly).

    u(t) = u_ref_k - K_k (x_hat - x_ref_k); MIT command composition
    matches the balance controller: board kp/kd on the reference,
    tau_ff compensates the DESIGN gains at the sensing instant.
    """

    def __init__(self, rm: RobotModel, tables: dict, direction: str,
                 ctrl_est: LqrController):
        p = "getup" if direction == "up" else "getdown"
        self.K = tables[f"{p}_K"]
        self.x_ref = tables[f"{p}_x_ref"]
        self.q_ref = tables[f"{p}_q_ref"]
        self.dq_ref = tables[f"{p}_dq_ref"]
        self.u_ref = tables[f"{p}_u_ref"]
        self.dt_knot = float(tables["getup_dt"])
        self.est = ctrl_est
        self.t = 0.0

    @property
    def done(self) -> bool:
        return self.t >= (self.K.shape[0] - 1) * self.dt_knot - 1e-9

    def mit_command(self, qpos, qvel, dt, accel_x=0.0):
        # round-to-nearest: accumulated float time sits a few ulp BELOW
        # the knot boundary (6*0.025 = 0.014999...), and truncation would
        # lag the whole schedule by one knot — a systematic feedforward
        # delay that measurably destabilizes playback
        k = min(int(round(self.t / self.dt_knot)), self.K.shape[0] - 1)
        self.t += dt
        x = self.est.estimated_state(qpos, qvel, accel_x, dt)
        u = self.u_ref[k] - self.K[k] @ (x - self.x_ref[k])
        q_cmd = self.q_ref[k]
        # Board gains are EXACTLY the ones the Riccati modeled (closed
        # into A via Kb): kp toward the knot reference, kd toward zero.
        # tau_ff compensates them at the sensing instant, so the applied
        # torque equals u there; between 100 Hz ticks the board PD acts
        # on the deviation — modeled, not parasitic.
        dq_cmd = np.zeros(8)
        tau = u - KP_SERVO * (q_cmd - qpos[self.est.rm.joint_qpos_adr]) \
                + KD_BOARD * qvel[self.est.rm.joint_dof_adr]
        from lqr.controller import MitCommand
        return MitCommand(
            q=q_cmd.copy(), dq=dq_cmd,
            kp=KP_SERVO.copy(), kd=KD_BOARD.copy(),
            tau=np.clip(tau, -TAU_PACKET, TAU_PACKET),
        )


if __name__ == "__main__":
    from lqr.model import build_model

    rm = build_model()
    tables = synthesize(rm)
    path = sys.argv[1] if len(sys.argv) > 1 else "getup_tables.npz"
    np.savez(path, **tables)
    print(f"wrote {path}")
