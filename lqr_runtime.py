"""Numpy-only LQR runtime for the Pi (no mujoco/scipy required).

A table-driven mirror of pineapple_lqr's LqrController + mode logic, fed by
lqr_tables.npz (produced by `uv run python -m lqr.export_runtime` in the
pineapple_lqr repo). Verified bit-equivalent to the full controller in
pineapple_lqr/tests/test_runtime_export.py before every table export.

This file is duplicated into motor_control/ (laptop ground truth, synced to
the Pi by sync_to_robot.sh). Keep it dependency-free: stdlib + numpy.

Joint order everywhere: L hip, L thigh(hip_fe), L calf(knee), L wheel(ankle),
R hip, R thigh, R calf, R wheel — matching robot_config ESC id order
[0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F].
"""

from __future__ import annotations

import dataclasses

import numpy as np

GRAVITY = 9.81
LEG_IDX = [0, 1, 2, 4, 5, 6]
WHEEL_IDX = [3, 7]
JOINT_NAMES = (
    "l_hip", "l_thigh", "l_calf", "l_wheel",
    "r_hip", "r_thigh", "r_calf", "r_wheel",
)


@dataclasses.dataclass
class MitCommand:
    """Per-joint MIT command, sim frame, joint order above."""

    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray


@dataclasses.dataclass
class Snapshot:
    """One tick of sensing, already converted to sim frame."""

    q: np.ndarray  # (8,) rad
    dq: np.ndarray  # (8,) rad/s
    quat: np.ndarray  # (4,) w,x,y,z body orientation
    gyro: np.ndarray  # (3,) rad/s body rates
    accel_x: float  # m/s^2 gravity-compensated forward accel (heading frame)


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, vec)
    return vec + w * t + np.cross(qv, t)


def quat_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_from_quat(quat: np.ndarray) -> tuple[float, float]:
    """(roll, pitch) tangent relative to the yaw-aligned frame — matches
    mujoco mju_quat2Vel(Rz(-yaw) * q) used in the design controller."""
    yaw = quat_yaw(quat)
    hy = -yaw / 2.0
    qy = np.array([np.cos(hy), 0.0, 0.0, np.sin(hy)])
    # quaternion product qy * quat
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = quat
    q_rel = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    # rotation-vector (axis * angle) of q_rel, roll/pitch components
    w = np.clip(q_rel[0], -1.0, 1.0)
    v = q_rel[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 0.0, 0.0
    angle = 2.0 * np.arctan2(n, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = v / n
    return float(axis[0] * angle), float(axis[1] * angle)


def forward_accel(quat: np.ndarray, accel_body: np.ndarray) -> float:
    """Gravity-compensated heading-frame forward acceleration from a raw
    body-frame accelerometer measurement f = R^T (a - g_vec)."""
    a_world = quat_rotate(quat, accel_body)
    a_world = a_world.copy()
    a_world[2] -= GRAVITY
    yaw = quat_yaw(quat)
    return float(np.cos(yaw) * a_world[0] + np.sin(yaw) * a_world[1])


class Calibration:
    """q_sim = sign*(q_motor - offset); command back = sign*q_sim + offset."""

    def __init__(self, signs=None, offsets=None):
        self.signs = np.ones(8) if signs is None else np.asarray(signs, float)
        self.offsets = np.zeros(8) if offsets is None else np.asarray(offsets, float)

    @staticmethod
    def load(path: str) -> "Calibration":
        import yaml

        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raw = {}
        return Calibration(raw.get("signs"), raw.get("offsets"))

    def to_sim(self, q_m, dq_m, tau_m):
        return (
            self.signs * (q_m - self.offsets),
            self.signs * dq_m,
            self.signs * tau_m,
        )

    def cmd_to_motor(self, cmd: MitCommand) -> MitCommand:
        return MitCommand(
            q=self.signs * cmd.q + self.offsets,
            dq=self.signs * cmd.dq,
            kp=cmd.kp.copy(),
            kd=cmd.kd.copy(),
            tau=self.signs * cmd.tau,
        )


class TableController:
    """Mirror of pineapple_lqr's LqrController, driven by lqr_tables.npz."""

    def __init__(self, tables_path: str = "lqr_tables.npz"):
        t = np.load(tables_path, allow_pickle=False)
        self.t = t
        self.labels = [str(s) for s in t["labels"]]
        self._idx = {lab: i for i, lab in enumerate(self.labels)}
        self.K = t["K"]
        self.Ki = t["Ki"]
        self.int_S = t["int_S"]
        self.u_eq = t["u_eq"]
        self.stance = t["stance"]
        self.dt_design = float(t["dt"])
        self._own_vel_col = t["own_vel_col"].astype(int)
        # Live tuning (GUI sliders; persisted in tuning.yaml). Defaults
        # reproduce the sim-verified controller exactly.
        #   wheel_kd   — emitted board damping on the wheels [Nm/(rad/s)]
        #   wheel_gain — scale on the wheel rows of K and Ki (chatter knob)
        #   vx_gain    — scale on the forward-velocity feedback column
        #   vi_gain    — scale on the velocity-integrator authority
        self.tune = {
            "wheel_kd": float(t["kd_emit"][3]),
            "wheel_gain": 1.0,
            "vx_gain": 1.0,
            "vi_gain": 1.0,
            # emitted hip kp above the design value: pure added stance
            # stiffness at the board rate (fights hip stiction creep /
            # track-width wander). 0 = sim-verified default.
            "hip_kp_extra": 0.0,
        }
        self._K_eff = self.K.copy()
        self._Ki_eff = self.Ki.copy()
        self._kd_emit_eff = t["kd_emit"].astype(float).copy()
        self._apply_tuning()
        self._v_smooth = 0.0
        self._w_smooth = 0.0
        self._integ = np.zeros(self.Ki.shape[1])
        self._vx_hat = 0.0

    def set_tuning(self, **kw):
        for k, v in kw.items():
            assert k in self.tune, k
            self.tune[k] = float(v)
        self._apply_tuning()

    def _apply_tuning(self):
        tn = self.tune
        K = self.K.copy()
        Ki = self.Ki.copy()
        K[[3, 7], :] *= tn["wheel_gain"]
        Ki[[3, 7], :] *= tn["wheel_gain"]
        dx = self._idx["dx"]
        K[:, dx] *= tn["vx_gain"]
        Ki[:, 0] *= tn["vi_gain"]
        self._K_eff = K
        self._Ki_eff = Ki
        kd = self.t["kd_emit"].astype(float).copy()
        kd[3] = kd[7] = tn["wheel_kd"]
        self._kd_emit_eff = kd
        kp = self.t["board_kp"].astype(float).copy()
        kp[0] += tn["hip_kp_extra"]
        kp[4] += tn["hip_kp_extra"]
        self._kp_emit_eff = kp

    def reset(self):
        self._v_smooth = 0.0
        self._w_smooth = 0.0
        self._integ[:] = 0.0
        self._vx_hat = 0.0

    # -- command shaping -----------------------------------------------------

    def govern(self, v, w):
        t = self.t
        if abs(v) <= float(t["gov_v_inplace"]):
            return v, w
        w_lim = min(float(t["gov_w_translate"]), float(t["gov_vw_max"]) / abs(v))
        return v, float(np.clip(w, -w_lim, w_lim))

    def slew(self, v_cmd, w_cmd, dt):
        v_cmd, w_cmd = self.govern(v_cmd, w_cmd)
        vs, ws = float(self.t["v_slew"]), float(self.t["w_slew"])
        self._v_smooth += float(np.clip(v_cmd - self._v_smooth, -vs * dt, vs * dt))
        self._w_smooth += float(np.clip(w_cmd - self._w_smooth, -ws * dt, ws * dt))
        return self._v_smooth, self._w_smooth

    # -- estimator -----------------------------------------------------------

    def estimated_state(self, snap: Snapshot, dt: float) -> np.ndarray:
        t = self.t
        x = np.zeros(len(self.labels))
        roll, pitch = tilt_from_quat(snap.quat)
        x[self._idx["roll"]] = roll
        x[self._idx["pitch"]] = pitch
        for k, jn in enumerate(
            ("hip_l", "thigh_l", "calf_l", "wheel_l", "hip_r", "thigh_r", "calf_r", "wheel_r")
        ):
            if jn in self._idx:
                x[self._idx[jn]] = snap.q[k] - self.stance[k]
            x[self._idx["d" + jn]] = snap.dq[k]
        x[self._idx["droll"]], x[self._idx["dpitch"]], x[self._idx["dyaw"]] = snap.gyro
        meas = np.concatenate([snap.gyro, snap.dq])
        vx_odo = float(t["odo_Mvx"] @ meas)
        v3 = t["odo_M3"] @ meas
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * float(t["vel_xover_hz"])))
        self._vx_hat = (1.0 - alpha) * (self._vx_hat + snap.accel_x * dt) + alpha * vx_odo
        x[self._idx["dx"]] = self._vx_hat
        x[self._idx["dy"]] = v3[1]
        x[self._idx["dz"]] = v3[2]
        pos_meas = np.concatenate([[roll, pitch, 0.0], snap.q - self.stance])
        x[self._idx["z"]] = float(t["odo_M3"][2] @ pos_meas)
        self.last_x = x  # telemetry (GUI) — updated on every estimate
        return x

    # -- control tick ----------------------------------------------------------

    def reference(self, v_cmd, w_cmd):
        t = self.t
        x_ref = np.zeros(len(self.labels))
        x_ref[self._idx["dx"]] = v_cmd
        x_ref[self._idx["dyaw"]] = w_cmd
        r = float(t["wheel_radius"])
        w_avg = v_cmd / r
        w_diff = w_cmd * float(t["half_track"]) / r
        x_ref[self._idx["dwheel_l"]] = w_avg - w_diff
        x_ref[self._idx["dwheel_r"]] = w_avg + w_diff
        return x_ref

    def mit_command(self, snap: Snapshot, v_cmd=0.0, w_cmd=0.0, dt=None) -> MitCommand:
        t = self.t
        dt = self.dt_design if dt is None else dt
        x = self.estimated_state(snap, dt)
        self.last_x = x
        x_ref = self.reference(v_cmd, w_cmd)
        err = x - x_ref
        u = self.u_eq - self._K_eff @ err - self._Ki_eff @ self._integ

        tilt = max(abs(x[self._idx["roll"]]), abs(x[self._idx["pitch"]]))
        if tilt > float(t["integ_reset_tilt"]):
            self._integ[:] = 0.0
        else:
            self._integ += dt * (self.int_S @ err)
            clamp = t["integ_clamp"]
            np.clip(self._integ, -clamp, clamp, out=self._integ)

        kp = self._kp_emit_eff
        kd = self._kd_emit_eff
        q_cmd = self.stance.copy()
        dq_cmd = np.zeros(8)
        for j in range(8):
            vc = self._own_vel_col[j]
            if vc >= 0:
                dq_cmd[j] = x_ref[vc]
        tau = u - t["board_kp"] * (q_cmd - snap.q) - t["kd_design"] * (dq_cmd - snap.dq)
        # last-resort sanity clip at the motor effort limits
        lim = t["effort_limit"]
        tau = np.clip(tau, -lim, lim)
        return MitCommand(q=q_cmd, dq=dq_cmd, kp=kp, kd=kd, tau=tau)


class RobotRuntime:
    """Mode logic + safety trips around TableController (mirror of
    pineapple_lqr's deploy_lqr.LqrRuntime, without mujoco/DDS)."""

    MODES = ("damp", "stand", "balance", "sit")
    SIT_ANGLES = np.array([0.093, 1.49, -3.14, 0.0, 0.093, 1.49, -3.14, 0.0])
    STAND_SECONDS = 3.0
    STAND_KP = 40.0
    STAND_KD = 1.0
    DAMP_KD = 1.0
    TRIP_TILT = 0.5
    DEADMAN_TIMEOUT = 0.5

    def __init__(self, ctrl: TableController, dt: float):
        self.ctrl = ctrl
        self.dt = dt
        self.mode = "damp"
        self._mode_t = 0.0
        self._mode_start_q = np.zeros(8)
        self._last_cmd = self.damp_cmd()
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.tripped = False
        self.trip_reason: str | None = None
        self._deadman_armed = False
        self._last_heartbeat = 0.0

    def set_mode(self, mode: str, snap: Snapshot):
        assert mode in self.MODES
        self.mode = mode
        self._mode_t = 0.0
        self._mode_start_q = snap.q.copy()
        if mode == "balance":
            self.ctrl.reset()
        self.tripped = False
        self.trip_reason = None
        if mode != "balance":
            self.v_cmd = 0.0
            self.w_cmd = 0.0

    def set_command(self, v, w, v_max=1.0, w_max=2.0):
        self.v_cmd = float(np.clip(v, -v_max, v_max))
        self.w_cmd = float(np.clip(w, -w_max, w_max))

    def note_heartbeat(self, t_mono: float):
        self._last_heartbeat = t_mono
        self._deadman_armed = True

    def check_deadman(self, t_mono: float) -> bool:
        if not self._deadman_armed:
            return False
        if t_mono - self._last_heartbeat <= self.DEADMAN_TIMEOUT:
            return False
        self.trip("OPERATOR LINK LOST")
        return True

    def trip(self, reason: str):
        if self.mode != "damp":
            self.mode = "damp"
            self.tripped = True
            self.trip_reason = reason
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self._last_cmd = self.damp_cmd()

    def damp_cmd(self) -> MitCommand:
        z = np.zeros(8)
        return MitCommand(
            q=z.copy(), dq=z.copy(), kp=z.copy(),
            kd=np.full(8, self.DAMP_KD), tau=z.copy(),
        )

    def _ramp_cmd(self, target: np.ndarray) -> MitCommand:
        phase = float(np.tanh(self._mode_t / (self.STAND_SECONDS / 2.5)))
        q_cmd = phase * target + (1.0 - phase) * self._mode_start_q
        kp = np.full(8, self.STAND_KP)
        kd = np.full(8, self.STAND_KD)
        kp[WHEEL_IDX] = 0.0
        kd[WHEEL_IDX] = 0.3
        return MitCommand(q=q_cmd, dq=np.zeros(8), kp=kp, kd=kd, tau=np.zeros(8))

    def tick(self, snap: Snapshot) -> MitCommand:
        self._mode_t += self.dt
        if self.mode == "damp":
            self._last_cmd = self.damp_cmd()
        elif self.mode == "stand":
            self._last_cmd = self._ramp_cmd(self.ctrl.stance)
        elif self.mode == "sit":
            self._last_cmd = self._ramp_cmd(self.SIT_ANGLES)
        elif self.mode == "balance":
            roll, pitch = tilt_from_quat(snap.quat)
            if max(abs(roll), abs(pitch)) > self.TRIP_TILT:
                self.trip("TILT LIMIT EXCEEDED")
            else:
                v, w = self.ctrl.slew(self.v_cmd, self.w_cmd, self.dt)
                self._last_cmd = self.ctrl.mit_command(snap, v, w, dt=self.dt)
        return self._last_cmd

    def telemetry(self) -> dict:
        out = {"mode": self.mode, "v_cmd": self.v_cmd, "w_cmd": self.w_cmd,
               "tripped": self.tripped, "reason": self.trip_reason}
        x = getattr(self.ctrl, "last_x", None)
        if x is not None:
            idx = self.ctrl._idx
            out.update(
                roll=float(x[idx["roll"]]), pitch=float(x[idx["pitch"]]),
                vx=float(x[idx["dx"]]), dyaw=float(x[idx["dyaw"]]),
                z=float(x[idx["z"]]),
                wheel_l=float(x[idx["dwheel_l"]]),
                wheel_r=float(x[idx["dwheel_r"]]),
                integ=[float(v) for v in self.ctrl._integ],
            )
        return out
