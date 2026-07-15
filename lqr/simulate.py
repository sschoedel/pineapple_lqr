"""Closed-loop simulation harness with hardware-realistic imperfections.

Structure mirrors the real system:
  physics @ 500 Hz  →  motor board applies the latest MIT command each step
  controller @ 500/decimation Hz  →  reads (possibly delayed + noisy) sensors,
                                      emits a new MIT command (possibly delayed)

Imperfections (all off by default, enabled by the robustness suite):
  * sensor noise: gaussian on joint pos/vel, gyro, and orientation tangent
  * sensor latency: controller sees measurements N physics steps old
  * actuation latency: a new MIT command takes N physics steps to reach the
    board (matches mjlab's 0-3 step training delay)
  * model mismatch is done by perturbing the *plant* model (see perturb.py);
    the controller keeps its nominal-model gains
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Callable

import mujoco
import numpy as np

from lqr.controller import LqrController, MitCommand
from lqr.model import RobotModel, reset_to_stance, torque_speed_clip


@dataclasses.dataclass
class SimConfig:
    duration: float = 10.0
    decimation: int = 1  # controller period in physics steps
    sensor_delay_steps: int = 0
    action_delay_steps: int = 0
    noise_joint_pos: float = 0.0  # rad
    noise_joint_vel: float = 0.0  # rad/s
    noise_gyro: float = 0.0  # rad/s
    noise_orientation: float = 0.0  # rad, on the roll/pitch tangent
    noise_base_vel: float = 0.0  # m/s, on estimated base linear velocity
    noise_accel: float = 0.0  # m/s^2, on the forward accelerometer
    # Loop-delay steps the controller compensates for. None = the true
    # configured delay (sensor + action); set explicitly to test robustness
    # to delay mis-calibration.
    comp_delay_steps: int | None = None
    seed: int = 0
    # command schedule: t -> (v_cmd, w_cmd)
    command_fn: Callable[[float], tuple[float, float]] | None = None
    # push schedule: t -> world-frame force (3,) on the base, or None
    push_fn: Callable[[float], np.ndarray | None] | None = None


@dataclasses.dataclass
class SimResult:
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tau_applied: np.ndarray
    fell: bool
    fall_time: float | None

    @property
    def base_height(self) -> np.ndarray:
        return self.qpos[:, 2]

    def pitch(self) -> np.ndarray:
        w, x, y, z = (self.qpos[:, 3 + i] for i in range(4))
        return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

    def roll(self) -> np.ndarray:
        w, x, y, z = (self.qpos[:, 3 + i] for i in range(4))
        return np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

    def yaw_rate(self) -> np.ndarray:
        return self.qvel[:, 5]

    def forward_vel(self) -> np.ndarray:
        """Base velocity along the heading."""
        w, x, y, z = (self.qpos[:, 3 + i] for i in range(4))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.cos(yaw) * self.qvel[:, 0] + np.sin(yaw) * self.qvel[:, 1]


FALL_HEIGHT = 0.25  # m; nominal is 0.381
FALL_TILT = 0.7  # rad on roll or pitch


def run(
    rm: RobotModel,
    ctrl: LqrController,
    cfg: SimConfig,
    plant: mujoco.MjModel | None = None,
) -> SimResult:
    """Roll out the closed loop. `plant` (default: nominal) is what physics
    integrates; the controller always uses its nominal-model quantities."""
    model = plant if plant is not None else rm.model
    data = mujoco.MjData(model)
    reset_to_stance(rm, data, base_height=ctrl.lin.height)
    ctrl.reset()
    ctrl.delay_comp_steps = (
        cfg.comp_delay_steps
        if cfg.comp_delay_steps is not None
        else cfg.sensor_delay_steps + cfg.action_delay_steps
    )
    rng = np.random.default_rng(cfg.seed)

    n_steps = int(round(cfg.duration / model.opt.timestep))
    base_body = model.body("base_link").id

    # Latency buffers. Sensor: history of (qpos, qvel, accel_x), so the
    # controller reads measurements sensor_delay_steps old. Action: commands
    # are timestamped and become active action_delay_steps after issue.
    sensor_buf: deque[tuple[np.ndarray, np.ndarray, float]] = deque(
        maxlen=cfg.sensor_delay_steps + 1
    )
    pending_cmds: deque[tuple[int, MitCommand]] = deque()
    prev_vx_h = 0.0

    active_cmd = ctrl.mit_command(data.qpos.copy(), data.qvel.copy())

    times = np.empty(n_steps)
    qpos_log = np.empty((n_steps, model.nq))
    qvel_log = np.empty((n_steps, model.nv))
    tau_log = np.empty((n_steps, 8))
    fell = False
    fall_time = None

    for i in range(n_steps):
        t = i * model.opt.timestep
        # Forward accelerometer: finite difference of heading-frame base
        # velocity (what an IMU accelerometer measures after gravity
        # removal), sampled at the physics rate.
        w, x_, y_, z_ = data.qpos[3:7]
        yaw = np.arctan2(2.0 * (w * z_ + x_ * y_), 1.0 - 2.0 * (y_ * y_ + z_ * z_))
        vx_h = np.cos(yaw) * data.qvel[0] + np.sin(yaw) * data.qvel[1]
        accel_x = (vx_h - prev_vx_h) / model.opt.timestep if i > 0 else 0.0
        prev_vx_h = vx_h
        sensor_buf.append((data.qpos.copy(), data.qvel.copy(), accel_x))

        if i % cfg.decimation == 0:
            qpos_m, qvel_m, accel_m = sensor_buf[0]
            qpos_m, qvel_m = _apply_noise(rm, cfg, rng, qpos_m, qvel_m)
            if cfg.noise_accel:
                accel_m += rng.normal(0.0, cfg.noise_accel)
            v_raw, w_raw = cfg.command_fn(t) if cfg.command_fn else (0.0, 0.0)
            v_cmd, w_cmd = ctrl.slew_command(
                v_raw, w_raw, cfg.decimation * model.opt.timestep
            )
            new_cmd = ctrl.mit_command(
                qpos_m, qvel_m, v_cmd, w_cmd,
                dt=cfg.decimation * model.opt.timestep,
                accel_x=accel_m,
            )
            pending_cmds.append((i, new_cmd))
        while pending_cmds and i - pending_cmds[0][0] >= cfg.action_delay_steps:
            active_cmd = pending_cmds.popleft()[1]

        # Board: MIT-mode PD + feedforward with true joint states, then the
        # DC-motor torque-speed clip.
        q = data.qpos[rm.joint_qpos_adr]
        v = data.qvel[rm.joint_dof_adr]
        tau = (
            active_cmd.kp * (active_cmd.q - q)
            + active_cmd.kd * (active_cmd.dq - v)
            + active_cmd.tau
        )
        tau = torque_speed_clip(tau, v)
        data.ctrl[rm.actuator_ids] = tau

        if cfg.push_fn is not None:
            force = cfg.push_fn(t)
            data.xfrc_applied[base_body, :3] = force if force is not None else 0.0

        mujoco.mj_step(model, data)

        times[i] = t
        qpos_log[i] = data.qpos
        qvel_log[i] = data.qvel
        tau_log[i] = tau

        tilt = _max_tilt(data.qpos[3:7])
        if data.qpos[2] < FALL_HEIGHT or tilt > FALL_TILT:
            fell = True
            fall_time = t
            times, qpos_log, qvel_log, tau_log = (
                times[: i + 1],
                qpos_log[: i + 1],
                qvel_log[: i + 1],
                tau_log[: i + 1],
            )
            break

    return SimResult(times, qpos_log, qvel_log, tau_log, fell, fall_time)


def _max_tilt(quat: np.ndarray) -> float:
    w, x, y, z = quat
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return max(abs(pitch), abs(roll))


def _apply_noise(rm, cfg: SimConfig, rng, qpos: np.ndarray, qvel: np.ndarray):
    if not (
        cfg.noise_joint_pos
        or cfg.noise_joint_vel
        or cfg.noise_gyro
        or cfg.noise_orientation
        or cfg.noise_base_vel
    ):
        return qpos, qvel
    qpos = qpos.copy()
    qvel = qvel.copy()
    qpos[rm.joint_qpos_adr] += rng.normal(0.0, cfg.noise_joint_pos, 8)
    qvel[rm.joint_dof_adr] += rng.normal(0.0, cfg.noise_joint_vel, 8)
    qvel[3:6] += rng.normal(0.0, cfg.noise_gyro, 3)
    qvel[0:3] += rng.normal(0.0, cfg.noise_base_vel, 3)
    if cfg.noise_orientation:
        tang = rng.normal(0.0, cfg.noise_orientation, 3)
        dq = np.empty(4)
        mujoco.mju_axisAngle2Quat(dq, tang / (np.linalg.norm(tang) + 1e-12),
                                  np.linalg.norm(tang))
        out = np.empty(4)
        mujoco.mju_mulQuat(out, qpos[3:7], dq)
        qpos[3:7] = out
    return qpos, qvel


if __name__ == "__main__":
    from lqr.linearize import linearize
    from lqr.model import build_model

    rm = build_model()
    lin = linearize(rm)
    ctrl = LqrController(rm, lin)
    print("K shape", ctrl.K.shape, "half_track", round(ctrl.half_track, 4))

    res = run(rm, ctrl, SimConfig(duration=10.0))
    print(
        f"balance 10s: fell={res.fell} "
        f"|pitch|max={np.abs(res.pitch()).max():.4f} rad "
        f"height final={res.base_height[-1]:.4f} "
        f"drift={np.hypot(res.qpos[-1,0], res.qpos[-1,1]):.3f} m"
    )

    res = run(
        rm, ctrl,
        SimConfig(duration=12.0, command_fn=lambda t: (0.8, 0.0) if 2 < t < 8 else (0.0, 0.0)),
    )
    vel_win = res.forward_vel()[(res.time > 4) & (res.time < 8)]
    print(
        f"vel cmd 0.8: fell={res.fell} mean v={vel_win.mean():.3f} "
        f"(target 0.8), final v={res.forward_vel()[-1]:.3f}"
    )

    res = run(
        rm, ctrl,
        SimConfig(duration=12.0, command_fn=lambda t: (0.0, 1.0) if 2 < t < 8 else (0.0, 0.0)),
    )
    yaw_win = res.yaw_rate()[(res.time > 4) & (res.time < 8)]
    print(f"yaw cmd 1.0: fell={res.fell} mean wz={yaw_win.mean():.3f} (target 1.0)")
