"""Get-up / go-to-ground through the exact runtime code path.

Sim -> Snapshot -> RobotRuntime(getup/getdown) at 100 Hz -> MIT command
applied by a 200 Hz board emulation, mirroring the hardware split.
"""

import numpy as np
import mujoco
import pytest

import lqr_runtime as rt
from lqr.export_runtime import export
from lqr.getup import SIT_ANGLES, settle_sit, _pitch_of
from lqr.model import PHYSICS_DT, build_model, torque_speed_clip


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rm = build_model()
    path = tmp_path_factory.mktemp("t") / "lqr_tables.npz"
    export(str(path))
    ctrl = rt.TableController(str(path))
    return rm, ctrl


def snapshot(rm, data, prev_vxh, rng=None, noise=False):
    q = data.qpos[rm.joint_qpos_adr].copy()
    dq = data.qvel[rm.joint_dof_adr].copy()
    quat = data.qpos[3:7].copy()
    gyro = data.qvel[3:6].copy()
    yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    vxh = np.cos(yaw) * data.qvel[0] + np.sin(yaw) * data.qvel[1]
    accel = (vxh - prev_vxh) / 0.01 if prev_vxh is not None else 0.0
    if noise:
        q = q + rng.normal(0, 0.002, 8)
        dq = dq + rng.normal(0, 0.1, 8)
        gyro = gyro + rng.normal(0, 0.02, 3)
        accel += rng.normal(0, 0.3)
    return rt.Snapshot(q=q, dq=dq, quat=quat, gyro=gyro, accel_x=accel), vxh


def run_mode(rm, runtime, data, mode, seconds, rng=None, noise=False):
    """Drive the runtime at 100 Hz over the sim; returns final pitch/z."""
    prev_vxh = None
    snap, prev_vxh = snapshot(rm, data, prev_vxh, rng, noise)
    runtime.set_mode(mode, snap)
    cmd = runtime.tick(snap)
    steps_per_tick = int(round(0.01 / PHYSICS_DT))
    for i in range(int(seconds / PHYSICS_DT)):
        if i % steps_per_tick == 0:
            snap, prev_vxh = snapshot(rm, data, prev_vxh, rng, noise)
            cmd = runtime.tick(snap)
        q = data.qpos[rm.joint_qpos_adr]
        dq = data.qvel[rm.joint_dof_adr]
        tau = cmd.kp * (cmd.q - q) + cmd.kd * (cmd.dq - dq) + cmd.tau
        data.ctrl[:] = torque_speed_clip(tau, dq)
        mujoco.mj_step(rm.model, data)
    assert not runtime.tripped, runtime.trip_reason


@pytest.mark.xfail(strict=True, reason="runtime-path getup tips at support "
                   "liftoff (design-side rollout succeeds — gap is in the "
                   "vx-estimate/roll-feedback shape; PROGRESS.md 2026-07-17)")
@pytest.mark.parametrize("noise", [False, True])
def test_getup_then_balance(world, noise):
    rm, ctrl = world
    data = mujoco.MjData(rm.model)
    settle_sit(rm, data)
    runtime = rt.RobotRuntime(ctrl, dt=0.01)
    runtime.note_heartbeat(0.0)
    rng = np.random.default_rng(1)
    run_mode(rm, runtime, data, "getup", 5.5, rng, noise)
    assert data.qpos[2] > 0.33, data.qpos[2]
    run_mode(rm, runtime, data, "balance", 3.0, rng, noise)
    assert abs(_pitch_of(data.qpos)) < 0.05
    assert data.qpos[2] > 0.35
    assert abs(data.qvel[0]) < 0.15


@pytest.mark.parametrize("noise", [False, True])
def test_getdown_settles(world, noise):
    rm, ctrl = world
    data = mujoco.MjData(rm.model)
    lin_qpos = None
    # start from a standing balance (settle via balance mode from stance)
    from lqr.linearize import linearize
    lin = linearize(rm)
    data.qpos[:] = lin.qpos_eq
    data.qvel[:] = 0.0
    mujoco.mj_forward(rm.model, data)
    runtime = rt.RobotRuntime(ctrl, dt=0.01)
    runtime.note_heartbeat(0.0)
    rng = np.random.default_rng(2)
    run_mode(rm, runtime, data, "getdown", 5.0, rng, noise)
    assert data.qpos[2] < 0.08, data.qpos[2]
    assert abs(_pitch_of(data.qpos)) < 0.3
    q_err = np.abs(data.qpos[rm.joint_qpos_adr] - SIT_ANGLES)[[1, 2, 5, 6]]
    assert q_err.max() < 0.4, q_err
