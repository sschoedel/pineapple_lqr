"""The numpy-only Pi runtime must match the full controller exactly."""

import numpy as np
import pytest

import lqr_runtime as rt
from lqr.controller import LqrController
from lqr.export_runtime import export
from lqr.linearize import linearize
from lqr.model import PHYSICS_DT, STANCE_JOINT_POS, build_model


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    rm = build_model()
    lin = linearize(rm)
    full = LqrController(rm, lin)
    path = tmp_path_factory.mktemp("tables") / "lqr_tables.npz"
    export(str(path))
    table = rt.TableController(str(path))
    return rm, full, table


def random_snapshot(rng):
    q = STANCE_JOINT_POS + rng.uniform(-0.15, 0.15, 8)
    dq = rng.uniform(-1.0, 1.0, 8)
    tangent = rng.uniform(-0.08, 0.08, 3)
    angle = np.linalg.norm(tangent)
    axis = tangent / (angle + 1e-12)
    quat = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    gyro = rng.uniform(-0.5, 0.5, 3)
    accel = rng.uniform(-1.0, 1.0)
    return q, dq, quat, gyro, accel


def to_mujoco_state(rm, q, dq, quat, gyro):
    qpos = np.zeros(rm.nq)
    qvel = np.zeros(rm.nv)
    qpos[3:7] = quat
    qpos[rm.joint_qpos_adr] = q
    qvel[3:6] = gyro
    qvel[rm.joint_dof_adr] = dq
    return qpos, qvel


def test_stateful_equivalence(pair):
    rm, full, table = pair
    full.reset()
    table.reset()
    rng = np.random.default_rng(7)
    for i in range(200):
        q, dq, quat, gyro, accel = random_snapshot(rng)
        qpos, qvel = to_mujoco_state(rm, q, dq, quat, gyro)
        v_raw = 0.6 * np.sin(i / 40.0)
        w_raw = 1.0 * np.cos(i / 25.0)
        v1, w1 = full.slew_command(v_raw, w_raw, PHYSICS_DT)
        v2, w2 = table.slew(v_raw, w_raw, PHYSICS_DT)
        assert abs(v1 - v2) < 1e-12 and abs(w1 - w2) < 1e-12
        c1 = full.mit_command(qpos, qvel, v1, w1, dt=PHYSICS_DT, accel_x=accel)
        snap = rt.Snapshot(q=q, dq=dq, quat=quat, gyro=gyro, accel_x=accel)
        c2 = table.mit_command(snap, v2, w2, dt=PHYSICS_DT)
        # the Pi runtime adds a last-resort effort clip; mirror it here
        from lqr.model import EFFORT_LIMIT

        c1_tau = np.clip(c1.tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        for name in ("q", "dq", "kp", "kd", "tau"):
            a = c1_tau if name == "tau" else getattr(c1, name)
            b = getattr(c2, name)
            assert np.allclose(a, b, atol=1e-9), (
                f"step {i} field {name}: max err {np.abs(a-b).max():.3e}"
            )


def test_tilt_matches_mujoco(pair):
    rm, full, _ = pair
    rng = np.random.default_rng(3)
    for _ in range(50):
        tangent = rng.uniform(-0.4, 0.4, 3)
        angle = np.linalg.norm(tangent)
        axis = tangent / (angle + 1e-12)
        quat = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
        qpos = np.zeros(rm.nq)
        qpos[3:7] = quat
        r1, p1 = full.tilt(qpos)
        r2, p2 = rt.tilt_from_quat(quat)
        assert abs(r1 - r2) < 1e-9 and abs(p1 - p2) < 1e-9


def test_calibration_roundtrip():
    rng = np.random.default_rng(0)
    cal = rt.Calibration(
        signs=[1, -1, 1, 1, -1, 1, 1, -1],
        offsets=rng.uniform(-1, 1, 8),
    )
    q = rng.uniform(-1, 1, 8)
    dq = rng.uniform(-2, 2, 8)
    tau = rng.uniform(-5, 5, 8)
    q_m = cal.signs * q + cal.offsets
    qs, dqs, taus = cal.to_sim(q_m, cal.signs * dq, cal.signs * tau)
    assert np.allclose(qs, q) and np.allclose(dqs, dq) and np.allclose(taus, tau)
    cmd = rt.MitCommand(q=q, dq=dq, kp=np.ones(8), kd=np.ones(8), tau=tau)
    m = cal.cmd_to_motor(cmd)
    assert np.allclose(cal.signs * (m.q - cal.offsets), q)
    assert np.allclose(cal.signs * m.tau, tau)
