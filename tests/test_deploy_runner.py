"""Sim-to-sim validation of the hardware runner's control core.

Drives MuJoCo through LqrRuntime's OWN code path — SensorSnapshot built
from IMU/encoder-equivalent signals only (never sim-only base states),
mode logic, safety trip, then the MIT command applied by an emulated
DAMIAO board. If these pass, the only hardware-untested code in
deploy_lqr.py is the thin DDS wrapper.
"""

import mujoco
import numpy as np
import pytest

from deploy_lqr import GRAVITY, LqrRuntime, RuntimeConfig, SensorSnapshot, quat_rotate
from lqr.model import STANCE_JOINT_POS, reset_to_stance, torque_speed_clip


@pytest.fixture(scope="module")
def runtime():
    return LqrRuntime(RuntimeConfig())


def snapshot_from_sim(rt, data, prev_qvel, dt):
    """Build the hardware-visible sensor set from the sim state."""
    rm = rt.rm
    quat = data.qpos[3:7].copy()
    # IMU accelerometer: f = R^T (a_world - g_vec), g_vec = (0,0,-g).
    a_world = (data.qvel[0:3] - prev_qvel[0:3]) / dt
    a_world[2] += GRAVITY
    w, x, y, z = quat
    quat_inv = np.array([w, -x, -y, -z])
    accel = quat_rotate(quat_inv, a_world)
    return SensorSnapshot(
        q=data.qpos[rm.joint_qpos_adr].copy(),
        dq=data.qvel[rm.joint_dof_adr].copy(),
        quat=quat,
        gyro=data.qvel[3:6].copy(),
        accel=accel,
    )


def board_apply(rt, data, cmd):
    """Emulate the DAMIAO board + torque-speed clip, then step."""
    rm = rt.rm
    q = data.qpos[rm.joint_qpos_adr]
    v = data.qvel[rm.joint_dof_adr]
    tau = cmd.kp * (cmd.q - q) + cmd.kd * (cmd.dq - v) + cmd.tau
    data.ctrl[rm.actuator_ids] = torque_speed_clip(tau, v)
    mujoco.mj_step(rm.model, data)


def rollout(rt, duration, mode="balance", v_cmd=0.0, w_cmd=0.0, push=None):
    rm = rt.rm
    data = mujoco.MjData(rm.model)
    reset_to_stance(rm, data, base_height=rt.lin.height)
    prev_qvel = data.qvel.copy()
    snap = snapshot_from_sim(rt, data, prev_qvel, rt.cfg.dt)
    rt.set_mode(mode, snap)
    rt.set_command(v_cmd, w_cmd)
    base = rm.model.body("base_link").id
    n = int(round(duration / rt.cfg.dt))
    log = {"pitch": [], "v": [], "wz": [], "z": []}
    for i in range(n):
        t = i * rt.cfg.dt
        snap = snapshot_from_sim(rt, data, prev_qvel, rt.cfg.dt)
        prev_qvel = data.qvel.copy()
        cmd = rt.tick(snap)
        if push is not None:
            f = push(t)
            data.xfrc_applied[base, :3] = f if f is not None else 0.0
        board_apply(rt, data, cmd)
        w, x, y, z = data.qpos[3:7]
        log["pitch"].append(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        log["v"].append(np.cos(yaw) * data.qvel[0] + np.sin(yaw) * data.qvel[1])
        log["wz"].append(data.qvel[5])
        log["z"].append(data.qpos[2])
        if data.qpos[2] < 0.25 or abs(log["pitch"][-1]) > 0.7:
            raise AssertionError(f"fell at t={t:.2f}s (mode={rt.mode})")
    return {k: np.asarray(vv) for k, vv in log.items()}


def test_runtime_balances_through_hardware_path(runtime):
    log = rollout(runtime, 8.0)
    assert abs(log["pitch"][-1]) < 0.02
    assert abs(log["v"][-1]) < 0.05


def test_runtime_velocity_tracking(runtime):
    log = rollout(runtime, 8.0, v_cmd=0.5)
    assert abs(log["v"][-500:].mean() - 0.5) < 0.06


def test_runtime_yaw_tracking(runtime):
    # config clamps teleop to 0.5 rad/s — full envelope is covered by the
    # controller suite; here we validate the runner path end to end.
    log = rollout(runtime, 8.0, w_cmd=0.5)
    assert abs(log["wz"][-500:].mean() - 0.5) < 0.08


def test_runtime_push_recovery(runtime):
    log = rollout(
        runtime, 8.0,
        push=lambda t: np.array([20.0, 0, 0]) if 3.0 <= t < 3.2 else None,
    )
    assert abs(log["pitch"][-1]) < 0.02
    assert abs(log["v"][-1]) < 0.1


def test_stand_ramp_reaches_stance(runtime):
    rm = runtime.rm
    data = mujoco.MjData(rm.model)
    # start from a slightly crouched pose
    reset_to_stance(rm, data, base_height=runtime.lin.height)
    crouch = STANCE_JOINT_POS.copy()
    crouch[[1, 5]] += 0.3
    crouch[[2, 6]] -= 0.3
    data.qpos[rm.joint_qpos_adr] = crouch
    data.qpos[2] -= 0.04
    mujoco.mj_forward(rm.model, data)
    prev_qvel = data.qvel.copy()
    snap = snapshot_from_sim(runtime, data, prev_qvel, runtime.cfg.dt)
    runtime.set_mode("stand", snap)
    for _ in range(int(4.0 / runtime.cfg.dt)):
        snap = snapshot_from_sim(runtime, data, prev_qvel, runtime.cfg.dt)
        prev_qvel = data.qvel.copy()
        board_apply(runtime, data, runtime.tick(snap))
    legs = [0, 1, 2, 4, 5, 6]
    err = data.qpos[rm.joint_qpos_adr][legs] - STANCE_JOINT_POS[legs]
    assert np.abs(err).max() < 0.1, f"stand ramp residual {err}"


def test_tilt_trip_drops_to_damp(runtime):
    snap = SensorSnapshot(
        q=STANCE_JOINT_POS.copy(), dq=np.zeros(8),
        quat=np.array([1.0, 0, 0, 0]), gyro=np.zeros(3),
        accel=np.array([0, 0, GRAVITY]),
    )
    runtime.set_mode("balance", snap)
    # 40 deg pitch: |tilt| > trip_tilt=0.5? 0.7 rad -> yes
    tilted = np.array([np.cos(0.35), 0.0, np.sin(0.35), 0.0])
    snap_tilted = SensorSnapshot(
        q=snap.q, dq=snap.dq, quat=tilted, gyro=snap.gyro, accel=snap.accel
    )
    cmd = runtime.tick(snap_tilted)
    assert runtime.tripped and runtime.mode == "damp"
    assert np.all(cmd.kp == 0.0) and np.all(cmd.tau == 0.0)
    assert np.all(cmd.kd == runtime.cfg.damp_kd)


def test_damp_mode_is_passive(runtime):
    snap = SensorSnapshot(
        q=STANCE_JOINT_POS.copy(), dq=np.zeros(8),
        quat=np.array([1.0, 0, 0, 0]), gyro=np.zeros(3),
        accel=np.array([0, 0, GRAVITY]),
    )
    runtime.set_mode("damp", snap)
    cmd = runtime.tick(snap)
    assert np.all(cmd.kp == 0.0) and np.all(cmd.tau == 0.0)


def test_teleop_limits_clamped(runtime):
    runtime.set_command(5.0, -9.0)
    assert runtime.v_cmd == runtime.cfg.max_lin_vel
    assert runtime.w_cmd == -runtime.cfg.max_ang_vel
    runtime.set_command(0.0, 0.0)


def test_watchdog_forces_damp_on_stale_state(runtime):
    snap = SensorSnapshot(
        q=STANCE_JOINT_POS.copy(), dq=np.zeros(8),
        quat=np.array([1.0, 0, 0, 0]), gyro=np.zeros(3),
        accel=np.array([0, 0, GRAVITY]),
    )
    runtime.set_mode("balance", snap)
    assert not runtime.check_watchdog(0.01)  # fresh: no trip
    assert runtime.mode == "balance"
    assert runtime.check_watchdog(0.2)  # stale: trip
    assert runtime.mode == "damp" and runtime.tripped
    cmd = runtime._last_cmd
    assert np.all(cmd.kp == 0.0) and np.all(cmd.tau == 0.0)
    # stays tripped while stale; once fresh again it stays parked in damp
    # until the operator re-arms a mode
    assert runtime.check_watchdog(0.2)
    assert not runtime.check_watchdog(0.001)
    assert runtime.mode == "damp"


def _bench_snap():
    return SensorSnapshot(
        q=STANCE_JOINT_POS.copy(), dq=np.zeros(8),
        quat=np.array([1.0, 0, 0, 0]), gyro=np.zeros(3),
        accel=np.array([0, 0, GRAVITY]),
    )


def test_deadman_unarmed_until_first_heartbeat(runtime):
    runtime.set_mode("balance", _bench_snap())
    runtime._deadman_armed = False
    assert not runtime.check_deadman(1e9)  # bench mode: no client ever
    assert runtime.mode == "balance"
    runtime.set_mode("damp", _bench_snap())


def test_deadman_trips_after_timeout(runtime):
    import time as _t
    runtime.set_mode("balance", _bench_snap())
    now = _t.monotonic()
    runtime.note_heartbeat(now)
    assert not runtime.check_deadman(now + 0.3)  # within 0.5 s
    assert runtime.mode == "balance"
    assert runtime.check_deadman(now + 0.6)  # heartbeats stopped
    assert runtime.mode == "damp" and runtime.tripped
    assert "OPERATOR LINK LOST" in runtime.trip_reason
    cmd = runtime._last_cmd
    assert np.all(cmd.kp == 0.0) and np.all(cmd.tau == 0.0)
    # explicit mode command re-arms
    runtime.note_heartbeat(now + 0.7)
    runtime.set_mode("balance", _bench_snap())
    assert not runtime.tripped and runtime.trip_reason is None
    runtime.set_mode("damp", _bench_snap())
    runtime._deadman_armed = False


def test_operator_link_end_to_end(runtime):
    import json
    import socket
    import time as _t
    from deploy_lqr import OperatorLink

    modes = []
    link = OperatorLink(
        runtime, "127.0.0.1", 0,
        mode_cb=lambda m: (modes.append(m), runtime.set_mode(m, _bench_snap())),
    )
    link.start()
    try:
        cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cli.settimeout(1.0)
        addr = ("127.0.0.1", link.port)

        def rpc(pkt):
            cli.sendto(json.dumps(pkt).encode(), addr)
            data, _ = cli.recvfrom(1024)
            return json.loads(data.decode())

        # heartbeat arms the deadman and echoes status
        r = rpc({"hb": 1})
        assert r["hb"] == 1 and r["mode"] == "damp"
        assert runtime._deadman_armed
        # mode + velocity commands flow through
        r = rpc({"hb": 2, "cmd": ["balance"]})
        assert modes == ["balance"] and r["mode"] == "balance"
        r = rpc({"hb": 3, "cmd": ["v", 0.3, -0.2]})
        assert abs(r["v"] - 0.3) < 1e-9 and abs(r["w"] + 0.2) < 1e-9
        # heartbeats stop -> deadman trips (checked as cmd_loop would)
        _t.sleep(0.05)
        assert not runtime.check_deadman(_t.monotonic())
        assert runtime.check_deadman(_t.monotonic() + runtime.cfg.deadman_timeout + 0.1)
        assert runtime.mode == "damp" and runtime.tripped
        r = rpc({"hb": 4})
        assert r["tripped"] and "OPERATOR LINK LOST" in r["reason"]
    finally:
        link.stop()
        runtime.set_mode("damp", _bench_snap())
        runtime._deadman_armed = False
        runtime.set_command(0.0, 0.0)
