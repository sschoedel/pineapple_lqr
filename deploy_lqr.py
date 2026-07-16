"""Hardware runner for the Pineapple V3 LQR controller.

Standalone sibling of pineapple_rl_deploy/mjlab_deploy.py: drives the real
robot through the same Unitree-SDK DDS transport (rt/lowcmd, rt/lowstate,
500 Hz, DAMIAO MIT mode) but with the LQR controller from this repo in
place of the RL policy. It imports NOTHING from pineapple_rl_deploy and
touches none of its files, so the RL deployment path stays byte-identical.

Structure:
  LqrRuntime  — pure logic (no DDS): sensor assembly from a LowState-shaped
                snapshot, modes (damp/stand/balance/sit), safety trips, and
                the LQR tick. Fully unit/sim-tested on this machine
                (tests/test_deploy_runner.py) — the only untested code on
                hardware is the thin DDS wrapper below it.
  main()      — DDS wrapper + 500 Hz command thread + console UI, modeled
                line-for-line on the proven deploy-stack pattern.

Usage (on the robot PC):
    uv run python deploy_lqr.py config/deploy_lqr.yaml

Console: 'stand' -> ramp to stance | 'balance' -> LQR active |
'sit' -> ramp down | 'damp' -> kd-only safe mode | 'v <vx> <wz>' -> command
| 'zero' -> zero commands | 'exit'.

SAFETY: start suspended/on a stand. 'balance' engages full torque control.
The tilt trip drops to damp mode automatically past TRIP_TILT rad.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
import traceback

import numpy as np
import yaml

from lqr.controller import LqrController, MitCommand
from lqr.linearize import linearize
from lqr.model import (
    JOINT_NAMES,
    PHYSICS_DT,
    STANCE_JOINT_POS,
    build_model,
)

GRAVITY = 9.81


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by quaternion (w, x, y, z) — world = R(q) @ body."""
    w, x, y, z = quat
    q_vec = np.array([x, y, z])
    t = 2.0 * np.cross(q_vec, vec)
    return vec + w * t + np.cross(q_vec, t)


@dataclasses.dataclass
class SensorSnapshot:
    """One tick of robot sensing, in JOINT_NAMES order (left leg first)."""

    q: np.ndarray  # (8,) joint positions, rad
    dq: np.ndarray  # (8,) joint velocities, rad/s
    quat: np.ndarray  # (4,) IMU orientation, (w, x, y, z)
    gyro: np.ndarray  # (3,) body angular rates, rad/s
    accel: np.ndarray  # (3,) IMU accelerometer (body frame, includes gravity)


@dataclasses.dataclass
class RuntimeConfig:
    dt: float = PHYSICS_DT  # low-level loop period (500 Hz)
    control_decimation: int = 1  # LQR every N low-level ticks
    joint_to_motor_idx: list[int] = dataclasses.field(
        default_factory=lambda: list(range(8))
    )
    stand_seconds: float = 3.0
    stand_kp: float = 40.0
    stand_kd: float = 1.0
    sit_angles: list[float] = dataclasses.field(
        # deploy-stack sit pose, JOINT_NAMES order
        default_factory=lambda: [0.093, 1.49, -3.14, 0.0, 0.093, 1.49, -3.14, 0.0]
    )
    damp_kd: float = 1.0
    trip_tilt: float = 0.5  # rad; balance mode drops to damp past this
    # Watchdog: if no fresh lowstate for this long, force damp mode (the
    # controller must never command torques against a frozen sensor
    # snapshot). 50 ms = 25 missed ticks at 500 Hz.
    state_timeout: float = 0.05
    # Operator deadman: once a laptop console client has connected (first
    # heartbeat), losing its heartbeats for this long forces damp mode —
    # if you can't reach the robot, the robot must not keep driving. The
    # deadman arms on the FIRST heartbeat: bench use via the local stdin
    # console (robot suspended!) works without a client, but then there is
    # NO laptop-loss protection.
    deadman_timeout: float = 0.5
    udp_port: int = 43613
    udp_bind: str = "0.0.0.0"
    # Motor calibration (signs/offsets) produced by calibrate.py.
    calibration_file: str = "config/calibration.yaml"
    max_lin_vel: float = 1.0  # conservative first-hardware defaults
    max_ang_vel: float = 1.0
    delay_comp_steps: int = 0  # set after calibrating the real loop latency

    @staticmethod
    def from_yaml(path: str) -> "RuntimeConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = RuntimeConfig()
        for key, val in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg


class LqrRuntime:
    """DDS-free control core: sensing -> mode logic -> MIT command."""

    MODES = ("damp", "stand", "balance", "sit")

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.rm = build_model()
        self.lin = linearize(self.rm)
        self.ctrl = LqrController(self.rm, self.lin)
        self.ctrl.delay_comp_steps = cfg.delay_comp_steps
        self.mode = "damp"
        self._mode_t = 0.0
        self._mode_start_q = np.zeros(8)
        self._tick_count = 0
        self._last_cmd = self._damp_cmd()
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.tripped = False
        self.trip_reason: str | None = None
        self._deadman_armed = False
        self._last_heartbeat = 0.0
        # scratch qpos/qvel in MuJoCo layout for the controller
        self._qpos = np.zeros(self.rm.nq)
        self._qvel = np.zeros(self.rm.nv)

    # -- mode switching ----------------------------------------------------

    def set_mode(self, mode: str, snap: SensorSnapshot) -> None:
        assert mode in self.MODES, mode
        self.mode = mode
        self._mode_t = 0.0
        self._tick_count = 0
        self._mode_start_q = snap.q.copy()
        if mode == "balance":
            self.ctrl.reset()
        # explicit operator command re-arms after any trip
        self.tripped = False
        self.trip_reason = None
        if mode != "balance":
            self.v_cmd = 0.0
            self.w_cmd = 0.0

    def set_command(self, v: float, w: float) -> None:
        self.v_cmd = float(np.clip(v, -self.cfg.max_lin_vel, self.cfg.max_lin_vel))
        self.w_cmd = float(np.clip(w, -self.cfg.max_ang_vel, self.cfg.max_ang_vel))

    def check_watchdog(self, state_age_s: float) -> bool:
        """Force damp mode if the sensor stream has gone stale. Returns
        True when the watchdog fired (newly or already tripped by age)."""
        if state_age_s <= self.cfg.state_timeout:
            return False
        self._trip("STALE LOWSTATE (sensor stream stopped)")
        return True

    # -- operator deadman ----------------------------------------------------

    def note_heartbeat(self, t_mono: float) -> None:
        """Record an operator-console heartbeat. Arms the deadman on the
        first call."""
        self._last_heartbeat = t_mono
        self._deadman_armed = True

    def check_deadman(self, t_mono: float) -> bool:
        """Force damp if the operator link is lost. Returns True when the
        deadman is currently holding the robot in damp."""
        if not getattr(self, "_deadman_armed", False):
            return False
        if t_mono - self._last_heartbeat <= self.cfg.deadman_timeout:
            return False
        self._trip("OPERATOR LINK LOST (laptop heartbeats stopped)")
        return True

    def _trip(self, reason: str) -> None:
        if self.mode != "damp":
            self.mode = "damp"
            self.tripped = True
            self.trip_reason = reason
        self._last_cmd = self._damp_cmd()

    # -- sensing -----------------------------------------------------------

    def tilt(self, quat: np.ndarray) -> float:
        self._qpos[3:7] = quat
        roll, pitch = self.ctrl.tilt(self._qpos)
        return float(max(abs(roll), abs(pitch)))

    def forward_accel(self, snap: SensorSnapshot) -> float:
        """Gravity-compensated forward acceleration in the heading frame.

        The IMU accelerometer measures f = R^T (a_world - g_vec) with
        g_vec = (0, 0, -g); so a_world = R f + (0, 0, -g). The heading-x
        component is what the complementary velocity filter integrates.
        """
        a_world = quat_rotate(snap.quat, snap.accel)
        a_world[2] -= GRAVITY
        w, x, y, z = snap.quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(np.cos(yaw) * a_world[0] + np.sin(yaw) * a_world[1])

    # -- main tick (call at cfg.dt from the low-level loop) -----------------

    def telemetry(self) -> dict:
        """Estimator/controller state for the GUI (sim-frame units)."""
        out = {"mode": self.mode, "v_cmd": self.v_cmd, "w_cmd": self.w_cmd}
        x = getattr(self.ctrl, "last_x", None)
        if x is not None and self.mode == "balance":
            idx = self.ctrl._idx
            out.update(
                roll=float(x[idx["roll"]]),
                pitch=float(x[idx["pitch"]]),
                vx=float(x[idx["dx"]]),
                dyaw=float(x[idx["dyaw"]]),
                z=float(x[idx["z"]]),
                wheel_l=float(x[idx["dwheel_l"]]),
                wheel_r=float(x[idx["dwheel_r"]]),
                integ=[float(v) for v in self.ctrl._integ],
            )
        return out

    def tick(self, snap: SensorSnapshot) -> MitCommand:
        self._mode_t += self.cfg.dt
        if self.mode == "damp":
            self._last_cmd = self._damp_cmd()
        elif self.mode == "stand":
            self._last_cmd = self._ramp_cmd(snap, STANCE_JOINT_POS)
        elif self.mode == "sit":
            self._last_cmd = self._ramp_cmd(snap, np.array(self.cfg.sit_angles))
        elif self.mode == "balance":
            self._balance_tick(snap)
        return self._last_cmd

    def _balance_tick(self, snap: SensorSnapshot) -> None:
        if self.tilt(snap.quat) > self.cfg.trip_tilt:
            self.tripped = True
            self.mode = "damp"
            self._last_cmd = self._damp_cmd()
            return
        if self._tick_count % self.cfg.control_decimation == 0:
            self._qpos[3:7] = snap.quat
            self._qpos[self.rm.joint_qpos_adr] = snap.q
            self._qvel[3:6] = snap.gyro
            self._qvel[self.rm.joint_dof_adr] = snap.dq
            dt = self.cfg.dt * self.cfg.control_decimation
            v_cmd, w_cmd = self.ctrl.slew_command(self.v_cmd, self.w_cmd, dt)
            self._last_cmd = self.ctrl.mit_command(
                self._qpos, self._qvel, v_cmd, w_cmd,
                dt=dt, accel_x=self.forward_accel(snap),
            )
        self._tick_count += 1

    # -- non-balance command builders ---------------------------------------

    def _damp_cmd(self) -> MitCommand:
        z = np.zeros(8)
        return MitCommand(
            q=z.copy(), dq=z.copy(), kp=z.copy(),
            kd=np.full(8, self.cfg.damp_kd), tau=z.copy(),
        )

    def _ramp_cmd(self, snap: SensorSnapshot, target: np.ndarray) -> MitCommand:
        # tanh ramp from the pose at mode entry, matching the deploy stack.
        phase = float(np.tanh(self._mode_t / (self.cfg.stand_seconds / 2.5)))
        q_cmd = phase * target + (1.0 - phase) * self._mode_start_q
        kp = np.full(8, self.cfg.stand_kp)
        kd = np.full(8, self.cfg.stand_kd)
        kp[[3, 7]] = 0.0  # wheels: velocity-damp only, like the deploy stand
        kd[[3, 7]] = 0.3
        return MitCommand(
            q=q_cmd, dq=np.zeros(8), kp=kp, kd=kd, tau=np.zeros(8),
        )


TRIP_BANNER = r"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  SAFETY TRIP ACTIVE — MOTORS IN DAMP MODE (kp=0, tau=0, kd=1)   !!
!!  reason: {reason:<55s}!!
!!  robot will NOT move until an explicit mode command re-arms it  !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


class OperatorLink:
    """UDP heartbeat/command server for the laptop console client.

    Packet in (JSON): {"hb": <seq>, "cmd": ["balance"] | ["v", vx, wz] | ...}
    Packet out (JSON): status echo {mode, tripped, reason, v, w, hb}.
    DDS-free and tested in tests/test_deploy_runner.py over localhost.
    """

    def __init__(self, runtime: LqrRuntime, bind: str, port: int,
                 mode_cb=None):
        import json
        import socket

        self._json = json
        self.runtime = runtime
        self.mode_cb = mode_cb  # called with (mode_name) — needs a snapshot
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self._running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._running = False
        self.thread.join(timeout=1.0)
        self.sock.close()

    def _serve(self):
        rt = self.runtime
        while self._running:
            try:
                data, addr = self.sock.recvfrom(1024)
            except (TimeoutError, OSError):
                continue
            try:
                msg = self._json.loads(data.decode())
            except ValueError:
                continue
            if "hb" in msg:
                rt.note_heartbeat(time.monotonic())
            cmd = msg.get("cmd")
            if cmd:
                if cmd[0] in LqrRuntime.MODES and self.mode_cb is not None:
                    self.mode_cb(cmd[0])
                elif cmd[0] == "v" and len(cmd) == 3:
                    rt.set_command(float(cmd[1]), float(cmd[2]))
                elif cmd[0] == "zero":
                    rt.set_command(0.0, 0.0)
            status = {
                "mode": rt.mode,
                "tripped": rt.tripped,
                "reason": rt.trip_reason,
                "v": rt.v_cmd,
                "w": rt.w_cmd,
                "hb": msg.get("hb"),
                "telemetry": rt.telemetry(),
            }
            try:
                self.sock.sendto(self._json.dumps(status).encode(), addr)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# DDS wrapper (hardware only; requires unitree_sdk2py on the robot PC).
# ---------------------------------------------------------------------------


def main(config_path: str) -> None:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    from calibrate import load_calibration
    from lqr.calibration import apply_calibration, command_to_motor_frame

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = RuntimeConfig.from_yaml(config_path)
    runtime = LqrRuntime(cfg)
    m2j = np.asarray(cfg.joint_to_motor_idx)
    cal_signs, cal_offsets = load_calibration(cfg.calibration_file)
    print(f"calibration: signs={cal_signs.tolist()}")
    print(f"             offsets={np.round(cal_offsets, 4).tolist()}")
    if np.all(cal_offsets == 0.0):
        print("!! offsets are all zero — run calibrate.py before trusting stance")

    ChannelFactoryInitialize(raw.get("dds_domain_id", 1), raw.get("net_interface", "eth0"))

    low_cmd = unitree_go_msg_dds__LowCmd_()
    low_cmd.head[0], low_cmd.head[1] = 0xFE, 0xEF
    low_cmd.level_flag = 0xFF
    for i in range(8):
        low_cmd.motor_cmd[i].mode = 0x01

    state_lock = threading.Lock()
    latest: dict = {"snap": None, "stamp": 0.0}

    def on_lowstate(msg: LowState_):
        q_m = np.array([msg.motor_state[m2j[j]].q for j in range(8)])
        dq_m = np.array([msg.motor_state[m2j[j]].dq for j in range(8)])
        tau_m = np.array([msg.motor_state[m2j[j]].tau_est for j in range(8)])
        q, dq, _ = apply_calibration(q_m, dq_m, tau_m, cal_signs, cal_offsets)
        snap = SensorSnapshot(
            q=q, dq=dq,
            quat=np.array(msg.imu_state.quaternion, dtype=float),
            gyro=np.array(msg.imu_state.gyroscope, dtype=float),
            accel=np.array(msg.imu_state.accelerometer, dtype=float),
        )
        with state_lock:
            latest["snap"] = snap
            latest["stamp"] = time.monotonic()

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 10)
    crc = CRC()

    running = {"on": True}
    last_banner = {"t": 0.0}

    def write_cmd(cmd) -> None:
        cmd = command_to_motor_frame(cmd, cal_signs, cal_offsets)
        for j in range(8):
            mc = low_cmd.motor_cmd[m2j[j]]
            mc.q = float(cmd.q[j])
            mc.dq = float(cmd.dq[j])
            mc.kp = float(cmd.kp[j])
            mc.kd = float(cmd.kd[j])
            mc.tau = float(cmd.tau[j])
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    def cmd_loop():
        while running["on"]:
            t0 = time.perf_counter()
            now = time.monotonic()
            with state_lock:
                snap = latest["snap"]
                stamp = latest["stamp"]
            if snap is not None:
                # Safety trips, most severe first. Both park the motors in
                # damp until an explicit operator command re-arms a mode:
                #  - stale lowstate: never command against frozen sensors
                #  - operator deadman: if the laptop can't reach us, we
                #    must not keep driving with no kill switch
                tripped = runtime.check_watchdog(now - stamp)
                tripped = runtime.check_deadman(now) or tripped
                if tripped:
                    write_cmd(runtime._damp_cmd())
                    if now - last_banner["t"] > 1.0:
                        last_banner["t"] = now
                        print(TRIP_BANNER.format(reason=str(runtime.trip_reason)))
                else:
                    write_cmd(runtime.tick(snap))
            sleep = cfg.dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
        # Shutdown path (console exit, SIGHUP from a dropped SSH session,
        # SIGTERM, or an uncaught error): actively park the motors in damp
        # for 0.3 s rather than vanishing mid-torque. Note this only covers
        # process-level exits — if the Pi itself dies, the motor bridge's
        # own lowcmd-timeout behavior is the last line of defense (verify
        # it with the bridge firmware before untethered runs).
        for _ in range(60):
            write_cmd(runtime._damp_cmd())
            time.sleep(cfg.dt)

    thread = threading.Thread(target=cmd_loop, daemon=False)
    thread.start()

    def link_mode_cb(mode: str):
        with state_lock:
            snap = latest["snap"]
        if snap is None:
            print("(link) mode command ignored — no lowstate yet")
            return
        runtime.set_mode(mode, snap)
        print(f"(link) mode -> {mode}")

    link = OperatorLink(runtime, cfg.udp_bind, cfg.udp_port, mode_cb=link_mode_cb)
    link.start()

    def shutdown(signum=None, frame=None):
        if running["on"]:
            print(f"\nshutdown (signal {signum}) — parking in damp mode")
            running["on"] = False

    import signal

    signal.signal(signal.SIGHUP, shutdown)  # SSH session died
    signal.signal(signal.SIGTERM, shutdown)

    print("LQR runner up. Modes: stand | balance | sit | damp | v <vx> <wz> | zero | exit")
    print(f"Operator link: UDP {cfg.udp_bind}:{cfg.udp_port} — drive from the")
    print("laptop with `uv run python deploy_lqr_console.py <pi-address>`;")
    print(f"deadman damps the robot after {cfg.deadman_timeout}s without heartbeats.")
    print("SAFETY: robot should be suspended before first 'balance'.")
    print("Run the RUNNER inside tmux on the Pi; run the CONSOLE on the laptop.")
    print("(local stdin console below works for suspended bench tests, but has")
    print(" NO laptop-loss protection until a console client connects)")

    while running["on"]:
        try:
            line = input("LQR> ").strip().split()
            if not line:
                continue
            with state_lock:
                snap = latest["snap"]
            if line[0] in LqrRuntime.MODES:
                if snap is None:
                    print("no lowstate yet")
                    continue
                runtime.set_mode(line[0], snap)
                print(f"mode -> {line[0]}")
            elif line[0] == "v" and len(line) == 3:
                runtime.set_command(float(line[1]), float(line[2]))
                print(f"cmd v={runtime.v_cmd:.2f} w={runtime.w_cmd:.2f}")
            elif line[0] == "zero":
                runtime.set_command(0.0, 0.0)
            elif line[0] == "exit":
                break
            if runtime.tripped:
                print("!! trip fired (tilt or stale lowstate) — now in damp mode")
        except (KeyboardInterrupt, EOFError):
            break
        except Exception:
            traceback.print_exc()
            break
    running["on"] = False
    thread.join(timeout=2.0)
    link.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python deploy_lqr.py config/deploy_lqr.yaml")
        sys.exit(1)
    main(sys.argv[1])
