"""Motor calibration for the Pineapple V3 (robot HOISTED, wheels free).

Usage (on the robot PC):
    uv run python calibrate.py config/deploy_lqr.yaml direction
    uv run python calibrate.py config/deploy_lqr.yaml range
    uv run python calibrate.py config/deploy_lqr.yaml both

direction  — run ONCE per robot (wiring/assembly property). Pulses each
             motor +tau then -tau; the console asks you to confirm which
             way the joint physically moved against the sim convention
             (open the MJCF in a viewer if unsure). Writes `signs` to
             config/calibration.yaml.
range      — rerun whenever encoders may have re-zeroed. Slow position-
             servo sweeps to both mechanical stops per leg joint (stop =
             tracking error while stationary; contact torque bounded by
             SWEEP_KP * ERR_STOP ~ 2 Nm), then aligns range midpoints
             with the XML ranges. Writes `offsets`, and reports the
             measured-vs-XML width error per joint as a sanity check
             (large width error = wrong signs, obstruction, or the CAD
             ranges don't match the real stops).

The resulting config/calibration.yaml is loaded by deploy_lqr.py, which
converts all sensing into sim frame and all commands into motor frame.

Sim-positive motion, for the direction prompts (right-hand rule about the
axis, robot viewed as in the MJCF):
    hips  (axis +x): positive rolls the leg plane to the robot's LEFT
    thigh (axis +y): positive swings the leg BACKWARD (pitch back)
    calf  (axis +y): positive extends the knee BACKWARD
    wheel (axis +y): positive rolls the robot FORWARD
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import yaml

from lqr.calibration import (
    LEG_IDX,
    DirectionCalibrator,
    RangeCalibrator,
    xml_joint_ranges,
)
from lqr.model import JOINT_NAMES, PHYSICS_DT, build_model

CAL_FILE = "config/calibration.yaml"

PROMPTS = {
    "hip": "leg plane rolled to the robot's LEFT",
    "thigh": "leg swung BACKWARD",
    "calf": "knee extended BACKWARD",
    "wheel": "wheel rolled FORWARD",
}


def load_calibration(path: str = CAL_FILE):
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raw = {}
    signs = np.asarray(raw.get("signs", [1.0] * 8), dtype=float)
    offsets = np.asarray(raw.get("offsets", [0.0] * 8), dtype=float)
    return signs, offsets


def save_calibration(signs, offsets, path: str = CAL_FILE):
    with open(path, "w") as f:
        yaml.safe_dump(
            {
                "signs": [float(s) for s in signs],
                "offsets": [float(o) for o in offsets],
                "joint_order": list(JOINT_NAMES),
            },
            f,
            sort_keys=False,
        )
    print(f"wrote {path}")


class DdsIo:
    """Minimal 500 Hz DDS command/state pump for the calibrators."""

    def __init__(self, config_path: str):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        self.m2j = np.asarray(raw.get("joint_to_motor_idx", list(range(8))))
        ChannelFactoryInitialize(
            raw.get("dds_domain_id", 1), raw.get("net_interface", "eth0")
        )
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head[0], self.low_cmd.head[1] = 0xFE, 0xEF
        self.low_cmd.level_flag = 0xFF
        for i in range(8):
            self.low_cmd.motor_cmd[i].mode = 0x01
        self.crc = CRC()
        self.lock = threading.Lock()
        self.state = None
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        with self.lock:
            self.state = (
                np.array([msg.motor_state[self.m2j[j]].q for j in range(8)]),
                np.array([msg.motor_state[self.m2j[j]].dq for j in range(8)]),
                np.array([msg.motor_state[self.m2j[j]].tau_est for j in range(8)]),
            )

    def read(self):
        with self.lock:
            return self.state

    def write(self, cmd):
        for j in range(8):
            mc = self.low_cmd.motor_cmd[self.m2j[j]]
            mc.q = float(cmd.q[j])
            mc.dq = float(cmd.dq[j])
            mc.kp = float(cmd.kp[j])
            mc.kd = float(cmd.kd[j])
            mc.tau = float(cmd.tau[j])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def pump(io: DdsIo, cal, on_joint_done=None):
    """Run a calibrator against hardware at 500 Hz."""
    last_joint = -1
    while not cal.done:
        t0 = time.perf_counter()
        st = io.read()
        if st is not None:
            q, dq, tau = st
            io.write(cal.tick(q, dq, tau))
            j = getattr(cal, "_joint", getattr(cal, "_legpos", -1))
            if j != last_joint and on_joint_done is not None and last_joint >= 0:
                on_joint_done(last_joint)
            last_joint = j
        dt = PHYSICS_DT - (time.perf_counter() - t0)
        if dt > 0:
            time.sleep(dt)
    if on_joint_done is not None and last_joint >= 0:
        on_joint_done(last_joint)


def run_direction(io: DdsIo, signs, offsets):
    print("\n=== DIRECTION CALIBRATION (confirm each motion visually) ===")
    print("Robot must be HOISTED with all joints free to move slightly.\n")
    new_signs = signs.copy()
    for j, name in enumerate(JOINT_NAMES):
        kind = name.split("_")[0]
        input(f"[{name}] press Enter to pulse (+ then -). Watch the joint...")
        cal = DirectionCalibrator(PHYSICS_DT)
        cal._joint = j  # calibrate this joint only
        cal.done = False

        def stop_after(jj, cal=cal, j=j):
            if jj >= j:
                cal.done = True

        pump(io, cal, on_joint_done=stop_after)
        if not cal.result.moved[j]:
            print(f"  !! {name} did not move — check motor/hoist. Sign unchanged.")
            continue
        ans = input(
            f"  During the FIRST (+) pulse, did the {kind} move so the "
            f"{PROMPTS[kind]}? [y/n] "
        ).strip().lower()
        # measured response sign is in motor frame; operator answer maps it
        # to the sim convention
        motor_response = cal.result.signs[j]
        new_signs[j] = motor_response if ans.startswith("y") else -motor_response
        print(f"  {name}: sign = {new_signs[j]:+.0f}")
    save_calibration(new_signs, offsets)
    return new_signs


def run_range(io: DdsIo, signs, offsets):
    rm = build_model()
    print("\n=== RANGE CALIBRATION (robot HOISTED) ===")
    print("Each leg joint sweeps slowly to both stops (~2 Nm max).\n")
    cal = RangeCalibrator(rm, PHYSICS_DT, signs=signs)
    pump(io, cal)
    if cal.timed_out:
        print(f"!! timed out on: {cal.timed_out} — offsets for those joints kept")
    ranges = xml_joint_ranges(rm)
    new_offsets = offsets.copy()
    for k, j in enumerate(LEG_IDX):
        if JOINT_NAMES[j] in cal.timed_out or np.isnan(cal.result.width_error[k]):
            continue
        new_offsets[j] = cal.result.offsets[j]
        lo, hi = ranges[j]
        print(
            f"{JOINT_NAMES[j]:14s} offset={new_offsets[j]:+.4f} "
            f"width err={cal.result.width_error[k]:+.3f} rad "
            f"(xml range [{lo:.2f}, {hi:.2f}])"
        )
        if abs(cal.result.width_error[k]) > 0.15:
            print("   ^^ WIDTH MISMATCH — check signs / obstructions before trusting")
    save_calibration(signs, new_offsets)
    return new_offsets


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("direction", "range", "both"):
        print("usage: python calibrate.py <config.yaml> direction|range|both")
        sys.exit(1)
    io = DdsIo(sys.argv[1])
    print("waiting for lowstate...")
    while io.read() is None:
        time.sleep(0.1)
    signs, offsets = load_calibration()
    print(f"current calibration: signs={signs.tolist()} offsets={np.round(offsets,4).tolist()}")
    if sys.argv[2] in ("direction", "both"):
        signs = run_direction(io, signs, offsets)
    if sys.argv[2] in ("range", "both"):
        run_range(io, signs, offsets)


if __name__ == "__main__":
    main()
