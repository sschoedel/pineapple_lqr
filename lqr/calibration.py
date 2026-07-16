"""Motor calibration state machines (direction + range-of-motion/zero).

Pure logic, DDS-free: each calibrator consumes (q, dq, tau_est) snapshots
at the loop rate and emits MIT-mode commands, so the whole procedure is
testable against MuJoCo (the sim's joint limits stand in for the real
mechanical stops — see tests/test_calibration.py). calibrate.py wraps
these for hardware.

Conventions:
  q_sim = sign * (q_motor - offset)        (sensor into sim frame)
  q_motor = sign * q_sim + offset          (command into motor frame)
Direction calibration finds `sign` per motor (run once per robot; signs
are wiring/assembly properties). Range calibration finds `offset` per leg
joint by driving to both mechanical stops at low torque (robot HOISTED)
and aligning the measured interval with the XML joint range. Wheels are
continuous: no offset needed (only velocity is used by the controller).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lqr.model import JOINT_NAMES, RobotModel
from lqr.controller import MitCommand

LEG_IDX = [0, 1, 2, 4, 5, 6]
WHEEL_IDX = [3, 7]


def xml_joint_ranges(rm: RobotModel) -> np.ndarray:
    """(8, 2) joint ranges in JOINT_NAMES order (wheels: [0, 0])."""
    out = np.zeros((8, 2))
    for k, name in enumerate(JOINT_NAMES):
        j = rm.model.joint(name)
        out[k] = j.range
    return out


@dataclasses.dataclass
class DirectionResult:
    signs: np.ndarray  # (8,) +1/-1: measured velocity sign under +tau
    moved: np.ndarray  # (8,) bool: joint responded at all


class DirectionCalibrator:
    """Differential ±tau pulses on one joint at a time; the sign of the
    velocity-response DIFFERENCE is the sign map, so a constant gravity
    bias on the hanging leg cancels. Non-active joints are position-held
    at their current pose. Robot hoisted.
    """

    PULSE_TAU = 2.0  # Nm
    PULSE_S = 0.35
    SETTLE_S = 0.8
    HOLD_KP = 15.0
    HOLD_KD = 1.0
    MIN_MOTION = 0.2  # rad/s peak-to-peak differential to count as "moved"

    def __init__(self, dt: float):
        self.dt = dt
        self._joint = 0
        self._phase = 0  # 0: +pulse, 1: settle, 2: -pulse, 3: settle
        self._t = 0.0
        self._peak_pos = 0.0
        self._peak_neg = 0.0
        self._hold_q: np.ndarray | None = None
        self.done = False
        self.result = DirectionResult(
            signs=np.ones(8), moved=np.zeros(8, dtype=bool)
        )

    def tick(self, q: np.ndarray, dq: np.ndarray, tau_est: np.ndarray) -> MitCommand:
        if self._hold_q is None:
            self._hold_q = q.copy()
        kp = np.full(8, self.HOLD_KP)
        kd = np.full(8, self.HOLD_KD)
        kp[WHEEL_IDX] = 0.0  # wheels are continuous: damp only
        tau = np.zeros(8)
        cmd_q = self._hold_q.copy()
        if self.done:
            return MitCommand(q=cmd_q, dq=np.zeros(8), kp=kp, kd=kd, tau=tau)
        j = self._joint
        self._t += self.dt
        pulsing = self._phase in (0, 2)
        if pulsing:
            kp[j] = 0.0
            kd[j] = 0.0
            sgn = 1.0 if self._phase == 0 else -1.0
            tau[j] = sgn * self.PULSE_TAU
            # record the largest-magnitude excursion regardless of its
            # direction — an inverted motor responds negatively to +tau
            if self._phase == 0:
                if abs(dq[j]) > abs(self._peak_pos):
                    self._peak_pos = dq[j]
            else:
                if abs(dq[j]) > abs(self._peak_neg):
                    self._peak_neg = dq[j]
            if self._t >= self.PULSE_S:
                self._phase += 1
                self._t = 0.0
        else:
            # settle back onto the hold servo
            if self._t >= self.SETTLE_S:
                if self._phase == 3:
                    diff = self._peak_pos - self._peak_neg
                    self.result.moved[j] = abs(diff) > self.MIN_MOTION
                    self.result.signs[j] = 1.0 if diff >= 0 else -1.0
                    self._joint += 1
                    self._peak_pos = 0.0
                    self._peak_neg = 0.0
                    self._hold_q = q.copy()
                    if self._joint >= 8:
                        self.done = True
                    self._phase = 0
                else:
                    self._phase = 2
                self._t = 0.0
        return MitCommand(q=cmd_q, dq=np.zeros(8), kp=kp, kd=kd, tau=tau)


@dataclasses.dataclass
class RangeResult:
    q_min: np.ndarray  # (6,) measured stop positions, motor frame, leg joints
    q_max: np.ndarray
    offsets: np.ndarray  # (8,) motor-frame zero offsets (wheels 0)
    width_error: np.ndarray  # (6,) measured width minus XML width, rad


class RangeCalibrator:
    """Find both mechanical stops of each leg joint at low torque.

    Robot HOISTED. One joint at a time: apply constant low torque toward a
    stop; a stop is declared when the joint has been (nearly) stationary
    for HOLD_S while torque is applied. Then reverse. Inputs are RAW
    motor-frame readings (no sign/offset correction). The measured
    interval is aligned to the XML range by matching midpoints (which
    cancels stop compliance symmetrically, and is invariant to sign):
        offset = mid(measured, motor frame) - sign * mid(xml range)
    """

    SWEEP_KP = 20.0  # servo carries gravity; stop torque bounded by KP*ERR
    SWEEP_KD = 1.0
    SWEEP_RATE = 0.5  # rad/s target ramp
    ERR_STOP = 0.10  # rad of tracking error at a stop (2 Nm at KP=20)
    STALL_VEL = 0.08  # rad/s: joint must be stationary for a stop
    ERR_HOLD_S = 0.25
    BACKOFF_S = 0.8
    TIMEOUT_S = 15.0
    HOLD_KP = 15.0
    HOLD_KD = 1.0

    def __init__(self, rm: RobotModel, dt: float, signs: np.ndarray | None = None):
        self.dt = dt
        self.ranges = xml_joint_ranges(rm)
        self.signs = np.ones(8) if signs is None else np.asarray(signs, float)
        self._legpos = 0  # index into LEG_IDX
        self._phase = 0  # 0: sweep down, 1: backoff, 2: sweep up, 3: backoff
        self._t = 0.0
        self._hold = 0.0
        self._target: float | None = None
        self._hold_q: np.ndarray | None = None
        self.done = False
        self.timed_out: list[str] = []
        n = len(LEG_IDX)
        self.result = RangeResult(
            q_min=np.full(n, np.nan), q_max=np.full(n, np.nan),
            offsets=np.zeros(8), width_error=np.full(n, np.nan),
        )

    def _finish_joint(self, q: np.ndarray):
        k = self._legpos
        j = LEG_IDX[k]
        lo, hi = self.ranges[j]
        mid_meas = 0.5 * (self.result.q_min[k] + self.result.q_max[k])
        mid_xml = 0.5 * (lo + hi)
        self.result.offsets[j] = mid_meas - self.signs[j] * mid_xml
        width_meas = self.result.q_max[k] - self.result.q_min[k]
        self.result.width_error[k] = width_meas - (hi - lo)
        self._legpos += 1
        self._phase = 0
        self._t = 0.0
        self._hold = 0.0
        self._target = None
        self._hold_q = q.copy()
        if self._legpos >= len(LEG_IDX):
            self.done = True

    def tick(self, q: np.ndarray, dq: np.ndarray, tau_est: np.ndarray) -> MitCommand:
        if self._hold_q is None:
            self._hold_q = q.copy()
        kp = np.full(8, self.HOLD_KP)
        kd = np.full(8, self.HOLD_KD)
        kp[WHEEL_IDX] = 0.0  # wheels: continuous, damp only
        cmd_q = self._hold_q.copy()
        z = np.zeros(8)
        if self.done:
            return MitCommand(q=cmd_q, dq=z, kp=kp, kd=kd, tau=z.copy())
        k = self._legpos
        j = LEG_IDX[k]
        self._t += self.dt
        if self._phase in (0, 2):
            direction = -1.0 if self._phase == 0 else 1.0
            if self._target is None:
                self._target = q[j]
            # slow position-servo sweep: the servo carries gravity in any
            # direction; a mechanical stop shows up as growing tracking
            # error with the contact torque bounded by SWEEP_KP * err.
            self._target += direction * self.SWEEP_RATE * self.dt
            kp[j] = self.SWEEP_KP
            kd[j] = self.SWEEP_KD
            cmd_q[j] = self._target
            err = direction * (self._target - q[j])
            # A stop = tracking error while the joint is STATIONARY.
            # Gravity sag also produces error, but the joint keeps moving
            # with the ramp, so the velocity condition rejects it.
            if err > self.ERR_STOP and abs(dq[j]) < self.STALL_VEL:
                self._hold += self.dt
            else:
                self._hold = 0.0
            if self._hold >= self.ERR_HOLD_S:
                if self._phase == 0:
                    self.result.q_min[k] = q[j]
                else:
                    self.result.q_max[k] = q[j]
                self._phase += 1
                self._t = 0.0
                self._hold = 0.0
                self._target = None
                self._hold_q = q.copy()
            elif self._t > self.TIMEOUT_S:
                self.timed_out.append(JOINT_NAMES[j])
                self._finish_joint(q)
        else:  # backoff: hold slightly off the stop, then next sweep
            if self._t >= self.BACKOFF_S:
                if self._phase == 3:
                    self._finish_joint(q)
                else:
                    self._phase = 2
                self._t = 0.0
        return MitCommand(q=cmd_q, dq=z, kp=kp, kd=kd, tau=z.copy())


def apply_calibration(
    q_motor: np.ndarray, dq_motor: np.ndarray, tau_motor: np.ndarray,
    signs: np.ndarray, offsets: np.ndarray,
):
    """Motor-frame sensing -> sim-frame (JOINT_NAMES order)."""
    q = signs * (q_motor - offsets)
    dq = signs * dq_motor
    tau = signs * tau_motor
    return q, dq, tau


def command_to_motor_frame(cmd: MitCommand, signs: np.ndarray, offsets: np.ndarray) -> MitCommand:
    """Sim-frame MIT command -> motor frame. kp/kd are invariant (they act
    on differences, and sign flips cancel: kp*(s*q_cmd - s*q) has the same
    magnitude with the torque sign handled by the tau/sign transform)."""
    return MitCommand(
        q=signs * cmd.q + offsets,
        dq=signs * cmd.dq,
        kp=cmd.kp.copy(),
        kd=cmd.kd.copy(),
        tau=signs * cmd.tau,
    )
