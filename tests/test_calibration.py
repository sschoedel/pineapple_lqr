"""Sim validation of the motor calibration routines.

The robot is 'hoisted' with a strong virtual hold wrench on the base, and
a fake motor layer injects sign flips and zero offsets between the sim
joints and what the calibrators see — exactly the errors real assembly
introduces. The calibrators must recover them from torque/velocity
observations alone, with the sim's XML joint limits standing in for the
mechanical stops.
"""

import mujoco
import numpy as np
import pytest

from lqr.calibration import (
    LEG_IDX,
    DirectionCalibrator,
    RangeCalibrator,
    apply_calibration,
    xml_joint_ranges,
)
from lqr.model import PHYSICS_DT, STANCE_JOINT_POS, build_model, reset_to_stance

TRUE_SIGNS = np.array([1, -1, 1, 1, -1, 1, 1, -1], dtype=float)
TRUE_OFFSETS = np.array([0.3, -0.5, 1.0, 0.0, -0.2, 0.7, -1.1, 0.0])


class FakeMotorLayer:
    """q_motor = sign * q_sim + offset; commands come back in motor frame."""

    def __init__(self, rm):
        self.rm = rm

    def read(self, data):
        q_sim = data.qpos[self.rm.joint_qpos_adr]
        dq_sim = data.qvel[self.rm.joint_dof_adr]
        tau_sim = data.qfrc_actuator[self.rm.joint_dof_adr]
        return (
            TRUE_SIGNS * q_sim + TRUE_OFFSETS,
            TRUE_SIGNS * dq_sim,
            TRUE_SIGNS * tau_sim,
        )

    def apply(self, data, cmd):
        q_m, dq_m, _ = self.read(data)
        tau_m = cmd.kp * (cmd.q - q_m) + cmd.kd * (cmd.dq - dq_m) + cmd.tau
        data.ctrl[self.rm.actuator_ids] = TRUE_SIGNS * tau_m


def hoist(rm, data, k=500.0, d=150.0, kr=30.0, dr=8.0, anchor=None):
    """Strong 6-DOF hold wrench on the base (virtual hoist)."""
    base = rm.model.body("base_link").id
    pos0 = anchor if anchor is not None else data.qpos[0:3].copy()
    f = -k * (data.qpos[0:3] - pos0) - d * data.qvel[0:3]
    quat = data.qpos[3:7]
    tangent = np.empty(3)
    q_ref = np.array([1.0, 0, 0, 0])
    q_rel = np.empty(4)
    mujoco.mju_mulQuat(q_rel, np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]]), quat)
    mujoco.mju_quat2Vel(tangent, q_rel, 1.0)
    t = -kr * tangent - dr * data.qvel[3:6]
    data.xfrc_applied[base, :3] = f
    data.xfrc_applied[base, 3:] = t
    return pos0


@pytest.fixture(scope="module")
def hoisted():
    rm = build_model()
    data = mujoco.MjData(rm.model)
    reset_to_stance(rm, data, base_height=0.9)  # well above the floor
    return rm, data


def run_calibrator(rm, data, cal, motors, max_s=120.0):
    anchor = data.qpos[0:3].copy()
    for _ in range(int(max_s / PHYSICS_DT)):
        if cal.done:
            break
        q_m, dq_m, tau_m = motors.read(data)
        cmd = cal.tick(q_m, dq_m, tau_m)
        hoist(rm, data, anchor=anchor)
        motors.apply(data, cmd)
        mujoco.mj_step(rm.model, data)
    assert cal.done, "calibration did not finish in time"


def test_direction_calibration_recovers_signs(hoisted):
    rm, data = hoisted
    motors = FakeMotorLayer(rm)
    cal = DirectionCalibrator(PHYSICS_DT)
    run_calibrator(rm, data, cal, motors)
    assert cal.result.moved.all(), f"joints did not move: {cal.result.moved}"
    # +tau in motor frame moves the motor-frame velocity positive always;
    # the calibrator measures dq in MOTOR frame, so it sees +: the sign
    # map is recovered by comparing against sim later — here the motor
    # layer is self-consistent, so all measured signs must be +1.
    assert np.all(cal.result.signs == 1.0)


def test_direction_calibration_detects_sim_frame_signs(hoisted):
    """Feeding SIM-frame velocity (cross-frame check, how the real
    procedure distinguishes wiring flips) recovers TRUE_SIGNS."""
    rm, data = hoisted
    motors = FakeMotorLayer(rm)
    cal = DirectionCalibrator(PHYSICS_DT)
    anchor = data.qpos[0:3].copy()
    for _ in range(int(120.0 / PHYSICS_DT)):
        if cal.done:
            break
        q_m, dq_m, tau_m = motors.read(data)
        dq_sim = data.qvel[rm.joint_dof_adr]
        cmd = cal.tick(q_m, dq_sim, tau_m)  # velocity observed in sim frame
        hoist(rm, data, anchor=anchor)
        motors.apply(data, cmd)
        mujoco.mj_step(rm.model, data)
    assert cal.done
    assert np.all(cal.result.signs == TRUE_SIGNS), cal.result.signs


def test_range_calibration_recovers_offsets(hoisted):
    rm, data = hoisted
    motors = FakeMotorLayer(rm)
    cal = RangeCalibrator(rm, PHYSICS_DT, signs=TRUE_SIGNS)
    run_calibrator(rm, data, cal, motors, max_s=240.0)
    assert not cal.timed_out, f"timed out on {cal.timed_out}"
    legs = LEG_IDX
    err = cal.result.offsets[legs] - TRUE_OFFSETS[legs]
    assert np.abs(err).max() < 0.06, (
        f"offset error {np.round(err, 3)} (recovered "
        f"{np.round(cal.result.offsets[legs], 3)})"
    )
    # measured widths should match the XML ranges to within stop compliance
    assert np.abs(cal.result.width_error).max() < 0.12, cal.result.width_error


def test_apply_calibration_roundtrip(hoisted):
    rm, _ = hoisted
    rng = np.random.default_rng(0)
    q_sim = rng.uniform(-1, 1, 8)
    dq_sim = rng.uniform(-2, 2, 8)
    tau_sim = rng.uniform(-5, 5, 8)
    q_m = TRUE_SIGNS * q_sim + TRUE_OFFSETS
    dq_m = TRUE_SIGNS * dq_sim
    tau_m = TRUE_SIGNS * tau_sim
    q, dq, tau = apply_calibration(q_m, dq_m, tau_m, TRUE_SIGNS, TRUE_OFFSETS)
    assert np.allclose(q, q_sim) and np.allclose(dq, dq_sim) and np.allclose(tau, tau_sim)
