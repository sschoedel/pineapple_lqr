"""Pineapple V3 armless model setup for the LQR controller.

Loads the ground-truth armless MJCF (which already carries the sysid joint
armature/friction/damping as joint defaults), injects one torque actuator per
joint via MjSpec (the XML ships without an <actuator> block; mjlab injects its
own at spec-edit time), and provides the default stance keyframe.

Joint order everywhere in this project is JOINT_NAMES order (left leg first,
matching the deploy configs), resolved by name — never by XML body order,
which is right-leg-first.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np

PINEAPPLE_V3_XML = (
    Path(__file__).resolve().parent.parent.parent
    / "pineappleV3"
    / "pineappleV3_mjcf"
    / "pineappleV3_armless.xml"
)

# Deploy-config joint order (left leg then right leg, wheels last per leg).
JOINT_NAMES = (
    "hip_l_joint",
    "thigh_l_joint",
    "calf_l_joint",
    "wheel_l_joint",
    "hip_r_joint",
    "thigh_r_joint",
    "calf_r_joint",
    "wheel_r_joint",
)
LEG_IDX = np.array([0, 1, 2, 4, 5, 6])
WHEEL_IDX = np.array([3, 7])

# Default stance (pineapple_v3_constants.INIT_STATE).
STANCE_JOINT_POS = np.array([0.0, 0.847, -1.479, 0.0, 0.0, 0.847, -1.479, 0.0])
STANDING_BASE_HEIGHT = 0.38
WHEEL_RADIUS = 0.0925

# Torque-speed curve params per joint (mjlab pineapple_v3_constants; effort ==
# saturation everywhere so the curve tapers from effort at v=0 to 0 at v_lim).
EFFORT_LIMIT = np.array([27.0, 27.0, 40.0, 11.0, 27.0, 27.0, 40.0, 11.0])
VELOCITY_LIMIT = np.array([20.94, 20.94, 16.76, 25.13, 20.94, 20.94, 16.76, 25.13])

# Hardware PD gains from the deploy configs (mjlab_deploy.yaml).
HW_KP = np.array([40.0, 25.0, 25.0, 0.0, 40.0, 25.0, 25.0, 0.0])
HW_KD = np.array([1.0, 0.5, 0.5, 0.3, 1.0, 0.5, 0.5, 0.3])
# Board gains emitted by the LQR controller. kp and the kd the design
# accounts for match the deploy stack (KD_DESIGN); the EMITTED wheel kd is
# raised 0.3 -> 1.5 (DAMIAO packet range allows kd <= 5): the extra
# 1.2 Nm/(rad/s) of onboard wheel damping acts at 500 Hz with zero delay
# and is what buys the balance/velocity regimes 20-30 ms of loop-delay
# tolerance (the same structural trick that lets the RL policy run at
# 50 Hz through DDS/CAN latency). It is deliberately NOT folded into the
# design model nor subtracted from tau_ff: folding it into the DARE
# redistributes gains in a way that breaks yaw maneuvers (verified), so it
# rides as benign unmodeled damping.
BOARD_KP = HW_KP.copy()
KD_DESIGN = HW_KD.copy()
KD_EMIT = np.array([1.0, 0.5, 0.5, 1.5, 1.0, 0.5, 0.5, 1.5])
# EMITTED hip_aa kp is raised 40 -> 70 (hardware-validated 2026-07-17):
# hip stiction let the legs splay/creep under load, showing up as a
# progressive roll list that eventually tipped the robot (the roll
# integrator was ruled out by A/B with ri_gain=0). +30 Nm/rad of board-rate
# hip stiffness fixed it on hardware. Same structure as KD_EMIT: NOT folded
# into the DARE design nor subtracted from tau_ff — it rides on top as
# zero-delay stance stiffness about the hip_aa setpoint.
KP_EMIT = BOARD_KP + np.array([30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0])

# DAMIAO MIT-mode packet gain ranges.
KP_RANGE = (0.0, 500.0)
KD_RANGE = (0.0, 5.0)

PHYSICS_DT = 0.005  # 500 Hz, matches deploy low-level loop and mjlab training.


@dataclasses.dataclass(frozen=True)
class RobotModel:
    model: mujoco.MjModel
    joint_qpos_adr: np.ndarray  # qpos index per JOINT_NAMES entry
    joint_dof_adr: np.ndarray  # qvel/dof index per JOINT_NAMES entry
    actuator_ids: np.ndarray  # actuator id per JOINT_NAMES entry

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv


def build_model(
    xml_path: Path = PINEAPPLE_V3_XML,
    timestep: float = PHYSICS_DT,
    floor_friction: float = 1.0,
) -> RobotModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.option.timestep = timestep
    # The armless XML ships without a floor (mjlab adds terrain separately).
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = (0.0, 0.0, 0.05)
    floor.friction[0] = floor_friction
    for name in JOINT_NAMES:
        act = spec.add_actuator()
        act.name = f"{name}_torque"
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = 1.0
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        # ctrlrange left unset: torque-speed clipping is done controller-side.
    model = spec.compile()

    qpos_adr = np.array(
        [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    )
    dof_adr = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
    act_ids = np.array([model.actuator(f"{n}_torque").id for n in JOINT_NAMES])
    return RobotModel(model, qpos_adr, dof_adr, act_ids)


def reset_to_stance(rm: RobotModel, data: mujoco.MjData, base_height: float = STANDING_BASE_HEIGHT) -> None:
    mujoco.mj_resetData(rm.model, data)
    data.qpos[:3] = (0.0, 0.0, base_height)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[rm.joint_qpos_adr] = STANCE_JOINT_POS
    data.qvel[:] = 0.0
    mujoco.mj_forward(rm.model, data)


def torque_speed_clip(tau: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """mjlab DcMotorActuator._clip_effort with effort_limit == saturation."""
    v = np.clip(qvel, -2.0 * VELOCITY_LIMIT, 2.0 * VELOCITY_LIMIT)
    top = np.clip(EFFORT_LIMIT * (1.0 - v / VELOCITY_LIMIT), None, EFFORT_LIMIT)
    bottom = np.clip(EFFORT_LIMIT * (-1.0 - v / VELOCITY_LIMIT), -EFFORT_LIMIT, None)
    return np.clip(tau, bottom, top)


# Contact-consistent joint torques at the default stance, from mjlab
# pineapple_v3_constants.PINEAPPLE_V3_DEFAULT_JOINT_TORQUE (computed with the
# base height solved so wheel contact exactly supports the weight). A naive
# mj_inverse here is wrong: without treating contact it returns torques that
# try to hover the base. JOINT_NAMES order.
STANCE_TORQUE = np.array(
    [-4.619007, 0.441768, 5.752541, 0.0, 4.547206, 0.441332, 5.754164, 0.0]
)


def settle(
    rm: RobotModel,
    data: mujoco.MjData,
    duration: float = 3.0,
    kp: np.ndarray = HW_KP,
    kd: np.ndarray = HW_KD,
    tau_ff: np.ndarray = STANCE_TORQUE,
) -> None:
    """Step the sim under stance PD + feedforward until transients die out."""
    steps = int(round(duration / rm.model.opt.timestep))
    for _ in range(steps):
        q = data.qpos[rm.joint_qpos_adr]
        v = data.qvel[rm.joint_dof_adr]
        tau = kp * (STANCE_JOINT_POS - q) + kd * (0.0 - v) + tau_ff
        data.ctrl[rm.actuator_ids] = torque_speed_clip(tau, v)
        mujoco.mj_step(rm.model, data)
