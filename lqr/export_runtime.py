"""Export the LQR controller to a table file for the Pi runtime.

The Pi runs a numpy-only mirror of the controller (lqr_runtime.py in the
motor_control repo) — no mujoco/scipy on the robot. This script bakes every
constant that runtime needs into lqr_tables.npz.

    uv run python -m lqr.export_runtime [out.npz]
"""

from __future__ import annotations

import sys

import numpy as np

from lqr.controller import LqrController
from lqr.linearize import linearize
from lqr.model import (
    BOARD_KP,
    EFFORT_LIMIT,
    KD_DESIGN,
    KD_EMIT,
    PHYSICS_DT,
    STANCE_JOINT_POS,
    VELOCITY_LIMIT,
    WHEEL_RADIUS,
    build_model,
)


def export(path: str = "lqr_tables.npz") -> dict:
    rm = build_model()
    lin = linearize(rm)
    c = LqrController(rm, lin)
    tables = dict(
        dt=PHYSICS_DT,
        labels=np.array(c.labels),
        K=c.K,
        Ki=c.Ki,
        int_S=c._int_S,
        u_eq=lin.u_eq,
        stance=STANCE_JOINT_POS,
        odo_Mvx=c._odo_Mvx,
        odo_M3=c._odo_M3,
        board_kp=BOARD_KP,
        kd_design=KD_DESIGN,
        kd_emit=KD_EMIT,
        effort_limit=EFFORT_LIMIT,
        velocity_limit=VELOCITY_LIMIT,
        wheel_radius=WHEEL_RADIUS,
        half_track=c.half_track,
        own_vel_col=c._own_vel_col,
        # scalars mirrored from LqrController class attrs
        v_slew=c.V_SLEW,
        w_slew=c.W_SLEW,
        vel_xover_hz=c.VEL_XOVER_HZ,
        integ_clamp=c.INTEG_CLAMP,
        integ_reset_tilt=c.INTEG_RESET_TILT,
        gov_v_inplace=c.GOV_V_INPLACE,
        gov_w_translate=c.GOV_W_TRANSLATE,
        gov_vw_max=c.GOV_VW_MAX,
    )
    np.savez(path, **tables)
    print(f"wrote {path}: K{c.K.shape}, {len(c.labels)} states")
    return tables


if __name__ == "__main__":
    export(*(sys.argv[1:] or ["lqr_tables.npz"]))
