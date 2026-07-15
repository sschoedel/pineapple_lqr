"""Render closed-loop rollouts to video (run with MUJOCO_GL=osmesa).

Usage: MUJOCO_GL=osmesa uv run python -m lqr.render [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

from lqr.controller import LqrController
from lqr.linearize import linearize
from lqr.model import build_model
from lqr.perturb import perturbed_plant
from lqr.simulate import SimConfig, run

FPS = 25
SIZE = (480, 640)  # (height, width)


def render_result(model, res, path: Path):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *SIZE)
    cam = mujoco.MjvCamera()
    cam.distance = 1.6
    cam.elevation = -18
    cam.azimuth = 135
    step = max(1, int(round(1.0 / (FPS * 0.005))))
    with imageio.get_writer(str(path), fps=FPS, macro_block_size=1) as w:
        for i in range(0, len(res.time), step):
            data.qpos[:] = res.qpos[i]
            data.qvel[:] = res.qvel[i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [res.qpos[i, 0], res.qpos[i, 1], 0.3]
            renderer.update_scene(data, camera=cam)
            w.append_data(renderer.render())
    renderer.close()
    print(f"wrote {path} ({len(res.time)} steps)")


def main(outdir: str = "videos"):
    out = Path(outdir)
    out.mkdir(exist_ok=True)
    rm = build_model()
    lin = linearize(rm)
    ctrl = LqrController(rm, lin)
    HWN = dict(
        noise_joint_pos=0.002, noise_joint_vel=0.1, noise_gyro=0.02,
        noise_orientation=0.008, noise_accel=0.3,
    )
    scenarios = {
        "push_25N_recovery": (
            SimConfig(duration=7.0, push_fn=lambda t: np.array([25.0, 0, 0]) if 3 <= t < 3.2 else None),
            None,
        ),
        "velocity_1ms_noise": (
            SimConfig(duration=10.0, seed=2, command_fn=lambda t: (1.0, 0.0) if 2 < t < 8 else (0.0, 0.0), **HWN),
            None,
        ),
        "yaw_2rads_inplace": (
            SimConfig(duration=10.0, command_fn=lambda t: (0.0, 2.0) if 2 < t < 8 else (0.0, 0.0)),
            None,
        ),
        "curve_governor_envelope": (
            SimConfig(duration=12.0, command_fn=lambda t: (0.5, 0.6) if 2 < t < 10 else (0.0, 0.0)),
            None,
        ),
        "stress_mass10_com5mm_noise_lat20ms_push": (
            SimConfig(
                duration=12.0, seed=8,
                command_fn=lambda t: (0.5, 0.0) if 2 < t < 10 else (0.0, 0.0),
                sensor_delay_steps=2, action_delay_steps=2,
                push_fn=lambda t: np.array([12.0, 5.0, 0]) if 6 <= t < 6.2 else None,
                **HWN,
            ),
            perturbed_plant(rm, mass_scale=0.10, com_offset=0.005, seed=7),
        ),
    }
    for name, (cfg, plant) in scenarios.items():
        res = run(rm, ctrl, cfg, plant=plant)
        status = "FELL" if res.fell else "ok"
        print(f"{name}: {status}")
        render_result(plant if plant is not None else rm.model, res, out / f"{name}.mp4")


if __name__ == "__main__":
    main(*sys.argv[1:])
