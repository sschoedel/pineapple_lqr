"""Perturbed plant models for model-mismatch robustness tests.

The controller always keeps its nominal-model gains; only the *plant* that
physics integrates is perturbed. Perturbations mimic real sources of error:
per-link mass/inertia scaling (CAD error), COM offsets (cabling, batteries),
and ground/wheel friction variation.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lqr.model import PINEAPPLE_V3_XML, PHYSICS_DT, RobotModel, build_model


def perturbed_plant(
    rm: RobotModel,
    mass_scale: float = 0.0,
    com_offset: float = 0.0,
    friction_scale: float = 1.0,
    seed: int = 0,
) -> mujoco.MjModel:
    """Build a plant with randomized per-link errors.

    mass_scale: each body's mass and rotational inertia scaled by an
        independent uniform factor in [1-mass_scale, 1+mass_scale].
    com_offset: each body's COM shifted by a uniform offset in
        [-com_offset, com_offset] per axis (meters).
    friction_scale: floor sliding friction multiplied by this factor.
    """
    rng = np.random.default_rng(seed)
    model = build_model(PINEAPPLE_V3_XML, PHYSICS_DT, floor_friction=1.0).model
    for b in range(1, model.nbody):
        if model.body_mass[b] <= 0.0:
            continue
        s = 1.0 + rng.uniform(-mass_scale, mass_scale)
        model.body_mass[b] *= s
        model.body_inertia[b] *= s
        model.body_ipos[b] += rng.uniform(-com_offset, com_offset, 3)
    floor = model.geom("floor").id
    model.geom_friction[floor, 0] *= friction_scale
    return model
