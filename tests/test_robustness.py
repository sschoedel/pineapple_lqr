"""Robustness suite for the Pineapple V3 LQR controller (final spec).

Every test rolls out the closed loop through the DAMIAO MIT-mode command
interface with DC-motor torque-speed clipping active, using only signals
the real robot has (encoders, tilt, gyro, forward accelerometer + wheel
odometry). Model-mismatch tests perturb only the plant; the controller
keeps its nominal-model design throughout.

Noise tiers are hardware-spec (DAMIAO encoders, EKF'd IMU), gaussian std:
joint pos 0.002 rad, joint vel 0.1 rad/s, gyro 0.02 rad/s, tilt 0.008 rad,
accel 0.3 m/s^2. Latency is in 500 Hz physics steps (1 step = 5 ms per
path). Known limitation (see PROGRESS.md iteration 6-7 evidence): yaw
maneuvers with loop delay >= ~10 ms fall — those cases are xfail(strict)
so a fix would surface as XPASS.
"""

import numpy as np
import pytest

from lqr.perturb import perturbed_plant
from lqr.simulate import SimConfig, SimResult, run

NOMINAL_HEIGHT = 0.3807

HW_NOISE = dict(
    noise_joint_pos=0.002,
    noise_joint_vel=0.1,
    noise_gyro=0.02,
    noise_orientation=0.008,
    noise_accel=0.3,
)

YAW_LATENCY_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="documented limitation: yaw maneuvers + loop delay >= ~10 ms "
    "(wheel-scrub stiction wind-up needs immediate reaction at gains that "
    "are provably irreducible; see PROGRESS.md iterations 6-7)",
)


def yawc(w, t0=2.0, t1=10.0):
    return lambda t: (0.0, w) if t0 < t < t1 else (0.0, 0.0)


def velc(v, t0=2.0, t1=10.0):
    return lambda t: (v, 0.0) if t0 < t < t1 else (0.0, 0.0)


def comboc(v, w, t0=2.0, t1=10.0):
    return lambda t: (v, w) if t0 < t < t1 else (0.0, 0.0)


def make_push(force_xyz, t_on=3.0, t_off=3.2):
    f = np.asarray(force_xyz, dtype=float)
    return lambda t: f if t_on <= t < t_off else None


def assert_upright(res: SimResult, tilt_tol=0.06, height_tol=0.04):
    assert not res.fell, f"robot fell at t={res.fall_time:.2f}s"
    assert abs(res.pitch()[-1]) < tilt_tol, f"final pitch {res.pitch()[-1]:.3f}"
    assert abs(res.roll()[-1]) < tilt_tol, f"final roll {res.roll()[-1]:.3f}"
    assert abs(res.base_height[-1] - NOMINAL_HEIGHT) < height_tol


def steady(res: SimResult, signal, t0=6.0, t1=10.0):
    mask = (res.time > t0) & (res.time < t1)
    assert mask.any(), "steady-state window empty (fell early?)"
    return signal[mask]


##
# Balance / station-keeping
##


def test_balance_long(rm, ctrl):
    res = run(rm, ctrl, SimConfig(duration=30.0))
    assert_upright(res)
    drift = np.hypot(res.qpos[-1, 0], res.qpos[-1, 1])
    assert drift < 0.05, f"drift {drift:.3f} m"
    assert np.abs(res.pitch()).max() < 0.01


def test_balance_noise(rm, ctrl):
    res = run(rm, ctrl, SimConfig(duration=10.0, seed=1, **HW_NOISE))
    assert_upright(res)
    v = steady(res, res.forward_vel())
    assert np.abs(v).mean() < 0.05


@pytest.mark.parametrize("delay", [1, 2, 3], ids=["10ms", "20ms", "30ms"])
def test_balance_noise_latency(rm, ctrl, delay):
    cfg = SimConfig(
        duration=10.0, seed=1,
        sensor_delay_steps=delay, action_delay_steps=delay, **HW_NOISE,
    )
    res = run(rm, ctrl, cfg)
    assert_upright(res)


##
# Velocity tracking
##


@pytest.mark.parametrize("v_cmd", [0.5, 1.0, 1.5, -0.5])
def test_velocity_tracking(rm, ctrl, v_cmd):
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=velc(v_cmd)))
    assert not res.fell
    v = steady(res, res.forward_vel(), 2.5 + abs(v_cmd) + 1.0, 10.0)
    assert abs(v.mean() - v_cmd) < max(0.05, 0.08 * abs(v_cmd))
    assert v.std() < 0.05, f"velocity oscillation std={v.std():.3f}"
    assert_upright(res)


def test_velocity_noise(rm, ctrl):
    cfg = SimConfig(duration=14.0, seed=2, command_fn=velc(0.8), **HW_NOISE)
    res = run(rm, ctrl, cfg)
    assert not res.fell
    v = steady(res, res.forward_vel())
    assert abs(v.mean() - 0.8) < 0.08


def test_velocity_noise_latency(rm, ctrl):
    cfg = SimConfig(
        duration=14.0, seed=2, command_fn=velc(0.8),
        sensor_delay_steps=2, action_delay_steps=2, **HW_NOISE,
    )
    res = run(rm, ctrl, cfg)
    assert not res.fell
    v = steady(res, res.forward_vel())
    assert abs(v.mean() - 0.8) < 0.1


##
# Yaw tracking (Sam's envelope: +-2 rad/s sustained, in place)
##


@pytest.mark.parametrize("w_cmd", [1.0, 2.0, -2.0])
def test_yaw_tracking(rm, ctrl, w_cmd):
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=yawc(w_cmd)))
    assert not res.fell
    wz = steady(res, res.yaw_rate())
    assert abs(wz.mean() - w_cmd) < 0.1, f"wz={wz.mean():.3f} vs {w_cmd}"
    assert_upright(res)


def test_yaw_full_rate_noise(rm, ctrl):
    cfg = SimConfig(duration=14.0, seed=3, command_fn=yawc(2.0), **HW_NOISE)
    res = run(rm, ctrl, cfg)
    assert not res.fell
    wz = steady(res, res.yaw_rate())
    assert abs(wz.mean() - 2.0) < 0.15
    assert_upright(res)


@YAW_LATENCY_XFAIL
def test_yaw_latency(rm, ctrl):
    cfg = SimConfig(
        duration=14.0, command_fn=yawc(1.0),
        sensor_delay_steps=1, action_delay_steps=1,
    )
    res = run(rm, ctrl, cfg)
    assert_upright(res)


##
# Combined curves (governor envelope: |w| <= min(0.6, 0.3/|v|) at speed)
##


@pytest.mark.parametrize("v,w", [(0.5, 0.6), (0.5, -0.6), (1.0, 0.3), (-0.5, 0.6)])
def test_curve_tracking(rm, ctrl, v, w):
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=comboc(v, w)))
    assert not res.fell
    vv = steady(res, res.forward_vel())
    wz = steady(res, res.yaw_rate())
    assert abs(vv.mean() - v) < 0.08
    assert abs(wz.mean() - w) < 0.1
    assert_upright(res)


def test_curve_noise(rm, ctrl):
    cfg = SimConfig(duration=14.0, seed=4, command_fn=comboc(0.5, 0.6), **HW_NOISE)
    res = run(rm, ctrl, cfg)
    assert not res.fell
    assert abs(steady(res, res.yaw_rate()).mean() - 0.6) < 0.12
    assert_upright(res)


def test_governor_clips_unsafe_curve(rm, ctrl):
    v, w = ctrl.govern_command(1.0, 2.0)
    assert v == 1.0 and abs(w) <= 0.3 + 1e-9
    v, w = ctrl.govern_command(0.0, 2.0)
    assert w == 2.0  # in-place turning keeps the full envelope


##
# Push disturbances (spec: 25 N x 0.2 s fore/aft, 8 N x 0.2 s lateral)
##


@pytest.mark.parametrize(
    "force", [(25, 0, 0), (-25, 0, 0), (0, 8, 0), (0, -8, 0), (18, 6, 0)],
    ids=["fwd", "back", "left", "right", "diag"],
)
def test_push_recovery(rm, ctrl, force):
    res = run(rm, ctrl, SimConfig(duration=9.0, push_fn=make_push(force)))
    assert_upright(res)
    late = res.time > 6.5
    assert np.abs(res.pitch()[late]).max() < 0.05
    assert np.abs(res.forward_vel()[late]).max() < 0.15


def test_push_noise_latency(rm, ctrl):
    cfg = SimConfig(
        duration=9.0, seed=5, push_fn=make_push((20, 0, 0)),
        sensor_delay_steps=2, action_delay_steps=2, **HW_NOISE,
    )
    res = run(rm, ctrl, cfg)
    assert_upright(res)


##
# Model mismatch (plant perturbed, controller nominal)
##


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mass_inertia_15pct(rm, ctrl, seed):
    plant = perturbed_plant(rm, mass_scale=0.15, seed=seed)
    res = run(rm, ctrl, SimConfig(duration=10.0), plant=plant)
    assert_upright(res)


def test_mass_15pct_full_yaw(rm, ctrl):
    plant = perturbed_plant(rm, mass_scale=0.15, seed=2)
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=yawc(2.0)), plant=plant)
    assert not res.fell
    assert abs(steady(res, res.yaw_rate()).mean() - 2.0) < 0.15


def test_com_offset_curve(rm, ctrl):
    plant = perturbed_plant(rm, mass_scale=0.10, com_offset=0.01, seed=3)
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=comboc(0.5, 0.6)), plant=plant)
    assert not res.fell
    assert abs(steady(res, res.forward_vel()).mean() - 0.5) < 0.08
    assert_upright(res, tilt_tol=0.08)


@pytest.mark.parametrize("fric", [0.5, 1.5], ids=["slippery", "grippy"])
def test_friction_variation(rm, ctrl, fric):
    plant = perturbed_plant(rm, friction_scale=fric, seed=6)
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=yawc(2.0)), plant=plant)
    assert not res.fell
    assert abs(steady(res, res.yaw_rate()).mean() - 2.0) < 0.15
    assert_upright(res)


def test_mismatch_velocity(rm, ctrl):
    plant = perturbed_plant(rm, mass_scale=0.15, com_offset=0.005, seed=5)
    res = run(rm, ctrl, SimConfig(duration=14.0, command_fn=velc(0.8)), plant=plant)
    assert not res.fell
    assert abs(steady(res, res.forward_vel()).mean() - 0.8) < 0.1


##
# Combined stress
##


def test_combined_stress(rm, ctrl):
    plant = perturbed_plant(rm, mass_scale=0.10, com_offset=0.005, seed=7)
    cfg = SimConfig(
        duration=14.0, seed=8, command_fn=velc(0.5),
        sensor_delay_steps=2, action_delay_steps=2,
        push_fn=make_push((12, 5, 0), t_on=6.0, t_off=6.2),
        **HW_NOISE,
    )
    res = run(rm, ctrl, cfg, plant=plant)
    assert_upright(res, tilt_tol=0.08)
    v = steady(res, res.forward_vel(), 3.5, 5.8)
    assert abs(v.mean() - 0.5) < 0.1
