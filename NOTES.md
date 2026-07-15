# Hardware / deploy facts mined from pineapple_rl_deploy and pineapple_rl_mjlab

Sources (read 2026-07-15, none modified):
- `pineapple_rl_deploy/pineapple_rl_deploy.py`, `mjlab_deploy.py`
- `pineapple_rl_deploy/config/pineapple_v2.yaml`, `config/mjlab_deploy.yaml`
- `pineapple_rl_mjlab/src/mjlab/asset_zoo/robots/pineapple_v3/pineapple_v3_constants.py`
- `pineapple_rl_mjlab/src/mjlab/actuator/dc_actuator.py`
- `pineappleV3/pineappleV3_mjcf/pineappleV3_armless.xml`

## Command path on the real robot

- Deploy PC publishes Unitree-SDK `LowCmd_` over DDS (`rt/lowcmd`, eth0,
  domain 1); an onboard bridge forwards to the DAMIAO motors in **MIT mode**.
  Per motor per tick: `q, dq, kp, kd, tau`; the motor board closes the loop:
  `tau_applied = kp*(q_cmd - q) + kd*(dq_cmd - dq) + tau_ff`.
- **Low-level command thread runs at 500 Hz** (`simulation_dt: 0.005`);
  the RL policy updates every 4 ticks (**50 Hz**, `control_decimation: 4`).
  Between policy updates the same MIT command is re-sent → onboard PD still
  reacts at high rate. An LQR should exploit the same structure: put local
  PD in kp/kd (board-rate feedback), coupling terms in `tau` at the
  controller rate.
- State feedback available: per-motor `q, dq, tau_est` + IMU quaternion and
  gyro (`LowState_`). No base linear-velocity sensor — must be estimated
  (wheel odometry + pitch kinematics).
- Deploy code applies **no software clipping** of q/dq/kp/kd/tau (only obs
  clipping to ±100 for the policy). Clipping happens at the motor.

## Gains used on hardware (deploy configs, joint order L_hip L_thigh L_calf L_wheel R_hip R_thigh R_calf R_wheel)

- `mjlab_deploy.yaml` (V2 hardware, mjlab policy): kp = [40, 25, 25, 0, ...],
  kd = [1.0, 0.5, 0.5, 0.3, ...]. Wheels: kp=0, kd=0.3 → pure velocity
  damping toward `dq_cmd` (vel_action_scale 5.0).
- `pineapple_v2.yaml` variant used kp hip 45; stand-up uses kp 40 / kd 1.
- These match mjlab V3 actuator cfg stiffness/damping exactly (hip 40/1.0,
  thigh+calf 25/0.5, wheel 0/0.3), i.e. sim gains == hardware gains, no
  rescaling anywhere.

## Torque / velocity limits (mjlab V3 constants — replicate in LQR sim)

| joint | effort = saturation (Nm) | vel limit (rad/s) | armature | frictionloss | viscous damping |
|-------|--------------------------|-------------------|----------|--------------|-----------------|
| hip   | 27  | 20.94 | 0.004   | 0.1    | 0.02   |
| thigh | 27  | 20.94 | 0.004   | 0.1    | 0.02   |
| calf  | 40  | 16.76 | 0.0161  | 0.2    | 0.091  |
| wheel | 11  | 25.13 | 0.00176 | 0.0841 | 0.0088 |

- Wheel armature/friction/viscous numbers are **sysid-identified** (on V2,
  same wheel motors). Calf numbers likewise differ from the default
  hip/thigh values — the armless XML already contains all of these as joint
  defaults, so a plain `MjModel` load gets them for free.
- mjlab's DC-motor model (`dc_actuator.py::_clip_effort`) clips applied
  torque to a **torque–speed curve**: max torque tapers linearly from
  `saturation_effort` at v=0 to 0 at `velocity_limit`, then is clamped to
  ±effort_limit. Since effort_limit == saturation_effort for every joint
  here, the effective limit is
  `tau_max(v) = clip(sat * (1 - v/v_lim), -eff, +eff)` (and mirrored below).
  Replicate this in the LQR sim actuator emulation.
- mjlab trains with a **command delay of 0–3 physics steps (0–15 ms)** held
  fixed per episode — evidence that ~15 ms actuation latency is realistic on
  this hardware. Our latency tests should cover at least that.

## DAMIAO MIT-mode packet gotchas (hardware, not visible in this code)

- MIT-mode CAN packets quantize commands into fixed ranges configured in
  motor flash (PMAX/VMAX/TMAX); typical DAMIAO firmware encodes
  **kp ∈ [0, 500], kd ∈ [0, 5]**. Deploy gains (kp ≤ 45, kd ≤ 1) fit easily,
  but an LQR gain decomposition must keep per-joint kd ≤ 5 and kp, kd ≥ 0
  (negative gains cannot be sent — fold any negative diagonal terms into tau
  instead).
- `tau` feedforward is clipped to the motor's TMAX; use the effort limits
  above as the sim-side clip.

## Robot / stance facts

- Armless V3 robot mass 10.92 kg; wheel radius 0.0925 m; standing base
  height 0.38 m; wheel axles under the COM at default stance.
- Default stance: hips 0, thigh 0.847, calf -1.479 (both legs), solved so
  wheel contact exactly supports the weight.
- Gravity-compensation torques at stance (mjlab constants): hip ∓4.6,
  thigh 0.44, calf 5.75, wheels 0 Nm — useful as LQR feedforward `u_eq`.
- XML body order is **right leg first** (hip_r, thigh_r, calf_r, wheel_r,
  then left); deploy joint order is left-first. Resolve joints by name,
  never by index.
- The armless XML has **no `<actuator>` block** (mjlab injects actuators at
  spec-edit time) → our sim must add torque actuators via `MjSpec` and
  implement the MIT-mode PD + torque-speed clipping ourselves.

## Environment quirks

- mjlab repo venv has stale shebangs; use `uv run python -m ...` if running
  anything there. This project uses its own uv venv (CPU-only mujoco).
- Render with `MUJOCO_GL=osmesa` (EGL hangs on this machine).
- Another agent is training on this box: no GPU, single-threaded sims only.
