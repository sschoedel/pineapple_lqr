# pineapple_lqr — model-based balancing controller for Pineapple V3

A full-state LQR controller (all 6 leg joints + 2 wheels) for the arm-less
Pineapple V3 wheeled biped: balancing, forward-velocity tracking to
±1.5 m/s, and yaw-rate tracking to ±2 rad/s, verified in CPU MuJoCo
against the ground-truth `pineappleV3_armless.xml` with the real DAMIAO
MIT-mode command interface and hardware clipping replicated. No learned
components; every runtime signal is one the real robot measures.

## Quick start

Requires the ground-truth robot model repo cloned as a **sibling
directory** (the loader resolves `../pineappleV3/pineappleV3_mjcf/
pineappleV3_armless.xml`):

```bash
git clone git@github.com:sschoedel/pineappleV3.git
git clone git@github.com:sschoedel/pineapple_lqr.git
cd pineapple_lqr
uv sync
uv run python -m lqr.simulate            # nominal smoke tests
uv run python -m pytest tests/           # full robustness suite (~8 s)
MUJOCO_GL=osmesa uv run python -m lqr.render videos   # render videos
```

Suite status: **36 passed, 1 xfailed** (the xfail is the documented
yaw-under-latency limitation, strict — a fix would surface as XPASS).

## Architecture

Controller tick (`lqr/controller.py`, design in `lqr/linearize.py`):

1. **Estimator** (real sensors only — encoders, IMU tilt + gyro, forward
   accelerometer): forward velocity from a complementary filter (no-slip
   wheel odometry through frozen contact Jacobians at DC + integrated
   forward acceleration above 1.5 Hz); base height and lateral/vertical
   velocity from static kinematic maps; everything else measured directly.
   Never reads simulator-only base states.
2. **LQR** on a 23-dim reduced tangent state (drops world x/y/yaw and
   wheel angles — the invariant directions) about the contact-consistent
   stance equilibrium (found by bisecting the base residual force;
   reproduces mjlab's published stance torques exactly). Discrete DARE at
   500 Hz with three designed-in integrators: forward velocity, yaw rate,
   and roll (the roll integrator auto-trims centrifugal load in curves and
   lateral COM mismatch).
3. **Command interface**: DAMIAO MIT mode (q, dq, kp, kd, tau per motor at
   500 Hz). kp/kd carry the deploy stack's proven PD gains; tau_ff carries
   the LQR delta, exact at the sensing instant. **The emitted wheel kd is
   1.5 while the design accounts for 0.3** — the extra 1.2 Nm/(rad/s) acts
   as zero-delay damping at the board rate and is what buys the 20-30 ms
   loop-delay tolerance (same structural trick that lets the RL policy run
   at 50 Hz through DDS/CAN latency). Torque-speed clipping (11-40 Nm,
   tapering to zero at the velocity limits) is applied exactly as mjlab
   models the motors.
4. **Command governor + slew**: commands ramp at 1 m/s² / 3 rad/s², and
   yaw is capped at `min(0.6, 0.3/|v|)` rad/s while translating (full
   ±2 rad/s near v = 0) — the verified safe envelope for turning while
   driving; mirrors the deploy stack's diamond constraint.

## Verified results (hardware-tier noise unless noted)

Noise tier (gaussian std): joint pos 0.002 rad, joint vel 0.1 rad/s, gyro
0.02 rad/s, tilt 0.008 rad, accel 0.3 m/s². Latency in 5 ms physics steps
per path (sensor + actuation).

| Scenario | Result |
|---|---|
| Balance 30 s, clean | drift-free, pitch < 0.01 rad |
| Balance + noise (+ latency to 30 ms) | passes at 10/20/30 ms |
| Velocity 0.5 / 1.0 / 1.5 / −0.5 m/s | tracks within 5-8 %, std < 0.05 |
| Velocity 0.8 + noise + 20 ms latency | 0.76 ± 0.02 m/s |
| Yaw ±2 rad/s in place (+ noise) | 2.000 ± 0.07, clean recovery |
| Curves in governor envelope (+ noise, ±1 cm COM) | tracks both refs |
| Push 25 N × 0.2 s fore/aft | recovers < 2.5 s |
| Push 8 N × 0.2 s lateral | recovers (10 N is the boundary — see limits) |
| Push 20 N + noise + 20 ms latency | recovers |
| ±15 % per-link mass/inertia (3 seeds), ±2 rad/s yaw under it | passes |
| Floor friction 0.5× / 1.5× at full yaw | passes |
| Combined stress (10 % mass + 5 mm COM + noise + 20 ms + push) | passes |

Videos of five key runs: `videos/*.mp4`.

## Known limitations (measured, evidence in PROGRESS.md)

1. **Yaw maneuvers require a fast control loop.** With total loop latency
   ≥ ~10 ms, yaw ramps ≥ 0.35 rad/s fall (balance/velocity/pushes are fine
   to 20-30 ms). Mechanism: wheel-scrub stiction wind-up during turning
   demands immediate reaction at gains that are provably irreducible for
   any LQR on this model (the COM-over-axle geometry leaves five weakly
   controllable unstable modes; minimum-energy stabilizing gain is
   ~215 Nm/rad on pitch). Model-based delay prediction cannot bridge it —
   the FD contact model forecasts the scrub regime worse than stale data.
   **Hardware guidance**: measure the real sensor→torque round trip. If
   < 10 ms (plausible for local CAN at 500 Hz), full yaw agility should
   hold; otherwise cap yaw rate in the governor. A gain-scheduled or MPC
   extension is the upgrade path.
2. **Lateral pushes above ~8 N × 0.2 s** (1.6 N·s) tip the robot —
   recovering more requires a steering/stepping maneuver outside any
   straight-stance linearization.
3. **Turning while driving** is envelope-limited (centrifugal + scrub are
   v·w product terms invisible to a single-point linearization); the
   governor enforces the verified region.
4. The armless XML ships without a floor or actuators; this repo adds a
   plane (friction 1.0) and direct torque actuators at model build.

## Motor calibration (before first stand)

Encoder zeros and rotation directions must match the sim joint frames.
With the robot **hoisted** (wheels free, legs able to swing):

```bash
uv run python calibrate.py config/deploy_lqr.yaml direction  # once per robot
uv run python calibrate.py config/deploy_lqr.yaml range      # after any re-zero
```

- **direction**: pulses each motor +tau then −tau (differential, so
  gravity on the hanging leg cancels) and asks you to confirm which way
  the joint moved against the sim convention (prompts describe it; open
  the MJCF in a viewer if unsure). Writes `signs`.
- **range**: slow position-servo sweeps to both mechanical stops per leg
  joint — a stop is declared when tracking error grows while the joint is
  stationary (contact torque bounded ≈ 2 Nm), then backs off. Offsets are
  set by aligning measured range midpoints with the XML ranges; the
  per-joint width error is reported as a sanity check (large mismatch =
  wrong sign, obstruction, or CAD/real stop disagreement). Wheels are
  continuous — no zero needed (the controller only uses wheel velocity).
- Results land in `config/calibration.yaml`; `deploy_lqr.py` loads it and
  converts all sensing into sim frame and all commands into motor frame
  (and warns loudly at startup if offsets are all zero).
- The whole procedure is validated in sim with injected sign flips and
  offsets (`tests/test_calibration.py`) — recovered to < 0.06 rad.

## Viser command center (laptop GUI)

```bash
uv run python deploy_lqr_gui.py <pi-address>   # opens http://localhost:8080
```

Buttons for E-STOP (damp) / stand / balance / sit, v–w sliders with
send/zero, live state-estimator telemetry (tilt, vx, yaw rate, height,
wheel speeds, integrators), and link/trip status banners. The GUI **is**
the operator deadman heartbeat: closing it (or losing wifi) damps the
robot within 0.5 s. The plain-text `deploy_lqr_console.py` speaks the
same protocol if you prefer a terminal.

## Hardware deployment notes

- Command path: identical to `pineapple_rl_deploy` — 500 Hz `LowCmd_` DDS
  → DAMIAO MIT mode. Emit exactly the suite-verified fields: kp
  [40,25,25,0]·2, kd [1,.5,.5,**1.5**]·2 (within DAMIAO packet ranges
  kp ∈ [0,500], kd ∈ [0,5]), tau from the controller.
- The stance ramp-up ("stand" mode) and sit sequencing should be reused
  from the deploy stack; this repo only implements the balancing mode.
- The forward accelerometer must be gravity-compensated (use the EKF tilt)
  before entering the complementary filter.
- Odometry/kinematic maps are constant matrices computed at the stance
  equilibrium — see `LqrController._build_odometry_map`; they can be
  copied as literals into embedded code.
- **Hardware runner: `deploy_lqr.py`** (config in `config/deploy_lqr.yaml`):
  `uv run python deploy_lqr.py config/deploy_lqr.yaml` on the robot PC.
  Same DDS transport as the RL stack (rt/lowcmd / rt/lowstate, 500 Hz),
  console modes `stand | balance | sit | damp | v <vx> <wz>`. Imports
  nothing from pineapple_rl_deploy — the RL deployment path is untouched.
  The control core (`LqrRuntime`) is sim-to-sim tested end to end through
  the exact hardware code path (sensor snapshot from IMU/encoders only →
  mode logic → MIT command; `tests/test_deploy_runner.py`), so the only
  hardware-untested code is the ~80-line DDS wrapper. Safety: tilt trip to
  damp mode at 0.5 rad; teleop clamped to ±0.5 m/s, ±0.5 rad/s for
  shakedown (raise in the yaml after). First run: suspend the robot,
  `stand`, then `balance` with wheels just touching.

### Comms-failure behavior (laptop ↔ Pi ↔ motor bridge)

Normal operation: **runner in tmux on the Pi, console on the laptop**:

```bash
# on the Pi (inside tmux):
uv run python deploy_lqr.py config/deploy_lqr.yaml
# on the laptop:
uv run python deploy_lqr_console.py <pi-address>
```

- **Operator deadman (laptop-loss kill switch)**: the console sends UDP
  heartbeats at 10 Hz; if the runner sees none for `deadman_timeout`
  (0.5 s) it forces damp mode (kp=0, tau=0, kd=1 — the robot goes limp)
  and prints a repeating warning banner on the Pi console. The laptop
  console likewise shouts the moment it loses replies. The robot stays
  limp until you reconnect and explicitly re-arm a mode. Wifi drop, ssh
  death, laptop sleep, network partition — all covered, because it is a
  positive heartbeat, not a disconnect signal. Note: the deadman arms on
  the FIRST heartbeat, so bench use via the Pi-local stdin console (robot
  suspended!) works without a client — but has no laptop-loss protection.
- **Stale lowstate watchdog**: if no fresh `rt/lowstate` arrives for
  50 ms (`state_timeout`), the runner forces damp — it never commands
  balance torques against a frozen sensor snapshot.
- **Process exit**: SIGHUP/SIGTERM/console exit/uncaught errors all route
  through a shutdown path that actively publishes damp commands for 0.3 s
  before the process ends.
- **Not covered — verify before untethered runs**: if the Pi itself dies
  (power, kernel), nothing can send damp. The last line of defense is
  the motor bridge's own behavior when `rt/lowcmd` stops arriving; MIT-
  mode motors typically HOLD the last kp/kd/tau otherwise. Check the CSL
  bridge firmware for a lowcmd-timeout → disable, and if absent, test by
  killing the runner with the robot suspended. The RL deploy stack has
  the same exposure.

## Repo layout

```
lqr/model.py       model build (floor + torque actuators), constants,
                   torque-speed clip, stance
lqr/linearize.py   equilibrium search, FD linearization, DARE synthesis
                   (with integrator augmentation)
lqr/controller.py  estimator + LQR + governor + MIT command composition
lqr/perturb.py     perturbed plants for mismatch tests
lqr/simulate.py    closed-loop harness (noise, latency, pushes, decimation)
lqr/render.py      video rendering (MUJOCO_GL=osmesa)
lqr/estimator.py   Kalman-filter experiment (kept for reference; unused —
                   see PROGRESS.md iteration 4 for why)
tests/             robustness suite
NOTES.md           everything mined from the deploy/mjlab repos
PROGRESS.md        full design/evidence log across all iterations
```
