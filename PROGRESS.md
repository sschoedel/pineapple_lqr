# LQR controller — progress

Goal: full-state LQR (legs included) for Pineapple V3 armless, balancing +
forward-velocity + yaw-rate tracking, verified in CPU MuJoCo with MIT-mode
actuator emulation, robust to pushes / noise / latency / ±15% model error.
Yaw-rate envelope (Sam, 2026-07-15): **±2 rad/s** sustained, matching the
RL teleop max_ang_vel. Suite must hold the controller to that bar.

## Milestones

- [x] **M1 — mine deploy + mjlab for actuator facts** → see NOTES.md.
      Key: 500 Hz MIT-mode loop (q,dq,kp,kd,tau per motor), 50 Hz policy rate,
      hw gains kp=[40,25,25,0] kd=[1,.5,.5,.3], torque-speed clipping with
      effort limits [27,27,40,11] Nm, 0–15 ms latency modeled in training.
- [x] **M2 — linearize + LQR synthesis + closed-loop sim** (iteration 1)
      `lqr/model.py` (XML + floor + torque actuators via MjSpec, torque-speed
      clip), `lqr/linearize.py` (equilibrium via bisection on base z-residual
      → h=0.380714, u_eq matches mjlab stance torques exactly; 23-state
      reduction, fully controllable, one unstable mode |λ|=1.021),
      `lqr/controller.py` (LQR + MIT decomposition + command slew limiting),
      `lqr/simulate.py` (closed-loop harness w/ noise+latency+push hooks).
      Nominal results @500 Hz: balance 10 s |pitch|<1e-4 rad, drift 6 mm;
      v=0.8 m/s tracks 0.802; yaw 1.0 rad/s tracks 0.962.
- [x] **M3 — robustness suite** (final: tests/test_robustness.py, 37 cases;
      hw-spec noise tiers, latency ladder to 30 ms, ±15% mass, 1 cm COM,
      friction 0.5-1.5x, pushes 25 N fwd / 8 N lat, tracking accuracy)
- [x] **M4 — iterate until suite passes** (iterations 3-7; final result:
      36 passed + 1 strict-xfail = the documented yaw+latency limitation)
- [x] **M5 — videos (videos/*.mp4, 5 scenarios) + README.md with results
      tables, envelope, limitations, and hardware guidance**
- [x] **M6 — hardware runner** (`deploy_lqr.py` + `config/deploy_lqr.yaml`):
      DDS rt/lowcmd + rt/lowstate at 500 Hz mirroring mjlab_deploy.py's
      transport, zero imports from pineapple_rl_deploy (RL path untouched —
      verified `git status` clean there). Pure-logic core `LqrRuntime`
      (sensor assembly from IMU/encoders only incl. gravity-compensated
      forward accel, damp/stand/balance/sit modes, tilt trip to damp,
      teleop clamps) is sim-to-sim tested end to end
      (tests/test_deploy_runner.py, 8 tests: balance, velocity/yaw
      tracking, push recovery, stand ramp, trip, damp, clamps). Full suite
      now 44 passed + 1 xfail. Only the ~80-line DDS wrapper is
      hardware-untested.

## Status (after iteration 2, 2026-07-15)

M3 suite written (`tests/test_robustness.py`, `lqr/perturb.py`); most
scenarios pass but the **sustained-yaw regime fails** and M4 iteration is in
progress. Full suite not yet green.

### The sustained-yaw problem (evidence trail)

Balance/velocity/push/noise/latency/mismatch all fine WITHOUT yaw. But:

1. Sustained yaw (1 rad/s, 7 s) in clean sim → robot falls ~3 s AFTER the
   maneuver. Mechanism: cylinder-width wheel contact scrubs during turning
   and *stiction holds the legs wound-up* (0.1–0.14 rad errors, board PD
   balanced by contact friction). The LQR leg-pos→wheel coupling gains
   (~55–100 Nm/rad, K[wheel, thigh/calf]) read this as impending fall and
   command wheel differential → yaw runaway (wz 0.04→0.08→1.1 over 2 s).
2. The coupling is insensitive to Q/R weights (structural: legs shift COM).
   Zeroing K[wheels, leg-pos] destroys pitch catch (falls on lateral push
   and during v-tracking). Full symmetrization (kill differential response)
   falls DURING the maneuver — the differential term is itself stabilizing
   for the leg-twist mode. Scaling the antisymmetric part by α: only α≈0.7
   survives nominal (α∈{0.15,0.3,0.5,1.0} all fall!) — fragile sweet spot,
   and α=0.7 still fails yaw+mass15%, yaw+noise, yaw+dec5+latency.
3. Separate failure, faster: yaw + sensor noise falls at yaw ramp start
   even at 0.5 rad/s, with unmodified K. Cause: dyaw weight 1000 (needed
   for tracking against scrub friction) turns gyro/wheel-vel noise into
   large wheel torque jitter. Linear closed loop is stable (max|eig|
   0.99945) in ALL variants — stiction/scrub effects are invisible to the
   linearization; only nonlinear rollouts discriminate.

### Iteration 3 findings (2026-07-15, long session) — path to LQG

Integral action (designed via augmented DARE, `lqr_gain_integral`) FIXED the
sustained-yaw wind-up: with truth-state feedback the 23-state design tracks
yaw 1.0/2.0 rad/s to ±0.5% incl. ±15% mass error, recovers cleanly
(endroll 0.0007), no post-maneuver runaway. Yaw envelope ±2 rad/s confirmed
feasible. BUT realistic sensing then became the blocker; everything below
is about replacing the two unmeasurable truth signals (base velocity dx,
and to a lesser degree tilt) with hardware-realistic sources:

1. **Root physical constraint**: COM sits over the wheel axle → the
   pitch/velocity subsystem has a near pole-zero cancellation (unstable
   pole 4.16 rad/s, NMP zero ~5.8 rad/s). Consequence: the stabilizing
   gains on pitch (~-220 Nm/rad) and dx (~-120 Nm/(m/s)) are PINNED —
   insensitive to Q/R over 4 orders of magnitude (verified). So noise on
   those channels cannot be traded away by weight tuning, and even small
   filter lag (1st-order 25 Hz) destabilizes tracking transients.
2. **What was tried and failed** (each with full test matrices):
   a. LPF on IMU base-vel + tilt: lag destabilizes velocity ramps at any
      cutoff tried (4-25 Hz); noise passes only where lag kills tracking.
   b. Drop dx/dy/dz from the state (wheel-speed anchoring only): balance
      and yaw fine, but absolute velocity is unanchored — pitch-offset ↔
      acceleration soft direction, v runs to ~2x cmd, station-keeping
      drifts (endtilt 0.044).
   c. Keep dx as design state, feed hand-derived wheel-chain odometry at
      runtime: sign/lever-arm subtleties matter enormously; even the
      correct contact-Jacobian version (odometry err RMS 0.002 m/s!) is
      **unstable**: the estimate is a linear combo of OTHER kept states,
      so substituting it reshapes the certified gain into an uncertified
      one (falls in 0.3 s under any perturbation; clean runs mask it).
   d. 20-state design + rank-1 cost anchor on the odometry functional
      W@x (design/runtime consistent, DARE-certified): stable but the
      velocity loop oscillates ±0.5-0.9 m/s (NMP + strong proportional
      anchor); pushes marginal, combo falls.
   e. Integral-only anchoring (QV_ODO=0, QI_V sweep): slow → drift,
      fast → oscillation/falls. No good corner.
3. **Odometry map worth keeping** (`LqrController._build_odometry_map`):
   no-slip contact-Jacobian least squares at eq gives
   vx = 0.380*dpitch + [0.2115, 0.137, 0.0462]·(dthigh,dcalf,dwheel)
   summed per side — constant 1x11 row vector, hardware-implementable,
   0.002 m/s RMS accurate in nominal rolling.
4. **Noise tiers recalibrated**: original tiers were mjlab's DOMAIN
   RANDOMIZATION ranges (deliberately extreme for RL), not sensor specs.
   Hardware-realistic (DAMIAO encoders, EKF'd IMU): jpos 0.002, jvel 0.1,
   gyro 0.02, tilt 0.008 (gaussian std) — use these as the suite's
   "realistic" tier, 2x as "harsh".
5. **MIT decomposition changed**: board-layer kp/kd now the deploy stack's
   proven HW PD gains (LQR's own diagonals can be zero/negative, e.g.
   calf), tau_ff carries the LQR delta. Exact at the sensing instant,
   deploy-grade posture hold between ticks.

### Iteration 4 findings — KF over FD contact model fails; band-limited
### anchor is close but velocity mode under-damped

1. **Kalman filter over the 23-state FD model** (lqr/estimator.py, kept for
   reference): observability rank OK, filter eig 0.955, LQR eig 0.999 —
   both certified stable, yet dx̂ error hits 10 m/s within 0.19 s under
   hardware-tier noise. The FD contact dynamics are stiff and heavily
   non-normal: transient (pseudospectral) amplification through the
   predict step swamps the corrections. Feeding clipped torque into the
   predict helped estimator honesty but not stability. Dynamics-model
   estimation of base velocity through this contact model is a dead end.
2. **Current architecture (in tree)**: 20-state all-measured basis
   + augmented low-pass state x_f on the odometry functional W@x
   (VEL_ANCHOR_HZ=0.4, below the ~0.9 Hz NMP zero) + DC integrators, all
   in one DARE. Results: every noise case passes (incl yaw2+hwN), 100 Hz
   + 20 ms latency passes, yaw tracking ±2 rad/s excellent (2.008±0.03).
   Remaining failures: velocity tracking oscillates (v=0.30±0.59 for 0.8
   cmd — anchor too slow to damp the velocity mode), station-keeping
   drifts ~0.1 m/s, lateral push falls, combo falls, yaw2+mass15 falls.
3. Odometry map extended: _build_odometry_map's underlying solve yields
   all 3 base-velocity rows (M3: vx, vy, vz) — currently only vx row used.

### Iteration 5 findings — complementary filter works; big spec progress;
### delay compensation is the last structural gap

**Architecture now in tree** (controller.py): 23-state LQR, THREE
integrators (dx via complementary-filtered odometry+accel, dyaw, roll),
QI=(200,200,300), estimator = complementary vx (odometry M3[0] @ rates at
DC + IMU forward accel above 1.5 Hz; accel synthesized in the harness,
noise_accel=0.3) + static kinematic maps for z/dy/dz (M3 rows), HW-PD MIT
decompose, command slew + governor.

Passing with margin (hardware-tier noise, ±15% mass, ±1cm COM, friction
0.5-1.5x): balance (drift-free station-keeping!), v to ±1.5 m/s
(±0.02 m/s tracking), yaw ±2 rad/s in place (±0.05), curves inside the
governor envelope, fwd/back 25 N pushes, lateral to 8 N.

Key facts established:
1. Complementary vx estimate (exogenous accel breaks the gain-reshaping
   trap) fixed velocity anchoring: v.8 tracks 0.809±0.000 clean,
   0.806±0.023 under noise. Crossover value barely matters (0.5-5 Hz).
2. Roll integrator (QI_R=300) auto-trims centrifugal load in curves and
   lateral COM mismatch — hand lean references actively hurt; deleted.
3. Curved-driving envelope is real physics (centrifugal + scrub outside
   the single-point linearization): safe = |w|<=0.6 while translating,
   taper 0.3/|v|, full ±2 rad/s only near v=0. Enforced by govern_command
   (mirrors the deploy diamond constraint). Gain scheduling = future work.
   Centripetal ff on the roll integrator helped one turn direction and
   hurt the mirrored one (scrub asymmetry) — removed.
4. Lateral push limit ~8 N x 0.2 s (1.6 N*s): recovering more demands a
   steering maneuver outside a straight-stance linearization. Documented
   spec boundary, not a tuning failure.
5. **Delay margin is the last gap**: any >=2-physics-step loop delay
   (4 ms!) fails during yaw ramps or under noise. Cause: the pinned
   dx/pitch gains (pole-zero geometry) buy velocity anchoring at the cost
   of delay margin — the earlier 'dec1+sd2ad2 ok' memory was from the
   20-state design (no dx channel), which had latency margin but no
   velocity anchor. Sweeps confirm insensitivity to integrator gains,
   dyaw weight, decompose style, complementary crossover.
6. Suite-hygiene lesson recorded: several 'regressions' were actually
   configs that had never been tested (pytest -x stopped early). The M3
   suite rewrite must run every case every time (no -x) and log a full
   result matrix.

### Iteration 6 findings — the delay problem is nonlinear stick-slip at
### provably-required high gain; predictors don't help

Hard facts established this iteration (all with test evidence):
1. Smith-predictor delay compensation (incl. a board-aware variant that
   models the 500 Hz PD recomputation exactly) is a no-op-to-harmful:
   the FD model predicts wheel/leg velocities WORSE than the stale
   measurement during scrub (pred err 0.55 vs stale 0.30 mean) — the FD
   contact dynamics cannot forecast the stick-slip regime.
2. The linear delayed closed loop is stable to >=5 steps (spectral radius
   0.99945 flat) — the delay failures live entirely in the nonlinear
   contact: high-frequency torque + stick-slip → growing limit cycle.
   Signature: any >=2-physics-step loop delay + (noise or yaw ramp).
3. The high gains are PROVABLY required by this linearization:
   R x 1e6 leaves K[wheel,pitch] = -215 (minimum-energy stabilizing
   gain). eig(A) has FIVE modes at/above 1 (1.0001..1.0035 + pendulum
   1.021); discounted design (alpha=0.99) halves gains but then even
   clean 30 s balance falls at 16 s — those slow modes are real physics
   needing stabilization, weakly controllable through the wheels because
   the COM sits over the axle. Softened-contact design models and native
   50/100 Hz designs do not reduce gains either (tested).
4. RL works at 50 Hz + 0-15 ms delay on this robot with tiny effective
   gains (wheel kd 0.3) — via a nonlinear anticipatory strategy, not
   linear feedback. Low-gain LINEAR feedback is impossible here (see 3).

Current status vs spec: everything passes EXCEPT loop delay >= 2 physics
steps (4 ms). Noise alone: passes. Mismatch: passes. Envelope: documented.

### Iteration 7 — SOLVED balance/velocity latency via live board damping;
### final config passes 16/16 decisive matrix; yaw+latency documented out

The winning structural move (in tree now): emit wheel kd = 1.5 in the MIT
command but subtract only the deploy-standard 0.3 in tau_ff — the extra
1.2 Nm/(rad/s) acts as zero-delay damping at the 500 Hz board rate,
unmodeled by the DARE (folding it into the design redistributes gains and
breaks yaw — verified; kd 0.5/0.75 lose delay immunity, 2.0-3.0 gain
nothing further; kd=1.5 also passes bal+N at 30 ms).

Decisive matrix 16/16 PASS (hw noise tiers, ±15% mass, 1 cm COM, friction
0.5x, delays to 20-30 ms): balance30 drift-free; v to 1.528±0.011 @1.5cmd;
v.8 under 20 ms+noise 0.762±0.019; yaw ±2 rad/s ±0.012-0.067 incl. noise/
mass/friction; combo 0.50/0.60; pushes 25 N fwd / 8 N lat incl. 20 N with
20 ms+noise.

**Documented limitation (evidence-complete): yaw maneuvers with loop delay
>= ~10 ms fall during/after the ramp at any yaw rate >= 0.35 rad/s** (even
0.2 rad/s at 20 ms). Mechanism: wheel-scrub stiction wind-up requires
immediate high-gain reaction; gains are provably irreducible (min-energy
K[w,pitch] = -215 — five weakly-controllable unstable modes); FD-model
prediction cannot forecast the scrub regime (worse than stale data); slew
rate, integrator gains, dyaw weight, board-aware redesign all invariant
(each tested). On hardware: yaw agility depends on the real loop latency —
if DDS+CAN round trip < ~10 ms, in-place yaw to ±2 rad/s should hold;
otherwise cap yaw rate via the governor. RL sidesteps this nonlinearly
(anticipatory policy) — a gain-scheduled or MPC extension is the upgrade
path, out of scope per the architecture mandate.

### Iteration 8 — DONE. Suite green (36 passed, 1 strict xfail, 7.3 s),
### five videos rendered, README written. Loop complete.

Remaining optional work (not part of the loop mandate):
- **M6 hardware runner** (`deploy_lqr.py`, Sam requested): mirror
  mjlab_deploy.py's DDS interface with LqrController; cannot be tested
  against the robot from this machine.
- Measure the real sensor→torque loop latency on hardware; if >= 10 ms,
  cap yaw rate in the governor before first tests.
- Upgrade path for full yaw agility under latency: gain scheduling on
  (v, w) or a short-horizon MPC — out of the full-state-LQR mandate.

### Superseded plan (iteration 7): actuator-lag-aware LQR

Augment the design model with a first-order torque filter per joint
(u_act' = u_act + dt/tau * (u_cmd - u_act), tau ~ 10-20 ms; 8 new states,
known at runtime — the controller implements the same filter on its
output). The DARE then designs FOR the lag: output torque is inherently
band-limited (starves the stick-slip pump) and small real delays are
dominated by the designed lag. This also matches real DAMIAO current-loop
+ CAN dynamics. If this fails too, invoke the task's stop clause: write
the fundamental-fragility case (high-gain requirement is geometric;
delay + stick-slip breaks it; RL sidesteps nonlinearly) and present
options (gain-scheduled/nonlinear extensions, or hardware loop < 4 ms).

### Superseded plan (iteration 6): explicit delay compensation, then suite

1. Smith-predictor-style compensation in LqrController: keep a ring
   buffer of applied u-deltas; predict x forward over the known loop
   delay k = sensor+action steps with the linear (A, B) before computing
   u (short-horizon OPEN-LOOP propagation — no covariance feedback, so
   the FD-contact ill-conditioning that killed the KF does not bite).
   mjlab RL handles 0-15 ms the same way implicitly. Test robustness to
   delay mis-estimate (±1-2 steps) — hardware delay must be calibrated.
2. Re-run the decisive matrix incl. latency ladder (sd/ad 1-3 at 500 Hz,
   dec5 with lat), 2x-noise tier as stretch goal.
3. Rewrite tests/test_robustness.py to the final spec (hw noise tiers,
   accel noise, ±2 rad/s in-place yaw, governor-envelope curves, push
   spec fwd 25 N / lat 8 N, latency with compensation, mismatch grid),
   run full suite without -x.
4. M5: videos + README with results tables + envelope documentation.

### Superseded plan (iteration 5): complementary dx estimator + 23-state LQR

The 23-state truth-fed LQR+integral met EVERY spec (incl. ±2 rad/s yaw
under ±15% mass). The whole remaining problem is estimating (dx, and
weakly z, dy, dz) without feedback-coupling pathologies:
- **dx**: complementary filter — odometry (M3[0] @ measured rates) at DC
  + integrated IMU forward acceleration at high frequency (crossover
  ~1-2 Hz). The accelerometer is EXOGENOUS (not a combination of fed-back
  states), which breaks the static-substitution gain-reshaping pathology
  (iteration-3 failure c). Real LowState IMU has an accelerometer; in sim,
  synthesize accel = finite-difference of base velocity + noise
  (~0.2 m/s² std) in the harness and pass to the controller.
- **z / dy / dz**: static kinematic maps from the same contact solve
  (M3 rows; z via the position-level version of the M3[2] row). Their
  LQR gains are small (5-25) so the reshaping concern is minor.
- Keep: integral action on dx/dyaw, slew limits, HW-PD MIT decomposition,
  lean reference (sign still uncalibrated — do it once combo survives).
- Then: re-run full matrix; recalibrate suite (hw noise tiers incl. accel
  noise, ±2 rad/s yaw cases, latency ladder, mismatch); M5 videos+README.

### Superseded plan (iteration 4): LQG — Kalman observer + 23-state LQR

The RL policy works from encoders+gyro+gravity with obs HISTORY — i.e. a
learned observer. The linear-control equivalent is a steady-state Kalman
filter over the 23-state linearization with y = [joint pos (8), joint vel
(8), tilt (2), gyro (3)] (21 outputs, all real sensors; never qvel[0:3] or
base z). dx/z become KF-inferred states (model-based fusion of exactly the
signals in the odometry map, but dynamic — no static-substitution gain
reshaping, separation principle applies). Steps:
1. `lqr/estimator.py`: steady-state discrete KF gain via DARE on the dual;
   check observability of (A23, C) first. Process noise: large on velocity
   states (pushes), small elsewhere; measurement noise from tier specs.
2. Controller LQG mode: x̂ propagation (predict with applied u, correct
   with y), LQR + integral action (on x̂ dx and gyro yaw) on the estimate.
   Command interface unchanged (slew, wheel refs, lean ref TBD).
3. Re-run the full matrix: the 23-state truth design passed everything
   except noise (now filtered by KF) — expect pushes/latency/mismatch to
   carry over; verify sustained yaw with KF (scrub = unmodeled dynamics,
   the KF may misestimate during turns — watch for it).
4. Then: suite update (hw noise tiers, ±2 rad/s yaw cases, combo, latency
   ladder), M4 sign-off, videos, README.

### Superseded plan (iteration 3) — integral action, designed not bolted on

Lower dyaw to ~50 and vx weight similarly; augment the design model with 2
integrator states ∫(vx−ref) and ∫(wz−ref) (A_aug block, redo discrete LQR on
the augmented system so guarantees hold); anti-windup: clamp integrators +
reset on tilt > 0.3 rad. This replaces the high-gain proportional fix for
friction-induced steady-state error with low-gain + integral, which should
also shrink the noise amplification and the wind-up loop gain. Then rerun
the full matrix incl. yaw+noise, yaw+mass15%, yaw+dec5+lat, yaw+v combo.
If integral action alone doesn't fix the post-maneuver wind-up runaway,
combine with α≈0.7 antisym scaling and stress-test α sensitivity; document
whichever ships. Suite must add sustained-yaw × {noise, mass, latency,
friction} cases (they were the blind spot of the original suite).

## Decisions / gotchas log

- Resolve joints by name (XML is right-leg-first, deploy is left-first).
- LQR designed at 500 Hz (one physics step); decimation tested in suite.
- kp/kd sent to motors must be ≥0 (kd ≤ 5); negative/off-diagonal feedback
  goes into tau; MIT decomposition puts own-joint diag K terms in kp/kd.
- Naive mj_inverse without contact is wrong for u_eq (tries to hover base);
  bisect base height on signed z-residual instead (contact breaks ~0.3811).
- Velocity step commands must be slew-limited (V_SLEW=1 m/s², W_SLEW=3
  rad/s²) or the 11 Nm wheel budget saturates and the robot falls (0.8 m/s
  step → fall without slew).
- ~~Yaw tracking needs dyaw weight 1000~~ REVERSED (iter 2): dyaw=1000
  amplifies sensor noise into instability during yaw. Use integral action
  for the scrub-friction steady-state error instead (see plan above).
- Push body is `base_link` (fixed in simulate.run, iter 2).
- Action latency semantics: physics-step timestamped queue (fixed iter 2).

## Get-up-from-ground / go-to-ground (2026-07-17, session with Sam)

Goal: TVLQR tracking a sit->stand interpolation (Sam's spec: simple joint
interpolation, Riccati gains at each step, terminal = standing gains).

### What works (sim-verified, `lqr/getup.py`)
- **Sit pose physics**: settled SIT_ANGLES rests on wheels + rear support
  geoms (geom25/32), base z=0.022, COM 3.8 cm behind the wheel contact.
  Naive position-ramp standup flips the robot backward (rolls to
  pitch -1.3, roll pi) — active balance through the rise is mandatory.
- **Per-knot quasi-static tables** (the trajectory analog of
  find_equilibrium): balance pitch theta(s) (COM-over-contact bisection;
  no solution below s=0.1 — deep fold), equilibrium height (base-fz
  bisection; +-1 mm penetration swings joint inverse-dynamics torques by
  +-50 Nm, exactness matters), static joint feedforward u_ff(s); below
  the first balanced knot, ff blends from settled-sit inverse dynamics.
- **Nominal rollout** at the REAL 100 Hz command rate (feedforward held,
  board PD per physics step): joint servo along the path + u_ff(s) +
  segway wheel loop A*pitch+B*dpitch+C*vx+D*clamp(ivx), (40,6,16,6),
  faded out below s=0.25 (statically supported there). Gets up in 5 s,
  ends z=0.375 pitch +0.10 speed 0.33 — **captured by the stance balance
  LQR** (with IMU accel fed — the complementary filter is useless without
  it; capture tests that omit accel_x fail spuriously). Reverse rollout
  (4 s) settles onto the supports at z=0.022. Both directions verified.
- Balance-LQR capture basin probes: yaw/wheel-angle/x invariance exact;
  rolling starts to 0.3 m/s captured (accel fed).

### What failed: time-locked TVLQR playback
Riccati gains along the recorded trajectory destabilize playback even
with ORACLE state feedback (estimator removed). Bugs found and fixed on
the way (all real, none sufficient):
1. **Knot-index float truncation**: accumulated t = 6*0.025 sits ulp
   below the boundary; int(t/dt) lagged the whole schedule one knot — a
   systematic 25 ms feedforward delay. Fix: round-to-nearest.
2. **tau_ff clipped at EFFORT_LIMIT corrupted the replay**: the
   compensation of board damping at wheel speed >10 rad/s legitimately
   exceeds the wheel's 11 Nm effort limit; clip at the DAMIAO MIT-packet
   range instead (hip 28 / thigh 28 / knee 120 / wheel 20 Nm).
3. **Board-PD closure**: playback emits kp/kd the board applies between
   ticks; the Riccati must design around A_closed = A - B*Kb or the
   emitted gains fight the schedule (KP_EMIT hips +30 and KD_EMIT wheel
   1.5 toward dq_ref were both measurably destabilizing).
4. **Rate consistency**: PHYSICS_DT is 5 ms (model.py comment says
   "500 Hz" — wrong, it is 200 Hz); knots must be 10 ms (DECIM=2).
   The 500 Hz-tuned segway loop runs away backward at 100 Hz (vx -2.2,
   wheels -23 rad/s into the torque-speed clip); re-swept at 100 Hz.
With all fixes: K=0 pure replay tracks the nominal bit-exactly for
~2.9 s (>half), then diverges benignly. But ANY Riccati K tried
(composed-window 10 ms, per-step 5 ms, support-phase fade, spike clamp
+-350) chatters the joints (dq errors 10-20 rad/s, saturation) and drops
the robot around the support->wheel contact transition. |K| profiles:
healthy ~120-290 (stance-like) in the pure wheel-balance phase, 3000-9000
spikes through contact transitions — mjd_transitionFD through sticking /
lifting contacts is noise, and backward Riccati both amplifies it and
bakes time-locked wheel-speed/scoot tracking into the gains (the least
repeatable part of the maneuver).

### Paths forward (Sam to pick)
A. **Ship the working schedule controller**: time-indexed q_ref(s(t)) +
   per-knot equilibrium feedforward + segway-structured wheel feedback
   (state-based, no nominal-locking). Works robustly in sim today; is
   honestly "time-varying reference + feedforward", NOT full TVLQR gain
   matrices.
B. **Gain-scheduled LQR over the quasi-static ridge** (closest sound
   realization of the TVLQR intent): DARE at each knot's STATIC
   equilibrium (theta_bal(s), clean settled contact — the same
   linearization recipe the proven stance LQR uses), schedule K(s) along
   a slow ramp that tracks the balance-pitch ridge (robot pitches nose
   -down ~0.8 rad onto its wheels at the deep-fold end, then rises
   level). No dynamic-trajectory FD, no time-locked u_ref; each knot is
   an infinite-horizon stabilizer. More synthesis work, and the maneuver
   LOOKS different (nose-down crouch) but every point is statically
   balanced.
