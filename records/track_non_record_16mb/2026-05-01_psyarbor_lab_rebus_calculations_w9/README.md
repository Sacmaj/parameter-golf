# PsyArbor W9 REBUS Calculations

W9 is the lab successor to W8. It keeps the W8 community-routed assembly model + REBUS Layer-2 supervisor, then ports four additional NumPy-only calculations from `rebus-lab` to make the supervisor more principled and to add observability.

## Intent

- Track: `track_non_record_16mb`
- Status: behavior-first exploratory lab branch
- Source baseline: `../2026-05-01_psyarbor_lab_rebus_supervisor_w8`
- Priority: principled signal estimation and confidence-bounded reporting on top of the W8 supervisor

## What Changed vs W8

W9 ports five calculation primitives from `rebus-lab` (NumPy-only — no cvxpy import, no SDP) and wires four of them into the supervisor:

1. **Log-det-cov dispersion entropy** (replaces softmax H). Adapted from `supervisor.compute_ensemble_entropy`. Stores a rolling buffer of recent fine-route probability vectors and computes `0.5 * logdet(Cov)` as the supervisor's input signal. This captures cross-step *dispersion* of the routing distribution (how much routing moves over time), not just instantaneous spread.

2. **Online pinball-loss quantile thresholds** (replaces frozen-after-calibration). Adapted from `identification.pinball_sum`. Each step the thresholds h_grow and h_recover get a small subgradient update (`h ← h ± lr * (q-or-(1-q))` depending on whether the observation is above or below). Threshold drifts continuously with the actual entropy distribution; no calibration window, no degenerate-quantile collapse.

3. **Huber-clipped per-assembly error attribution** (replaces mean attribution). Adapted from `identification.huber_sum` semantics. Bounds the contribution of any single (loss, count) pair to ±M, robust to outlier-loss passes.

4. **Moving-block bootstrap CI on per-step routing metrics** (manifest observability). Adapted from `identification.moving_block_indices` / `bootstrap_take`. Reports 95% CIs on `query_lm_loss`, `cross_community_route_rate`, and `supervisor_entropy` aggregated across episodic steps.

5. **Pinball-loss kappa envelope on lm_loss vs episode_idx** (manifest observability). Adapted from `identification.fit_kappa_quantile` (cvxpy version) but reimplemented as a NumPy subgradient solver. Fits `lm_loss ≈ κ * episode_idx` as a q-th-percentile upper envelope; reports whether the most-recent value is above the envelope (off-trajectory detection).

## What Stays Exactly as W8 v1.1

- Coarse router and `community_assignment` machinery
- Dense-subgraph helpers (W6 origin)
- 12-assembly scaffold, `TRAIN_SEQ_LEN=256`, int8/zlib quantization path
- Mode FSM (CLOSED/GROW/RECOVER), sparse softmax fine routing, per-assembly health, clone-and-perturb growth (with W8 v1.1 trigger fix)
- Kill switch via `REBUS_SUPERVISOR_ENABLED=0` reproduces W8 kill-switch behavior

## Dependency Policy

W9 mirrors the rebus-lab math locally inside `train_gpt.py`. It does **not** import from `C:\Users\jjor3\Dev\rebus-lab`, does not use `cvxpy` or any solver, does not depend on rebus-lab Layer 1 identification or Layer 3 controller paths. The five new calculations are pure NumPy/PyTorch.

## W9 Knobs (additions on top of W8)

- `REBUS_USE_DISPERSION_ENTROPY` (default `1`) — when set, supervisor input signal is log-det-cov over the rolling buffer; falls back to softmax H until the buffer has ≥2 valid rows.
- `REBUS_DISPERSION_WINDOW` (default `16`) — rolling buffer size for the probability vectors.
- `REBUS_USE_ONLINE_QUANTILES` (default `1`) — when set, h_grow / h_recover update online via pinball-loss subgradient; calibration window is bypassed.
- `REBUS_PINBALL_LR` (default `0.05`) — learning rate for the online quantile update.
- `REBUS_USE_HUBER_ATTRIBUTION` (default `1`) — when set, per-assembly error EMA uses Huber-clipped delta instead of mean.
- `REBUS_HUBER_M` (default `1.0`) — Huber clip threshold.
- `REBUS_TRACK_LM_ENVELOPE` (default `1`) — when set, fit a kappa envelope at run end and report off-trajectory status.
- `REBUS_LM_ENVELOPE_Q` (default `0.85`) — quantile for the envelope.
- `REBUS_BOOTSTRAP_CI` (default `1`) — when set, compute moving-block bootstrap CIs on per-step series.
- `REBUS_BOOTSTRAP_N` (default `200`) — bootstrap replicates.
- `REBUS_BOOTSTRAP_BLOCK_LEN` (default `4`) — block length.

Each W9 feature has its own boolean kill switch — set to `0` to fall back to W8 v1.1 behavior for that piece.

## W9 Signals

Manifest `routing_stats.supervisor_stats` adds:

- `use_dispersion_entropy`, `use_online_quantiles`, `use_huber_attribution` (bools — what was active)
- `dispersion_entropy_last` (last log-det-cov value)
- `bootstrap_ci.query_lm_loss_mean_ci`, `cross_community_route_rate_ci`, `supervisor_entropy_ci` (each with `lo`, `hi`, `n_samples`)
- `lm_envelope` (kappa, q, last_query_lm_loss, envelope_at_last_episode, above_envelope_now, n_episodes)

Per-episodic-step train_metrics also gain `softmax_entropy` and `dispersion_entropy` fields (so we can see both signals).

## Self-Tests

`RUN_SELF_TESTS=1` covers everything W8 v1.1 covered, plus six new W9 cases:

- `test_pinball_quantile_convergence` — online updates converge to `np.quantile(samples, q)` within tolerance
- `test_log_det_cov_entropy` — uniform-prob buffer yields lower entropy than spread buffer; <2 rows returns 0
- `test_huber_clip_attribute` — in-band passes through, out-of-band clips to ±M, edge cases (zero count, NaN loss) safe
- `test_moving_block_bootstrap_ci` — 95% CI brackets the empirical mean; empty input safe
- `test_fit_kappa_envelope` — recovers kappa=2 for noiseless y=2x at q=0.5; q=0.85 envelope sits above the median for positively-skewed noise
- `test_w9_dispersion_supervisor_step` — integration: dispersion entropy + online quantiles drive `supervisor_step` correctly through 20 steps

## How to run

```powershell
# CPU self-tests
$env:RUN_SELF_TESTS = "1"
.\.venv\Scripts\python.exe records\track_non_record_16mb\2026-05-01_psyarbor_lab_rebus_calculations_w9\train_gpt.py

# Local end-to-end smoke (16/64, full data)
.\scripts\run_psyarbor_w9_local_smoke.ps1 -PretrainEpisodes 16 -Iterations 64 -FullData -RunId my_w9_run

# Kill-switch parity check (should reproduce W6 baseline bit-for-bit)
.\scripts\run_psyarbor_w9_local_smoke.ps1 -SupervisorOff -PretrainEpisodes 16 -Iterations 64 -FullData -RunId my_w9_kill
```
